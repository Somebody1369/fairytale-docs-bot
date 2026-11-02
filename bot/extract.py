"""Text extraction from PDF, Word (.docx and the old .doc) and plain text files.

Some archive PDFs are scans on a birch-bark background with no text layer,
so pages without text are rendered and passed through Tesseract OCR.
"""

import io
import re
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path

import docx
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
import pytesseract
from PIL import ImageOps

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}
MIN_PAGE_TEXT = 20
MAX_OCR_PAGES = 30  # a scanned page takes a few seconds to read; a longer scan is refused rather than cut
# Pixels on the longer side, so an oversized page (a poster, A0) cannot take gigabytes; A4 and A3 still get 300 dpi.
MAX_RENDER_SIDE = 5000
# Tesseract is stopped after this, and the file counts as unreadable rather than being stored without the page.
OCR_PAGE_SECONDS = 120
# python-docx holds every part of a .docx in memory, unpacked: a 1.5 MB zip bomb took a gigabyte. Pictures in
# a .docx are packed already, so even a 20 MB file (Telegram's limit) unpacks to not much more.
MAX_DOCX_UNPACKED = 50 * 1024 * 1024

# PDFium is not thread-safe, not even for different documents (pypdfium2 docs), and PDFs are read in threads:
# the archive sync and every animal's file at once. Two PDFs read together crashed the process.
PDFIUM = threading.Lock()


class TooManyScannedPages(ValueError):
    def __init__(self, pages: int):
        super().__init__(f"{pages} scanned pages, at most {MAX_OCR_PAGES} are read")
        self.pages = pages


class UnreadableScan(ValueError):
    """Tesseract did not finish a scanned page in OCR_PAGE_SECONDS, or pages with pictures or drawings gave no text."""


class OcrFailed(UnreadableScan):
    """Tesseract itself failed on a page (a language not installed, killed for lack of memory): the scan may be fine."""


class UnsupportedFormat(Exception):
    pass


def extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)  # "Назва (1).pdf" is the same document
    return re.sub(r"[_\s]+", " ", stem).strip()


def old_word_supported() -> bool:
    """The old Word format is read by antiword (installed in the Docker image)."""
    return shutil.which("antiword") is not None


def extract_text(data: bytes, filename: str, ocr_langs: str = "ukr", ext: str | None = None) -> str:
    ext = ext or extension(filename)
    if ext == ".txt":
        return _clean(_decode_text(data))
    if ext == ".docx":
        return _clean(_docx_text(data))
    if ext == ".doc":
        return _clean(_doc_text(data))
    if ext == ".pdf":
        return _clean(_pdf_text(data, ocr_langs))
    raise UnsupportedFormat(ext)


def file_at_fault(error: Exception) -> bool:
    """The file itself cannot be read, however often it is tried: too many scanned pages, a scan page Tesseract
    does not finish, a .docx too big unpacked, a broken PDF or .docx. Not when Tesseract, memory or the disk fail."""
    return isinstance(error, (ValueError, pdfium.PdfiumError, zipfile.BadZipFile)) and not isinstance(error, OcrFailed)


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    # UTF-8 with a stray byte (0xA0, a non-breaking space from Windows) also decodes as cp1251, every letter garbled.
    # Text in cp1251 fails as UTF-8 on almost every letter, so one bad byte, or one in a hundred, is UTF-8 with a gap.
    text = data.decode("utf-8-sig", errors="replace")
    if text.count("\ufffd") <= max(1, len(text) // 100):
        return text
    try:
        return data.decode("cp1251")
    except UnicodeDecodeError:
        return text


def _docx_text(data: bytes) -> str:
    # The sizes a zip declares are the most it unpacks: zipfile stops there and fails the check sum.
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        unpacked = sum(member.file_size for member in archive.infolist())
    if unpacked > MAX_DOCX_UNPACKED:
        raise ValueError(f"the .docx unpacks to {unpacked} bytes, more than {MAX_DOCX_UNPACKED}")
    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n".join(parts)


def _doc_text(data: bytes) -> str:
    if not old_word_supported():
        raise UnsupportedFormat(".doc")
    with tempfile.NamedTemporaryFile(suffix=".doc") as file:
        file.write(data)
        file.flush()
        result = subprocess.run(["antiword", "-m", "UTF-8.txt", "-w", "0", file.name],
                                capture_output=True, timeout=60, check=True)
    return result.stdout.decode("utf-8", errors="replace")


def _pdf_text(data: bytes, ocr_langs: str) -> str:
    # The document is closed inside the lock too, so no PDFium call is left for later in another thread.
    # OCR stays inside as well: simpler, and a second PDF waits for at most one file's scanned pages.
    with PDFIUM, pdfium.PdfDocument(data) as pdf:
        pages = [pdf[index].get_textpage().get_text_range() for index in range(len(pdf))]
        # A scan is a picture: a blank page, or one with a heading alone, has nothing for OCR to read.
        scanned = [index for index, text in enumerate(pages)
                   if len(text.strip()) < MIN_PAGE_TEXT and _has_image(pdf[index])]
        if len(scanned) > MAX_OCR_PAGES:
            raise TooManyScannedPages(len(scanned))
        for index in scanned:
            pages[index] = _ocr_page(pdf[index], ocr_langs) or pages[index]
        # No text at all: a blank PDF is just an empty file, but pages with pictures or drawings are a scan
        # (or text turned into curves) the bot could not read, and the sender is asked for a text layer.
        if not any(text.strip() for text in pages) and any(_has_drawing(pdf[index]) for index in range(len(pdf))):
            raise UnreadableScan("no text on any page")
    return "\n\n".join(pages)


def _has_image(page) -> bool:
    return next(page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]), None) is not None


def _has_drawing(page) -> bool:
    # Text objects already gave their text: spaces alone, as in a blank text file printed to PDF, are no content.
    drawn = [pdfium_c.FPDF_PAGEOBJ_PATH, pdfium_c.FPDF_PAGEOBJ_IMAGE, pdfium_c.FPDF_PAGEOBJ_SHADING]
    return next(page.get_objects(filter=drawn), None) is not None


def _ocr_page(page, ocr_langs: str) -> str:
    scale = min(300 / 72, MAX_RENDER_SIDE / max(page.get_size()))  # PDF sizes are in points, 72 per inch
    image = page.render(scale=scale).to_pil().convert("L")
    image = ImageOps.autocontrast(image)
    # Text is near-black; the bark background and its stains are lighter.
    image = image.point(lambda value: 0 if value < 110 else 255)
    # A timeout fails the whole file: the sender is asked for a document with text, as for any scan that could not
    # be read, and the archive sync reads the file again once it changes on Drive or EXTRACT_VERSION is bumped,
    # instead of storing the scan without this page. So does an error of Tesseract itself, but the archive sync
    # tries that file again at the next sync, as Tesseract may be set up right by then.
    try:
        raw = pytesseract.image_to_string(image, lang=ocr_langs, config="--psm 4", timeout=OCR_PAGE_SECONDS)
    except pytesseract.TesseractError as error:
        raise OcrFailed(str(error)) from error
    except RuntimeError as error:  # pytesseract's timeout; TesseractError above is a RuntimeError too
        raise UnreadableScan(str(error)) from error
    return _clean_ocr(raw)


def _clean_ocr(raw: str) -> str:
    """Drop specks of the bark background that OCR reads as stray symbols."""
    lines = []
    for line in raw.splitlines():
        line = re.sub(r"^[^\w«\"(]+", "", line).strip()
        # OCR reads the digit 3 as the Cyrillic letter З: "становить З шишки", "1 рибина за З гілки".
        line = re.sub(r"(?<=[а-яіїєґ0-9,:] )З(?= )", "3", line)
        if len(re.findall(r"[А-Яа-яІіЇїЄєҐґA-Za-z]", line)) >= 3:
            lines.append(line)
        elif not line and lines and lines[-1]:
            lines.append("")
    return "\n".join(lines)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("­", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
