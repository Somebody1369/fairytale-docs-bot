import io

import pytest
from datetime import datetime, timezone

import docx
from pathlib import Path

from bot import archive as archive_module
from bot.archive import DriveFile, parse_folder_page, parse_shown_date, sync_archive
from bot.classify import classify_document, scope_from_caption
from bot.extract import _clean_ocr, extract_text, old_word_supported, title_from_filename
from bot.places import CONIFER, DECIDUOUS, FOREST_SCOPE, Scope

from .conftest import FOREST_MAP_TEXT, FakeLLM

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)

FOLDER_PAGE = """
<div class="flip-entry" id="entry-abc123" tabindex="0" role="link"><div class="flip-entry-info">
<a href="https://drive.google.com/file/d/abc123/view"><div class="flip-entry-title">Календар сезонів.txt</div></a>
</div><div class="flip-entry-last-modified"><div>Oct 13</div></div></div>
<div class="flip-entry" id="entry-def456" tabindex="0" role="link"><div class="flip-entry-info">
<a href="https://drive.google.com/file/d/def456/view"><div class="flip-entry-title">Карта лісу.pdf</div></a>
</div><div class="flip-entry-last-modified"><div>3:45 PM</div></div></div>
"""


def test_folder_page_is_parsed():
    files = parse_folder_page(FOLDER_PAGE, NOW)
    assert [(f.id, f.name) for f in files] == [("abc123", "Календар сезонів.txt"), ("def456", "Карта лісу.pdf")]
    assert files[0].modified == datetime(2026, 10, 13, tzinfo=timezone.utc)
    assert files[1].modified == NOW


def test_google_docs_document_in_public_folder_is_exported_as_text():
    page = FOLDER_PAGE + """
<div class="flip-entry" id="entry-gdoc1" tabindex="0" role="link"><div class="flip-entry-info">
<a href="https://docs.google.com/document/d/gdoc1/edit?usp=drive_web"><div class="flip-entry-title">Нові правила</div></a>
</div><div class="flip-entry-last-modified"><div>Oct 20</div></div></div>
"""
    files = {f.id: f for f in parse_folder_page(page, NOW)}
    assert files["gdoc1"].name == "Нові правила.txt" and files["gdoc1"].export_mime == "text/plain"
    assert files["abc123"].export_mime is None


def test_shown_dates():
    assert parse_shown_date("Oct 13, 2025", NOW) == datetime(2025, 10, 13, tzinfo=timezone.utc)
    assert parse_shown_date("10/13/25", NOW) == datetime(2025, 10, 13, tzinfo=timezone.utc)


class FakeArchive:
    api_key = None
    ocr_langs = "ukr"

    def __init__(self, files: dict[str, tuple[str, str, str]]):
        self.files = files  # id -> (name, marker, text)

    async def list_files(self, session):
        return [DriveFile(id=i, name=n, marker=m, modified=NOW) for i, (n, m, _) in self.files.items()]

    async def download(self, session, file):
        return self.files[file.id][2].encode("utf-8")


async def test_sync_adds_updates_and_removes(store, forest):
    archive = FakeArchive({
        "1": ("Календар сезонів.txt", "Oct 13", "Цей календар для всіх мешканців лісу. Гриби з серпня."),
        "2": ("Небезпеки хвойного лісу.txt", "Oct 13", "Живиця липне до шерсті."),
    })
    report = await sync_archive(store, forest, archive)
    assert (report.added, report.updated) == (2, 0)
    scopes = {d.title: d.scope for d in store.active_documents()}
    assert scopes == {"Календар сезонів": FOREST_SCOPE, "Небезпеки хвойного лісу": Scope("part", CONIFER)}

    archive.files["2"] = ("Небезпеки хвойного лісу.txt", "4:10 PM", "Живиця липне. Відмивайте олією.")
    del archive.files["1"]
    report = await sync_archive(store, forest, archive)
    assert (report.updated, report.removed) == (1, 1)
    assert [d.text for d in store.active_documents()] == ["Живиця липне. Відмивайте олією."]


async def test_renamed_archive_file_gets_its_new_title(store, forest):
    text = "Ця пам'ятка для мешканців біля річки. Броди мілкі в серпні."
    archive = FakeArchive({"1": ("Броди.txt", "Oct 13", text)})
    await sync_archive(store, forest, archive)
    archive.files["1"] = ("Броди і коли бобри відкривають греблю.txt", "Oct 14", text)
    await sync_archive(store, forest, archive)
    [doc] = store.active_documents()
    assert doc.title == "Броди і коли бобри відкривають греблю"
    assert doc.modified_at == NOW  # the advice itself did not change


async def test_sync_reads_forest_map(store, forest):
    changed = FOREST_MAP_TEXT.replace("Біля мурашника — хвойний ліс.", "Біля мурашника — листяний ліс.")
    await sync_archive(store, forest, FakeArchive({"m": ("Карта лісу.txt", "Oct 13", changed)}))
    assert forest.mitten_parts["Біля мурашника"] == {"deciduous"}


def test_txt_in_windows_encoding_and_docx():
    assert extract_text("Порада для лісу".encode("cp1251"), "a.txt") == "Порада для лісу"
    document = docx.Document()
    document.add_paragraph("Гриби під ялинами")
    buffer = io.BytesIO()
    document.save(buffer)
    assert extract_text(buffer.getvalue(), "a.docx") == "Гриби під ялинами"


def test_title_from_filename():
    assert title_from_filename("Гриби_під_ялинами (1).pdf") == "Гриби під ялинами"


async def test_document_place_from_title_skips_model(forest):
    llm = FakeLLM()
    scope = await classify_document("Чергування Під дубом", "текст", forest, llm)
    assert scope == Scope("mitten", "Під дубом") and llm.prompts == []


async def test_document_place_from_text_without_model(forest):
    llm = FakeLLM()
    assert await classify_document("Порада", "Під ялинами після дощу ростуть рижики.", forest, llm) \
        == Scope("part", CONIFER)
    assert await classify_document("Нагадування", "Шановні мешканці рукавички Під дубом!", forest, llm) \
        == Scope("mitten", "Під дубом")
    assert await classify_document("Вогонь", "Ця порада для всіх мешканців лісу.", forest, llm) == FOREST_SCOPE
    assert llm.prompts == []


async def test_document_without_any_place_is_not_guessed(forest):
    llm = FakeLLM({"kind": "forest", "value": ""})
    assert await classify_document("Нотатка", "Збирайте гриби після дощу.", forest, llm) is None
    assert llm.prompts == []


async def test_model_chooses_only_between_mentioned_places(forest):
    text = "Гриби ростуть серед сосен, а ягоди біля річки."
    llm = FakeLLM({"kind": "part", "value": "біля річки"})
    assert await classify_document("Порада", text, forest, llm) == Scope("part", "river")
    llm = FakeLLM({"kind": "mitten", "value": "Під дубом"})  # not mentioned in the text
    assert await classify_document("Порада", text, forest, llm) is None


def test_caption_names_the_place(forest):
    assert scope_from_caption("це для Під вербою", forest) == Scope("mitten", "Під вербою")
    assert scope_from_caption("для хвойного лісу", forest) == Scope("part", CONIFER)
    assert scope_from_caption("для всього лісу", forest) == FOREST_SCOPE
    assert scope_from_caption("ось моя порада", forest) is None
    assert scope_from_caption(None, forest) is None


async def test_empty_drive_listing_keeps_the_archive(store, forest):
    archive = FakeArchive({"1": ("Календар сезонів.txt", "Oct 13", "Цей календар для всіх мешканців лісу.")})
    await sync_archive(store, forest, archive)
    archive.files.clear()  # a consent page or a new page layout looks like an empty folder
    with pytest.raises(RuntimeError):
        await sync_archive(store, forest, archive)
    assert [d.title for d in store.active_documents()] == ["Календар сезонів"]


def test_ocr_digit_three_read_as_letter():
    assert _clean_ocr("Внесок становить З шишки з дорослого звіра.") == "Внесок становить 3 шишки з дорослого звіра."
    assert _clean_ocr("Видра може платити рибою: 1 рибина за З гілки.") == "Видра може платити рибою: 1 рибина за 3 гілки."
    assert _clean_ocr("З берега не стрибайте.") == "З берега не стрибайте."


async def test_better_reading_keeps_drive_date(store, forest, monkeypatch):
    archive = FakeArchive({"1": ("Внески Під ялиною.txt", "Oct 13", "Внесок З шишки.")})
    await sync_archive(store, forest, archive)
    first = store.active_documents()[0]

    # The bot learns to read the file better; the file on Drive did not change.
    monkeypatch.setattr(archive_module, "EXTRACT_VERSION", "test")
    archive.files["1"] = ("Внески Під ялиною.txt", "Oct 13", "Внесок 3 шишки.")
    await sync_archive(store, forest, archive)
    [current] = store.active_documents()
    assert current.text == "Внесок 3 шишки." and current.modified_at == first.modified_at


async def test_addressee_decides_over_places_inside_the_advice(forest):
    llm = FakeLLM()
    assert await classify_document("Гриби", "Порада для листяного лісу: білі гриби шукайте під дубом.", forest, llm) \
        == Scope("part", DECIDUOUS)
    assert await classify_document(
        "Шишки", "Ця порада для мешканців хвойного лісу. Найкращі шишки у соснах біля стежки.", forest, llm) \
        == Scope("part", CONIFER)
    assert await classify_document("Збори", "Шановні мешканці рукавички Під вербою! Збори в суботу.", forest, llm) \
        == Scope("mitten", "Під вербою")
    assert llm.prompts == []


async def test_mitten_and_part_without_addressee_are_not_decided_by_keywords(forest):
    # A mitten mentioned next to a forest part may be just a place inside general advice: ask the model.
    text = "У хвойному лісі восени багато маслюків, особливо у соснах."
    llm = FakeLLM({"kind": "part", "value": "хвойний ліс"})
    assert await classify_document("Маслюки", text, forest, llm) == Scope("part", CONIFER)
    assert len(llm.prompts) == 1


@pytest.mark.skipif(not old_word_supported(), reason="antiword is installed in the Docker image")
def test_old_word_document():
    data = (Path(__file__).parent / "fixtures" / "old_word.doc").read_bytes()
    assert extract_text(data, "Гриби.doc").startswith("Порада для листяного лісу.\nБілі гриби шукайте під дубом")


async def test_title_with_two_places_is_not_decided_by_the_title(forest):
    llm = FakeLLM()
    assert await classify_document("Гриби Під дубом і в соснах", "Збирайте після дощу.", forest, llm) is None
    assert await classify_document("Внески Під дубом, листяний ліс", "текст", forest, llm) == Scope("mitten", "Під дубом")
    assert llm.prompts == []


def test_long_scan_is_refused_instead_of_cut(monkeypatch):
    import pypdfium2 as pdfium
    from bot import extract

    pdf = pdfium.PdfDocument.new()
    for _ in range(extract.MAX_OCR_PAGES + 1):
        pdf.new_page(200, 200)
    data = io.BytesIO()
    pdf.save(data)
    monkeypatch.setattr(extract, "_ocr_page", lambda page, langs: pytest.fail("no page is read before the check"))
    with pytest.raises(extract.TooManyScannedPages):
        extract_text(data.getvalue(), "scan.pdf")
