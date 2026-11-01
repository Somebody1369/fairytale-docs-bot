"""Keeps the bot in sync with the public "Лісовий архів" folder on Google Drive.

With GOOGLE_API_KEY the Drive API gives exact modification times. Without a key
the public folder page is read instead; it shows coarse dates, so a changed file
is detected by its content and dated by the moment the bot noticed the change.
"""

import asyncio
import html
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import aiohttp

from .extract import SUPPORTED_EXTENSIONS, extension, extract_text, title_from_filename
from .places import Forest
from .storage import Store, content_hash, norm_title

log = logging.getLogger(__name__)

DRIVE_API = "https://www.googleapis.com/drive/v3/files"
FOLDER_PAGE = "https://drive.google.com/embeddedfolderview?id={folder_id}"
DOWNLOAD = "https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
# A Google Docs document in a public folder has no file to download, but can be exported as text.
PUBLIC_DOC_EXPORT = "https://docs.google.com/document/d/{file_id}/export?format=txt"
GOOGLE_DOC_EXPORTS = {"application/vnd.google-apps.document": "text/plain"}
# Bump when text extraction improves (e.g. the OCR fix for "З" read instead of "3"):
# every archive file is then read again, and documents whose text changed get a new version.
EXTRACT_VERSION = "2"

MONTHS_EN = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


@dataclass
class DriveFile:
    id: str
    name: str
    marker: str  # changes whenever Drive reports a change
    modified: datetime | None
    export_mime: str | None = None


@dataclass
class SyncReport:
    files: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    failed: int = 0


class DriveArchive:
    def __init__(self, folder_id: str, api_key: str | None, ocr_langs: str):
        self.folder_id, self.api_key, self.ocr_langs = folder_id, api_key, ocr_langs

    async def list_files(self, session: aiohttp.ClientSession) -> list[DriveFile]:
        if self.api_key:
            return await self._list_api(session)
        return await self._list_public_page(session)

    async def _list_api(self, session: aiohttp.ClientSession) -> list[DriveFile]:
        files, token = [], None
        while True:
            params = {
                "q": f"'{self.folder_id}' in parents and trashed = false",
                "fields": "nextPageToken, files(id, name, mimeType, modifiedTime, md5Checksum)",
                "pageSize": "1000",
            }
            if token:
                params["pageToken"] = token
            async with session.get(DRIVE_API, params=params, headers=self._key_header()) as response:
                response.raise_for_status()
                payload = await response.json()
            for item in payload.get("files", []):
                export = GOOGLE_DOC_EXPORTS.get(item["mimeType"])
                name = export_name(item["name"]) if export else item["name"]
                modified = datetime.fromisoformat(item["modifiedTime"].replace("Z", "+00:00"))
                files.append(DriveFile(
                    id=item["id"], name=name, marker=f"{item['modifiedTime']}|{item.get('md5Checksum', '')}",
                    modified=modified, export_mime=export,
                ))
            token = payload.get("nextPageToken")
            if not token:
                return files

    async def _list_public_page(self, session: aiohttp.ClientSession) -> list[DriveFile]:
        async with session.get(FOLDER_PAGE.format(folder_id=self.folder_id)) as response:
            response.raise_for_status()
            page = await response.text()
        return parse_folder_page(page, datetime.now(timezone.utc))

    async def download(self, session: aiohttp.ClientSession, file: DriveFile) -> bytes:
        # Only the Drive API gets the key; the public download and export pages never needed it.
        if file.export_mime and self.api_key:
            url, params, headers = f"{DRIVE_API}/{file.id}/export", {"mimeType": file.export_mime}, self._key_header()
        elif file.export_mime:
            url, params, headers = PUBLIC_DOC_EXPORT.format(file_id=file.id), None, None
        else:
            url, params, headers = DOWNLOAD.format(file_id=file.id), None, None
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.read()
        if data[:15].lower().startswith((b"<!doctype html", b"<html")):
            raise RuntimeError("Drive returned a web page instead of the file")
        return data

    def _key_header(self) -> dict[str, str]:
        # In a header, not in the URL: a failed request's error, and so the log, shows the URL.
        # Google recommends the header too: https://docs.cloud.google.com/docs/authentication/api-keys-use
        return {"X-Goog-Api-Key": self.api_key}


def export_name(name: str) -> str:
    """A Google Docs document is exported as text and its title has no extension:
    "Внески від 01.11" is "Внески від 01.11.txt", not a file of type ".11"."""
    return name if extension(name) == ".txt" else f"{name}.txt"


def parse_folder_page(page: str, now: datetime) -> list[DriveFile]:
    files = []
    for entry in re.finditer(
        r'<div class="flip-entry" id="entry-([\w-]+)".*?<a href="([^"]*)".*?<div class="flip-entry-title">(.*?)</div>'
        r'.*?<div class="flip-entry-last-modified"><div>(.*?)</div>',
        page,
        re.S,
    ):
        file_id, link = entry.group(1), html.unescape(entry.group(2))
        name, shown_date = html.unescape(entry.group(3)), html.unescape(entry.group(4))
        export = None
        if "docs.google.com/document/" in link:  # a Google Docs document, not an uploaded file
            export, name = "text/plain", export_name(name)
        files.append(DriveFile(id=file_id, name=name, marker=shown_date, modified=parse_shown_date(shown_date, now),
                               export_mime=export))
    return files


def parse_shown_date(shown: str, now: datetime) -> datetime | None:
    """Drive shows "Oct 13", "Oct 13, 2025", "10/13/25" or a time like "3:45 PM" for today."""
    shown = shown.strip()
    if re.fullmatch(r"\d{1,2}:\d{2}\s*(AM|PM)?", shown, re.I):
        return now
    match = re.fullmatch(r"([A-Za-z]{3})\w*\s+(\d{1,2})(?:,\s*(\d{4}))?", shown)
    if match and match.group(1).lower() in MONTHS_EN:
        year = int(match.group(3) or now.year)
        return datetime(year, MONTHS_EN[match.group(1).lower()], int(match.group(2)), tzinfo=timezone.utc)
    match = re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", shown)
    if match:
        year = int(match.group(3))
        year = year + 2000 if year < 100 else year
        return datetime(year, int(match.group(1)), int(match.group(2)), tzinfo=timezone.utc)
    return None


async def sync_archive(store: Store, forest: Forest, archive: DriveArchive) -> SyncReport:
    report = SyncReport()
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        files = [f for f in await archive.list_files(session) if extension(f.name) in SUPPORTED_EXTENSIONS]
        if not files and store.has_archive():
            # A changed page layout or a consent page looks like an empty folder; never wipe the archive for it.
            raise RuntimeError("Drive returned no files, keeping the current archive")
        report.files = len(files)
        for file in files:
            marker = f"{file.marker}|x{EXTRACT_VERSION}"
            known = store.latest_archive_version(file.id)
            if known and known.present and known.drive_marker == marker:
                continue
            try:
                data = await archive.download(session, file)
                text = await asyncio.to_thread(extract_text, data, file.name, archive.ocr_langs)
            except Exception as error:  # one broken file must not stop the rest
                log.warning("Archive file %s (%s) skipped: %s", file.name, file.id, error)
                report.failed += 1
                continue
            if not text.strip():
                log.warning("Archive file %s has no readable text", file.name)
                report.failed += 1
                continue

            now = datetime.now(timezone.utc)
            title = title_from_filename(file.name)
            same_text = known and known.content_hash == content_hash(text)
            if same_text and known.title == title:
                store.set_drive_marker(known.id, marker)
                continue
            if known and (same_text or known.drive_marker.rsplit("|x", 1)[0] == file.marker):
                # Only renamed, or the same file the bot now reads better: the advice keeps its date.
                modified = known.modified_at
                report.updated += 1
            elif known:
                # Without exact Drive times the moment of noticing the change is the best version date.
                modified = file.modified if archive.api_key and file.modified else now
                report.updated += 1
            else:
                modified = file.modified or now
                report.added += 1
            store.add_archive_version(
                drive_file_id=file.id, marker=marker, filename=file.name, title=title,
                scope=forest.archive_scope(title, text), text=text, modified_at=modified, now=now,
            )

        report.removed = store.mark_missing_archive_files({f.id for f in files})
    store.recompute_archive_status()
    if refresh_map(store, forest):
        log.info("Forest map changed: %s", map_summary(forest))
    return report


def refresh_map(store: Store, forest: Forest) -> bool:
    """Re-read "Карта лісу" from the saved archive; whether the map the bot uses changed."""
    for doc in store.active_documents():
        if doc.from_archive and doc.norm_title == norm_title("Карта лісу"):
            before = list(forest.mitten_parts.items())  # in order: mitten buttons go by position
            return forest.update_from_map(doc.text) and list(forest.mitten_parts.items()) != before
    return False


def map_summary(forest: Forest) -> dict[str, list[str]]:
    return {mitten: sorted(parts) for mitten, parts in forest.mitten_parts.items()}


async def run_periodic_sync(store: Store, forest: Forest, archive: DriveArchive, minutes: int) -> None:
    while True:
        await asyncio.sleep(minutes * 60)
        try:
            report = await sync_archive(store, forest, archive)
            if report.added or report.updated or report.removed:
                log.info("Archive sync: %s", report)
        except Exception:
            log.exception("Archive sync failed, keeping the previous state")
