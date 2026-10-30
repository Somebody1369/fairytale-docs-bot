"""SQLite storage for documents and animals' names.

Every version of a document is kept as its own row. Only rows with
status "active" take part in answers, so replaced or deleted advice is
never shown while its history (who sent what and when) is preserved.
"""

import hashlib
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from .places import Scope

INVITE_DAYS = 7

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    tg_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS members (
    tg_id INTEGER PRIMARY KEY,
    joined_at TEXT NOT NULL,
    invited_by INTEGER,            -- who shared the invite; NULL for the first animals
    tg_name TEXT                   -- Telegram name at joining, so /invited shows who it is before they name themselves
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invites (
    token TEXT PRIMARY KEY,
    created_by INTEGER,            -- NULL when made from the command line
    created_at TEXT NOT NULL,
    used_by INTEGER,
    used_at TEXT
);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,          -- 'archive' | 'animal'
    title TEXT NOT NULL,
    norm_title TEXT NOT NULL,
    filename TEXT NOT NULL,
    scope TEXT NOT NULL,           -- 'mitten:Під дубом' | 'part:conifer' | 'forest:'
    author_id INTEGER,
    author_name TEXT,
    drive_file_id TEXT,
    drive_marker TEXT,
    present INTEGER NOT NULL DEFAULT 1,  -- archive: the file's latest content is still on Drive
    content_hash TEXT NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,      -- when the animal sent it / when the bot first saw it
    modified_at TEXT NOT NULL,     -- version time used to pick the newest advice
    status TEXT NOT NULL           -- 'active' | 'superseded' | 'deleted' | 'revoked'
);
CREATE INDEX IF NOT EXISTS documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS documents_drive ON documents(drive_file_id);
"""


@dataclass
class Document:
    id: int
    source: str
    title: str
    norm_title: str
    filename: str
    scope: Scope
    author_id: int | None
    author_name: str | None
    drive_file_id: str | None
    drive_marker: str | None
    present: bool
    content_hash: str
    text: str
    created_at: datetime
    modified_at: datetime
    status: str

    @property
    def from_archive(self) -> bool:
        return self.source == "archive"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm_title(title: str) -> str:
    return " ".join(title.lower().replace("«", "").replace("»", "").replace('"', "").split())


class Store:
    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(members)")}
        if "invited_by" not in columns:
            self.db.execute("ALTER TABLE members ADD COLUMN invited_by INTEGER")
        if "tg_name" not in columns:
            self.db.execute("ALTER TABLE members ADD COLUMN tg_name TEXT")
        self.revision = 0  # bumped on every change so the search index knows to rebuild

    # Users

    def user_name(self, tg_id: int) -> str | None:
        row = self.db.execute("SELECT name FROM users WHERE tg_id = ?", (tg_id,)).fetchone()
        return row["name"] if row else None

    def set_user_name(self, tg_id: int, name: str, now: datetime) -> None:
        self.db.execute(
            "INSERT INTO users (tg_id, name, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(tg_id) DO UPDATE SET name = excluded.name",
            (tg_id, name, now.isoformat()),
        )
        self.db.commit()

    def rename_author(self, tg_id: int, name: str) -> None:
        self.db.execute("UPDATE documents SET author_name = ? WHERE source = 'animal' AND author_id = ?", (name, tg_id))
        self._changed()

    def meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, value))
        self.db.commit()

    # Invites: with INVITES_ONLY a neighbour's one-time link is the way in

    def is_member(self, tg_id: int) -> bool:
        return self.db.execute("SELECT 1 FROM members WHERE tg_id = ?", (tg_id,)).fetchone() is not None

    def create_invite(self, created_by: int | None, now: datetime) -> str:
        token = secrets.token_urlsafe(9)
        self.db.execute("INSERT INTO invites (token, created_by, created_at) VALUES (?, ?, ?)",
                        (token, created_by, now.isoformat()))
        self.db.commit()
        return token

    def has_archive(self) -> bool:
        return self.db.execute(
            "SELECT 1 FROM documents WHERE source = 'archive' AND status = 'active' LIMIT 1").fetchone() is not None

    def has_members(self) -> bool:
        return self.db.execute("SELECT 1 FROM members LIMIT 1").fetchone() is not None

    def first_invite(self, now: datetime) -> str:
        """The invite for the very first animal: reuse a still valid unused one, so restarts don't pile them up."""
        since = (now - timedelta(days=INVITE_DAYS)).isoformat()
        row = self.db.execute(
            "SELECT token FROM invites WHERE created_by IS NULL AND used_by IS NULL AND created_at > ? "
            "ORDER BY created_at DESC LIMIT 1",
            (since,),
        ).fetchone()
        return row["token"] if row else self.create_invite(None, now)

    def invited_by_chain(self, inviter_id: int) -> list[tuple[int, str | None, datetime, int]]:
        """Everyone this member brought in, directly or through the animals they invited:
        (id, the name they gave the bot or else their Telegram name, when they joined, who invited them)."""
        found, queue = [], [inviter_id]
        while queue:
            current = queue.pop(0)
            rows = self.db.execute(
                "SELECT m.tg_id, COALESCE(u.name, m.tg_name), m.joined_at, m.invited_by "
                "FROM members m LEFT JOIN users u ON u.tg_id = m.tg_id "
                "WHERE m.invited_by = ? ORDER BY m.joined_at",
                (current,),
            ).fetchall()
            for row in rows:
                found.append((row[0], row[1], datetime.fromisoformat(row[2]), row[3]))
                queue.append(row[0])
        return found

    def revoke_member(self, member_id: int, by_id: int) -> list[int] | None:
        """Close access for an animal brought in by `by_id` (directly or further down the invite chain),
        and for everyone that animal invited in turn. Their documents stop being used in answers
        (kept with status "revoked") and their unused invites stop working. No admins are needed:
        whoever stands above in the chain can do it, so a sleeping inviter does not leave a stranger in."""
        chain = {member[0] for member in self.invited_by_chain(by_id)}
        if member_id not in chain:
            return None
        removed, queue = [], [member_id]
        while queue:
            current = queue.pop()
            removed.append(current)
            queue += [r[0] for r in self.db.execute("SELECT tg_id FROM members WHERE invited_by = ?", (current,))]
        for tg_id in removed:
            self.db.execute("DELETE FROM members WHERE tg_id = ?", (tg_id,))
            self.db.execute("DELETE FROM invites WHERE used_by IS NULL AND created_by = ?", (tg_id,))
            self.db.execute(
                "UPDATE documents SET status = 'revoked' WHERE source = 'animal' AND author_id = ? AND status = 'active'",
                (tg_id,),
            )
        self._changed()
        return removed

    def use_invite(self, token: str, tg_id: int, now: datetime, tg_name: str | None = None) -> bool:
        """Let the animal in if the invite exists, is unused and not older than INVITE_DAYS."""
        row = self.db.execute("SELECT * FROM invites WHERE token = ?", (token,)).fetchone()
        if not row or row["used_by"] is not None:
            return False
        if now - datetime.fromisoformat(row["created_at"]) > timedelta(days=INVITE_DAYS):
            return False
        creator = row["created_by"]
        if creator is not None and (creator == tg_id or not self.is_member(creator)):
            return False  # one's own spare link, or a link from someone whose access was closed
        self.db.execute("UPDATE invites SET used_by = ?, used_at = ? WHERE token = ?", (tg_id, now.isoformat(), token))
        self.db.execute("INSERT OR IGNORE INTO members (tg_id, joined_at, invited_by, tg_name) VALUES (?, ?, ?, ?)",
                        (tg_id, now.isoformat(), creator, tg_name))
        # A neighbour vouched for this animal again: documents closed with their access count again.
        self.db.execute(
            "UPDATE documents SET status = 'active' WHERE source = 'animal' AND author_id = ? AND status = 'revoked'",
            (tg_id,),
        )
        self._changed()
        return True

    # Documents

    def active_documents(self) -> list[Document]:
        rows = self.db.execute("SELECT * FROM documents WHERE status = 'active' ORDER BY id").fetchall()
        return [self._doc(row) for row in rows]

    def get(self, doc_id: int) -> Document | None:
        row = self.db.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def documents_of(self, author_id: int) -> list[Document]:
        rows = self.db.execute(
            "SELECT * FROM documents WHERE source = 'animal' AND author_id = ? AND status = 'active' "
            "ORDER BY created_at DESC",
            (author_id,),
        ).fetchall()
        return [self._doc(row) for row in rows]

    def add_animal_document(
        self, *, title: str, filename: str, scope: Scope, author_id: int, author_name: str, text: str, now: datetime
    ) -> tuple[Document, Document | None]:
        """Save an animal's document. The same animal sending a document with the same
        title for the same place replaces its previous version."""
        replaced_row = self.db.execute(
            "SELECT * FROM documents WHERE source = 'animal' AND status = 'active' "
            "AND author_id = ? AND norm_title = ? AND scope = ?",
            (author_id, norm_title(title), scope.encode()),
        ).fetchone()
        replaced = self._doc(replaced_row) if replaced_row else None
        if replaced:
            self.db.execute("UPDATE documents SET status = 'superseded' WHERE id = ?", (replaced.id,))
        doc_id = self._insert(
            source="animal", title=title, filename=filename, scope=scope, author_id=author_id,
            author_name=author_name, text=text, created_at=now, modified_at=now,
        )
        self._changed()
        return self.get(doc_id), replaced

    def delete_animal_document(self, doc_id: int, author_id: int) -> Document | None:
        doc = self.get(doc_id)
        if not doc or doc.source != "animal" or doc.author_id != author_id or doc.status != "active":
            return None
        self.db.execute("UPDATE documents SET status = 'deleted' WHERE id = ?", (doc_id,))
        self._changed()
        return doc

    def change_scope(self, doc_id: int, author_id: int, scope: Scope) -> Document | None:
        doc = self.get(doc_id)
        if not doc or doc.source != "animal" or doc.author_id != author_id or doc.status != "active":
            return None
        # The animal's own document with this title already written for the new place is an older version.
        self.db.execute(
            "UPDATE documents SET status = 'superseded' WHERE source = 'animal' AND status = 'active' "
            "AND author_id = ? AND norm_title = ? AND scope = ? AND id != ?",
            (author_id, doc.norm_title, scope.encode(), doc_id),
        )
        self.db.execute("UPDATE documents SET scope = ? WHERE id = ?", (scope.encode(), doc_id))
        self._changed()
        return self.get(doc_id)

    # Archive (Google Drive)

    def latest_archive_version(self, drive_file_id: str) -> Document | None:
        row = self.db.execute(
            "SELECT * FROM documents WHERE source = 'archive' AND drive_file_id = ? ORDER BY id DESC LIMIT 1",
            (drive_file_id,),
        ).fetchone()
        return self._doc(row) if row else None

    def set_drive_marker(self, doc_id: int, marker: str) -> None:
        self.db.execute("UPDATE documents SET drive_marker = ?, present = 1 WHERE id = ?", (marker, doc_id))
        self.db.commit()

    def add_archive_version(
        self, *, drive_file_id: str, marker: str, filename: str, title: str, scope: Scope, text: str,
        modified_at: datetime, now: datetime,
    ) -> int:
        self.db.execute(
            "UPDATE documents SET present = 0 WHERE source = 'archive' AND drive_file_id = ?", (drive_file_id,)
        )
        return self._insert(
            source="archive", title=title, filename=filename, scope=scope, text=text, created_at=now,
            modified_at=modified_at, drive_file_id=drive_file_id, drive_marker=marker,
        )

    def mark_missing_archive_files(self, present_ids: set[str]) -> int:
        rows = self.db.execute(
            "SELECT DISTINCT drive_file_id FROM documents WHERE source = 'archive' AND present = 1"
        ).fetchall()
        missing = [row["drive_file_id"] for row in rows if row["drive_file_id"] not in present_ids]
        for file_id in missing:
            self.db.execute("UPDATE documents SET present = 0 WHERE drive_file_id = ?", (file_id,))
        return len(missing)

    def recompute_archive_status(self) -> None:
        """One active version per document: the newest present file with that title and place."""
        rows = self.db.execute("SELECT * FROM documents WHERE source = 'archive'").fetchall()
        best: dict[tuple[str, str], sqlite3.Row] = {}
        for row in rows:
            if not row["present"]:
                continue
            key = (row["norm_title"], row["scope"])
            current = best.get(key)
            if current is None or (row["modified_at"], row["id"]) > (current["modified_at"], current["id"]):
                best[key] = row
        winners = {row["id"] for row in best.values()}
        for row in rows:
            status = "active" if row["id"] in winners else "superseded"
            if row["status"] != status:
                self.db.execute("UPDATE documents SET status = ? WHERE id = ?", (status, row["id"]))
        self._changed()

    # Internals

    def _insert(
        self, *, source: str, title: str, filename: str, scope: Scope, text: str, created_at: datetime,
        modified_at: datetime, author_id: int | None = None, author_name: str | None = None,
        drive_file_id: str | None = None, drive_marker: str | None = None,
    ) -> int:
        cursor = self.db.execute(
            "INSERT INTO documents (source, title, norm_title, filename, scope, author_id, author_name, "
            "drive_file_id, drive_marker, present, content_hash, text, created_at, modified_at, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 'active')",
            (
                source, title, norm_title(title), filename, scope.encode(), author_id, author_name,
                drive_file_id, drive_marker, content_hash(text), text, created_at.isoformat(),
                modified_at.isoformat(),
            ),
        )
        self.db.commit()
        return cursor.lastrowid

    def _changed(self) -> None:
        self.db.commit()
        self.revision += 1

    @staticmethod
    def _doc(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            norm_title=row["norm_title"],
            filename=row["filename"],
            scope=Scope.decode(row["scope"]),
            author_id=row["author_id"],
            author_name=row["author_name"],
            drive_file_id=row["drive_file_id"],
            drive_marker=row["drive_marker"],
            present=bool(row["present"]),
            content_hash=row["content_hash"],
            text=row["text"],
            created_at=datetime.fromisoformat(row["created_at"]),
            modified_at=datetime.fromisoformat(row["modified_at"]),
            status=row["status"],
        )
