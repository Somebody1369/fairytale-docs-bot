"""Dialog state in SQLite, so a document waiting for its sender's name or place survives a restart."""

import json
import sqlite3
from typing import Any

from aiogram.fsm.state import State
from aiogram.fsm.storage.base import BaseStorage, DefaultKeyBuilder, StorageKey


class SQLiteStorage(BaseStorage):
    def __init__(self, db: sqlite3.Connection):
        self.db = db
        self.keys = DefaultKeyBuilder()
        self.db.execute("CREATE TABLE IF NOT EXISTS fsm (key TEXT PRIMARY KEY, state TEXT, data TEXT NOT NULL DEFAULT '{}')")
        self.db.commit()

    async def set_state(self, key: StorageKey, state: str | State | None = None) -> None:
        value = state.state if isinstance(state, State) else state
        self.db.execute(
            "INSERT INTO fsm (key, state) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET state = excluded.state",
            (self.keys.build(key), value),
        )
        self.db.commit()

    async def get_state(self, key: StorageKey) -> str | None:
        row = self.db.execute("SELECT state FROM fsm WHERE key = ?", (self.keys.build(key),)).fetchone()
        return row[0] if row else None

    async def set_data(self, key: StorageKey, data: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO fsm (key, data) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET data = excluded.data",
            (self.keys.build(key), json.dumps(data, ensure_ascii=False)),
        )
        self.db.commit()

    async def get_data(self, key: StorageKey) -> dict[str, Any]:
        row = self.db.execute("SELECT data FROM fsm WHERE key = ?", (self.keys.build(key),)).fetchone()
        return json.loads(row[0]) if row else {}

    async def close(self) -> None:
        pass
