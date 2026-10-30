from datetime import datetime, timedelta, timezone

import pytest

from bot.answering import PICK_SYSTEM
from bot.places import Forest, Scope
from bot.storage import Store

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)

FOREST_MAP_TEXT = """Карта лісу
Ця карта для всіх мешканців лісу. Вона показує, до якої частини лісу
належить кожна рукавичка. Частин три: хвойний ліс, листяний ліс і біля річки.
Легенда карти
–
Під ялиною — хвойний ліс.
–
У соснах — хвойний ліс.
–
Біля мурашника — хвойний ліс.
–
Під дубом — листяний ліс.
–
У ліщині — листяний ліс.
–
Під вербою — і листяний ліс, і біля річки водночас.
–
Біля бобрової греблі — біля річки.
–
На узліссі — ні хвойний, ні листяний ліс, і до річки далеко.
Дві особливі рукавички
Рукавичка «Під вербою» належить одразу до двох частин."""


class FakeLLM:
    """Records prompts and returns prepared JSON answers in order. Picking documents by title
    has its own queue of answers (`picks`), by default "none of them"."""

    def __init__(self, *answers: dict, picks: list[dict] | None = None):
        self.answers = list(answers)
        self.picks = list(picks or [])
        self.prompts: list[tuple[str, str]] = []
        self.pick_prompts: list[str] = []
        self.reasoning: list[str] = []

    async def json(self, system: str, user: str, max_tokens: int = 700, reasoning: str = "low",
                   fast: bool = False) -> dict:
        if system == PICK_SYSTEM:
            self.pick_prompts.append(user)
            return self.picks.pop(0) if self.picks else {"ids": []}
        self.prompts.append((system, user))
        self.reasoning.append(reasoning)
        return self.answers.pop(0) if self.answers else {}


@pytest.fixture
def forest() -> Forest:
    return Forest()


@pytest.fixture
def store() -> Store:
    return Store(":memory:")


def add_archive(store: Store, forest: Forest, title: str, text: str, file_id: str | None = None,
                modified: datetime = NOW - timedelta(days=10)) -> int:
    doc_id = store.add_archive_version(
        drive_file_id=file_id or title, marker="m1", filename=f"{title}.txt", title=title,
        scope=forest.archive_scope(title, text), text=text, modified_at=modified, now=NOW,
    )
    store.recompute_archive_status()
    return doc_id


def add_animal(store: Store, title: str, text: str, scope: Scope, author_id: int = 1, name: str = "Зайчик",
               when: datetime = NOW):
    return store.add_animal_document(title=title, filename=f"{title}.txt", scope=scope, author_id=author_id,
                                     author_name=name, text=text, now=when)
