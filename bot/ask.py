"""Ask the bot a question from the terminal, without Telegram:

    docker compose run --rm bot python -m bot.ask "скільки внесок під дубом?"

Uses the same database as the bot, so the archive and animals' documents are already there.
"""

import asyncio
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .answering import Answerer
from .archive import DriveArchive, sync_archive
from .config import load_settings
from .llm import LLM
from .places import Forest
from .storage import Store


async def main(question: str) -> None:
    # Read as the bot reads them ("" is the default, "a, b" two models); docker compose demands the
    # Telegram token that load_settings wants anyway.
    settings = load_settings()
    store = Store(settings.db_path)
    forest = Forest()
    await sync_archive(store, forest, DriveArchive(settings.drive_folder_id, settings.google_api_key, settings.ocr_langs))
    answerer = Answerer(store, forest, LLM(settings.groq_api_key, settings.groq_models), ZoneInfo(settings.timezone))
    reply = await answerer.ask(question, datetime.now(timezone.utc))
    print("Потрібно уточнити рукавичку." if reply.kind == "need_place" else reply.text)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "що робити при пожежі?"))
