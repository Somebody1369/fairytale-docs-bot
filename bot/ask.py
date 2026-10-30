"""Ask the bot a question from the terminal, without Telegram:

    docker compose run --rm bot python -m bot.ask "скільки внесок під дубом?"

Uses the same database as the bot, so the archive and animals' documents are already there.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .answering import Answerer
from .archive import DriveArchive, sync_archive
from .config import DEFAULT_DRIVE_FOLDER_ID, DEFAULT_GROQ_MODELS, env
from .llm import LLM
from .places import Forest
from .storage import Store


async def main(question: str) -> None:
    # docker-compose passes unset settings as empty strings, so "" means the default too.
    store = Store(env("DB_PATH", "data/bot.db"))
    forest = Forest()
    archive = DriveArchive(os.getenv("DRIVE_FOLDER_ID") or DEFAULT_DRIVE_FOLDER_ID, os.getenv("GOOGLE_API_KEY") or None,
                           env("OCR_LANGS", "ukr"))
    await sync_archive(store, forest, archive)
    models = (os.getenv("GROQ_MODELS") or DEFAULT_GROQ_MODELS).split(",")
    answerer = Answerer(store, forest, LLM(os.environ["GROQ_API_KEY"], models), ZoneInfo(env("TZ_NAME", "Europe/Kyiv")))
    reply = await answerer.ask(question, datetime.now(timezone.utc))
    print("Потрібно уточнити рукавичку." if reply.kind == "need_place" else reply.text)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "що робити при пожежі?"))
