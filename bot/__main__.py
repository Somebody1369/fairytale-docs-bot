import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand, FSInputFile, InputProfilePhotoStatic

from .answering import Answerer
from .archive import DriveArchive, refresh_map, run_periodic_sync, sync_archive
from .config import load_settings
from .fsm import SQLiteStorage
from . import texts
from .handlers import App, invite_link, router
from .llm import LLM
from .places import Forest
from .storage import Store

log = logging.getLogger("bot")


AVATAR = Path(__file__).parent / "assets" / "avatar.jpg"


async def set_avatar(bot: Bot, store: Store) -> None:
    """Birch bark with an oak leaf and a page: set once, and again only when the picture changes."""
    if not AVATAR.exists():
        return
    digest = hashlib.sha256(AVATAR.read_bytes()).hexdigest()
    if store.meta("avatar") == digest:
        return
    try:
        await bot.set_my_profile_photo(InputProfilePhotoStatic(photo=FSInputFile(AVATAR)))
        store.set_meta("avatar", digest)
        log.info("Bot avatar updated")
    except TelegramAPIError as error:
        log.warning("Could not set the bot's avatar: %s", error)


async def keep_archive_in_sync(store: Store, forest: Forest, archive: DriveArchive, settings, ready: asyncio.Event) -> None:
    try:
        report = await sync_archive(store, forest, archive)
        log.info("Archive loaded: %s", report)
    except Exception:
        log.exception("Archive is not reachable now, the bot answers from what it already has")
    finally:
        ready.set()
    await run_periodic_sync(store, forest, archive, settings.sync_minutes)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    tz = ZoneInfo(settings.timezone)

    bot = Bot(settings.telegram_token)
    try:
        me = await bot.get_me()  # fail fast on a wrong token, before the long first archive load
        log.info("Telegram bot @%s", me.username)
        invites = [
            BotCommand(command="invite", description="Запрошення для сусіда"),
            BotCommand(command="invited", description="Кого я запросив"),
        ] if settings.invites_only else []
        await bot.set_my_commands([
            BotCommand(command="start", description="Почати"),
            BotCommand(command="help", description="Як працює бот"),
            BotCommand(command="my", description="Мої документи"),
            *invites,
            BotCommand(command="name", description="Змінити ім'я"),
            BotCommand(command="cancel", description="Скасувати поточну дію"),
        ])
        store = Store(settings.db_path)
        if settings.set_profile:
            try:
                if me.first_name != texts.BOT_NAME:
                    await bot.set_my_name(texts.BOT_NAME)
                await bot.set_my_short_description(texts.BOT_SHORT_DESCRIPTION)
                await bot.set_my_description(texts.BOT_DESCRIPTION_INVITES if settings.invites_only
                                             else texts.BOT_DESCRIPTION)
            except TelegramAPIError as error:  # Telegram limits how often these can change
                log.warning("Could not update the bot's name or description: %s", error)
            await set_avatar(bot, store)
        forest = Forest()
        refresh_map(store, forest)
        llm = LLM(settings.groq_api_key, settings.groq_models)
        await llm.check_models()
        archive = DriveArchive(settings.drive_folder_id, settings.google_api_key, settings.ocr_langs)
        # The bot listens at once. On the very first start, while the archive is read (OCR of the
        # scans takes a couple of minutes), questions wait for it; after that the saved archive is used.
        archive_ready = asyncio.Event()
        if store.has_archive():
            archive_ready.set()
        sync_task = asyncio.create_task(keep_archive_in_sync(store, forest, archive, settings, archive_ready))

        if settings.invites_only and not store.has_members():
            link = invite_link(me.username, store.first_invite(datetime.now(timezone.utc)))
            log.warning("The bot is open only by invitation. Invite for the first animal: %s", link)

        app = App(settings=settings, store=store, forest=forest, llm=llm,
                  answerer=Answerer(store, forest, llm, tz), tz=tz, archive_ready=archive_ready)
        dispatcher = Dispatcher(storage=SQLiteStorage(store.db), app=app)
        dispatcher.include_router(router)
        try:
            await dispatcher.start_polling(bot)
        finally:
            sync_task.cancel()
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
