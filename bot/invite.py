"""A new one-time invite link from the command line, e.g. for the first animals (the bot also logs one at start):

    docker compose run --rm bot python -m bot.invite

Every run prints a new link for one animal, who can then invite neighbours with /invite.
"""

import asyncio
from datetime import datetime, timezone

from aiogram import Bot

from .config import load_settings
from .handlers import invite_link
from .storage import INVITE_DAYS, Store


async def main() -> None:
    settings = load_settings()
    token = Store(settings.db_path).create_invite(None, datetime.now(timezone.utc))
    bot = Bot(settings.telegram_token)
    try:
        me = await bot.get_me()
    finally:
        await bot.session.close()
    print(f"{invite_link(me.username, token)}\n(one use, valid for {INVITE_DAYS} days)")
    if not settings.invites_only:
        print("Note: INVITES_ONLY is off, so the bot answers everyone and the invite is not needed.")


if __name__ == "__main__":
    asyncio.run(main())
