"""Whole conversations through aiogram with a fake Telegram server."""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from itertools import count
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import GetFile, GetMe, SendMessage
from aiogram.types import CallbackQuery, Chat, Document, File, Message, Update, User

from bot.answering import Answerer
from bot.config import Settings
from bot.fsm import SQLiteStorage
from bot import texts
from bot.handlers import App, clean_name, router
from bot.places import FOREST_SCOPE

from .conftest import FakeLLM, add_archive

TZ = ZoneInfo("Europe/Kyiv")
USER = User(id=10, is_bot=False, first_name="Bilka")
CHAT = Chat(id=10, type="private")
ids = count(1)


class FakeTelegram(BaseSession):
    def __init__(self):
        super().__init__()
        self.sent: list[SendMessage] = []
        self.file_bytes = b""

    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, SendMessage):
            self.sent.append(method)
            return Message(message_id=next(ids), date=datetime.now(timezone.utc), chat=CHAT, text=method.text)
        if isinstance(method, GetMe):
            return User(id=1, is_bot=True, first_name="Довідник лісу", username="fairytale_docs_bot")
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path="documents/file.txt")
        return True

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield self.file_bytes

    async def close(self):
        pass


def message(user: User = USER, **fields) -> Update:
    chat = Chat(id=user.id, type="private")
    msg = Message(message_id=next(ids), date=datetime.now(timezone.utc), chat=chat, from_user=user, **fields)
    return Update(update_id=next(ids), message=msg)


def callback(data: str) -> Update:
    msg = Message(message_id=next(ids), date=datetime.now(timezone.utc), chat=CHAT, text="...")
    query = CallbackQuery(id=str(next(ids)), from_user=USER, chat_instance="c", data=data, message=msg)
    return Update(update_id=next(ids), callback_query=query)


def buttons(sent: SendMessage) -> list[tuple[str, str]]:
    return [(b.text, b.callback_data) for row in sent.reply_markup.inline_keyboard for b in row]


async def flow_state(dispatcher, bot) -> str | None:
    return await dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id).get_state()


def press(sent: SendMessage, text: str) -> Update:
    """The button with this text under a message the bot sent."""
    return callback(dict(buttons(sent))[text])


@pytest.fixture
def setup(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    add_archive(store, forest, "Членські внески рукавички У соснах", "Внесок 3 шишки на місяць.")
    llm = FakeLLM()
    settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=False)  # dialogs; invites tested below
    app = App(settings=settings, store=store, forest=forest, llm=llm, answerer=Answerer(store, forest, llm, TZ),
              tz=TZ, locks=defaultdict(asyncio.Lock))
    telegram = FakeTelegram()
    bot = Bot("123:TEST", session=telegram)
    dispatcher = Dispatcher(storage=SQLiteStorage(store.db), app=app)
    router._parent_router = None  # the module-level router is reused by every test's dispatcher
    dispatcher.include_router(router)
    return dispatcher, bot, telegram, llm, store


async def test_question_without_place_then_button(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    assert "Про яку рукавичку" in telegram.sent[-1].text

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Під дубом"))
    assert telegram.sent[-1].text.startswith("4 жолуді на місяць.")
    assert "Членські внески рукавички Під дубом" in telegram.sent[-1].text


async def test_first_document_asks_name_then_unclear_place(setup):
    dispatcher, bot, telegram, llm, store = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    llm.answers.append({"kind": "unclear", "value": ""})

    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f1", file_unique_id="u1", file_name="Порада про гриби.txt", file_size=40)))
    assert "як Вас звати" in telegram.sent[-1].text

    await dispatcher.feed_update(bot, message(text="Мене звати білочка."))
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')
    await dispatcher.feed_update(bot, callback("nm:yes"))
    assert "для якого місця ця порада" in telegram.sent[-1].text

    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Хвойний ліс"))
    saved = telegram.sent[-1].text
    assert 'Дякую, Білочка! Документ "Порада про гриби" збережено' in saved and "хвойний ліс" in saved
    doc = store.documents_of(USER.id)[0]
    assert doc.author_name == "Білочка" and doc.scope.value == "conifer"

    await dispatcher.feed_update(bot, message(text="/my"))
    assert "Ваші документи" in telegram.sent[-1].text
    await dispatcher.feed_update(bot, callback(f"del:{doc.id}"))
    assert store.documents_of(USER.id)  # nothing is deleted before the animal confirms
    await dispatcher.feed_update(bot, callback(f"delok:{doc.id}"))
    assert store.documents_of(USER.id) == []


async def test_photo_and_wrong_format_are_refused(setup):
    dispatcher, bot, telegram, _, _ = setup
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f2", file_unique_id="u2", file_name="scan.jpg", file_size=10)))
    assert "PDF, Word або TXT" in telegram.sent[-1].text


async def test_caption_sets_document_place(setup):
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    telegram.file_bytes = "Опеньки ростуть на старих пеньках.".encode()
    await dispatcher.feed_update(bot, message(caption="це для Під вербою", document=Document(
        file_id="f3", file_unique_id="u3", file_name="Опеньки.txt", file_size=40)))
    assert 'рукавичка "Під вербою"' in telegram.sent[-1].text
    assert llm.prompts == []


async def test_invites_let_neighbours_in(setup):
    dispatcher, bot, telegram, llm, store = setup
    app = dispatcher["app"]
    app.settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=True)
    neighbour = User(id=20, is_bot=False, first_name="Fox")
    stranger = User(id=30, is_bot=False, first_name="Stranger")

    # Nobody has joined yet: whoever started the bot is told where the first invite is.
    await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    assert "журналі запуску бота" in telegram.sent[-1].text and llm.prompts == []

    # The first invite comes from the start log or the command line (python -m bot.invite).
    first = store.create_invite(None, datetime.now(timezone.utc))
    await dispatcher.feed_update(bot, message(text=f"/start inv_{first}"))
    assert telegram.sent[-1].text.startswith("Запрошення прийнято, тепер Ви з нами.")
    await dispatcher.feed_update(bot, message(user=stranger, text=f"/start inv_{first}"))
    assert "вже використане" in telegram.sent[-1].text and not store.is_member(stranger.id)
    await dispatcher.feed_update(bot, message(user=stranger, text="привіт"))
    assert "попросіть, будь ласка, запрошення в сусіда" in telegram.sent[-1].text

    # A member invites a neighbour; the bot remembers who invited whom.
    await dispatcher.feed_update(bot, message(text="/invite"))
    link = telegram.sent[-1].text.split("\n")[1]
    assert link.startswith("https://t.me/fairytale_docs_bot?start=inv_")
    await dispatcher.feed_update(bot, message(user=neighbour, text="/start " + link.split("start=")[1]))
    assert store.is_member(neighbour.id)
    invited_by = store.db.execute("SELECT invited_by FROM members WHERE tg_id = ?", (neighbour.id,)).fetchone()[0]
    assert invited_by == USER.id

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(user=neighbour, text="скільки внесок під дубом?"))
    assert telegram.sent[-1].text.startswith("4 жолуді на місяць.")


async def test_invite_is_not_needed_when_invites_are_off(setup):
    dispatcher, bot, telegram, _, _ = setup
    await dispatcher.feed_update(bot, message(text="/invite"))
    assert "запрошення не потрібні" in telegram.sent[-1].text


async def test_start_and_help_speak_of_invites_only_when_the_bot_is_closed(setup):
    dispatcher, bot, telegram, _, _ = setup
    await dispatcher.feed_update(bot, message(text="/start"))
    await dispatcher.feed_update(bot, message(text="/help"))
    assert all("/invite" not in sent.text for sent in telegram.sent[-2:])

    dispatcher["app"].settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=True)
    store = dispatcher["app"].store
    await dispatcher.feed_update(bot, message(text=f"/start inv_{store.create_invite(None, datetime.now(timezone.utc))}"))
    await dispatcher.feed_update(bot, message(text="/help"))
    assert "/invited" in telegram.sent[-2].text and texts.HELP_INVITES == telegram.sent[-1].text


def test_name_is_cleaned():
    assert clean_name("Мене звати Білочка.") == "Білочка"
    assert clean_name("я - Ведмідь") == "Ведмідь"
    assert clean_name("  зайчик-побігайчик ") == "Зайчик-побігайчик"


async def test_waiting_document_survives_restart(setup, store, forest):
    dispatcher, bot, telegram, llm, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f4", file_unique_id="u4", file_name="Гриби.txt", file_size=30)))
    assert "як Вас звати" in telegram.sent[-1].text

    # A new dispatcher on the same database, as after a restart, still remembers the document.
    restarted = Dispatcher(storage=SQLiteStorage(store.db), app=dispatcher["app"])
    router._parent_router = None
    restarted.include_router(router)
    await restarted.feed_update(bot, message(text="Їжак"))
    await restarted.feed_update(bot, callback("nm:yes"))
    assert "для якого місця ця порада" in telegram.sent[-1].text


async def test_stale_place_button_says_question_was_lost(setup):
    dispatcher, bot, telegram, _, _ = setup
    await dispatcher.feed_update(bot, callback("qp:m:3"))
    assert "загубив це питання" in telegram.sent[-1].text


async def test_inviter_closes_access_in_the_bot(setup):
    dispatcher, bot, telegram, _, store = setup
    dispatcher["app"].settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=True)
    fox = User(id=40, is_bot=False, first_name="Fox")
    store.use_invite(store.create_invite(None, datetime.now(timezone.utc)), USER.id, datetime.now(timezone.utc))
    store.use_invite(store.create_invite(USER.id, datetime.now(timezone.utc)), fox.id, datetime.now(timezone.utc))

    await dispatcher.feed_update(bot, message(text="/invited"))
    assert "Звірі, яких Ви запросили" in telegram.sent[-1].text
    await dispatcher.feed_update(bot, callback(f"rev:{fox.id}"))
    assert store.is_member(fox.id)  # nothing happens before the confirmation
    await dispatcher.feed_update(bot, callback(f"revok:{fox.id}"))
    assert not store.is_member(fox.id)
    await dispatcher.feed_update(bot, message(user=fox, text="скільки внесок під дубом?"))
    assert "попросіть, будь ласка, запрошення" in telegram.sent[-1].text


async def test_typed_yes_confirms_the_name(setup):
    dispatcher, bot, telegram, llm, store = setup
    telegram.file_bytes = "Ця порада для мешканців хвойного лісу. Рижики ростуть після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f5", file_unique_id="u5", file_name="Рижики.txt", file_size=60)))
    await dispatcher.feed_update(bot, message(text="Білочка"))
    await dispatcher.feed_update(bot, message(text="Так"))
    assert store.user_name(USER.id) == "Білочка"
    assert 'Дякую, Білочка! Документ "Рижики" збережено' in telegram.sent[-1].text


async def test_typed_no_asks_the_name_again(setup):
    dispatcher, bot, telegram, _, store = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f6", file_unique_id="u6", file_name="Гриби.txt", file_size=30)))
    await dispatcher.feed_update(bot, message(text="Білка"))
    await dispatcher.feed_update(bot, message(text="ні, не так"))
    assert "своє ім'я ще раз" in telegram.sent[-1].text and store.user_name(USER.id) is None


async def test_greeting_and_thanks_do_not_go_to_the_model(setup):
    dispatcher, bot, telegram, llm, _ = setup
    await dispatcher.feed_update(bot, message(text="Привіт!"))
    assert telegram.sent[-1].text.startswith("Вітаю! Поставте, будь ласка, питання")
    await dispatcher.feed_update(bot, message(text="Дякую, дуже!"))
    assert telegram.sent[-1].text.startswith("Будь ласка!")
    assert llm.prompts == [] and llm.pick_prompts == []


async def test_short_follow_up_asks_the_same_question_for_another_place(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    llm.answers.append({"found": True, "answer": "3 шишки на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="А у соснах?"))
    assert telegram.sent[-1].text.startswith("3 шишки на місяць.")
    prompt = llm.prompts[-1][1]
    assert "Питання звіра: скільки внесок у соснах?" in prompt
    assert "Членські внески рукавички У соснах" in prompt


async def test_plain_text_file_of_another_kind_is_accepted(setup):
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    telegram.file_bytes = "Порада для листяного лісу: горіхи збирайте у вересні.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f7", file_unique_id="u7", file_name="Горіхи.md", mime_type="text/markdown", file_size=60)))
    assert 'Документ "Горіхи" збережено' in telegram.sent[-1].text and "листяний ліс" in telegram.sent[-1].text


async def test_question_waits_for_the_first_archive_load(setup):
    dispatcher, bot, telegram, llm, _ = setup
    ready = asyncio.Event()
    dispatcher["app"].archive_ready = ready
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    asking = asyncio.create_task(dispatcher.feed_update(bot, message(text="скільки внесок під дубом?")))
    await asyncio.sleep(0.05)
    assert "вперше читаю Лісовий архів" in telegram.sent[-1].text and llm.prompts == []
    ready.set()
    await asking
    assert telegram.sent[-1].text.startswith("4 жолуді на місяць.")


async def test_old_word_without_antiword_asks_for_docx(setup, monkeypatch):
    from bot import handlers
    dispatcher, bot, telegram, _, _ = setup
    monkeypatch.setattr(handlers, "old_word_supported", lambda: False)
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f8", file_unique_id="u8", file_name="Гриби.doc", file_size=20000)))
    assert "старий формат Word" in telegram.sent[-1].text


async def send_unclear_document(dispatcher, bot, telegram, llm, store, name="Report.txt"):
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()  # no place at all, so the model is not asked
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f9", file_unique_id="u9", file_name=name, file_size=30)))
    assert "для якого місця ця порада" in telegram.sent[-1].text


async def test_question_while_document_waits_for_place_is_answered_not_filed(setup):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store, "Гриби.txt")

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом"))
    assert telegram.sent[-2].text.startswith("4 жолуді на місяць.")
    assert 'А для документа "Гриби" оберіть' in telegram.sent[-1].text and telegram.sent[-1].reply_markup
    assert store.documents_of(USER.id) == []  # the question named a mitten, the document was not filed there

    await dispatcher.feed_update(bot, message(text="для всього лісу"))
    assert "Він стосується: увесь ліс" in telegram.sent[-1].text


async def test_typed_whole_forest_and_part_are_places(setup):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    await dispatcher.feed_update(bot, message(text="увесь ліс"))
    assert "Він стосується: увесь ліс" in telegram.sent[-1].text
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    await dispatcher.feed_update(bot, message(text="у хвойному лісі"))
    assert "Він стосується: хвойний ліс" in telegram.sent[-1].text
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    await dispatcher.feed_update(bot, message(text="не знаю"))
    assert "напишіть лише місце" in telegram.sent[-1].text


async def test_forest_part_as_the_place_offers_its_mittens(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    await dispatcher.feed_update(bot, message(text="у хвойному лісі"))
    buttons = [b.text for row in telegram.sent[-1].reply_markup.inline_keyboard for b in row]
    assert "хвойний ліс" in telegram.sent[-1].text and buttons == ["Під ялиною", "У соснах", "Біля мурашника"]

    llm.answers.append({"found": True, "answer": "3 шишки на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="у соснах"))
    assert telegram.sent[-1].text.startswith("3 шишки на місяць.")
    assert "Питання звіра: скільки внесок?" in llm.prompts[-1][1]


async def test_new_question_instead_of_the_mitten_is_a_new_question(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    llm.answers.append({"found": True, "answer": "4 жолуді.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом для дорослих?"))
    assert "Питання звіра: Скільки внесок під дубом для дорослих?" in llm.prompts[-1][1]


async def test_question_instead_of_name_is_answered(setup):
    dispatcher, bot, telegram, llm, store = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f10", file_unique_id="u10", file_name="Гриби.txt", file_size=30)))
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом"))
    assert telegram.sent[-2].text.startswith("4 жолуді на місяць.")
    assert "напишіть, будь ласка, своє ім'я" in telegram.sent[-1].text and store.user_name(USER.id) is None
    await dispatcher.feed_update(bot, message(text="Білочка"))
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')


async def test_cancel_says_which_documents_were_dropped(setup):
    dispatcher, bot, telegram, _, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f11", file_unique_id="u11", file_name="Гриби.txt", file_size=30)))
    await dispatcher.feed_update(bot, message(text="/cancel"))
    assert '"Гриби"' in telegram.sent[-1].text and "надішліть його ще раз" in telegram.sent[-1].text


async def test_name_can_be_changed_later(setup):
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Білка", datetime.now(timezone.utc))
    telegram.file_bytes = "Порада для листяного лісу: горіхи збирайте у вересні.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f12", file_unique_id="u12", file_name="Горіхи.txt", file_size=60)))
    await dispatcher.feed_update(bot, message(text="/name"))
    await dispatcher.feed_update(bot, message(text="Білочка"))
    await dispatcher.feed_update(bot, callback("nm:yes"))
    assert "тепер я підписую Ваші документи: Білочка" in telegram.sent[-1].text
    assert store.user_name(USER.id) == "Білочка" and store.documents_of(USER.id)[0].author_name == "Білочка"


async def test_unexpected_error_gets_a_polite_reply(setup, monkeypatch):
    dispatcher, bot, telegram, _, _ = setup

    async def broken(*args, **kwargs):
        raise AttributeError("'list' object has no attribute 'get'")

    monkeypatch.setattr(dispatcher["app"].answerer, "ask", broken)
    await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    assert "щось пішло не так" in telegram.sent[-1].text


async def test_invited_list_shows_the_telegram_name(setup):
    dispatcher, bot, telegram, _, store = setup
    dispatcher["app"].settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=True)
    store.use_invite(store.create_invite(None, datetime.now(timezone.utc)), USER.id, datetime.now(timezone.utc))
    fox = User(id=50, is_bot=False, first_name="Foxy", last_name="Red", username="foxy")
    await dispatcher.feed_update(bot, message(text="/invite"))
    link = telegram.sent[-1].text.split("\n")[1]
    await dispatcher.feed_update(bot, message(user=fox, text="/start " + link.split("start=")[1]))
    await dispatcher.feed_update(bot, message(text="/invited"))
    assert "Foxy Red @foxy" in telegram.sent[-1].reply_markup.inline_keyboard[0][0].text


async def test_document_sent_while_a_question_waits_for_its_mitten(setup):
    # The old mitten button answers the question, and the document still waits for its place.
    dispatcher, bot, telegram, llm, store = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    mittens = telegram.sent[-1]
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    place_buttons = buttons(telegram.sent[-1])

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, press(mittens, "Під дубом"))
    assert telegram.sent[-2].text.startswith("4 жолуді на місяць.")
    assert 'А для документа "Report" оберіть' in telegram.sent[-1].text
    assert await flow_state(dispatcher, bot) == "Flow:waiting_doc_place"

    await dispatcher.feed_update(bot, callback(dict(place_buttons)["Увесь ліс"]))
    assert "Він стосується: увесь ліс" in telegram.sent[-1].text and store.documents_of(USER.id)


async def test_name_after_a_pending_question_is_a_name(setup):
    dispatcher, bot, telegram, llm, store = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    mittens = telegram.sent[-1]
    telegram.file_bytes = "Ця порада для мешканців хвойного лісу. Рижики ростуть після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f13", file_unique_id="u13", file_name="Рижики.txt", file_size=60)))
    assert "як Вас звати" in telegram.sent[-1].text

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, press(mittens, "Під дубом"))
    assert "напишіть, будь ласка, своє ім'я" in telegram.sent[-1].text  # the document still needs the name
    await dispatcher.feed_update(bot, message(text="Білочка"))
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')
    await dispatcher.feed_update(bot, callback("nm:yes"))
    assert 'Документ "Рижики" збережено' in telegram.sent[-1].text


async def test_question_needing_a_mitten_does_not_push_out_a_document(setup):
    # The answer takes a few seconds; a document comes meanwhile and asks for its place.
    from bot.answering import Reply
    dispatcher, bot, telegram, llm, store = setup
    release = asyncio.Event()

    async def slow_ask(question, now, mitten=None):
        await release.wait()
        return Reply("need_place")

    dispatcher["app"].answerer.ask = slow_ask
    asking = asyncio.create_task(dispatcher.feed_update(bot, message(text="скільки внесок?")))
    await asyncio.sleep(0.01)
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    release.set()
    await asking
    assert "назвіть у ньому рукавичку" in telegram.sent[-2].text
    # The last word is the document's again, so "під дубом" typed next is clearly its place.
    assert 'А для документа "Report" оберіть' in telegram.sent[-1].text and telegram.sent[-1].reply_markup
    assert await flow_state(dispatcher, bot) == "Flow:waiting_doc_place"


async def test_old_place_buttons_do_not_file_the_next_document(setup):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store, "Перший.txt")
    first = dict(buttons(telegram.sent[-1]))
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f14", file_unique_id="u14", file_name="Другий.txt", file_size=30)))
    assert "збережу його одразу після попереднього" in telegram.sent[-1].text

    await dispatcher.feed_update(bot, callback(first["Під дубом"]))
    assert 'для якого місця ця порада' in telegram.sent[-1].text and '"Другий"' in telegram.sent[-1].text
    second = dict(buttons(telegram.sent[-1]))

    await dispatcher.feed_update(bot, callback(first["Увесь ліс"]))  # pressed under the first document again
    assert telegram.sent[-1].text.startswith('Документ "Перший" уже збережено, він стосується: рукавичка "Під дубом"')
    await dispatcher.feed_update(bot, callback("dp:f:"))  # a keyboard from before documents had ids
    assert 'Ці кнопки були для попереднього документа. Для якого місця документ "Другий"?' == telegram.sent[-1].text
    assert dict(buttons(telegram.sent[-1])) == second
    assert [d.title for d in store.documents_of(USER.id)] == ["Перший"]

    await dispatcher.feed_update(bot, callback(second["Увесь ліс"]))
    assert {d.title: d.scope.describe() for d in store.documents_of(USER.id)} == {
        "Перший": 'рукавичка "Під дубом"', "Другий": "увесь ліс"}


async def test_question_naming_a_forest_part_offers_its_mittens(setup):
    dispatcher, bot, telegram, llm, store = setup
    add_archive(store, dispatcher["app"].forest, "Членські внески рукавички Під вербою", "Внесок 3 горіхи на місяць.")
    add_archive(store, dispatcher["app"].forest, "Членські внески рукавички Біля бобрової греблі", "Внесок 2 рибини.")
    await dispatcher.feed_update(bot, message(text="Скільки внесок біля річки?"))
    assert "Оберіть, будь ласка, рукавичку (біля річки)" in telegram.sent[-1].text
    options = dict(buttons(telegram.sent[-1]))
    assert list(options) == ["Під вербою", "Біля бобрової греблі"]

    llm.answers.append({"found": True, "answer": "2 рибини на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, callback(options["Біля бобрової греблі"]))
    assert telegram.sent[-1].text.startswith("2 рибини на місяць.")
    assert "Питання звіра: Скільки внесок біля річки?" in llm.prompts[-1][1]


async def test_saved_document_does_not_leave_an_old_question_waiting_for_text(setup):
    # The mitten buttons still answer the old question, but "Дякую!" after the document is small talk again.
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    mittens = telegram.sent[-1]
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(caption="для всього лісу", document=Document(
        file_id="f15", file_unique_id="u15", file_name="Гриби.txt", file_size=30)))
    assert "Він стосується: увесь ліс" in telegram.sent[-1].text
    await dispatcher.feed_update(bot, message(text="Дякую!"))
    assert telegram.sent[-1].text.startswith("Будь ласка!")
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, press(mittens, "Під дубом"))
    assert telegram.sent[-1].text.startswith("4 жолуді на місяць.")


async def test_buttons_under_an_older_question_answer_that_question(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers += [{"needs_place": True}, {"needs_place": True}]
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    first = telegram.sent[-1]
    await dispatcher.feed_update(bot, message(text="до якого числа здавати внесок?"))
    second = telegram.sent[-1]
    llm.answers.append({"found": True, "answer": "4 жолуді.", "sources": [1]})
    await dispatcher.feed_update(bot, press(first, "Під дубом"))
    assert telegram.sent[-1].text.startswith("4 жолуді.") and "Питання звіра: скільки внесок?" in llm.prompts[-1][1]
    llm.answers.append({"found": True, "answer": "До 10 числа.", "sources": [1]})
    await dispatcher.feed_update(bot, press(second, "Під дубом"))
    assert telegram.sent[-1].text.startswith("До 10 числа.")
    assert "Питання звіра: до якого числа здавати внесок?" in llm.prompts[-1][1]


async def test_slow_question_does_not_push_out_a_document_waiting_for_the_name(setup):
    from bot.answering import Reply
    dispatcher, bot, telegram, llm, store = setup
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_ask(question, now, mitten=None):
        started.set()
        await release.wait()
        return Reply("need_place")

    dispatcher["app"].answerer.ask = slow_ask
    asking = asyncio.create_task(dispatcher.feed_update(bot, message(text="скільки внесок?")))
    await started.wait()
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f16", file_unique_id="u16", file_name="Гриби.txt", file_size=30)))
    assert "як Вас звати" in telegram.sent[-1].text
    release.set()
    await asking
    assert "назвіть у ньому рукавичку" in telegram.sent[-2].text and "своє ім'я" in telegram.sent[-1].text
    assert await flow_state(dispatcher, bot) == "Flow:waiting_name"
    await dispatcher.feed_update(bot, message(text="Білочка"))
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')


async def test_question_while_the_name_is_confirmed_brings_the_confirmation_back(setup):
    dispatcher, bot, telegram, llm, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, message(document=Document(
        file_id="f17", file_unique_id="u17", file_name="Гриби.txt", file_size=30)))
    await dispatcher.feed_update(bot, message(text="Білочка"))
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    assert telegram.sent[-2].text.startswith("4 жолуді на місяць.")
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')
    assert [data for _, data in buttons(telegram.sent[-1])] == ["nm:yes", "nm:no"]


async def test_mitten_button_during_rename_asks_for_the_new_name(setup):
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Білка", datetime.now(timezone.utc))
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    mittens = telegram.sent[-1]
    await dispatcher.feed_update(bot, message(text="/name"))
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, press(mittens, "Під дубом"))
    assert telegram.sent[-1].text == texts.ASK_NEW_NAME  # no document waits, so none is mentioned


async def test_document_left_in_the_queue_by_the_old_code_is_saved(setup):
    # The old code could lose the dialog state while a document stayed queued, with an old keyboard.
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    context = dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id)
    await context.set_state("Flow:waiting_question_place")
    await context.set_data({"question": "скільки внесок?", "queue": [{
        "filename": "Старий.txt", "title": "Старий", "text": "Збирайте гриби.", "sent_at": "2026-10-28T12:00:00+00:00"}]})
    await dispatcher.feed_update(bot, callback("dp:m:3"))
    assert 'Документ "Старий" збережено' in telegram.sent[-1].text and store.documents_of(USER.id)


def document_update(name: str, file_id: str, caption: str | None = None) -> Update:
    return message(caption=caption, document=Document(file_id=file_id, file_unique_id=file_id, file_name=name,
                                                      file_size=30))


async def test_cancel_while_the_model_places_a_document_drops_it(setup):
    # /cancel answers at once; the document the model was still placing is not saved and nothing waits after.
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_json(system, user, **kwargs):
        started.set()
        await release.wait()
        return {"kind": "unclear", "value": ""}

    llm.json = slow_json
    telegram.file_bytes = "Білі гриби шукайте під дубом після дощу.".encode()
    sending = asyncio.create_task(dispatcher.feed_update(bot, document_update("Гриби.txt", "c1")))
    await started.wait()
    await dispatcher.feed_update(bot, message(text="/cancel"))
    assert telegram.sent[-1].text == texts.CANCELLED_DOCS.format(titles='"Гриби"')
    release.set()
    await sending
    assert telegram.sent[-1].text == texts.CANCELLED_DOCS.format(titles='"Гриби"')  # no place question after it
    assert store.documents_of(USER.id) == [] and await flow_state(dispatcher, bot) is None


async def test_file_still_being_read_at_cancel_is_dropped(setup, monkeypatch):
    from bot import handlers
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    started, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_extract(data, filename, langs, ext):
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return "Порада для всього лісу: гриби після дощу."

    monkeypatch.setattr(handlers, "extract_text", slow_extract)
    sending = asyncio.create_task(dispatcher.feed_update(bot, document_update("Скан.pdf", "c2")))
    await started.wait()
    await dispatcher.feed_update(bot, message(text="/cancel"))
    release.set()
    await sending
    assert telegram.sent[-1].text == texts.CANCELLED_DOCS.format(titles='"Скан"') and store.documents_of(USER.id) == []


async def test_start_in_the_middle_says_which_documents_were_dropped(setup):
    dispatcher, bot, telegram, _, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, document_update("Гриби.txt", "c3"))
    await dispatcher.feed_update(bot, message(text="/start"))
    assert telegram.sent[-2].text == texts.CANCELLED_DOCS.format(titles='"Гриби"')
    assert telegram.sent[-1].text == texts.START


async def test_typed_mitten_after_a_document_answers_the_waiting_question(setup):
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="До якого числа здавати внесок?"))
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Увесь ліс"))
    llm.answers.append({"found": True, "answer": "До 10 числа.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="у соснах"))
    assert "Питання звіра: До якого числа здавати внесок?" in llm.prompts[-1][1]
    assert 'рукавичка "У соснах"' in llm.prompts[-1][1]


async def test_question_buttons_answer_once(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    mittens = telegram.sent[-1]
    llm.answers += [{"found": True, "answer": "4 жолуді на місяць.", "sources": [1]}] * 2
    await asyncio.gather(dispatcher.feed_update(bot, press(mittens, "Під дубом")),
                         dispatcher.feed_update(bot, press(mittens, "Під дубом")))
    answers = [sent.text for sent in telegram.sent if sent.text.startswith("4 жолуді")]
    assert len(answers) == 1 and len(llm.prompts) == 2  # the need_place question and one answer
    n = len(telegram.sent)
    await dispatcher.feed_update(bot, press(mittens, "У соснах"))  # the same buttons later: nothing to say
    assert len(telegram.sent) == n


async def test_document_during_rename_waits_for_the_name(setup):
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Білка", datetime.now(timezone.utc))
    await dispatcher.feed_update(bot, message(text="/name"))
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, document_update("Гриби.txt", "c4"))
    assert telegram.sent[-1].text == texts.DOC_AFTER_NAME.format(title="Гриби")


async def test_saved_document_buttons_say_where_it_is_now(setup):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store, "Гриби.txt")
    keyboard = telegram.sent[-1]
    await dispatcher.feed_update(bot, press(keyboard, "Під дубом"))
    doc = store.documents_of(USER.id)[0]
    store.change_scope(doc.id, USER.id, FOREST_SCOPE)  # "Змінити місце"
    await dispatcher.feed_update(bot, press(keyboard, "Хвойний ліс"))
    assert telegram.sent[-1].text == texts.DOC_ALREADY_SAVED.format(title="Гриби", scope="увесь ліс")


async def test_old_queue_without_ids_is_not_filed_by_a_second_old_button(setup):
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    context = dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id)
    await context.set_state("Flow:waiting_doc_place")
    await context.set_data({"queue": [
        {"filename": f"{title}.txt", "title": title, "text": "Збирайте гриби.", "sent_at": "2026-10-28T12:00:00+00:00"}
        for title in ("Перший", "Другий")]})
    await dispatcher.feed_update(bot, callback("dp:m:3"))
    await dispatcher.feed_update(bot, callback("dp:m:3"))  # the same old keyboard again
    assert [d.title for d in store.documents_of(USER.id)] == ["Перший"]
    assert telegram.sent[-1].text.startswith("Ці кнопки були для попереднього документа")


async def test_document_stranded_by_the_old_code_is_asked_about_after_an_answer(setup):
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    context = dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id)
    await context.set_data({"queue": [{"filename": "Старий.txt", "title": "Старий", "text": "Збирайте гриби.",
                                       "sent_at": "2026-10-28T12:00:00+00:00"}]})
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    assert telegram.sent[-2].text.startswith("4 жолуді") and 'А для документа "Старий"' in telegram.sent[-1].text
    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Увесь ліс"))
    assert 'Документ "Старий" збережено' in telegram.sent[-1].text
