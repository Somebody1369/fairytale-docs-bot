"""Whole conversations through aiogram with a fake Telegram server."""

import asyncio
from datetime import datetime, timezone
from itertools import count
from zoneinfo import ZoneInfo

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, EditMessageReplyMarkup, EditMessageText, GetFile, GetMe, SendMessage
from aiogram.types import CallbackQuery, Chat, Document, File, InaccessibleMessage, Message, Update, User

from bot.answering import Answerer, Reply
from bot.config import Settings
from bot.extract import UnreadableScan
from bot.fsm import SQLiteStorage
from bot import handlers, texts
from bot.handlers import App, Flow, clean_name, router
from bot.llm import LLMUnavailable
from bot.places import FOREST_SCOPE

from .conftest import FOREST_MAP_TEXT, FakeLLM, add_archive

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


def callback(data: str, user: User = USER) -> Update:
    chat = Chat(id=user.id, type="private")
    msg = Message(message_id=next(ids), date=datetime.now(timezone.utc), chat=chat, text="...")
    query = CallbackQuery(id=str(next(ids)), from_user=user, chat_instance="c", data=data, message=msg)
    return Update(update_id=next(ids), callback_query=query)


def same_message_callback(message_id: int, data: str, inaccessible: bool = False) -> Update:
    """A button under one given message: tapped twice, both taps come from it. A message the bot can no
    longer see (deleted, or too old) comes as an InaccessibleMessage, which has no edit_* methods."""
    msg = (InaccessibleMessage(chat=CHAT, message_id=message_id) if inaccessible
           else Message(message_id=message_id, date=datetime.now(timezone.utc), chat=CHAT, text="..."))
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
              tz=TZ)
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


async def test_gate_is_silent_in_groups_and_alerts_under_buttons(setup, monkeypatch):
    # The bot answers in private chats only, so a group hears nothing. A non-member is asked for an invite,
    # and a button they press (under an old message of the bot) answers with the same words in an alert.
    dispatcher, bot, telegram, _, store = setup
    dispatcher["app"].settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=True)
    store.use_invite(store.create_invite(None, datetime.now(timezone.utc)), USER.id, datetime.now(timezone.utc))
    stranger = User(id=99, is_bot=False, first_name="Stranger")
    group = Chat(id=-5, type="group")
    await dispatcher.feed_update(bot, Update(update_id=next(ids), message=Message(
        message_id=next(ids), date=datetime.now(timezone.utc), chat=group, from_user=stranger, text="привіт")))
    assert telegram.sent == []
    await dispatcher.feed_update(bot, message(user=stranger, text="/help"))
    assert telegram.sent[-1].text == texts.NEED_INVITE

    alerts = []
    make_request = telegram.make_request

    async def record(bot_, method, timeout=None):
        if isinstance(method, AnswerCallbackQuery):
            alerts.append((method.text, method.show_alert))
        return await make_request(bot_, method, timeout)

    monkeypatch.setattr(telegram, "make_request", record)
    for data in ("del:1", "qp:abc:m:1", "nm:yes"):
        await dispatcher.feed_update(bot, callback(data, user=stranger))
    assert alerts == [(texts.NEED_INVITE, True)] * 3


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
    await dispatcher.feed_update(bot, callback("qp:gone1234:m:3"))
    assert "загубив це питання" in telegram.sent[-1].text


async def test_buttons_the_bot_cannot_read_get_a_polite_reply(setup):
    dispatcher, bot, telegram, llm, store = setup
    await dispatcher.feed_update(bot, callback("qp:gone1234:m:99"))  # a mitten no longer on the map
    assert telegram.sent[-1].text == texts.QUESTION_LOST
    await dispatcher.feed_update(bot, callback("dp:gone1234:f:"))
    assert telegram.sent[-1].text == texts.DOC_LOST

    # A mitten no longer on the map under a question still known: the mittens are offered again.
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    key = dict(buttons(telegram.sent[-1]))["Під дубом"].split(":")[1]
    await dispatcher.feed_update(bot, callback(f"qp:{key}:m:99"))
    assert telegram.sent[-1].text == texts.QUESTION_PLACE_GONE and llm.answers == []
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Під дубом"))
    assert telegram.sent[-1].text.startswith("4 жолуді на місяць.")

    # Under the waiting document itself: its own buttons again, not "for the previous document".
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    own = dict(buttons(telegram.sent[-1]))
    key = own["Увесь ліс"].split(":")[1]
    await dispatcher.feed_update(bot, callback(f"dp:{key}:m:99"))
    assert telegram.sent[-1].text == texts.DOC_PLACE_GONE.format(title="Report")
    assert dict(buttons(telegram.sent[-1])) == own and store.documents_of(USER.id) == []
    # Once the document is saved, such a button says where it is.
    await dispatcher.feed_update(bot, callback(own["Увесь ліс"]))
    await dispatcher.feed_update(bot, callback(f"dp:{key}:m:99"))
    assert telegram.sent[-1].text == texts.DOC_ALREADY_SAVED.format(title="Report", scope="увесь ліс")
    # "Змінити місце" for a mitten no longer on the map: the places are offered again, the document stays put.
    doc = store.documents_of(USER.id)[0]
    await dispatcher.feed_update(bot, callback(f"cs:{doc.id}:m:99"))
    assert telegram.sent[-1].text == texts.CHANGE_PLACE_GONE and store.get(doc.id).scope == FOREST_SCOPE
    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Під дубом"))
    assert telegram.sent[-1].text == texts.PLACE_CHANGED.format(title="Report", scope='рукавичка "Під дубом"')


async def test_button_data_the_bot_does_not_make_stops_spinning_quietly(setup, monkeypatch):
    # Telegram warns that a client may send any data under a button: such a button is answered, so it stops
    # spinning, and no apology or other message follows.
    dispatcher, bot, telegram, _, _ = setup
    answers = []
    make_request = telegram.make_request

    async def record(bot_, method, timeout=None):
        if isinstance(method, AnswerCallbackQuery):
            answers.append(method.text)
        return await make_request(bot_, method, timeout)

    monkeypatch.setattr(telegram, "make_request", record)
    odd = ("del:abc", "delok:", "rev:x", "revok:1:2", "chg:x", "cs:x:m:1", "nm:maybe", "qp:m", "dp:x", "zz",
           # numbers SQLite cannot hold
           "del:99999999999999999999", "delok:-99999999999999999999", "revok:99999999999999999999",
           "chg:99999999999999999999", "cs:99999999999999999999:f:")
    for data in odd:
        await dispatcher.feed_update(bot, callback(data))
    assert answers == [texts.UNKNOWN_BUTTON] * len(odd) and telegram.sent == []


async def test_buttons_made_before_the_map_changed_keep_their_mitten(setup):
    # "Карта лісу" is read again at every archive sync. A mitten added to it before "Під дубом" must not make
    # an older "Під дубом" button answer for another mitten or file a document there.
    dispatcher, bot, telegram, llm, store = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    question_buttons = telegram.sent[-1]
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    document_buttons = telegram.sent[-1]
    new_map = FOREST_MAP_TEXT.replace("Під дубом — листяний ліс.", "У ялівці — хвойний ліс.\n–\nПід дубом — листяний ліс.")
    assert dispatcher["app"].forest.update_from_map(new_map)

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    asked = len(llm.prompts)
    await dispatcher.feed_update(bot, press(question_buttons, "Під дубом"))
    # Asked about it: "У ялівці" has no documents, so the model would not be asked at all.
    assert len(llm.prompts) == asked + 1 and 'Місце з питання: рукавичка "Під дубом"' in llm.prompts[-1][1]
    await dispatcher.feed_update(bot, press(document_buttons, "Під дубом"))
    assert [doc.scope.describe() for doc in store.documents_of(USER.id)] == ['рукавичка "Під дубом"']


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


async def test_document_read_while_access_is_closed_is_not_used(setup, monkeypatch):
    # A big scan takes minutes to read, and the fox's access is closed meanwhile: the document is kept like the
    # fox's other ones, not used in answers, and the fox is not told it is. A new invite brings it back.
    dispatcher, bot, telegram, _, store = setup
    dispatcher["app"].settings = Settings(telegram_token="1:x", groq_api_key="x", invites_only=True)
    fox, now = User(id=40, is_bot=False, first_name="Fox"), datetime.now(timezone.utc)
    store.use_invite(store.create_invite(None, now), USER.id, now)
    store.use_invite(store.create_invite(USER.id, now), fox.id, now)
    store.set_user_name(fox.id, "Лис", now)
    started, release = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_extract(data, filename, langs, ext):
        loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), loop).result()
        return "Порада для всього лісу: при пожежі збирайтеся біля річки."

    monkeypatch.setattr(handlers, "extract_text", slow_extract)
    sending = asyncio.create_task(dispatcher.feed_update(bot, message(user=fox, document=Document(
        file_id="r1", file_unique_id="r1", file_name="Пожежа.pdf", file_size=30))))
    await started.wait()
    await dispatcher.feed_update(bot, callback(f"revok:{fox.id}"))
    release.set()
    await sending
    assert telegram.sent[-1].text == texts.NEED_INVITE and store.documents_of(fox.id) == []
    assert store.use_invite(store.create_invite(USER.id, now), fox.id, now)
    assert [doc.title for doc in store.documents_of(fox.id)] == ["Пожежа"]


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


async def test_document_stranded_by_a_restart_is_asked_about_after_an_answer(setup, monkeypatch):
    # The queue is stored before the model places the document: a restart meanwhile leaves the document
    # waiting with no question about it and no dialog state.
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    deciding = asyncio.Event()

    async def classify(*args):
        deciding.set()
        await asyncio.Event().wait()  # the model is deciding when the container stops

    monkeypatch.setattr(handlers, "classify_document", classify)
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    sending = asyncio.create_task(dispatcher.feed_update(bot, message(document=Document(
        file_id="f9", file_unique_id="u9", file_name="Гриби.txt", file_size=30))))
    await deciding.wait()
    # A question while the model still decides does not ask about the document yet.
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    assert telegram.sent[-1].text.startswith("4 жолуді на місяць.")
    sending.cancel()  # the restart
    await asyncio.gather(sending, return_exceptions=True)
    assert await flow_state(dispatcher, bot) is None

    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    assert telegram.sent[-2].text.startswith("4 жолуді") and 'А для документа "Гриби"' in telegram.sent[-1].text
    assert await flow_state(dispatcher, bot) == Flow.waiting_doc_place.state
    await dispatcher.feed_update(bot, press(telegram.sent[-1], "Увесь ліс"))
    assert 'Документ "Гриби" збережено' in telegram.sent[-1].text


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


async def test_text_in_a_document_state_with_no_document_is_an_ordinary_message(setup):
    # The reply about saving the last document failed, so the dialog still waits for a place while no document
    # waits: "Дякую" is small talk again, not "оберіть місце кнопкою" for a document that is not there.
    dispatcher, bot, telegram, _, _ = setup
    await dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id).set_state(Flow.waiting_doc_place)
    await dispatcher.feed_update(bot, message(text="Дякую"))
    assert telegram.sent[-1].text == texts.THANKS and await flow_state(dispatcher, bot) is None


async def test_document_after_a_dialog_left_waiting_for_a_place_is_placed(setup):
    # The same failed reply, then the next file: it is placed at once, not "збережу його одразу після
    # попереднього" behind a document that is not there.
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    await dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id).set_state(Flow.waiting_doc_place)
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()  # no place at all, so the model is not asked
    await dispatcher.feed_update(bot, document_update("Гриби.txt", "w1"))
    assert "для якого місця ця порада" in telegram.sent[-1].text and telegram.sent[-1].reply_markup


async def test_scan_that_cannot_be_read_in_time_asks_for_a_document_with_text(setup, monkeypatch):
    # As agreed for scans that cannot be read: not "maybe it is damaged, send it again", which would time out again.
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))

    def unreadable(data, filename, langs, ext):
        raise UnreadableScan("Tesseract process timeout")

    monkeypatch.setattr(handlers, "extract_text", unreadable)
    await dispatcher.feed_update(bot, document_update("Скан.pdf", "t1"))
    assert telegram.sent[-1].text == texts.DOC_EMPTY.format(filename="Скан.pdf") and store.documents_of(USER.id) == []


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


async def test_place_with_a_question_mark_answers_the_waiting_question(setup):
    # /help teaches the short form with a question mark: "А у соснах?" answers "Про яку рукавичку?" too.
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    llm.answers.append({"found": True, "answer": "3 шишки на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="А у соснах?"))
    assert telegram.sent[-1].text.startswith("3 шишки на місяць.")
    assert "Питання звіра: скільки внесок?" in llm.prompts[-1][1] and 'рукавичка "У соснах"' in llm.prompts[-1][1]


async def test_part_with_a_question_mark_offers_its_mittens(setup):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    await dispatcher.feed_update(bot, message(text="у хвойному лісі?"))
    assert [text for text, _ in buttons(telegram.sent[-1])] == ["Під ялиною", "У соснах", "Біля мурашника"]


async def test_thanks_while_a_question_waits_for_its_mitten_is_small_talk(setup):
    # No Groq request is spent on "Дякую", and the question still waits: a place typed next answers it.
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    picks = len(llm.pick_prompts)
    await dispatcher.feed_update(bot, message(text="Дякую"))
    assert telegram.sent[-1].text == texts.THANKS and len(llm.pick_prompts) == picks and len(llm.prompts) == 1
    llm.answers.append({"found": True, "answer": "3 шишки на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="у соснах"))
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


async def test_only_so_many_documents_wait_for_the_name(setup):
    # Each waiting document keeps its text in the dialog state, and /cancel names them all in one message.
    dispatcher, bot, telegram, _, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    for i in range(handlers.QUEUE_LIMIT + 1):
        await dispatcher.feed_update(bot, document_update(f"Порада {i}.txt", f"q{i}"))
    assert telegram.sent[-1].text == texts.DOC_QUEUE_FULL.format(title=f"Порада {handlers.QUEUE_LIMIT}")
    data = await dispatcher.fsm.get_context(bot, chat_id=USER.id, user_id=USER.id).get_data()
    assert len(data["queue"]) == handlers.QUEUE_LIMIT


async def test_file_beyond_the_queue_is_not_read(setup, monkeypatch):
    # A scan takes minutes to read: a file that will be refused is not downloaded and read first.
    dispatcher, bot, telegram, _, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    for i in range(handlers.QUEUE_LIMIT):
        await dispatcher.feed_update(bot, document_update(f"Порада {i}.txt", f"q{i}"))
    read = []
    monkeypatch.setattr(handlers, "extract_text", lambda *args: read.append(args) or "")
    await dispatcher.feed_update(bot, document_update("Ще одна.txt", "extra"))
    assert telegram.sent[-1].text == texts.DOC_QUEUE_FULL.format(title="Ще одна") and not read


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
    data = telegram.sent[-1].reply_markup.inline_keyboard[0][0].callback_data
    await dispatcher.feed_update(bot, same_message_callback(next(ids), data, inaccessible=True))  # a new message
    assert telegram.sent[-1].text == texts.CONFIRM_REVOKE.format(name="Foxy Red @foxy")


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
    await dispatcher.feed_update(bot, callback("dp:dropped1:f:"))  # under a document dropped with /cancel
    assert 'Це місце було для попереднього документа. Для якого місця документ "Другий"?' == telegram.sent[-1].text
    assert dict(buttons(telegram.sent[-1])) == second
    assert [d.title for d in store.documents_of(USER.id)] == ["Перший"]

    await dispatcher.feed_update(bot, callback(second["Увесь ліс"]))
    assert {d.title: d.scope.describe() for d in store.documents_of(USER.id)} == {
        "Перший": 'рукавичка "Під дубом"', "Другий": "увесь ліс"}


async def test_place_typed_twice_does_not_file_the_next_document(setup, monkeypatch):
    # "Під дубом" sent twice, the second while the bot still reports saving the first document: the next document,
    # not asked about yet, is not filed there, and the bot says where the first one is.
    dispatcher, bot, telegram, llm, store = setup
    lock = dispatcher["app"].locks[USER.id] = WatchedLock()
    await send_unclear_document(dispatcher, bot, telegram, llm, store, "Перший.txt")
    await dispatcher.feed_update(bot, document_update("Другий.txt", "t2"))
    make_request, saving, go = telegram.make_request, asyncio.Event(), asyncio.Event()

    async def slow(bot_, method, timeout=None):
        if isinstance(method, SendMessage) and "збережено" in method.text and not saving.is_set():
            saving.set()
            await go.wait()
        return await make_request(bot_, method, timeout)

    monkeypatch.setattr(telegram, "make_request", slow)
    first = asyncio.create_task(dispatcher.feed_update(bot, message(text="під дубом")))
    await saving.wait()
    second = asyncio.create_task(dispatcher.feed_update(bot, message(text="під дубом")))
    await lock.waited.wait()
    go.set()
    await asyncio.gather(first, second)
    assert [d.title for d in store.documents_of(USER.id)] == ["Перший"]
    assert telegram.sent[-1].text == texts.DOC_ALREADY_SAVED.format(title="Перший", scope='рукавичка "Під дубом"')
    await dispatcher.feed_update(bot, message(text="увесь ліс"))
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


async def test_double_tap_on_the_name_saves_the_document_once(setup, monkeypatch):
    # The second tap of "Так" waits for the lock while another file is being taken, then finds the name accepted
    # already: the waiting document is asked about once, and no apology comes.
    dispatcher, bot, telegram, _, _ = setup
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, document_update("Гриби.txt", "n1"))
    await dispatcher.feed_update(bot, message(text="Білочка"))
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')
    make_request, edited, taking = telegram.make_request, set(), asyncio.Event()

    async def telegram_like(bot_, method, timeout=None):
        if isinstance(method, (EditMessageReplyMarkup, EditMessageText)):
            if method.message_id in edited:
                raise TelegramBadRequest(method=method, message="Bad Request: message is not modified")
            edited.add(method.message_id)
        if isinstance(method, SendMessage) and "щойно Ви назветеся" in method.text:
            taking.set()
            await asyncio.sleep(0.05)  # one Telegram round trip under the user's lock
        return await make_request(bot_, method, timeout)

    monkeypatch.setattr(telegram, "make_request", telegram_like)
    second = asyncio.create_task(dispatcher.feed_update(bot, document_update("Ягоди.txt", "n2")))
    await taking.wait()
    pressed = next(ids)
    await asyncio.gather(dispatcher.feed_update(bot, same_message_callback(pressed, "nm:yes")),
                         dispatcher.feed_update(bot, same_message_callback(pressed, "nm:yes")), second)
    sent = [sent.text for sent in telegram.sent]
    assert sum("для якого місця ця порада" in text for text in sent) == 1 and texts.SOMETHING_WRONG not in sent


async def test_double_tap_answers_once_without_an_apology(setup, monkeypatch):
    # Both taps come from one message, and Telegram refuses to take its keyboard off a second time
    # ("message is not modified"): the second tap has nothing to do, and no apology comes next to the answer.
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers.append({"needs_place": True})
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    data = dict(buttons(telegram.sent[-1]))["Під дубом"]
    edited: set[int] = set()
    make_request = telegram.make_request

    async def telegram_like(bot_, method, timeout=None):
        if isinstance(method, (EditMessageReplyMarkup, EditMessageText)):
            await asyncio.sleep(0.01)
            if method.message_id in edited:
                raise TelegramBadRequest(method=method, message="Bad Request: message is not modified")
            edited.add(method.message_id)
        return await make_request(bot_, method, timeout)

    monkeypatch.setattr(telegram, "make_request", telegram_like)
    llm.answers.append({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    pressed = next(ids)
    await asyncio.gather(dispatcher.feed_update(bot, same_message_callback(pressed, data)),
                         dispatcher.feed_update(bot, same_message_callback(pressed, data)))
    sent = [sent.text for sent in telegram.sent]
    assert len([text for text in sent if text.startswith("4 жолуді")]) == 1 and texts.SOMETHING_WRONG not in sent


@pytest.mark.parametrize("refusal, apology", [("message is not modified", False), ("message can't be edited", True)])
async def test_refused_edit_of_the_pressed_message(setup, monkeypatch, refusal, apology):
    # A double tap of "Ні" edits the message to the text it shows already, and Telegram refuses: the answer is
    # there. Any other refusal is still a failure the animal hears about.
    dispatcher, bot, telegram, _, _ = setup
    make_request = telegram.make_request

    async def refuse(bot_, method, timeout=None):
        if isinstance(method, EditMessageText):
            raise TelegramBadRequest(method=method, message=f"Bad Request: {refusal}")
        return await make_request(bot_, method, timeout)

    monkeypatch.setattr(telegram, "make_request", refuse)
    await dispatcher.feed_update(bot, callback("delno"))
    assert [sent.text for sent in telegram.sent] == ([texts.SOMETHING_WRONG] if apology else [])


async def test_buttons_under_an_inaccessible_message_still_work(setup):
    # A message the bot can no longer see has nothing to edit: the button still does its work, and what
    # would replace the message comes as a new one.
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    data = dict(buttons(telegram.sent[-1]))["Увесь ліс"]
    await dispatcher.feed_update(bot, same_message_callback(next(ids), data, inaccessible=True))
    assert "Він стосується: увесь ліс" in telegram.sent[-1].text
    doc = store.documents_of(USER.id)[0]
    await dispatcher.feed_update(bot, same_message_callback(next(ids), f"delok:{doc.id}", inaccessible=True))
    assert telegram.sent[-1].text == texts.DELETED.format(title=doc.title)


async def test_document_during_rename_waits_for_the_name(setup):
    dispatcher, bot, telegram, _, store = setup
    store.set_user_name(USER.id, "Білка", datetime.now(timezone.utc))
    await dispatcher.feed_update(bot, message(text="/name"))
    telegram.file_bytes = "Збирайте гриби після дощу.".encode()
    await dispatcher.feed_update(bot, document_update("Гриби.txt", "c4"))
    assert telegram.sent[-1].text == texts.DOC_AFTER_NAME.format(title="Гриби")


async def test_name_command_while_the_model_places_a_document_still_asks_the_name(setup):
    # /name waits till the model has placed the document, so the end of that dialog does not wipe it,
    # and the name typed next is a name, not a question.
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_json(system, user, **kwargs):
        started.set()
        await release.wait()
        return {"kind": "forest", "value": ""}

    llm.json = slow_json
    telegram.file_bytes = "Білі гриби шукайте під дубом після дощу. Ця порада для мешканців усього лісу.".encode()
    sending = asyncio.create_task(dispatcher.feed_update(bot, document_update("Гриби.txt", "n1")))
    await started.wait()
    naming = asyncio.create_task(dispatcher.feed_update(bot, message(text="/name")))
    await asyncio.sleep(0.01)
    assert not naming.done()  # it waits for the document's lock
    release.set()
    await asyncio.gather(sending, naming)
    assert telegram.sent[-1].text == texts.ASK_NEW_NAME
    await dispatcher.feed_update(bot, message(text="Білочка"))
    assert telegram.sent[-1].text.startswith('Вас звати "Білочка"?')


async def test_saved_document_buttons_say_where_it_is_now(setup):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store, "Гриби.txt")
    keyboard = telegram.sent[-1]
    await dispatcher.feed_update(bot, press(keyboard, "Під дубом"))
    doc = store.documents_of(USER.id)[0]
    store.change_scope(doc.id, USER.id, FOREST_SCOPE)  # "Змінити місце"
    await dispatcher.feed_update(bot, press(keyboard, "Хвойний ліс"))
    assert telegram.sent[-1].text == texts.DOC_ALREADY_SAVED.format(title="Гриби", scope="увесь ліс")


async def test_document_for_two_mittens_is_sent_back_to_be_split(setup):
    # Mishka: one document is for one place, so the animal is asked to split it. It is not saved, but waits:
    # the animal may still say the one place the whole document is for.
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    telegram.file_bytes = ("У рукавичці Під дубом внесок 4 жолуді.\n"
                           "А в рукавичці Під вербою внесок 3 горіхи.").encode()
    await dispatcher.feed_update(bot, document_update("Внески.txt", "s1"))
    assert telegram.sent[-1].text == texts.DOC_SPLIT.format(title="Внески", mittens="Під дубом, Під вербою")
    assert store.documents_of(USER.id) == [] and await flow_state(dispatcher, bot) == Flow.waiting_doc_place.state
    await dispatcher.feed_update(bot, message(text="для листяного лісу"))
    assert [doc.scope.describe() for doc in store.documents_of(USER.id)] == ["листяний ліс"]
    # Or /cancel, to send it again as separate documents.
    await dispatcher.feed_update(bot, document_update("Внески.txt", "s2"))
    await dispatcher.feed_update(bot, message(text="/cancel"))
    assert telegram.sent[-1].text == texts.CANCELLED_DOCS.format(titles='"Внески"') and len(store.documents_of(USER.id)) == 1


async def test_empty_file_is_not_taken_for_a_scan(setup):
    dispatcher, bot, telegram, _, store = setup
    telegram.file_bytes = b"   \n"
    await dispatcher.feed_update(bot, document_update("Порада.txt", "e1"))
    assert telegram.sent[-1].text == texts.DOC_NO_TEXT.format(filename="Порада.txt")


@pytest.fixture
def no_pauses(monkeypatch):
    """A busy model is asked again at once instead of after half a minute, three times in all."""
    monkeypatch.setattr(handlers, "LATER_PAUSES", (0, 0, 0))


async def answered_later(dispatcher):
    """Wait for the questions the bot promised to answer once a model is free."""
    await asyncio.gather(*dispatcher["app"].background)


async def stop_waiting(dispatcher):
    """Cancel the questions still waiting for a free model, as at the end of a test that keeps it busy."""
    tasks = list(dispatcher["app"].background)
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def test_busy_model_answers_the_question_later(setup, no_pauses):
    # Every Groq model rests after its rate limit: the bot does not give up, it answers a few minutes later.
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers += [LLMUnavailable("busy"), LLMUnavailable("busy"),
                    {"found": True, "answer": "4 жолуді на місяць.", "sources": [1]}]
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    assert telegram.sent[-1].text == texts.ANSWER_LATER
    await answered_later(dispatcher)
    later = telegram.sent[-1].text
    assert later.startswith('На Ваше питання "Скільки внесок під дубом?":\n\n4 жолуді на місяць.')
    assert "Членські внески рукавички Під дубом" in later and not dispatcher["app"].later[USER.id]
    # The answer given later is the last one, so a short follow-up asks the same for another place.
    llm.answers.append({"found": True, "answer": "3 шишки на місяць.", "sources": [1]})
    await dispatcher.feed_update(bot, message(text="А у соснах?"))
    assert telegram.sent[-1].text.startswith("3 шишки на місяць.")
    assert "Питання звіра: скільки внесок у соснах?" in llm.prompts[-1][1]


async def test_busy_model_asks_for_the_mitten_later(setup, no_pauses):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers += [LLMUnavailable("busy"), {"needs_place": True}]
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    await answered_later(dispatcher)
    asked = telegram.sent[-1]
    assert asked.text.startswith('На Ваше питання "скільки внесок?":\n\n') and "Про яку рукавичку" in asked.text
    assert await flow_state(dispatcher, bot) == Flow.waiting_question_place.state

    # The mitten chosen with a button is kept for the next try, and the later answer names it.
    llm.answers += [LLMUnavailable("busy"), {"found": True, "answer": "4 жолуді на місяць.", "sources": [1]}]
    await dispatcher.feed_update(bot, press(asked, "Під дубом"))
    assert telegram.sent[-1].text == texts.ANSWER_LATER
    await answered_later(dispatcher)
    assert telegram.sent[-1].text.startswith(
        'На Ваше питання "скільки внесок?" для рукавички "Під дубом":\n\n4 жолуді на місяць.')
    assert 'рукавичка "Під дубом"' in llm.prompts[-1][1]


async def test_mitten_asked_later_while_a_document_waits_is_asked_in_the_text(setup, no_pauses):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    llm.answers += [LLMUnavailable("busy"), {"needs_place": True}]
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    assert telegram.sent[-2].text == texts.ANSWER_LATER and 'А для документа "Report"' in telegram.sent[-1].text
    await answered_later(dispatcher)
    assert telegram.sent[-2].text == 'На Ваше питання "скільки внесок?":\n\n' + texts.ASK_PLACE_IN_TEXT
    # The last word is the document's again, so "під дубом" typed next is clearly its place.
    assert 'А для документа "Report" оберіть' in telegram.sent[-1].text and telegram.sent[-1].reply_markup
    assert await flow_state(dispatcher, bot) == Flow.waiting_doc_place.state
    await dispatcher.feed_update(bot, message(text="під дубом"))
    assert [doc.scope.describe() for doc in store.documents_of(USER.id)] == ['рукавичка "Під дубом"']


async def test_model_busy_for_every_try_brings_an_apology(setup, no_pauses):
    dispatcher, bot, telegram, llm, _ = setup
    llm.answers += [LLMUnavailable("busy")] * 4  # at once and after each of the three pauses
    await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    await answered_later(dispatcher)
    assert telegram.sent[-1].text == 'На Ваше питання "скільки внесок під дубом?":\n\n' + texts.ANSWER_LATER_FAILED
    assert len(llm.prompts) == 4 and llm.answers == [] and not dispatcher["app"].later[USER.id]


async def test_apology_while_a_document_waits_is_followed_by_the_document(setup, no_pauses):
    dispatcher, bot, telegram, llm, store = setup
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    llm.answers += [LLMUnavailable("busy")] * 4
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    await answered_later(dispatcher)
    assert telegram.sent[-2].text.endswith(texts.ANSWER_LATER_FAILED)
    assert 'А для документа "Report" оберіть' in telegram.sent[-1].text and store.documents_of(USER.id) == []


async def test_only_a_few_questions_of_an_animal_wait_for_a_free_model(setup, monkeypatch):
    dispatcher, bot, telegram, llm, _ = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (3600,))  # the model stays busy while the animal asks on
    fox = User(id=60, is_bot=False, first_name="Fox")
    llm.answers += [LLMUnavailable("busy")] * 5
    for _ in range(4):
        await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    assert [sent.text for sent in telegram.sent] == [texts.ANSWER_LATER] * 3 + [texts.LLM_UNAVAILABLE]
    await dispatcher.feed_update(bot, message(user=fox, text="скільки внесок під дубом?"))
    assert telegram.sent[-1].text == texts.ANSWER_LATER  # a neighbour's question still waits
    later = dispatcher["app"].later
    assert len(later[USER.id]) == 3 and len(later[fox.id]) == 1
    await stop_waiting(dispatcher)


async def test_questions_forwarded_together_do_not_pass_the_cap(setup, monkeypatch):
    # aiogram handles them at once, and each reply to Telegram takes a while: every question counts before it.
    dispatcher, bot, telegram, llm, _ = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (3600,))
    make_request = telegram.make_request

    async def slow(bot, method, timeout=None):
        await asyncio.sleep(0.01)
        return await make_request(bot, method, timeout)

    monkeypatch.setattr(telegram, "make_request", slow)
    llm.answers += [LLMUnavailable("busy")] * 5
    await asyncio.gather(*[dispatcher.feed_update(bot, message(text="скільки внесок під дубом?")) for _ in range(5)])
    sent = [sent.text for sent in telegram.sent]
    assert sorted(sent) == sorted([texts.ANSWER_LATER] * 3 + [texts.LLM_UNAVAILABLE] * 2)
    assert len(dispatcher["app"].later[USER.id]) == 3
    await stop_waiting(dispatcher)


async def test_error_while_answering_later_still_gets_a_polite_reply(setup, no_pauses, caplog):
    dispatcher, bot, telegram, _, _ = setup
    replies = [Reply("unavailable")]

    async def ask(question, now, mitten=None):
        if replies:
            return replies.pop(0)
        raise AttributeError("'list' object has no attribute 'get'")

    dispatcher["app"].answerer.ask = ask
    await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    await answered_later(dispatcher)  # the task logs and swallows its error
    assert telegram.sent[-1].text == 'На Ваше питання "скільки внесок під дубом?":\n\n' + texts.SOMETHING_WRONG
    assert "later failed" in caplog.text


async def test_cancel_stops_a_question_waiting_for_a_free_model(setup, monkeypatch):
    # The animal no longer waits for the answer promised later: /cancel stops it and names the question.
    dispatcher, bot, telegram, llm, _ = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (3600,))
    llm.answers.append(LLMUnavailable("busy"))
    await dispatcher.feed_update(bot, message(text="скільки внесок\nпід дубом?"))
    waiting = list(dispatcher["app"].background)
    await dispatcher.feed_update(bot, message(text="/cancel"))
    assert telegram.sent[-1].text == (
        f"{texts.CANCELLED} " + texts.CANCELLED_QUESTIONS.format(questions='"скільки внесок під дубом?"'))
    await asyncio.gather(*waiting, return_exceptions=True)
    assert all(task.cancelled() for task in waiting) and not dispatcher["app"].later[USER.id]


async def test_start_drops_the_waiting_document_and_question_together(setup, monkeypatch):
    dispatcher, bot, telegram, llm, store = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (3600,))
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    llm.answers.append(LLMUnavailable("busy"))
    await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    waiting = list(dispatcher["app"].background)
    await dispatcher.feed_update(bot, message(text="/start"))
    assert telegram.sent[-2].text == (texts.CANCELLED_DOCS.format(titles='"Report"') + " "
                                      + texts.CANCELLED_QUESTIONS.format(questions='"скільки внесок під дубом?"'))
    await asyncio.gather(*waiting, return_exceptions=True)
    assert all(task.cancelled() for task in waiting) and store.documents_of(USER.id) == []


@pytest.mark.parametrize("busy", [False, True])
async def test_question_cancelled_while_the_model_answers_gets_nothing_after(setup, busy):
    # /cancel answers at once and names the question; the answer that comes after it, or the promise to
    # answer later, is not sent.
    dispatcher, bot, telegram, _, _ = setup
    started, release = asyncio.Event(), asyncio.Event()

    async def ask(question, now, mitten=None):
        started.set()
        await release.wait()
        return Reply("unavailable") if busy else Reply("answer", "4 жолуді на місяць.")

    dispatcher["app"].answerer.ask = ask
    asking = asyncio.create_task(dispatcher.feed_update(bot, message(text="скільки внесок під дубом?")))
    await started.wait()
    await dispatcher.feed_update(bot, message(text="/cancel"))
    release.set()
    await asking
    assert telegram.sent[-1].text == (
        f"{texts.CANCELLED} " + texts.CANCELLED_QUESTIONS.format(questions='"скільки внесок під дубом?"'))
    assert not dispatcher["app"].later[USER.id] and not dispatcher["app"].asking[USER.id]


async def test_cancel_names_the_question_being_answered_and_the_one_waiting(setup, monkeypatch):
    dispatcher, bot, telegram, _, _ = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (3600,))
    started, release = asyncio.Event(), asyncio.Event()
    replies = [Reply("unavailable")]

    async def ask(question, now, mitten=None):
        if replies:
            return replies.pop(0)
        started.set()
        await release.wait()
        return Reply("answer", "3 шишки на місяць.")

    dispatcher["app"].answerer.ask = ask
    await dispatcher.feed_update(bot, message(text="скільки внесок під дубом?"))
    waiting = list(dispatcher["app"].background)
    asking = asyncio.create_task(dispatcher.feed_update(bot, message(text="скільки внесок у соснах?")))
    await started.wait()
    await dispatcher.feed_update(bot, message(text="/cancel"))
    release.set()
    await asking
    await asyncio.gather(*waiting, return_exceptions=True)
    assert telegram.sent[-1].text == f"{texts.CANCELLED} " + texts.CANCELLED_QUESTIONS.format(
        questions='"скільки внесок під дубом?", "скільки внесок у соснах?"')


class WatchedLock(asyncio.Lock):
    """A chat lock that tells when someone has to wait for it."""

    def __init__(self):
        super().__init__()
        self.waited = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.waited.set()
        return await super().acquire()


async def test_mitten_question_waiting_for_a_document_is_dropped_at_cancel(setup):
    # The model wants the mitten while a document is still being placed and holds the chat's lock; /cancel
    # comes meanwhile and names the question, no mitten buttons follow it and the dialog stays cleared.
    dispatcher, bot, telegram, llm, store = setup
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    lock = dispatcher["app"].locks[USER.id] = WatchedLock()
    placing, release = asyncio.Event(), asyncio.Event()

    async def slow_json(system, user, **kwargs):
        placing.set()
        await release.wait()
        return {"kind": "unclear", "value": ""}

    async def ask(question, now, mitten=None):
        return Reply("need_place")

    llm.json = slow_json
    dispatcher["app"].answerer.ask = ask
    telegram.file_bytes = "Білі гриби шукайте під дубом після дощу.".encode()
    sending = asyncio.create_task(dispatcher.feed_update(bot, document_update("Report.txt", "c4")))
    await placing.wait()
    question = asyncio.create_task(dispatcher.feed_update(bot, message(text="скільки внесок?")))
    await lock.waited.wait()  # the model has answered, and the question waits for the document's lock
    assert not question.done()
    await dispatcher.feed_update(bot, message(text="/cancel"))
    release.set()
    await asyncio.gather(sending, question)
    assert telegram.sent[-1].text == (texts.CANCELLED_DOCS.format(titles='"Report"') + " "
                                      + texts.CANCELLED_QUESTIONS.format(questions='"скільки внесок?"'))
    assert await flow_state(dispatcher, bot) is None and not dispatcher["app"].asking[USER.id]


async def test_question_promised_later_waiting_for_a_document_is_named_at_cancel(setup, monkeypatch):
    # The same, when the mitten is asked in the answer that comes later.
    dispatcher, bot, telegram, llm, store = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (0,))
    store.set_user_name(USER.id, "Їжак", datetime.now(timezone.utc))
    lock = dispatcher["app"].locks[USER.id] = WatchedLock()
    placing, release = asyncio.Event(), asyncio.Event()
    replies = [Reply("unavailable"), Reply("need_place")]

    async def slow_json(system, user, **kwargs):
        placing.set()
        await release.wait()
        return {"kind": "unclear", "value": ""}

    async def ask(question, now, mitten=None):
        return replies.pop(0)

    llm.json = slow_json
    dispatcher["app"].answerer.ask = ask
    telegram.file_bytes = "Білі гриби шукайте під дубом після дощу.".encode()
    sending = asyncio.create_task(dispatcher.feed_update(bot, document_update("Report.txt", "c5")))
    await placing.wait()
    await dispatcher.feed_update(bot, message(text="скільки внесок?"))
    assert telegram.sent[-1].text == texts.ANSWER_LATER
    waiting = list(dispatcher["app"].background)
    await lock.waited.wait()  # the answer given later waits for the document's lock
    await dispatcher.feed_update(bot, message(text="/cancel"))
    release.set()
    await asyncio.gather(sending, *waiting)
    assert telegram.sent[-1].text == (texts.CANCELLED_DOCS.format(titles='"Report"') + " "
                                      + texts.CANCELLED_QUESTIONS.format(questions='"скільки внесок?"'))
    assert await flow_state(dispatcher, bot) is None


async def test_cancel_right_after_a_late_answer_does_not_take_it_back(setup, monkeypatch):
    # The late answer is out and the bot is reminding about the waiting document when /cancel comes: the
    # reply names the document, not the question just answered.
    dispatcher, bot, telegram, llm, store = setup
    monkeypatch.setattr(handlers, "LATER_PAUSES", (0,))
    await send_unclear_document(dispatcher, bot, telegram, llm, store)
    make_request = telegram.make_request
    reminding, go = asyncio.Event(), asyncio.Event()

    async def slow(bot_, method, timeout=None):
        if "А для документа" in (getattr(method, "text", "") or "") and not reminding.is_set():
            reminding.set()
            await go.wait()
        return await make_request(bot_, method, timeout)

    llm.answers += [LLMUnavailable("busy"), {"found": True, "answer": "4 жолуді на місяць.", "sources": [1]}]
    await dispatcher.feed_update(bot, message(text="Скільки внесок під дубом?"))
    monkeypatch.setattr(telegram, "make_request", slow)
    waiting = list(dispatcher["app"].background)
    await reminding.wait()
    await dispatcher.feed_update(bot, message(text="/cancel"))
    go.set()
    await asyncio.gather(*waiting, return_exceptions=True)
    sent = [sent.text for sent in telegram.sent]
    assert any(text.startswith('На Ваше питання "Скільки внесок під дубом?":\n\n4 жолуді') for text in sent)
    assert texts.CANCELLED_DOCS.format(titles='"Report"') in sent and not any("вже не відповім" in text for text in sent)
