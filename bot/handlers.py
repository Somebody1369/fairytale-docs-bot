"""Telegram handlers: text is a question, a file is advice."""

import asyncio
import hashlib
import logging
import re
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from aiogram import BaseMiddleware, Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.types import CallbackQuery, ErrorEvent, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender
from pydantic import Field

from . import texts
from .answering import Answerer, Reply, format_date
from .classify import SPLIT, WHOLE_FOREST_CAPTION, classify_document, scope_from_caption
from .config import Settings
from .extract import (MAX_OCR_PAGES, SUPPORTED_EXTENSIONS, TooManyScannedPages, UnreadableScan, extension, extract_text,
                      old_word_supported, title_from_filename)
from .llm import LLM
from .places import FOREST_SCOPE, PART_NAMES, Forest, Scope, normalize
from .search import STOPWORDS
from .storage import INVITE_DAYS, Store

log = logging.getLogger(__name__)


@dataclass
class App:
    settings: Settings
    store: Store
    forest: Forest
    llm: LLM
    answerer: Answerer
    tz: ZoneInfo
    locks: defaultdict = field(default_factory=lambda: defaultdict(asyncio.Lock))
    archive_ready: asyncio.Event | None = None  # set once the archive is read; None means ready
    # Questions each chat waits for while every model is busy, asked again in the background: task -> question.
    # A task leaves once its answer goes out, so /cancel names and stops only the questions still waiting.
    later: defaultdict = field(default_factory=lambda: defaultdict(dict))
    background: set = field(default_factory=set)  # those tasks till they end, so none is garbage-collected half way
    # Questions the model is answering right now in each chat, so /cancel can say they are dropped too.
    asking: defaultdict = field(default_factory=lambda: defaultdict(list))


BIG_PDF_BYTES = 1024 * 1024  # a scan this big takes a while to read, so the sender is told
FOLLOW_UP_MINUTES = 30
# When every Groq model rests after its rate limit for longer than a question may take, the bot does not
# give up: Telegram lets it answer whenever it likes, so it asks again after each of these pauses (seconds).
# The tries count too, each up to QUESTION_SECONDS: 7 minutes of pauses and 5 tries keep the answer within
# the "about 10 minutes" promised to the customer (docs/dialog.md) even when every try takes all its time.
LATER_PAUSES = (30, 60, 90, 120, 120)
LATER_PER_CHAT = 3  # beyond this many waiting questions the animal is asked to try again itself
# Documents waiting for their sender's name or place keep their whole text in the dialog state, and /cancel
# lists their titles in one message: beyond this many the animal is asked to send the file again later.
QUEUE_LIMIT = 10


NAME_PREFIX = re.compile(r"^(?:мене\s+(?:звати|звуть)|моє\s+ім'?я|я\s*[-–:]?|це)\s+", re.IGNORECASE)


def clean_name(text: str) -> str:
    """ "Мене звати Білочка." -> "Білочка" """
    name = " ".join(text.replace("’", "'").replace("ʼ", "'").split()).strip(" .,!")
    name = NAME_PREFIX.sub("", name).strip(" .,!")
    return name[:1].upper() + name[1:]


def valid_name(name: str) -> bool:
    return 2 <= len(name) <= 40 and len(name.split()) <= 4 and not any(c.isdigit() or c in "?/" for c in name)


YES_WORDS = {"так", "да", "ага", "угу", "вірно", "правильно", "точно", "звісно", "yes", "ок", "окей", "ok", "+"}
NO_WORDS = {"ні", "нет", "no", "не", "неправильно", "невірно", "інакше"}


def yes_or_no(text: str) -> bool | None:
    """ "Так" typed instead of pressing the button is a confirmation, not a name."""
    words = set(re.findall(r"[\w+']+", text.lower()))
    if words and words <= YES_WORDS:
        return True
    if words & NO_WORDS and words <= NO_WORDS | {"так"}:
        return False
    return None


GREETING_WORDS = {"привіт", "привітик", "вітаю", "добрий", "доброго", "добрийдень", "день", "дня", "ранок", "ранку",
                  "вечір", "вечора", "здрастуйте", "здоровенькі", "були", "hello", "hi", "hey", "салют", "хай",
                  "довіднику", "боте", "бот"}
THANKS_WORDS = {"дякую", "дякуємо", "дяка", "дуже", "щиро", "велике", "спасибі", "спасибо", "thanks", "thank", "you",
                "ок", "окей", "ok", "добре", "гаразд", "зрозуміло", "ясно", "чудово", "супер", "клас", "все"}


def small_talk(text: str) -> str | None:
    """A reply to "Привіт" or "Дякую" without asking the model to find advice about it."""
    words = re.findall(r"[a-zа-яіїєґ']+", text.lower().replace("’", "'"))
    if not words:
        return texts.THANKS if re.fullmatch(r"[\W_]+", text) else None  # 👍, 🙏
    if all(word in GREETING_WORDS for word in words):
        return texts.GREETING
    if all(word in THANKS_WORDS | GREETING_WORDS for word in words):
        return texts.THANKS
    return None


FILLER_WORDS = {"щодо", "стосовно", "ну", "тоді", "теж", "то", "там", "тут", "а", "як", "що", "у", "в", "нас", "вас"}


def follow_up(text: str, last: str | None, forest: Forest) -> str | None:
    """ "А під ялиною?" right after "Що їсти під вербою?" is the same question for another place."""
    if not last or not (forest.detect_mittens(text) or forest.detect_parts(text)):
        return None
    rest = re.findall(r"[а-яіїєґ']+", forest.strip_places(text))
    if any(word not in STOPWORDS | FILLER_WORDS for word in rest):
        return None  # a new question that names its own place
    topic = re.sub(r"\s+", " ", forest.strip_places(last)).strip(" ?.!,")
    place = re.sub(r"^(?:а якщо|а як|а що|а|і|й|ну|щодо|стосовно)\s+", "", text.strip().lower()).strip(" ?.!,")
    return f"{topic} {place}?" if topic else None


PLACE_REPLY_WORDS = {"рукавичка", "рукавички", "рукавичці", "рукавичку", "частина", "частини", "частину", "ліс", "лісу",
                     "лісі", "мешканців", "звірів", "всіх", "усіх", "будь", "ласка", "мабуть", "напевно", "порада",
                     "документ"}


def typed_place(text: str, forest: Forest) -> Scope | None:
    """ "Під дубом" or "для всього лісу" typed instead of pressing a button. A message with anything
    more is not an answer about the place: "Скільки внесок під дубом" is a question, and it must not
    file a waiting document under that mitten."""
    if "?" in text:
        return None
    rest = re.findall(r"[а-яіїєґ']+", forest.strip_places(re.sub(WHOLE_FOREST_CAPTION, " ", normalize(text))))
    if any(word not in STOPWORDS | PLACE_REPLY_WORDS for word in rest):
        return None
    return scope if (scope := scope_from_caption(text, forest)) and scope.kind != SPLIT else None


QUESTION_WORDS = {"скільки", "що", "де", "коли", "як", "хто", "чому", "навіщо", "нащо", "який", "яка", "яке", "які", "чи",
                  "куди", "звідки", "чим", "кому", "кого", "можна", "треба", "потрібно", "підкажіть", "скажіть",
                  "розкажіть"}


def looks_like_question(text: str, forest: Forest) -> bool:
    """A question typed while the bot waits for a name or a place, even without "?":
    it starts with "скільки", "де"... or names a place in a few words ("внесок під дубом")."""
    words = re.findall(r"[а-яіїєґ']+", normalize(text))
    if "?" in text or (words and words[0] in QUESTION_WORDS):
        return True
    return len(words) >= 3 and bool(forest.detect_mittens(text) or forest.detect_parts(text))


class Flow(StatesGroup):
    waiting_name = State()
    confirming_name = State()
    waiting_doc_place = State()
    waiting_question_place = State()


# A document waits for its sender's name or its place. A question waiting for its mitten keeps
# its text in the data and its buttons work in any state, so it never pushes a document out.
DOCUMENT_STATES = (Flow.waiting_name.state, Flow.confirming_name.state, Flow.waiting_doc_place.state)


def invite_link(bot_username: str, token: str) -> str:
    return f"https://t.me/{bot_username}?start=inv_{token}"


def invite_token(text: str | None) -> str | None:
    match = re.fullmatch(r"/start\s+inv_([\w-]+)", (text or "").strip())
    return match.group(1) if match else None


class InviteGate(BaseMiddleware):
    """With INVITES_ONLY, only animals who came by a neighbour's one-time invite link may use the bot.
    There are no admins: any member can invite, and the bot remembers who invited whom."""

    async def __call__(self, handler, event, data):
        app: App = data["app"]
        user, chat = data.get("event_from_user"), data.get("event_chat")
        if not app.settings.invites_only or not user or (chat and chat.type != ChatType.PRIVATE) \
                or app.store.is_member(user.id):
            return await handler(event, data)
        token = invite_token(event.text) if isinstance(event, Message) else None
        # A bot nobody has joined yet tells whoever started it where the first invite is.
        need_invite = texts.NEED_INVITE if app.store.has_members() else texts.NEED_FIRST_INVITE
        # The Telegram name helps the inviter recognise the animal in /invited before it names itself.
        tg_name = user.full_name + (f" @{user.username}" if user.username else "")
        if token and app.store.use_invite(token, user.id, now_utc(), tg_name):
            await event.answer(f"{texts.INVITE_WELCOME}\n\n{texts.START_INVITES}")
        elif isinstance(event, Message):
            await event.answer(texts.INVITE_BAD if token else need_invite)
        else:
            await event.answer(texts.NEED_INVITE, show_alert=True)


router = Router()
router.message.filter(F.chat.type == ChatType.PRIVATE)
router.callback_query.filter(F.message.chat.type == ChatType.PRIVATE)
router.message.outer_middleware(InviteGate())
router.callback_query.outer_middleware(InviteGate())


def now_utc() -> datetime:
    return datetime.now(UTC)


# Buttons. A client may send any data under a button, so each kind is a CallbackData: data that does not
# unpack into one reaches no handler but unknown_button. The data stays "prefix:field:field", within
# Telegram's 64 bytes.

# A document or a Telegram user: both ids are positive, and SQLite raises on an integer beyond 64 bits.
Id = Annotated[int, Field(gt=0, lt=2**63)]


class DeleteDoc(CallbackData, prefix="del"):
    doc_id: Id


class DeleteYes(CallbackData, prefix="delok"):
    doc_id: Id


class DeleteNo(CallbackData, prefix="delno"):
    pass


class Revoke(CallbackData, prefix="rev"):
    member_id: Id


class RevokeYes(CallbackData, prefix="revok"):
    member_id: Id


class RevokeNo(CallbackData, prefix="revno"):
    pass


class NameAnswer(CallbackData, prefix="nm"):
    answer: Literal["yes", "no"]


class DocPlace(CallbackData, prefix="dp"):
    """A place for the waiting document `key`. `kind`: "m" a mitten (`value` is its mitten_key),
    "p" a forest part, "f" the whole forest."""
    key: str
    kind: str
    value: str


class QuestionPlace(CallbackData, prefix="qp"):
    """A mitten for the waiting question `key`, as in DocPlace."""
    key: str
    kind: str
    value: str


class ChangePlace(CallbackData, prefix="chg"):
    doc_id: Id


class NewPlace(CallbackData, prefix="cs"):
    """Another place for a saved document, as in DocPlace."""
    doc_id: Id
    kind: str
    value: str


def mitten_key(name: str) -> str:
    """A mitten in button data. Not its position on the map: "Карта лісу" is read again at every archive sync,
    and a button made before a change must not answer for another mitten."""
    return hashlib.blake2s(name.encode(), digest_size=4).hexdigest()


# Keyboards. `button` makes one button's data from its kind and value, e.g. partial(DocPlace, key=...).

def mitten_keyboard(forest: Forest, button: Callable[..., CallbackData],
                    only: list[str] | None = None) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(text=m, callback_data=button(kind="m", value=mitten_key(m)).pack())
               for m in forest.mittens if only is None or m in only]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def place_keyboard(forest: Forest, button: Callable[..., CallbackData]) -> InlineKeyboardMarkup:
    keyboard = mitten_keyboard(forest, button).inline_keyboard
    parts = [InlineKeyboardButton(text=name.capitalize(), callback_data=button(kind="p", value=key).pack())
             for key, name in PART_NAMES.items()]
    whole = [InlineKeyboardButton(text=texts.WHOLE_FOREST, callback_data=button(kind="f", value="").pack())]
    return InlineKeyboardMarkup(inline_keyboard=[*keyboard, parts, whole])


def doc_place_keyboard(forest: Forest, pending: dict) -> InlineKeyboardMarkup:
    """Buttons for one waiting document: an old keyboard must not file the next document."""
    return place_keyboard(forest, partial(DocPlace, key=pending["id"]))


def question_keyboard(forest: Forest, key: str, only: list[str] | None = None) -> InlineKeyboardMarkup:
    """Mitten buttons for one waiting question: buttons under an older question must not answer this one."""
    return mitten_keyboard(forest, partial(QuestionPlace, key=key), only)


def answered(data: dict) -> dict:
    """FSM data once the waiting question is answered: its buttons then answer nothing more."""
    return {"question": None, "question_id": None,
            "questions": data.get("questions", {}) | {data["question_id"]: None}}


def scope_from_callback(forest: Forest, kind: str, value: str) -> Scope | None:
    """The place a button names; None for a mitten no longer on the map, which still belongs to its
    document or question."""
    if kind == "m":
        mitten = {mitten_key(m): m for m in forest.mittens}.get(value)
        return Scope("mitten", mitten) if mitten else None
    if kind == "p" and value in PART_NAMES:
        return Scope("part", value)
    if kind == "f":
        return FOREST_SCOPE
    return None


async def drop_keyboard(callback: CallbackQuery) -> None:
    """Take the pressed buttons off. A double tap finds them gone already, and Telegram refuses an edit that
    changes nothing ("message is not modified"); a message the bot can no longer see comes as an
    InaccessibleMessage, with nothing to edit. Either way the button still does its work."""
    if isinstance(callback.message, Message):
        with suppress(TelegramBadRequest):
            await callback.message.edit_reply_markup(reply_markup=None)


async def show(callback: CallbackQuery, text: str, reply_markup: InlineKeyboardMarkup | None = None) -> None:
    """Put the outcome in place of the pressed message, or below it when the bot can no longer see that one.
    A double tap of "Ні" edits the text to what it is already, and Telegram refuses ("message is not
    modified"): the answer is there, so that is no failure. Any other refusal still reaches the error handler."""
    if not isinstance(callback.message, Message):
        await callback.message.answer(text, reply_markup=reply_markup)
        return
    try:
        await callback.message.edit_text(text, reply_markup=reply_markup)
    except TelegramBadRequest as error:
        if "message is not modified" not in error.message:
            raise


# Commands

@router.message(CommandStart())
async def start(message: Message, state: FSMContext, app: App) -> None:
    dropped = await drop_dialog(message, state, app)
    if dropped:
        await message.answer(dropped)
    await message.answer(texts.START_INVITES if app.settings.invites_only else texts.START)


@router.message(Command("help"))
async def help_command(message: Message, app: App) -> None:
    await message.answer(texts.HELP_INVITES if app.settings.invites_only else texts.HELP)


@router.message(Command("cancel"))
async def cancel(message: Message, state: FSMContext, app: App) -> None:
    await message.answer(await drop_dialog(message, state, app) or texts.CANCELLED)


async def drop_dialog(message: Message, state: FSMContext, app: App) -> str:
    """/cancel and /start: forget waiting documents and questions, those promised an answer later too, and
    say which were dropped ("" when none). A new epoch tells a document still being read or placed by the
    model, or a question the model still answers, to stop: it was dropped too."""
    later = app.later.pop(message.chat.id, {})
    for task in later:
        task.cancel()
    asking = app.asking.pop(message.chat.id, [])
    data = await state.get_data()
    await state.clear()
    await state.update_data(saved=data.get("saved", {}), epoch=data.get("epoch", 0) + 1)
    titles = ", ".join(f'"{pending["title"]}"' for pending in data.get("queue", []))
    dropped = texts.CANCELLED_DOCS.format(titles=titles) if titles else ""
    if later or asking:
        questions = ", ".join(f'"{short_question(question)}"' for question in [*later.values(), *asking])
        dropped = f"{dropped or texts.CANCELLED} {texts.CANCELLED_QUESTIONS.format(questions=questions)}"
    return dropped


@router.message(Command("name"))
async def change_name(message: Message, state: FSMContext, app: App) -> None:
    async with app.locks[message.from_user.id]:  # a document being placed now would end the dialog after it
        await state.set_state(Flow.waiting_name)
        await state.update_data(renaming=True)
    await message.answer(texts.ASK_NEW_NAME)


@router.message(Command("my"))
async def my_documents(message: Message, app: App) -> None:
    docs = app.store.documents_of(message.from_user.id)
    if not docs:
        await message.answer(texts.MY_EMPTY)
        return
    rows = [[InlineKeyboardButton(
        text=f"🗑 {doc.title} ({doc.scope.describe()}, {doc.created_at.astimezone(app.tz):%d.%m})",
        callback_data=DeleteDoc(doc_id=doc.id).pack(),
    )] for doc in docs[:40]]
    await message.answer(texts.MY_LIST, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


@router.message(Command("invite"))
async def invite(message: Message, app: App, bot: Bot) -> None:
    if not app.settings.invites_only:
        await message.answer(texts.INVITES_OFF)
        return
    token = app.store.create_invite(message.from_user.id, now_utc())
    me = await bot.me()
    await message.answer(texts.INVITE_LINK.format(link=invite_link(me.username, token), days=INVITE_DAYS))


@router.message(Command("invited"))
async def invited(message: Message, app: App) -> None:
    if not app.settings.invites_only:
        await message.answer(texts.INVITES_OFF)
        return
    animals = app.store.invited_by_chain(message.from_user.id)
    if not animals:
        await message.answer(texts.INVITED_NONE)
        return
    names = {tg_id: name for tg_id, name, _, _ in animals}
    rows = []
    for tg_id, name, joined, inviter in animals[:40]:
        via = "" if inviter == message.from_user.id else texts.INVITED_VIA.format(name=names.get(inviter) or texts.INVITED_UNNAMED)
        rows.append([InlineKeyboardButton(
            text=f"🚫 {name or texts.INVITED_UNNAMED}{via} (з {joined.astimezone(app.tz):%d.%m})",
            callback_data=Revoke(member_id=tg_id).pack(),
        )])
    await message.answer(texts.INVITED_LIST, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(Revoke.filter())
async def confirm_revoke(callback: CallbackQuery, callback_data: Revoke, app: App) -> None:
    member_id = callback_data.member_id
    await callback.answer()
    if not app.store.in_invite_chain(member_id, callback.from_user.id):
        await show(callback, texts.NOT_YOUR_INVITEE)
        return
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.REVOKE_YES, callback_data=RevokeYes(member_id=member_id).pack()),
        InlineKeyboardButton(text=texts.REVOKE_NO, callback_data=RevokeNo().pack()),
    ]])
    name = app.store.member_name(member_id) or texts.INVITED_UNNAMED
    await show(callback, texts.CONFIRM_REVOKE.format(name=name), keyboard)


@router.callback_query(RevokeYes.filter())
async def revoke(callback: CallbackQuery, callback_data: RevokeYes, app: App) -> None:
    removed = app.store.revoke_member(callback_data.member_id, callback.from_user.id)
    await callback.answer()
    if not removed:
        await show(callback, texts.NOT_YOUR_INVITEE)
        return
    animals = "1 звіра" if len(removed) == 1 else f"{len(removed)} звірів"
    await show(callback, texts.REVOKED.format(animals=animals))


@router.callback_query(RevokeNo.filter())
async def keep_access(callback: CallbackQuery) -> None:
    await callback.answer()
    await show(callback, texts.REVOKE_KEPT)


@router.callback_query(DeleteDoc.filter())
async def confirm_delete(callback: CallbackQuery, callback_data: DeleteDoc, app: App) -> None:
    doc = app.store.get(callback_data.doc_id)
    await callback.answer()
    if not doc or doc.author_id != callback.from_user.id or doc.status != "active":
        await show(callback, texts.NOT_YOURS)
        return
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.DELETE_YES, callback_data=DeleteYes(doc_id=doc.id).pack()),
        InlineKeyboardButton(text=texts.DELETE_NO, callback_data=DeleteNo().pack()),
    ]])
    await show(callback, texts.CONFIRM_DELETE.format(title=doc.title), keyboard)


@router.callback_query(DeleteYes.filter())
async def delete_document(callback: CallbackQuery, callback_data: DeleteYes, app: App) -> None:
    doc = app.store.delete_animal_document(callback_data.doc_id, callback.from_user.id)
    await callback.answer()
    await show(callback, texts.DELETED.format(title=doc.title) if doc else texts.NOT_YOURS)


@router.callback_query(DeleteNo.filter())
async def keep_document(callback: CallbackQuery) -> None:
    await callback.answer()
    await show(callback, texts.KEPT)


# Documents (advice)

@router.message(F.photo)
async def photo(message: Message) -> None:
    await message.answer(texts.NO_PHOTOS)


@router.message(F.document)
async def document(message: Message, state: FSMContext, app: App, bot: Bot) -> None:
    file = message.document
    filename = file.file_name or "документ"
    ext = extension(filename)
    if ext not in SUPPORTED_EXTENSIONS and (file.mime_type or "").startswith("text/"):
        ext = ".txt"  # Markdown, CSV and other plain text
    if ext == ".doc" and not old_word_supported():
        await message.answer(texts.DOC_OLD_WORD)
        return
    if ext not in SUPPORTED_EXTENSIONS:
        await message.answer(texts.DOC_WRONG_FORMAT)
        return
    if file.file_size and file.file_size > app.settings.max_file_mb * 1024 * 1024:
        await message.answer(texts.DOC_TOO_BIG.format(limit=app.settings.max_file_mb))
        return

    dialog = await state.get_data()
    epoch = dialog.get("epoch", 0)
    title = title_from_filename(filename)
    if len(dialog.get("queue", [])) >= QUEUE_LIMIT:  # not downloaded and read (a scan takes minutes) to be refused
        await message.answer(texts.DOC_QUEUE_FULL.format(title=title))
        return
    if ext == ".pdf" and (file.file_size or 0) > BIG_PDF_BYTES:
        await message.answer(texts.DOC_READING.format(title=title))
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            content = (await bot.download(file)).read()
            text = await asyncio.to_thread(extract_text, content, filename, app.settings.ocr_langs, ext)
    except TooManyScannedPages as error:
        await message.answer(texts.DOC_TOO_MANY_SCANS.format(filename=filename, pages=error.pages, limit=MAX_OCR_PAGES))
        return
    except UnreadableScan as error:
        log.warning("Cannot read the scan %s: %s", filename, error)
        await message.answer(texts.DOC_EMPTY.format(filename=filename))
        return
    except Exception as error:  # broken PDF, wrong encoding, etc.
        log.warning("Cannot read %s: %s", filename, error, exc_info=True)
        await message.answer(texts.DOC_UNREADABLE.format(filename=filename))
        return
    if not text.strip():  # a PDF with a picture or drawing but no text came as UnreadableScan: this file is empty
        await message.answer(texts.DOC_NO_TEXT.format(filename=filename))
        return

    pending = {"id": uuid.uuid4().hex[:8], "filename": filename, "title": title,
               "text": text, "sent_at": now_utc().isoformat()}
    caption_scope = scope_from_caption(message.caption, app.forest)
    if caption_scope:
        pending["scope"] = caption_scope.encode()
    async with app.locks[message.from_user.id]:
        data = await state.get_data()
        if data.get("epoch", 0) != epoch:  # /cancel or /start while the file was being read
            await message.answer(texts.CANCELLED_DOCS.format(titles=f'"{pending["title"]}"'))
            return
        waiting = data.get("queue", [])
        if len(waiting) >= QUEUE_LIMIT:
            await message.answer(texts.DOC_QUEUE_FULL.format(title=pending["title"]))
            return
        await state.update_data(queue=waiting + [pending])
        current = await state.get_state()
        # Waiting for a place with no document left means the reply about the last one failed: this file is next.
        if current in DOCUMENT_STATES and (waiting or current != Flow.waiting_doc_place.state):
            waits_for_name = current != Flow.waiting_doc_place.state  # a first document or /name
            await message.answer((texts.DOC_AFTER_NAME if waits_for_name else texts.DOC_QUEUED).format(title=pending["title"]))
            return
        if not app.store.user_name(message.from_user.id):
            await state.set_state(Flow.waiting_name)
            await message.answer(texts.ASK_NAME)
            return
        await process_queue(message, state, app, message.from_user.id)


@router.message(Flow.waiting_name, F.text, ~F.text.startswith("/"))
@router.message(Flow.confirming_name, F.text, ~F.text.startswith("/"))
async def got_name(message: Message, state: FSMContext, app: App, bot: Bot) -> None:
    confirming = await state.get_state() == Flow.confirming_name.state
    answer = yes_or_no(message.text)
    if confirming:
        name = (await state.get_data()).get("name")
        if answer and name:
            await accept_name(message, state, app, message.from_user.id, name)
            return
        if answer is False:
            await state.set_state(Flow.waiting_name)
            await message.answer(texts.ASK_NAME_AGAIN)
            return
    if answer is None and looks_like_question(message.text, app.forest):
        # "Скільки внесок під дубом" is not a name: answer it, then ask for the name again.
        await answer_question(message, state, app, bot, message.text.strip(), keep_state=True)
        await remind_document(message, state, app)
        return
    name = clean_name(message.text)
    if not valid_name(name):
        await message.answer(texts.BAD_NAME)
        return
    await state.update_data(name=name)
    await state.set_state(Flow.confirming_name)
    await message.answer(texts.CONFIRM_NAME.format(name=name), reply_markup=name_keyboard())


def name_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.NAME_YES, callback_data=NameAnswer(answer="yes").pack()),
        InlineKeyboardButton(text=texts.NAME_NO, callback_data=NameAnswer(answer="no").pack()),
    ]])


@router.callback_query(NameAnswer.filter())
async def name_confirmed(callback: CallbackQuery, callback_data: NameAnswer, state: FSMContext, app: App) -> None:
    await callback.answer()
    await drop_keyboard(callback)
    name = (await state.get_data()).get("name")
    if await state.get_state() != Flow.confirming_name.state or not name:
        return
    if callback_data.answer == "no":
        await state.set_state(Flow.waiting_name)
        await callback.message.answer(texts.ASK_NAME_AGAIN)
        return
    await accept_name(callback.message, state, app, callback.from_user.id, name)


async def accept_name(message: Message, state: FSMContext, app: App, user_id: int, name: str) -> None:
    async with app.locks[user_id]:
        if await state.get_state() != Flow.confirming_name.state:
            return  # a double tap of "Так": the first tap took the name while this one waited for the lock
        renaming = (await state.get_data()).get("renaming")
        app.store.set_user_name(user_id, name, now_utc())
        await state.update_data(renaming=None)
        await state.set_state(None)
        if renaming:  # /name: the documents already sent are signed with the new name too
            app.store.rename_author(user_id, name)
            await message.answer(texts.NAME_CHANGED.format(name=name))
        await process_queue(message, state, app, user_id)


async def process_queue(message: Message, state: FSMContext, app: App, user_id: int) -> None:
    """Save queued documents one by one; stop and ask when a document's place is unclear."""
    while True:
        data = await state.get_data()
        queue = data.get("queue", [])
        if not queue:
            # A question asked in between keeps its text, so its mitten buttons still answer it, but typed
            # text is a new message again: the bot is not asking for that mitten any more.
            await state.set_state(None)
            return
        pending = queue[0]
        if pending.get("scope"):
            scope = Scope.decode(pending["scope"])
        else:
            scope = await classify_document(pending["title"], pending["text"], app.forest, app.llm)
            if (await state.get_data()).get("epoch", 0) != data.get("epoch", 0):
                return  # /cancel or /start while the model decided: the document is dropped
        if scope is None or scope.kind == SPLIT:
            # Mishka: one document is for one place, so advice for two mittens comes as two documents. The animal
            # splits it, or says the one place the whole document is for; the bot does not choose.
            ask = (texts.ASK_DOC_PLACE.format(title=pending["title"]) if scope is None
                   else texts.DOC_SPLIT.format(title=pending["title"], mittens=scope.value))
            await state.set_state(Flow.waiting_doc_place)
            await state.update_data(asked=pending["id"])  # the document a place typed next is for
            await message.answer(ask, reply_markup=doc_place_keyboard(app.forest, pending))
            return
        if not await file_pending(message, state, app, user_id, queue, scope):
            return


async def remind_document(message: Message, state: FSMContext, app: App) -> None:
    """After a question answered in the middle, ask again what the waiting document still needs.
    `message` may be the bot's own (under a button), so the animal is the dialog's user."""
    current, data, user_id = await state.get_state(), await state.get_data(), state.key.user_id
    queue = data.get("queue")
    if queue and current not in DOCUMENT_STATES and not app.locks[user_id].locked():
        # The queue is stored before the model places the document, so a restart (or an error) meanwhile
        # leaves it waiting with nothing asked about it: the dialog goes back to waiting for its place, or for
        # the name first. The lock is held while it is being placed now.
        current = Flow.waiting_doc_place.state if app.store.user_name(user_id) else Flow.waiting_name.state
        await state.set_state(current)
    if current == Flow.waiting_doc_place.state and queue:
        await state.update_data(asked=queue[0]["id"])
        await message.answer(texts.DOC_STILL_WAITING.format(title=queue[0]["title"]),
                             reply_markup=doc_place_keyboard(app.forest, queue[0]))
    elif current == Flow.confirming_name.state and data.get("name"):
        await message.answer(texts.CONFIRM_NAME.format(name=data["name"]), reply_markup=name_keyboard())
    elif current == Flow.waiting_name.state:
        await message.answer(texts.ASK_NAME_STILL if queue else texts.ASK_NEW_NAME)  # or /name


async def file_pending(message: Message, state: FSMContext, app: App, user_id: int, queue: list[dict],
                       scope: Scope) -> bool:
    """Save the first waiting document; False when it is not used in answers and the queue stops.
    Its id is kept, so an older keyboard for it says it is saved. The store and the queue change before
    the reply is sent, so a /cancel meanwhile cannot bring it back."""
    pending, name = queue[0], app.store.user_name(user_id) or "невідомо"
    # Access closed while the file was read or placed: the document is kept like the author's other ones,
    # and a new invite brings it back with them.
    closed = app.settings.invites_only and not app.store.is_member(user_id)
    doc, replaced = app.store.add_animal_document(
        title=pending["title"], filename=pending["filename"], scope=scope, author_id=user_id,
        author_name=name, text=pending["text"], now=datetime.fromisoformat(pending["sent_at"]),
        status="revoked" if closed else "active",
    )
    saved = dict(list((await state.get_data()).get("saved", {}).items())[-19:]) | {pending["id"]: doc.id}
    await state.update_data(queue=queue[1:], saved=saved)
    if closed:
        await state.set_state(None)  # the rest of the queue waits for a new invite
        await message.answer(texts.NEED_INVITE)
        return False
    reply = texts.DOC_SAVED.format(name=name, title=doc.title, scope=scope.describe(),
                                   date=format_date(doc.created_at, app.tz))
    if replaced:
        reply += texts.DOC_REPLACED.format(title=replaced.title)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.CHANGE_PLACE, callback_data=ChangePlace(doc_id=doc.id).pack())]])
    await message.answer(reply, reply_markup=keyboard)
    return True


async def place_document(message: Message, state: FSMContext, app: App, user_id: int, key: str,
                         scope: Scope | None) -> None:
    """File the waiting document `key` under the place chosen for it with a button or typed; the caller holds
    the animal's lock. A place given for another document does not file the one waiting now: buttons under an
    earlier document, or "під дубом" sent twice, when the first one filed that document."""
    data = await state.get_data()
    queue = data.get("queue", [])
    saved = data.get("saved", {}).get(key)
    if saved:  # a second keyboard for a document already saved: where it is now (after "Змінити місце")
        doc = app.store.get(saved)
        await message.answer(texts.DOC_ALREADY_SAVED.format(title=doc.title, scope=doc.scope.describe())
                             if doc and doc.status == "active" else texts.NOT_YOURS)
        return
    if not queue:
        if await state.get_state() == Flow.waiting_doc_place.state:
            await state.set_state(None)
        await message.answer(texts.DOC_LOST)
        return
    if key != queue[0]["id"]:
        # Given for an earlier document, or not a button the bot makes: this document's buttons again.
        await message.answer(texts.DOC_PLACE_FOR_ANOTHER.format(title=queue[0]["title"]),
                             reply_markup=doc_place_keyboard(app.forest, queue[0]))
        return
    if await state.get_state() in (Flow.waiting_name.state, Flow.confirming_name.state):
        await remind_document(message, state, app)  # /name in between: the name comes first
        return
    if scope is None:  # its own button, for a mitten no longer on the map
        await message.answer(texts.DOC_PLACE_GONE.format(title=queue[0]["title"]),
                             reply_markup=doc_place_keyboard(app.forest, queue[0]))
        return
    if await file_pending(message, state, app, user_id, queue, scope):
        await state.set_state(None)
        await process_queue(message, state, app, user_id)


@router.callback_query(DocPlace.filter())
async def doc_place_chosen(callback: CallbackQuery, callback_data: DocPlace, state: FSMContext, app: App) -> None:
    scope = scope_from_callback(app.forest, callback_data.kind, callback_data.value)
    await callback.answer()
    await drop_keyboard(callback)
    async with app.locks[callback.from_user.id]:
        await place_document(callback.message, state, app, callback.from_user.id, callback_data.key, scope)


@router.message(Flow.waiting_doc_place, F.text, ~F.text.startswith("/"))
async def doc_place_typed(message: Message, state: FSMContext, app: App, bot: Bot) -> None:
    data = await state.get_data()
    if not data.get("queue"):
        # No document waits any more (the reply about saving the last one failed): the text is an ordinary message.
        await state.set_state(None)
        await question(message, state, app, bot)
        return
    scope = typed_place(message.text, app.forest)
    if scope is None and looks_like_question(message.text, app.forest):
        # A question while a document waits for its place: answer it, the document keeps waiting.
        await answer_question(message, state, app, bot, message.text.strip(), keep_state=True)
        await remind_document(message, state, app)
        return
    if scope is None:
        await message.answer(texts.PICK_WITH_BUTTONS)
        return
    # The place is for the document the bot asked about, not for whatever waits first when the lock is free:
    # with "під дубом" sent twice (or twice in one batch of updates after a reconnect) the first one files that
    # document, and the queue moves on before the bot asks about the next one. A dialog saved before the bot
    # kept "asked" is about its first document.
    asked = data.get("asked", data["queue"][0]["id"])
    async with app.locks[message.from_user.id]:
        await place_document(message, state, app, message.from_user.id, asked, scope)


@router.callback_query(ChangePlace.filter())
async def change_place(callback: CallbackQuery, callback_data: ChangePlace, app: App) -> None:
    await callback.answer()
    await callback.message.answer(texts.CHANGE_PLACE_ASK,
                                  reply_markup=place_keyboard(app.forest, partial(NewPlace, doc_id=callback_data.doc_id)))


@router.callback_query(NewPlace.filter())
async def place_changed(callback: CallbackQuery, callback_data: NewPlace, app: App) -> None:
    scope = scope_from_callback(app.forest, callback_data.kind, callback_data.value)
    await callback.answer()
    await drop_keyboard(callback)
    if scope is None:  # a mitten no longer on the map: the places there are now
        await callback.message.answer(texts.CHANGE_PLACE_GONE, reply_markup=place_keyboard(
            app.forest, partial(NewPlace, doc_id=callback_data.doc_id)))
        return
    doc = app.store.change_scope(callback_data.doc_id, callback.from_user.id, scope)
    await callback.message.answer(
        texts.PLACE_CHANGED.format(title=doc.title, scope=scope.describe()) if doc else texts.NOT_YOURS)


# Questions

@router.callback_query(QuestionPlace.filter())
async def question_place_chosen(callback: CallbackQuery, callback_data: QuestionPlace, state: FSMContext, app: App,
                                bot: Bot) -> None:
    key = callback_data.key
    scope = scope_from_callback(app.forest, callback_data.kind, callback_data.value)
    await callback.answer()
    await drop_keyboard(callback)
    async with app.locks[callback.from_user.id]:  # a double tap answers once
        data = await state.get_data()
        # Each keyboard answers its own question, also under an older one.
        questions = data.get("questions", {})
        question = questions.get(key)
        if key in questions and question is None:
            return  # the same question's buttons pressed again: the answer is already above
        if not question:  # forgotten, or not a button the bot makes
            await callback.message.answer(texts.QUESTION_LOST)
            return
        if scope is None:  # a mitten no longer on the map: the question is still known, so the mittens there are now
            await callback.message.answer(texts.QUESTION_PLACE_GONE, reply_markup=question_keyboard(app.forest, key))
            return
        if key == data.get("question_id"):
            # A document may have come in between and wait for a name or a place: its state stays.
            if await state.get_state() == Flow.waiting_question_place.state:
                await state.set_state(None)
            await state.update_data(question=None, question_id=None)
        await state.update_data(questions=questions | {key: None})
    await answer_question(callback.message, state, app, bot, question, mitten=scope.value)
    await remind_document(callback.message, state, app)


@router.message(F.text, ~F.text.startswith("/"))
async def question(message: Message, state: FSMContext, app: App, bot: Bot) -> None:
    text, mitten = message.text.strip(), None
    reply = small_talk(text)
    if reply:  # "Дякую" after "Про яку рукавичку?" too: the question keeps waiting, and no Groq request is spent
        await message.answer(reply)
        return
    data = await state.get_data()
    pending = data.get("question")
    waiting = await state.get_state() == Flow.waiting_question_place.state
    asked_at, last_at = data.get("question_at"), data.get("last_at")
    is_recent = lambda at: (now_utc() - datetime.fromisoformat(at)).total_seconds() < FOLLOW_UP_MINUTES * 60
    # A question still waiting for its mitten after a document came in between, with nothing answered since:
    # a bare place typed now ("у соснах") answers it.
    fresh = pending and asked_at and asked_at > (last_at or "") and is_recent(asked_at)
    # "А у соснах?" answers "Про яку рукавичку?" too, since /help teaches the place with a question mark.
    place = typed_place(text.rstrip(" ?!"), app.forest) if pending and (waiting or fresh) else None
    if waiting or place:
        if place and place.kind != "mitten":
            # "У хвойному лісі": the answer differs by mitten, so offer the mittens of that part.
            only = app.forest.mittens if place.kind == "forest" else app.forest.mittens_in([place.value])
            if len(only) != 1:
                ask = (texts.ASK_MITTEN_IN_PART.format(part=PART_NAMES[place.value]) if place.kind == "part"
                       else texts.ASK_PLACE)
                await message.answer(ask, reply_markup=question_keyboard(app.forest, data["question_id"], only or None))
                return
            place = Scope("mitten", only[0])
        await state.set_state(None)
        # A new question instead of the mitten: the old one's buttons still answer it.
        await state.update_data(**(answered(data) if place else {"question": None, "question_id": None}))
        if place:
            # The animal answered "під дубом" instead of pressing a button.
            text, mitten = pending, place.value
    else:
        recent = last_at and is_recent(last_at)
        # "А у соснах?": the question still waiting for its mitten first, else the last one answered.
        again = fresh and follow_up(text, pending, app.forest)
        if again:
            await state.update_data(**answered(data))
        text = again or (recent and follow_up(text, data.get("last_question"), app.forest)) or text
    await answer_question(message, state, app, bot, text, mitten)
    await remind_document(message, state, app)


async def answer_question(message: Message, state: FSMContext, app: App, bot: Bot, text: str,
                          mitten: str | None = None, keep_state: bool = False) -> None:
    """`keep_state`: a document is waiting for a name or a place, so a question that needs the mitten
    is asked to name it in the text instead of switching the dialog to mitten buttons."""
    epoch = (await state.get_data()).get("epoch", 0)
    asking = app.asking[message.chat.id]
    asking.append(text)
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            if app.archive_ready and not app.archive_ready.is_set():
                await message.answer(texts.ARCHIVE_LOADING)
                await app.archive_ready.wait()
            reply = await app.answerer.ask(text, now_utc(), mitten=mitten)
    finally:
        asking.remove(text)
    if (await state.get_data()).get("epoch", 0) != epoch:
        return  # /cancel or /start while the model answered: the question is dropped
    if reply.kind == "unavailable":
        await answer_later(message, state, app, text, mitten, epoch)
    else:
        await deliver(message, state, app, text, reply, epoch, keep_state)


async def deliver(message: Message, state: FSMContext, app: App, question: str, reply: Reply, epoch: int,
                  keep_state: bool = False, lead: str = "") -> None:
    """Send the answer, or the mitten buttons when the answer depends on the mitten.
    `epoch`: the dialog's epoch when the question came; a /cancel since then drops it.
    `lead`: an answer given later starts with the question it answers (see later_lead)."""
    if reply.kind == "need_place":
        # The answer took a few seconds, or minutes when it came later: a document may have come meanwhile
        # and now waits for a name or a place, and typed words go to it. Then the question is asked again
        # with its mitten. While a document holds the lock the question is still unanswered, so /cancel names it.
        asking = app.asking[message.chat.id]
        asking.append(question)
        try:
            async with app.locks[state.key.user_id]:  # `message` may be the bot's own, under a button
                if (await state.get_data()).get("epoch", 0) != epoch:
                    return  # /cancel or /start while a document held the lock: the question is dropped
                busy = keep_state or await state.get_state() in DOCUMENT_STATES
                if not busy:
                    key = uuid.uuid4().hex[:8]
                    questions = dict(list((await state.get_data()).get("questions", {}).items())[-9:]) | {key: question}
                    await state.set_state(Flow.waiting_question_place)
                    await state.update_data(question=question, question_id=key, questions=questions,
                                            question_at=now_utc().isoformat())
        finally:
            asking.remove(question)
        if busy:
            await message.answer(lead + texts.ASK_PLACE_IN_TEXT)  # the callers then remind about the document
        else:
            await message.answer(lead + (reply.text or texts.ASK_PLACE),
                                 reply_markup=question_keyboard(app.forest, key, reply.mittens or None))
        return
    await state.update_data(last_question=question, last_at=now_utc().isoformat())
    await message.answer(lead + reply.text)


def short_question(question: str, limit: int = 200) -> str:
    """The question as the bot quotes it back: on one line, and cut when long, so the message still fits."""
    question = " ".join(question.split())
    return question[:limit].rstrip() + "…" if len(question) > limit else question


def later_lead(question: str, mitten: str | None) -> str:
    """The start of a message sent later: other messages may have come in between, so it names its question
    (and the mitten chosen with a button)."""
    question = short_question(question)
    if mitten:
        return texts.ANSWER_LATER_LEAD_MITTEN.format(question=question, mitten=mitten)
    return texts.ANSWER_LATER_LEAD.format(question=question)


async def answer_later(message: Message, state: FSMContext, app: App, question: str, mitten: str | None,
                       epoch: int) -> None:
    """No model answered in time, as all rest after their rate limits: promise the answer and ask again
    in the background."""
    waiting = app.later[message.chat.id]
    if len(waiting) >= LATER_PER_CHAT:
        await message.answer(texts.LLM_UNAVAILABLE)
        return
    # The task counts before the first await: questions forwarded together are handled at once and would
    # all pass the check above. It pauses before it asks, so the promise still comes first.
    task = asyncio.create_task(ask_later(message, state, app, question, mitten, epoch))
    waiting[task] = question
    task.add_done_callback(lambda done: waiting.pop(done, None))  # also when asking fails half way
    app.background.add(task)
    task.add_done_callback(app.background.discard)
    await message.answer(texts.ANSWER_LATER)


async def ask_later(message: Message, state: FSMContext, app: App, question: str, mitten: str | None,
                    epoch: int) -> None:
    """Ask again after each pause. The first reply is delivered as it would have been at once, and then the
    bot's last word is again about what still waits for a typed answer. The router's error handler does not
    see this task, so it logs its own errors and still tells the animal."""
    lead = later_lead(question, mitten)
    try:
        for pause in LATER_PAUSES:
            await asyncio.sleep(pause)
            reply = await app.answerer.ask(question, now_utc(), mitten=mitten)
            if reply.kind != "unavailable":
                break
        # Answered or given up: a /cancel from now on no longer says the bot will not answer it.
        app.later.get(message.chat.id, {}).pop(asyncio.current_task(), None)
        if reply.kind == "unavailable":
            log.warning("No model was free for a question of chat %s in %s s", message.chat.id, sum(LATER_PAUSES))
            await message.answer(lead + texts.ANSWER_LATER_FAILED)
        else:
            await deliver(message, state, app, question, reply, epoch, lead=lead)
        await remind_document(message, state, app)
    except Exception:
        log.exception("Answering a question of chat %s later failed", message.chat.id)
        with suppress(TelegramAPIError):
            await message.answer(lead + texts.SOMETHING_WRONG)


@router.message()
async def anything_else(message: Message) -> None:
    await message.answer(texts.OTHER_CONTENT)


@router.callback_query()
async def unknown_button(callback: CallbackQuery) -> None:
    """Data no button of the bot carries: the button stops spinning, and nothing else happens."""
    await callback.answer(texts.UNKNOWN_BUTTON)


@router.errors()
async def unexpected_error(event: ErrorEvent, bot: Bot) -> None:
    """Whatever breaks, the animal hears back instead of waiting in silence."""
    update = event.update
    log.error("Update %s failed", update.update_id, exc_info=event.exception)
    chat = update.message.chat if update.message else \
        update.callback_query.message.chat if update.callback_query and update.callback_query.message else None
    if chat:
        with suppress(TelegramAPIError):
            await bot.send_message(chat.id, texts.SOMETHING_WRONG)
