"""Telegram handlers: text is a question, a file is advice."""

import asyncio
import logging
import re
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from aiogram import BaseMiddleware, Bot, F, Router
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.exceptions import TelegramAPIError
from aiogram.types import CallbackQuery, ErrorEvent, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender

from . import texts
from .answering import Answerer, format_date
from .classify import WHOLE_FOREST_CAPTION, classify_document, scope_from_caption
from .config import Settings
from .extract import (MAX_OCR_PAGES, SUPPORTED_EXTENSIONS, TooManyScannedPages, extension, extract_text,
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


BIG_PDF_BYTES = 1024 * 1024  # a scan this big takes a while to read, so the sender is told
FOLLOW_UP_MINUTES = 30


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
    place = re.sub(r"^(?:а|і|й|ну|а як|а що|а якщо|щодо|стосовно)\s+", "", text.strip().lower()).strip(" ?.!,")
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
    return scope_from_caption(text, forest)


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
        user = data.get("event_from_user")
        chat = event.chat if isinstance(event, Message) else event.message.chat if event.message else None
        if not app.settings.invites_only or not user or (chat and chat.type != ChatType.PRIVATE) \
                or app.store.is_member(user.id):
            return await handler(event, data)
        token = invite_token(event.text) if isinstance(event, Message) else None
        # A bot nobody has joined yet tells whoever started it where the first invite is.
        need_invite = texts.NEED_INVITE if app.store.has_members() else texts.NEED_FIRST_INVITE
        # The Telegram name helps the inviter recognise the animal in /invited before it names itself.
        tg_name = user.full_name + (f" @{user.username}" if user.username else "")
        if token and app.store.use_invite(token, user.id, now_utc(), tg_name):
            await event.answer(f"{texts.INVITE_WELCOME}\n\n{texts.START}")
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
    return datetime.now(timezone.utc)


# Keyboards

def mitten_keyboard(forest: Forest, prefix: str, only: list[str] | None = None) -> InlineKeyboardMarkup:
    buttons = [InlineKeyboardButton(text=m, callback_data=f"{prefix}:m:{i}")
               for i, m in enumerate(forest.mittens) if only is None or m in only]
    rows = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def place_keyboard(forest: Forest, prefix: str) -> InlineKeyboardMarkup:
    keyboard = mitten_keyboard(forest, prefix).inline_keyboard
    parts = [InlineKeyboardButton(text=name.capitalize(), callback_data=f"{prefix}:p:{key}")
             for key, name in PART_NAMES.items()]
    whole = [InlineKeyboardButton(text=texts.WHOLE_FOREST, callback_data=f"{prefix}:f:")]
    return InlineKeyboardMarkup(inline_keyboard=[*keyboard, parts, whole])


def doc_place_keyboard(forest: Forest, pending: dict) -> InlineKeyboardMarkup:
    """Buttons for one waiting document: an old keyboard must not file the next document."""
    return place_keyboard(forest, f"dp:{pending['id']}" if pending.get("id") else "dp")


def answered(data: dict) -> dict:
    """FSM data once the waiting question is answered: its buttons then answer nothing more."""
    done = {data["question_id"]: None} if data.get("question_id") else {}
    return {"question": None, "question_id": None, "questions": data.get("questions", {}) | done}


def question_prefix(data: dict) -> str:
    """Mitten buttons for the waiting question only: buttons under an older question must not answer it."""
    return f"qp:{data['question_id']}" if data.get("question_id") else "qp"


def scope_from_callback(forest: Forest, kind: str, value: str) -> Scope | None:
    if kind == "m" and value.isdigit() and int(value) < len(forest.mittens):
        return Scope("mitten", forest.mittens[int(value)])
    if kind == "p" and value in PART_NAMES:
        return Scope("part", value)
    if kind == "f":
        return FOREST_SCOPE
    return None


# Commands

@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    dropped = await drop_dialog(message, state)
    if dropped:
        await message.answer(texts.CANCELLED_DOCS.format(titles=dropped))
    await message.answer(texts.START)


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(texts.HELP)


@router.message(Command("cancel"))
async def cancel(message: Message, state: FSMContext) -> None:
    dropped = await drop_dialog(message, state)
    await message.answer(texts.CANCELLED_DOCS.format(titles=dropped) if dropped else texts.CANCELLED)


async def drop_dialog(message: Message, state: FSMContext) -> str:
    """/cancel and /start: forget waiting documents and questions, and say which documents were dropped.
    A new epoch tells a document still being read or placed by the model to stop: it was dropped too."""
    data = await state.get_data()
    await state.clear()
    await state.update_data(saved=data.get("saved", {}), epoch=data.get("epoch", 0) + 1)
    return ", ".join(f'"{pending["title"]}"' for pending in data.get("queue", []))


@router.message(Command("name"))
async def change_name(message: Message, state: FSMContext) -> None:
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
        callback_data=f"del:{doc.id}",
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
            callback_data=f"rev:{tg_id}",
        )])
    await message.answer(texts.INVITED_LIST, reply_markup=InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data.startswith("rev:"))
async def confirm_revoke(callback: CallbackQuery, app: App) -> None:
    member_id = int(callback.data.split(":")[1])
    await callback.answer()
    names = {tg_id: name for tg_id, name, _, _ in app.store.invited_by_chain(callback.from_user.id)}
    if member_id not in names:
        await callback.message.edit_text(texts.NOT_YOUR_INVITEE)
        return
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.REVOKE_YES, callback_data=f"revok:{member_id}"),
        InlineKeyboardButton(text=texts.REVOKE_NO, callback_data="revno"),
    ]])
    name = names[member_id] or texts.INVITED_UNNAMED
    await callback.message.edit_text(texts.CONFIRM_REVOKE.format(name=name), reply_markup=keyboard)


@router.callback_query(F.data.startswith("revok:"))
async def revoke(callback: CallbackQuery, app: App) -> None:
    removed = app.store.revoke_member(int(callback.data.split(":")[1]), callback.from_user.id)
    await callback.answer()
    if not removed:
        await callback.message.edit_text(texts.NOT_YOUR_INVITEE)
        return
    animals = "1 звіра" if len(removed) == 1 else f"{len(removed)} звірів"
    await callback.message.edit_text(texts.REVOKED.format(animals=animals))


@router.callback_query(F.data == "revno")
async def keep_access(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(texts.REVOKE_KEPT)


@router.callback_query(F.data.startswith("del:"))
async def confirm_delete(callback: CallbackQuery, app: App) -> None:
    doc = app.store.get(int(callback.data.split(":")[1]))
    await callback.answer()
    if not doc or doc.author_id != callback.from_user.id or doc.status != "active":
        await callback.message.edit_text(texts.NOT_YOURS)
        return
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.DELETE_YES, callback_data=f"delok:{doc.id}"),
        InlineKeyboardButton(text=texts.DELETE_NO, callback_data="delno"),
    ]])
    await callback.message.edit_text(texts.CONFIRM_DELETE.format(title=doc.title), reply_markup=keyboard)


@router.callback_query(F.data.startswith("delok:"))
async def delete_document(callback: CallbackQuery, app: App) -> None:
    doc = app.store.delete_animal_document(int(callback.data.split(":")[1]), callback.from_user.id)
    await callback.answer()
    await callback.message.edit_text(texts.DELETED.format(title=doc.title) if doc else texts.NOT_YOURS)


@router.callback_query(F.data == "delno")
async def keep_document(callback: CallbackQuery) -> None:
    await callback.answer()
    await callback.message.edit_text(texts.KEPT)


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

    epoch = (await state.get_data()).get("epoch", 0)
    if ext == ".pdf" and (file.file_size or 0) > BIG_PDF_BYTES:
        await message.answer(texts.DOC_READING.format(title=title_from_filename(filename)))
    try:
        async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
            data = (await bot.download(file)).read()
            text = await asyncio.to_thread(extract_text, data, filename, app.settings.ocr_langs, ext)
    except TooManyScannedPages as error:
        await message.answer(texts.DOC_TOO_MANY_SCANS.format(filename=filename, pages=error.pages, limit=MAX_OCR_PAGES))
        return
    except Exception as error:  # broken PDF, wrong encoding, etc.
        log.warning("Cannot read %s: %s", filename, error)
        await message.answer(texts.DOC_UNREADABLE.format(filename=filename))
        return
    if not text.strip():
        await message.answer(texts.DOC_EMPTY.format(filename=filename))
        return

    pending = {"id": uuid.uuid4().hex[:8], "filename": filename, "title": title_from_filename(filename),
               "text": text, "sent_at": now_utc().isoformat()}
    caption_scope = scope_from_caption(message.caption, app.forest)
    if caption_scope:
        pending["scope"] = caption_scope.encode()
    async with app.locks[message.from_user.id]:
        data = await state.get_data()
        if data.get("epoch", 0) != epoch:  # /cancel or /start while the file was being read
            await message.answer(texts.CANCELLED_DOCS.format(titles=f'"{pending["title"]}"'))
            return
        queue = data.get("queue", []) + [pending]
        await state.update_data(queue=queue)
        current = await state.get_state()
        if current in DOCUMENT_STATES:
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
        InlineKeyboardButton(text=texts.NAME_YES, callback_data="nm:yes"),
        InlineKeyboardButton(text=texts.NAME_NO, callback_data="nm:no"),
    ]])


@router.callback_query(F.data.startswith("nm:"))
async def name_confirmed(callback: CallbackQuery, state: FSMContext, app: App) -> None:
    await callback.answer()
    await callback.message.edit_reply_markup(reply_markup=None)
    name = (await state.get_data()).get("name")
    if await state.get_state() != Flow.confirming_name.state or not name:
        return
    if callback.data == "nm:no":
        await state.set_state(Flow.waiting_name)
        await callback.message.answer(texts.ASK_NAME_AGAIN)
        return
    await accept_name(callback.message, state, app, callback.from_user.id, name)


async def accept_name(message: Message, state: FSMContext, app: App, user_id: int, name: str) -> None:
    async with app.locks[user_id]:
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
        if scope is None:
            await state.set_state(Flow.waiting_doc_place)
            await message.answer(texts.ASK_DOC_PLACE.format(title=pending["title"]),
                                 reply_markup=doc_place_keyboard(app.forest, pending))
            return
        await file_pending(message, state, app, user_id, queue, scope)


async def remind_document(message: Message, state: FSMContext, app: App) -> None:
    """After a question answered in the middle, ask again what the waiting document still needs."""
    current, data = await state.get_state(), await state.get_data()
    if current == Flow.waiting_doc_place.state and data.get("queue"):
        await message.answer(texts.DOC_STILL_WAITING.format(title=data["queue"][0]["title"]),
                             reply_markup=doc_place_keyboard(app.forest, data["queue"][0]))
    elif current == Flow.confirming_name.state and data.get("name"):
        await message.answer(texts.CONFIRM_NAME.format(name=data["name"]), reply_markup=name_keyboard())
    elif current == Flow.waiting_name.state:
        await message.answer(texts.ASK_NAME_STILL if data.get("queue") else texts.ASK_NEW_NAME)  # or /name
    elif data.get("queue") and current not in DOCUMENT_STATES and not app.locks[message.chat.id].locked():
        # Left by the old code, which could lose the dialog state while a document waited.
        if app.store.user_name(message.chat.id):
            await state.set_state(Flow.waiting_doc_place)
            await message.answer(texts.DOC_STILL_WAITING.format(title=data["queue"][0]["title"]),
                                 reply_markup=doc_place_keyboard(app.forest, data["queue"][0]))
        else:
            await state.set_state(Flow.waiting_name)
            await message.answer(texts.ASK_NAME_STILL)


async def file_pending(message: Message, state: FSMContext, app: App, user_id: int, queue: list[dict],
                       scope: Scope) -> None:
    """Save the first waiting document. Its id is kept, so an older keyboard for it says it is saved.
    The store and the queue change before the reply is sent, so a /cancel meanwhile cannot bring it back."""
    pending, name = queue[0], app.store.user_name(user_id) or "невідомо"
    doc, replaced = app.store.add_animal_document(
        title=pending["title"], filename=pending["filename"], scope=scope, author_id=user_id,
        author_name=name, text=pending["text"], now=datetime.fromisoformat(pending["sent_at"]),
    )
    saved = (await state.get_data()).get("saved", {})
    if pending.get("id"):
        saved = dict(list(saved.items())[-19:]) | {pending["id"]: doc.id}
    # Documents queued by the old code have no ids: give them ids now, so an old keyboard pressed again
    # cannot file the next one.
    rest = [{"id": uuid.uuid4().hex[:8], **waiting} if not waiting.get("id") else waiting for waiting in queue[1:]]
    await state.update_data(queue=rest, saved=saved)
    reply = texts.DOC_SAVED.format(name=name, title=doc.title, scope=scope.describe(),
                                   date=format_date(doc.created_at, app.tz))
    if replaced:
        reply += texts.DOC_REPLACED.format(title=replaced.title)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text=texts.CHANGE_PLACE, callback_data=f"chg:{doc.id}")]])
    await message.answer(reply, reply_markup=keyboard)


@router.callback_query(F.data.startswith("dp:"))
async def doc_place_chosen(callback: CallbackQuery, state: FSMContext, app: App) -> None:
    fields = callback.data.split(":")
    key = fields[1] if len(fields) == 4 else None  # "dp:<document>:m:3"; buttons sent before ids were "dp:m:3"
    scope = scope_from_callback(app.forest, fields[-2], fields[-1])
    await callback.answer()
    if scope is None:
        return
    await callback.message.edit_reply_markup(reply_markup=None)
    async with app.locks[callback.from_user.id]:
        data = await state.get_data()
        queue = data.get("queue", [])
        saved = data.get("saved", {}).get(key or "")
        if saved:  # a second keyboard for a document already saved: where it is now (after "Змінити місце")
            doc = app.store.get(saved) if isinstance(saved, int) else None
            await callback.message.answer(texts.DOC_ALREADY_SAVED.format(title=doc.title, scope=doc.scope.describe())
                                          if doc and doc.status == "active" else texts.NOT_YOURS)
            return
        if not queue:
            if await state.get_state() == Flow.waiting_doc_place.state:
                await state.set_state(None)
            await callback.message.answer(texts.DOC_LOST)
            return
        if key != queue[0].get("id"):
            # Buttons under an earlier document's question, or sent before documents had ids.
            await callback.message.answer(texts.DOC_PLACE_OLD_BUTTONS.format(title=queue[0]["title"]),
                                          reply_markup=doc_place_keyboard(app.forest, queue[0]))
            return
        if await state.get_state() in (Flow.waiting_name.state, Flow.confirming_name.state):
            await remind_document(callback.message, state, app)  # /name in between: the name comes first
            return
        # Waiting for its place, or left in the queue by the old code that lost the dialog state.
        await file_pending(callback.message, state, app, callback.from_user.id, queue, scope)
        await state.set_state(None)
        await process_queue(callback.message, state, app, callback.from_user.id)


@router.message(Flow.waiting_doc_place, F.text, ~F.text.startswith("/"))
async def doc_place_typed(message: Message, state: FSMContext, app: App, bot: Bot) -> None:
    scope = typed_place(message.text, app.forest)
    if scope is None and looks_like_question(message.text, app.forest):
        # A question while a document waits for its place: answer it, the document keeps waiting.
        await answer_question(message, state, app, bot, message.text.strip(), keep_state=True)
        await remind_document(message, state, app)
        return
    if scope is None:
        await message.answer(texts.PICK_WITH_BUTTONS)
        return
    async with app.locks[message.from_user.id]:
        queue = (await state.get_data()).get("queue", [])
        if queue:
            await file_pending(message, state, app, message.from_user.id, queue, scope)
        await state.set_state(None)
        await process_queue(message, state, app, message.from_user.id)


@router.callback_query(F.data.startswith("chg:"))
async def change_place(callback: CallbackQuery, app: App) -> None:
    doc_id = callback.data.split(":")[1]
    await callback.answer()
    await callback.message.answer(texts.CHANGE_PLACE_ASK, reply_markup=place_keyboard(app.forest, f"cs:{doc_id}"))


@router.callback_query(F.data.startswith("cs:"))
async def place_changed(callback: CallbackQuery, app: App) -> None:
    _, doc_id, kind, value = callback.data.split(":", 3)
    scope = scope_from_callback(app.forest, kind, value)
    await callback.answer()
    doc = app.store.change_scope(int(doc_id), callback.from_user.id, scope) if scope else None
    await callback.message.edit_reply_markup(reply_markup=None)
    await callback.message.answer(
        texts.PLACE_CHANGED.format(title=doc.title, scope=scope.describe()) if doc else texts.NOT_YOURS)


# Questions

@router.callback_query(F.data.startswith("qp:"))
async def question_place_chosen(callback: CallbackQuery, state: FSMContext, app: App, bot: Bot) -> None:
    fields = callback.data.split(":")
    key = fields[1] if len(fields) == 4 else None  # "qp:<question>:m:3"; buttons sent before ids were "qp:m:3"
    scope = scope_from_callback(app.forest, fields[-2], fields[-1])
    await callback.answer()
    if not scope:
        return
    await callback.message.edit_reply_markup(reply_markup=None)
    async with app.locks[callback.from_user.id]:  # a double tap answers once
        data = await state.get_data()
        # Each keyboard answers its own question, also under an older one; old buttons had no id.
        questions = data.get("questions", {})
        question = questions.get(key) if key else data.get("question")
        if key in questions and question is None:
            return  # the same question's buttons pressed again: the answer is already above
        if not question:
            await callback.message.answer(texts.QUESTION_LOST)
            return
        if not key or key == data.get("question_id"):
            # A document may have come in between and wait for a name or a place: its state stays.
            if await state.get_state() == Flow.waiting_question_place.state:
                await state.set_state(None)
            await state.update_data(question=None, question_id=None)
        if key:
            await state.update_data(questions=questions | {key: None})
    await answer_question(callback.message, state, app, bot, question, mitten=scope.value)
    await remind_document(callback.message, state, app)


@router.message(F.text, ~F.text.startswith("/"))
async def question(message: Message, state: FSMContext, app: App, bot: Bot) -> None:
    text, mitten = message.text.strip(), None
    data = await state.get_data()
    pending = data.get("question")
    waiting = await state.get_state() == Flow.waiting_question_place.state
    # A question still waiting for its mitten after a document came in between, with nothing answered since:
    # a bare place typed now ("у соснах") answers it.
    asked_at = data.get("question_at")
    fresh = pending and asked_at and asked_at > (data.get("last_at") or "") and \
        (now_utc() - datetime.fromisoformat(asked_at)).total_seconds() < FOLLOW_UP_MINUTES * 60
    place = typed_place(text, app.forest) if pending and (waiting or fresh) else None
    if waiting or place:
        if place and place.kind != "mitten":
            # "У хвойному лісі": the answer differs by mitten, so offer the mittens of that part.
            only = app.forest.mittens if place.kind == "forest" else app.forest.mittens_in([place.value])
            if len(only) != 1:
                ask = (texts.ASK_MITTEN_IN_PART.format(part=PART_NAMES[place.value]) if place.kind == "part"
                       else texts.ASK_PLACE)
                await message.answer(ask, reply_markup=mitten_keyboard(app.forest, question_prefix(data), only or None))
                return
            place = Scope("mitten", only[0])
        await state.set_state(None)
        # A new question instead of the mitten: the old one's buttons still answer it.
        await state.update_data(**(answered(data) if place else {"question": None, "question_id": None}))
        if place:
            # The animal answered "під дубом" instead of pressing a button.
            text, mitten = pending, place.value
    else:
        reply = small_talk(text)
        if reply:
            await message.answer(reply)
            return
        last_at = data.get("last_at")
        recent = last_at and (now_utc() - datetime.fromisoformat(last_at)).total_seconds() < FOLLOW_UP_MINUTES * 60
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
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        if app.archive_ready and not app.archive_ready.is_set():
            await message.answer(texts.ARCHIVE_LOADING)
            await app.archive_ready.wait()
        reply = await app.answerer.ask(text, now_utc(), mitten=mitten)
    if reply.kind == "need_place":
        # The answer took a few seconds: a document may have come meanwhile and now waits for a name
        # or a place, and typed words go to it. Then the question is asked again with its mitten.
        async with app.locks[message.chat.id]:
            busy = keep_state or await state.get_state() in DOCUMENT_STATES
            if not busy:
                key = uuid.uuid4().hex[:8]
                questions = dict(list((await state.get_data()).get("questions", {}).items())[-9:]) | {key: text}
                await state.set_state(Flow.waiting_question_place)
                await state.update_data(question=text, question_id=key, questions=questions,
                                        question_at=now_utc().isoformat())
        if busy:
            await message.answer(texts.ASK_PLACE_IN_TEXT)  # the callers then remind about the document
        else:
            await message.answer(reply.text or texts.ASK_PLACE, reply_markup=mitten_keyboard(
                app.forest, question_prefix(await state.get_data()), reply.mittens or None))
        return
    await state.update_data(last_question=text, last_at=now_utc().isoformat())
    await message.answer(reply.text)


@router.message()
async def anything_else(message: Message) -> None:
    await message.answer(texts.OTHER_CONTENT)


@router.errors()
async def unexpected_error(event: ErrorEvent, bot: Bot) -> None:
    """Whatever breaks, the animal hears back instead of waiting in silence."""
    log.error("Update %s failed", event.update.update_id, exc_info=event.exception)
    update = event.update
    chat = update.message.chat if update.message else \
        update.callback_query.message.chat if update.callback_query and update.callback_query.message else None
    if chat:
        try:
            await bot.send_message(chat.id, texts.SOMETHING_WRONG)
        except TelegramAPIError:
            pass
