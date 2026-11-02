"""From a question to an answer: pick the place, find the documents, ask Groq, cite sources."""

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import NamedTuple
from zoneinfo import ZoneInfo

from .llm import LLM, LLMUnavailable
from .places import CONIFER, DECIDUOUS, PART_NAMES, RIVER, Forest, Scope, resident_homes, residents_from_statute
from .search import COMMON_STEMS, STOPWORDS, SYNONYMS, Hit, Index, expand, looks_like_heading, split_chunks, tokens
from .storage import Document, Store, norm_title
from . import texts

MAX_DOCS = 7  # documents in one answer prompt; archive documents are short (under 1100 characters)
DOC_CHARS = 2500  # a long guide from an animal is cut to the parts that match the question
# Groq's free tier admits a request only when its prompt plus max_completion_tokens fits the model's 8000 tokens
# a minute; qwen also takes at most 7000 for the prompt alone. Measured on 2 November 2026, these prompts are
# 2.55 characters a token on gpt-oss, and qwen counted the same prompt 1% longer: 2.5 fits both.
TOKENS_PER_MINUTE = 8000
INPUT_TOKENS_PER_MINUTE = 7000
CHARS_PER_TOKEN = 2.5
# Left free for a count a little off. A second try's reminders may still go over: llm.py then asks with a smaller
# reply budget, and if that fails too, the first answer holds.
SPARE_TOKENS = 150
# Keyword search keeps the best documents of every level (mitten, each forest part, whole forest)
# separately, so a strong whole-forest match cannot push the mitten's own advice out. A mitten has
# only a handful of documents and they matter most, so any of them that matches at all may go.
LEVEL_LIMIT = {"mitten": 3, "part": 2, "forest": 2}
LEVEL_CUTOFF = {"mitten": 0.0, "part": 0.3, "forest": 0.3}
PICK_LIMIT = 4
# A topic varies by place when some mitten's or forest part's advice on it (the topic in its title) scores at least
# this share of the
# whole forest's best: "утеплювати" (the conifer forest says October, the calendar November), but not the hunters'
# cry, which only the whole forest's memo has ("Біля греблі мисливці!" names where the hunters are).
PLACE_VARIES = 0.6
# Seconds an animal waits for an answer at most, rate-limit pauses and slower fallback models included;
# picking documents by title is a help, not a must, so it gets a short part of them.
QUESTION_SECONDS = 45
PICK_SECONDS = 10
# The pick is only {"ids": [...]}: 47-130 tokens seen, and Groq counts the whole budget against the minute's limit.
# At least llm.MIN_OUTPUT, so a model whose learned output cap is below that is still skipped instead of being
# asked for a cut-off reply.
PICK_TOKENS = 400
LOCATIVE = {"під", "біля", "коло", "поблизу", "у", "в", "на", "серед", "над", "за", "між"}  # a place at a line's start
LEVEL_NAMES = {"mitten": "рукавичка", "part": "частина лісу", "forest": "увесь ліс"}
PICK_ANIMAL_TITLES = 150  # newest animals' documents shown to the model when it picks by title, as many as fit
# How much of the calendar a question gets. Groq counts the prompt and the reply's budget against the 8000 tokens
# a minute, and while "коли" brought nearly every question the whole calendar and longer thinking, 10 of 12
# questions in a row in a live check fell back to weaker models, which answered worse.
WEEKDAY_STEMS = r"понеділ|вівтор|серед(?:а|и|у|ою|ам|ах)\b|четвер(?:\b|г)|п.?ятниц|субот|неділ|вихідн|будн"
MONTH_STEMS = r"(?:січ|берез|квіт|трав|черв|лип|серп|верес|жовт|груд)(?:ень|ня|ню|ні|нем)\b|лют(?:ий|ого|ому)\b|листопад"
SEASON_STEMS = r"зим|весн|\bліт(?:о|а|у|і|ом)\b|[ву]літку|\bлітн|\bв?осін|\bв?осен"
# Counting from today or by weekdays gets every day of this and next month and longer thinking: "сьогодні",
# "завтра", "вчора", "зараз", "вже" anywhere in a "чи" question, "можна вже", "вже можна", "вже" before a verb
# in the past ("толока вже відбулася?"), "чи ще можна", "чи не пізно", "чи скоро", "досі", "скільки днів до",
# "скільки ще чекати", "скільки (ще) до толоки", "через/за скільки днів" with no "після", "ще буде", "якого
# (котрого) числа", "дата", "які дати", "до якого часу", "якого (котрого) дня", "тиждень", the weekdays (also
# "пятниця" with no apostrophe), "наступного місяця", "наступного разу", "минулої осені". Endings and neighbouring words are
# spelled out where a bare stem would catch others: "серед дубів", "четвертий", "заразитися", "численні",
# "в тому числі", "дати" (to give), "до якої частини лісу", "що ще можна їсти", "чи ще хтось".
DATE_WORDS = re.compile(
    r"сьогодн|завтр|[ву]чора|\bзараз\b|\bдосі\b|\bчи\b[^?]*\b(?:вже|уже)\b"
    r"|\bчи\s+(?:час|пора|скоро|не\s+пізно|не\s+рано|не\s+запізно|ще\s+(?:можна|не|встиг\w*|час|пора|рано))\b"
    r"|\b(?:можна|час|пора|пізно|рано|треба)\s+(?:вже|уже)\b"
    r"|\b(?:вже|уже)\s+(?:(?:можна|час|пора|пізно|рано|треба|скоро)\b|почина|\w+(?:в|ла|ло|ли|ся|сь)\b)"
    r"|\bскільки\s+(?:ще\s+)?(?:(?:днів|тижнів|місяців|часу|чекати|лишил\w*|залишил\w*)\s+)*до\b"
    r"|\b(?:ще|довго)\s+чекати\b|\b(?:через|за)\s+скільки\b(?![^?]*\bпісля\b)|\bще\s+буде\b"
    r"|\bчисл(?:о|а|ом|ах)\b|\bдат(?:а|у|і|ою|ам|ами|ах)?\b|\b(?:як|котр)\w+\s+дати\b"
    r"|\bдо\s+(?:як|котр)\w*\s+(?:дн|дат|час(?:у|ом)?\b|пор[иуі]\b|термін|місяц|тижн)"
    r"|\b(?:як|котр)\w*\s+(?:дн[яюі]|день)\b|тижн|тижд|" + WEEKDAY_STEMS
    + r"|\b(?:наступн|найближч|минул|позаминул)\w*\s+(?:раз|дн|день|тижн|тижд|місяц|рок|рік|сезон|"
    + "|".join((WEEKDAY_STEMS, MONTH_STEMS, SEASON_STEMS)) + ")")
# Words that only name a time of year get the first weekdays of last, this and next month: "коли осіння толока?"
# needs the first Saturday of October to say it already was. Counting days is no date ("через скільки днів після
# дощу" is a plain fact), nor are the dues "на місяць" and "щомісяця", "терміново" or "літати".
TIME_WORDS = re.compile(
    r"\b(?:від|до\s+)?коли\b|\bдоки\b|сезон|\bпор(?:а|и|і|у|ою)\b|\bтермін(?:у|и|ом|ів)?\b"
    r"|(?<!на\s)(?<!за\s)(?<!що)місяц|\bр(?:ік|оку|оком)\b|" + MONTH_STEMS + "|" + SEASON_STEMS)
WEEKDAYS = ["понеділок", "вівторок", "середа", "четвер", "пʼятниця", "субота", "неділя"]
WEEKDAYS_SHORT = ["пн", "вт", "ср", "чт", "пт", "сб", "нд"]
MONTHS_NOMINATIVE = ["січень", "лютий", "березень", "квітень", "травень", "червень", "липень", "серпень", "вересень",
                     "жовтень", "листопад", "грудень"]
MONTHS = ["січня", "лютого", "березня", "квітня", "травня", "червня", "липня", "серпня", "вересня", "жовтня",
          "листопада", "грудня"]

ANSWER_SYSTEM = """Ти бот-помічник звірів лісу. Ти відповідаєш на питання лише на основі наданих документів із порадами.

Правила:
1. Бери інформацію тільки з наданих документів. Нічого не вигадуй і не додавай від себе: жодних дат, умов чи уточнень, яких немає в тексті документів.
2. Якщо в документах немає відповіді на питання, поверни "found": false.
3. Конкретніша порада важливіша за загальну: рівень "рукавичка" важливіший за "частина лісу", а "частина лісу" важливіша за "увесь ліс". Рівень документа написано в рядку "Рівень", і місця, названі в тексті документа, його не змінюють. Якщо в документі для рукавички є хоч щось на тему питання, починай відповідь із нього. Якщо поради суперечать, у "answer" пиши лише конкретнішу, а із загальнішої тільки те, що їй не суперечить. Загальнішу пораду, що радить інакше, не змішуй із відповіддю: коротко перекажи її в полі "other", скажи там, для кого вона і що тут діє конкретніша, і додай її документ у "sources" (наприклад: "Загальна пам'ятка для всього лісу радить збиратися біля бобрової греблі, але для рукавички Під дубом діє конкретніша порада.").
4. Якщо поради однаково конкретні й написані для того самого місця: документ із Лісового архіву важливіший за документ від звіра. Серед документів архіву, як і серед документів від звірів, важливіший новіший (дата й час у рядку "Звідки"; якщо вони однакові, новіший той, що стоїть у переліку вище). Пораду, що поступилася й радить інакше, коротко перекажи в полі "other", як у правилі 3, і додай її документ у "sources".
5. Якщо питання про час, пору року, день тижня чи "чи вже можна", враховуй сьогоднішню дату. Дні тижня й дати бери лише з наданого календаря (там уже пораховано, наприклад, яка дата перша субота місяця), не обчислюй їх сам. Порівнюй дату події із сьогоднішньою: якщо вона раніше за сьогодні, пиши в минулому часі, що подія вже була, і не вигадуй дату наступної, якщо її немає в документах чи календарі. Про щорічну подію за правилом ("у першу суботу жовтня", "у вересні"), чия дата цього року вже минула, пиши, що цього року вона вже була (з датою з календаря, якщо вона там є), а наступна буде за тим самим правилом наступного року. Про минулу подію ніколи не пиши в майбутньому часі.
6. Пиши грамотною українською, ввічливо, просто, доброзичливо й по суті, без казковості та без звертань на кшталт "любий звіре". Якщо звертаєшся до звіра, то на "Ви" (з великої літери): "Вам", "здайте", "біжіть", "якщо почуєте". Відповідь коротка (1-5 речень), але переказуй усі пункти документа, що стосуються питання, нічого з них не пропускай: скільки, коли, кому й куди віддавати чи приносити, чим можна замінити, хто звільнений.
7. Якщо рукавичка належить до кількох частин лісу (як "Під вербою": і листяний ліс, і біля річки), їй підходять поради кожної з цих частин: наведи в "answer" поради всіх її частин лісу, що стосуються питання, навіть якщо вони не суперечать одна одній, і скажи, до якої частини лісу належить кожна. Поради різних частин лісу однієї рукавички одна одній не поступаються, тож жодна з них не йде в "other". Порада самої рукавички, як і завжди, іде першою.
8. Не згадуй в "answer" та "other" номери й назви документів: джерела бот додасть сам.
9. Якщо в документі є окремі розділи для різних місць, бери розділ для місця з питання. Але рівень розділу той самий, що й у всього документа (рядок "Рівень"): розділ "Під дубом" у путівнику для всього лісу є порадою рівня "увесь ліс", тож документ для рукавички Під дубом або для її частини лісу важливіший за нього.
10. Місце в питанні ("під дубом", "біля сосон", "біля верби") означає рукавичку з рядка "Місце з питання", а не саме дерево: відповідай усіма порадами, що підходять цій рукавичці, навіть якщо в них це дерево не згадано, а названо інші дерева чи місця ("білі гриби тримаються старих дубів" теж відповідь на "де шукати гриби біля верби"). Звір не обов'язково там живе: ніколи не пиши "ваш…" перед рукавичкою, частиною лісу, деревом чи домом, а називай місце з рядка "Для кого" документа ("у хвойному лісі", навіть коли питали про «У соснах»).
11. Якщо в питанні названо рукавичку чи частину лісу, а документ для всього лісу відповідає на це питання ("після дощу" — і на "через скільки днів", не на "де"), його порада обов'язкова: що не суперечить конкретнішій, дай в "answer", що радить інакше, — в "other" (правило 3), а її документ — у "sources". Якщо в рядку "Поради на цю тему різні в різних місцях" стоїть "так", скажи, для якого місця кожна порада ("У хвойному лісі …"), а пораду всього лісу завжди починай словами "Загальна порада для всього лісу — …" й не змішуй із порадою місця ("загальна" — лише про весь ліс). Якщо ж відповідь лише з поради для всього лісу, почни так: "Окремої поради про … для … немає, тож діє загальна порада для всього лісу: …". Якщо там "ні", цих слів не пиши: порада для всього лісу однакова для всіх (усе одно перекажи всі її пункти на тему питання).

Поверни JSON: {"facts": ["кожен пункт документів, з яких відповідаєш, на тему питання, окремим рядком, навіть якщо про нього прямо не питали (на \"скільки внесок\" це й кому та куди віддавати, і чим можна замінити): скільки, коли, кому, куди, чим замінити, хто звільнений; на початку рядка номер документа і його рівень, наприклад [2, частина лісу]"], "found": true або false, "answer": "текст відповіді для місця з питання, у якому є всі ці пункти, крім тих, що пішли в other", "other": "порада, що радить інакше й тут поступається відповіді, з поясненням чому, або \"\"", "sources": [номери документів, з яких узято відповідь, найважливіший перший]}"""

PLACE_UNKNOWN_RULE = """

Місце в питанні не вказано, а серед документів є поради для окремих рукавичок чи частин лісу. Якщо відповідь залежить від того, про яку рукавичку питає звір (для різних рукавичок чи частин лісу поради різні, або відповідь є лише в порадах для окремих місць і для кожного своя), не відповідай, а поверни "needs_place": true. Якщо ж відповідь однакова для всього лісу, або питання про конкретного звіра, який у документах живе в певній рукавичці (наприклад, "коли Ведмідь лягає спати?" чи "де живе Зайчик?"), відповідай як звичайно, назвавши цю рукавичку, і поверни "needs_place": false. Порада для рукавички цього звіра важливіша за загальну. Якщо відповідь є лише для одного місця, відповідай і прямо скажи, для якого місця ця порада."""

PICK_SYSTEM = """Тобі дають питання звіра і перелік назв документів із порадами. Обери до чотирьох документів, у яких найімовірніше є відповідь. Зважай на зміст, а не лише на однакові слова: "що їсти" і "чим поживитись" про те саме, "хто живе" є у статуті. Якщо на тему питання є кілька документів для різних місць або від різних звірів, обери їх усі. Якщо жоден не підходить, поверни порожній список.
Поверни JSON: {"ids": [номери документів]}"""

LEVEL_REMINDER = """

Зверни увагу: документ [{better}] (рівень "{better_level}") конкретніший за документ [{weaker}] (рівень "{weaker_level}"), навіть якщо в [{weaker}] є розділ про місце з питання. Якщо в [{better}] є відповідь на питання, відповідай за ним і постав його першим у "sources", а пораду [{weaker}], якщо вона радить інакше, перекажи в полі "other"."""

PARTS_REMINDER = """

Зверни увагу: рукавичка "{mitten}" належить до кількох частин лісу, і їй підходять поради кожної з них. У відповіді {cited}немає поради для {missing}. Якщо в документі {docs} є щось на тему питання "{topic}" (навіть без дерева, названого в питанні, чи з іншими деревами: дерево з питання лише вказує на рукавичку), додай у відповідь кожен його пункт, що відповідає на це питання, сказавши, {label}, і додай номер документа в "sources".{quotes}"""

FACTS_REMINDER = """

Зверни увагу: у документах, з яких ти відповідаєш, є на тему питання пункти, яких немає ні в "answer", ні в "other": {facts}. Додай їх у "answer" (можна коротко), нічого не вигадуючи, а пункт поради, що тут поступається конкретнішій чи архіву, — коротко в "other"."""

ARCHIVE_REMINDER = """

Зверни увагу: перша порада у відповіді — з документа звіра [{animal}], а документ Лісового архіву {archive} написаний для того самого місця. Якщо в архіві теж є відповідь на питання, за правилом 4 важливіший архів, навіть якщо документ звіра новіший: відповідай за архівом і постав його першим у "sources", а пораду [{animal}], якщо вона радить інакше, перекажи в полі "other". Якщо відповіді в архіві немає, відповідай як раніше."""

SOURCES_REMINDER = """

Зверни увагу: у "sources" немає номерів документів, з яких узято відповідь. Без них бот відповідь не покаже: постав у "sources" номери документів, з яких відповідаєш, найважливіший перший."""

OTHER_REMINDER = """

Зверни увагу: у "sources" лише один документ, а "other" переказує пораду з іншого документа. Без нього бот цю пораду не покаже: додай у "sources" після джерел відповіді номер документа, пораду з якого переказує "other". Якщо такого документа немає, поверни "other": ""."""

FOREST_REMINDER = """

Зверни увагу: у "sources" немає документа для всього лісу, а в документі {docs} (рівень "увесь ліс") є порада на тему питання "{topic}". За правилом 11 вона обов'язкова: кожен її пункт на це питання, що не суперечить конкретнішій пораді, додай у "answer"{label}, а що радить інакше — коротко в "other", як у правилі 3, і додай номер документа в "sources".{quotes}"""

UKRAINIAN_REMINDER = """

Зверни увагу: у відповіді є російські літери ({letters}). Пиши лише українською: перекажи "answer" і "other" грамотною українською, нічого не змінюючи по суті."""

# The model, given "Де шукати гриби біля верби?", kept to the willow and skipped the deciduous guide's "Білі гриби
# тримаються старих дубів" even when reminded of it; quoting the sentences on the topic brought them in. Asked for
# "саме те", the live second try took the white mushrooms and left out the birch boletes quoted beside them.
QUOTE_NOTE = (" На цю тему в документі [{number}] сказано: «{quote}» Перекажи з цього кожен пункт, що відповідає на "
              "питання, жодного не пропускаючи.")
QUOTE_CHARS = 300
# Words of searching and going say nothing of the topic: "де шукати гриби" shares them with "Гриби шукайте після
# дощу" (when, not where), "коли йти по гриби" with no advice at all.
SEARCH_STEMS = frozenset({"шука", "знай", "знах", "йти", "іти", "піти", "ходи"})
WHEN = re.compile(r"\bколи\b", re.I)
# The line saying whom a document is for: "Ця пам'ятка для всіх мешканців лісу", "Це звернення до всіх мешканців".
FOR_EVERYONE = re.compile(r"\b(?:всіх|усіх)\s+мешканців\b|\b(?:всього|усього)\s+лісу\b", re.I)
# A document's own question answers the animal's when it asks the same: "Де шукати білі гриби?" and "А де
# підберезники?" for "де шукати гриби", not "Коли йти по гриби?".
QUESTION_WORDS = re.compile(r"\b(?:де|коли|скільки|чим|як|куди|хто|що)\b", re.I)
# "Гриби" asks about each kind ("А де підберезники?" answers "де шукати гриби"), one kind about itself only: "де
# шукати підберезники" asks nothing about the white mushrooms, nor "де шукати опеньки" about the birch boletes.
GENERAL_WORDS = frozenset({"гриби", "гриб", "грибів"})
KINDS = frozenset(term for group in SYNONYMS if group & GENERAL_WORDS for word in group - GENERAL_WORDS
                  for term in tokens(word))

RESIDENT_NOTE = " (місце визначено за звіром {name}, який тут живе; якщо порада однакова для всього лісу, на рукавичці не наголошуй)"


@dataclass
class Reply:
    kind: str  # "answer" | "not_found" | "need_place" | "unavailable"
    text: str = ""
    mittens: list[str] = field(default_factory=list)  # need_place: the mittens to choose from, empty means all


class Draft(NamedTuple):
    """The model's answer with its sources, kept while the model is asked to put it right."""
    answer: str
    other: str
    sources: list[Document]
    faults: int  # how many of the rules Answerer._review counts it breaks, the whole forest's advice aside
    general: bool  # the whole forest's advice on the topic of a place asked left out (rule 11)
    facts: list  # the points the model listed from the documents
    points: list[str]  # the documents' own answers to the very question asked (Answerer._asked_points)


def prompt_chars(max_tokens: int) -> int:
    """How many characters a prompt, system prompt included, may take for Groq's free tier to admit it with
    this budget for the reply, on every model."""
    tokens = min(TOKENS_PER_MINUTE - max_tokens, INPUT_TOKENS_PER_MINUTE) - SPARE_TOKENS
    return int(tokens * CHARS_PER_TOKEN)


def format_date(moment: datetime, tz: ZoneInfo) -> str:
    local = moment.astimezone(tz)
    return f"{local:%d.%m.%Y} о {local:%H:%M}"


def human_date(moment: datetime, tz: ZoneInfo) -> str:
    local = moment.astimezone(tz)
    return f"{local.day} {MONTHS[local.month - 1]} {local.year}"


def calendar_need(question: str) -> str:
    """How much of the calendar the question needs: "days", "months" or "today" (DATE_WORDS, TIME_WORDS)."""
    question = question.lower()
    return "days" if DATE_WORDS.search(question) else "months" if TIME_WORDS.search(question) else "today"


def calendar_note(moment: datetime, tz: ZoneInfo, need: str = "days") -> str:
    """Today with its weekday; for "months" also the first weekdays of last, this and next month, so the model
    reads "перша субота жовтня" off the calendar instead of counting weekdays itself, and on 1 November sees that
    it was 3 October; for "days" also every day of this and next month with its weekday."""
    local = moment.astimezone(tz).date()
    lines = [f"Сьогодні: {WEEKDAYS[local.weekday()]}, {local.day} {MONTHS[local.month - 1]} {local.year}."]
    if need == "today":
        return lines[0]
    this = local.replace(day=1)
    last = (this - timedelta(days=1)).replace(day=1)  # in January, December of the year before
    following = (this + timedelta(days=31)).replace(day=1)
    for first in (last, this, following):
        month, label = first.month, f"{MONTHS_NOMINATIVE[first.month - 1]} {first.year}"
        if need == "days" and first != last:
            days = []
            day = first
            while day.month == month:
                days.append(f"{day.day} {WEEKDAYS_SHORT[day.weekday()]}")
                day += timedelta(days=1)
            lines.append(f"{label.capitalize()}: " + ", ".join(days))
        firsts = []
        for offset in range(7):
            day = first + timedelta(days=offset)
            ordinal = "перша" if day.weekday() in (2, 4, 5, 6) else "перший"
            firsts.append(f"{ordinal} {WEEKDAYS[day.weekday()]} {MONTHS[month - 1]} = {day.day} {MONTHS[month - 1]}")
        # Each line names its month and year, as last month's comes without the list of its days.
        lines.append(f"Перші дні тижня, {label}: " + ", ".join(firsts))
    return "\n".join(lines)


def source_line(doc: Document, tz: ZoneInfo) -> str:
    """Archive: the document and its place. An animal's document: also its real name and when it was sent."""
    title, place = f'"{doc.title}"', doc.scope.describe()
    if doc.from_archive:
        return f"{title} (Лісовий архів, місце: {place})"
    return (f"{title}, надіслано {format_date(doc.created_at, tz)}, автор: {doc.author_name or 'невідомо'}, "
            f"місце: {place}")


# Numbers written as words count as the numbers: "чотири жолуді" keeps "4 жолуді".
NUMERALS = {word: digits for digits, words in {
    "1": "один одна одне одну одного одній", "2": "два дві двох двома", "3": "три трьох трьома тричі",
    "4": "чотири чотирьох чотирьома", "5": "п'ять п'яти п'ятьма", "6": "шість шести шістьма", "7": "сім семи",
    "8": "вісім восьми", "9": "дев'ять дев'яти", "10": "десять десяти десятого", "15": "п'ятнадцять п'ятнадцятого",
    "20": "двадцять двадцяти двадцятого", "30": "тридцять тридцяти", "50": "п'ятдесят п'ятдесяти",
}.items() for word in words.split()}
ENDINGS = sorted("ами ями ові еві єві ого ому ими іми ою ею єю ом ем ам ям ах ях ів їв ий ій ої ей им ім "
                 "а я у ю е є о и і ї ь".split(), key=len, reverse=True)


def stem(word: str) -> str:
    """A rough stem that meets the inflections: "щомісяця"/"місяць", "узимку"/"взимку", "мохом"/"мох"."""
    if word.startswith("що") and len(word) > 6:
        word = word[2:]
    if len(word) > 3 and word[0] == "у" and word[1] not in "аеєиіїоуюя":
        word = "в" + word[1:]
    ending = next((ending for ending in ENDINGS if word.endswith(ending) and len(word) - len(ending) >= 2), "")
    return word[:len(word) - len(ending)][:4]


# A day of the month written as a word: "до першого числа" is "до 1 числа". Only before "числа", as
# "після першого морозу" has no number in it.
DAY_ORDINALS = {"першого": "1", "другого": "2", "третього": "3", "четвертого": "4", "п'ятого": "5", "десятого": "10",
                "п'ятнадцятого": "15", "двадцятого": "20"}


def fact_terms(text: str) -> set[str]:
    """Numbers and the stems of meaningful words: "Кабанові в кошик біля входу" -> {"каба", "кош", "вход"}."""
    text = re.sub(r"\b([а-яіїєґ']+)(?=\s+числ)", lambda match: DAY_ORDINALS.get(match.group(1), match.group(1)),
                  text.lower().replace("’", "'").replace("ʼ", "'"))
    words = re.findall(r"[а-щьюяіїєґa-z0-9']+", text)
    return {word if word.isdigit() else NUMERALS.get(word) or stem(word) for word in words
            if word.isdigit() or word in NUMERALS
            or (len(word) >= 3 and word not in STOPWORDS and not word.startswith(COMMON_STEMS))}


FACT = re.compile(r"\s*[-*•«\"(]*\s*\[\s*(?:№|документ\s*)?(\d+)[^\]]*\]\**\s*(.+)", re.S | re.I)


def missing_facts(facts, docs: list[Document], sources: list[Document], text: str, question: str = "",
                  points: list[str] | tuple = ()) -> list[str]:
    """The facts ("[1, рукавичка] ...") the model listed from the documents it cites, and the `points` the documents
    give themselves as answers to the question, that the text leaves out, each once: checked sentence by sentence
    (and line by line for lists), a sentence is left out when the text lacks one of its numbers or half of its word
    stems. Words of the question need not be repeated ("у хвойному лісі" in "Коли утеплювати у соснах?" -> "У
    жовтні"). Facts from documents the answer does not cite are not required."""
    cited = {docs.index(doc) + 1 for doc in sources}
    have, asked = fact_terms(text), fact_terms(question)
    sentences = []
    for fact in facts if isinstance(facts, list) else []:
        match = FACT.match(str(fact))
        if match and int(match.group(1)) in cited:
            sentences += re.split(r"(?<=[.!?;])\s+|\n+", match.group(2).strip())
    left_out = []
    for sentence in sentences + list(points):
        terms = fact_terms(sentence)
        numbers = {term for term in terms if term.isdigit()}
        terms = (terms - asked) or terms
        if (numbers - have or (terms and len(terms & have) * 2 < len(terms))) and sentence.strip() not in left_out:
            left_out.append(sentence.strip())
    return left_out


def answered_sentences(doc: Document, skip_intro: bool = False) -> list[tuple[str, str, str]]:
    """The document's sentences other than questions, each with its heading and the question just before it, which it
    answers: ("", "Де шукати білі гриби?", "Білі гриби тримаються старих дубів."). The title line is no heading, nor a
    question the opening answers ("Де шукати гриби?" as an animal's file name), and a PDF line wrapped mid-sentence is
    joined to the one before. With `skip_intro`, the opening paragraph saying whom the document is for goes: "Хто
    знає різницю, той повертається додому…" is no advice on "коли повертатися"."""
    lines = []
    for line in (line.strip() for line in doc.text.splitlines()):
        if lines and line and not lines[-1].endswith((".", "!", "?", "…", ":", ";")) and line[0].islower():
            lines[-1] += " " + line  # a PDF line wrapped mid-sentence
        elif line:
            lines.append(line)
    title = lambda line: norm_title(line) == doc.norm_title
    body = [line for line in lines if not looks_like_heading(line) and not title(line)]
    if skip_intro and len(body) > 1 and FOR_EVERYONE.search(body[0]):
        lines.remove(body[0])
    sentences, heading, asked = [], "", ""
    for line in lines:
        if title(line) or looks_like_heading(line):
            heading, asked = ("" if title(line) else line), ""
            continue
        for sentence in re.split(r"(?<=[.!?…])\s+", line):
            if sentence.endswith("?"):
                asked = sentence
            else:
                sentences.append((heading, asked, sentence))
                asked = ""
    return sentences


def question_word(text: str) -> str:
    """The word a question or a heading asks with, or "": the first of QUESTION_WORDS in its last asking sentence,
    before any comma ("Біля греблі мисливці! Як кричати?" -> "як", "Коли шукати гриби" -> "коли"), so the "коли" of
    "Що робити, коли біля греблі мисливці?" ("whenever") asks nothing."""
    sentences = [sentence for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()) if sentence]
    asking = [sentence for sentence in sentences if sentence.endswith("?")] or sentences or [""]
    match = QUESTION_WORDS.search(asking[-1].split(",")[0])
    return match.group(0).lower() if match else ""


# Document numbers as the model writes them: "[1]", "[2, рукавичка]" and runs of them like "[1], [2]" or "[1] і [2]".
REFERENCES = r"\[\s*\d+(?:,[^\]]*)?\](?:\s*(?:,|;|і|й|та)\s*\[\s*\d+(?:,[^\]]*)?\])*"


def clean_text(value, documents: int) -> str:
    """The model's text without the document numbers it now and then puts in ("у документі [2]", "[1, рукавичка]")
    or a bare "(документ)"; other numbers in brackets ("[2026]") stay."""
    # An object or a number where a text belongs is no text: "{'text': '4 жолуді.'}" must never reach an animal.
    if isinstance(value, list):
        text = " ".join(part for part in value if isinstance(part, str))
    else:
        text = value if isinstance(value, str) else ""
    # A run goes only when all its numbers are documents' numbers, so "[1] і [2026]" stays whole.
    numbered = lambda match: "" if all(1 <= int(n) <= documents for n in re.findall(r"\[\s*(\d+)", match.group(0))) \
        else match.group(0)
    # "(документ)", "(у документі [2])", "(у документах [1] і [2])", "(з документа «…»)", but not
    # "(у документі сказано: до 10 числа)". Any one bracket goes too ("(у документі [№2])", "(документ [Внески])"):
    # the run is tried first, so "[1] і [2]" goes whole.
    text = re.sub(r"\s*\((?:згідно\s+з\s+|див\.\s*|з\s+|із\s+|у\s+|в\s+)?документ\w*"
                  rf"(?:\s*{REFERENCES}|\s*\[[^\]]*\]|\s+\d+|\s*«[^»]{{0,80}}»|\s*\"[^\"]{{0,80}}\")?\s*\)", "", text)
    text = re.sub(rf"(?:\s+(?:у\s+|в\s+)?документ\w*)?\s*{REFERENCES}", numbered, text)
    text = re.sub(r"\s*\(\s*\)", "", text)  # "([1], [2])" once its numbers are gone
    return re.sub(r"\s{2,}", " ", text).strip()


PLACEHOLDER = re.compile(r"(?:немає|нема|-+|—|–|null|none|n/a|порожній рядок|\"\")\.?"
                         r"|(?:немає|нема)\s+ін\w*.*|інших\s+порад\s+немає.*", re.I)


# The model talking about its own JSON ("тому її не включаємо в \"other\"") is not advice for an animal.
SERVICE_WORDS = re.compile(r"\b(?:other|answer|sources|facts|json)\b", re.I)
# "…, але це не суперечить конкретній пораді…" (live for "Коли Ведмідь лягає спати?") is the model's word on its own
# reply, not advice: the clause goes (a sentence that starts with it goes whole), the advice before it stays. Dropping
# the paragraph would lose advice its document is still cited for, the whole forest's that rule 11 makes a must too.
NO_CONFLICT_CLAUSE = re.compile(r"\s*\([^()]*\bне\s+супереч[^()]*\)|\s*[,;:—–][^,;:—–(.!?…]*\bне\s+супереч[^.!?…]*",
                                re.I)
NO_CONFLICT_SENTENCE = re.compile(r"[^.!?…]*\bне\s+супереч[^.!?…]*[.!?…]*\s*", re.I)


def clean_other(value, answer: str, documents: int) -> str:
    """The advice that gives way, or "" for the stubs the model writes instead of nothing ("Немає", "-", "null"),
    for a repeat of the answer and for the model's words about its own reply ("це не суперечить …" is cut out)."""
    other = clean_text(value, documents)
    other = NO_CONFLICT_SENTENCE.sub("", NO_CONFLICT_CLAUSE.sub("", other)).strip()
    if not other or PLACEHOLDER.fullmatch(other) or SERVICE_WORDS.search(other) \
            or fact_terms(other) <= fact_terms(answer):
        return ""
    return other


# Places "ваш" must not stand before: the animal may live elsewhere, and who asks does not matter (Mishka). Live on
# 2 November, despite rule 10: "У вашій рукавичці «У соснах»", "біля вашої верби", "У вашій частині лісу — хвойному —".
PLACE_WORDS = (r"рукавич\w*|частин\w*\s+лісу\b|ліс(?:у|і|ом|а|и|ів|ах)?\b|верб\w*|дуб\w*|сосн\w*|ялин\w*|ліщин\w*"
               r"|мурашник\w*|гребл\w*|узлісс\w*|нор(?:а|и|і|у|ою|ах)?\b|дім\b|дом(?:у|і|ом|а|івк\w*)\b")
YOUR_PLACE = re.compile(rf"\b([Вв])аш(?:а|ої|ій|у|ою|е|ого|ому|і|их|им|ими)?\s+(?P<place>{PLACE_WORDS})")
# "У вашій частині лісу — хвойному —" and "у вашій частині лісу", the part named when the asked place has one.
YOUR_PART = re.compile(r"\b([УуВв])\s+[Вв]ашій\s+частині\s+лісу(?:\s*[—–-]\s*(\w+ому)\s*[—–-])?(\s*)")
# Rule 6 writes "Ваш" with a capital anywhere, so only a sentence's first word keeps its capital.
SENTENCE_START = re.compile(r"(?:^|[.!?…\n])[\s\-–—•*]*$")
PART_LOCATIVE = {CONIFER: "у хвойному лісі", DECIDUOUS: "у листяному лісі", RIVER: "біля річки"}
# "Загальна порада" is the whole forest's label only: "тож діє загальна порада для хвойного лісу" (live for
# "У соснах") is the conifer forest's own advice.
PART_GENERAL = re.compile(r"\b([Зз])агальн(?:а|ої|у|ою|ій|і|их)\s+(порад\w*)(?=\s+(?:для\s+)?(?:хвойн|листян"
                          r"|частин\w*\s+лісу\s+біля\s+річки|біля\s+річки))")


def clean_wording(text: str, mitten: str | None = None, part: str | None = None) -> str:
    """The model's text without "ваш" before a place ("біля вашої верби" -> "біля верби", "у вашій рукавичці" ->
    "у рукавичці «Під вербою»" with the one mitten asked, "У вашій частині лісу — хвойному —" -> "У хвойному лісі"),
    and without "загальна" before a forest part's advice. Other "ваш" stay: "перевірте ваші запаси"."""
    capital = lambda word, upper: word[0].upper() + word[1:] if upper else word

    def part_place(match: re.Match) -> str:
        place = f"у {match.group(2)} лісі" if match.group(2) else PART_LOCATIVE.get(part or "", "у частині лісу")
        return capital(place, match.group(1).isupper()) + match.group(3)

    def place(match: re.Match) -> str:
        word = match.group("place")
        # The asked mitten's name, unless it is already there: "у рукавичці «У соснах»".
        if mitten and word.startswith("рукавич") and not re.match(rf"\s*(?:[«\"„]|{re.escape(mitten)})",
                                                                  match.string[match.end():], re.I):
            word += f" «{mitten}»"
        return capital(word, match.group(1) == "В" and bool(SENTENCE_START.search(match.string[:match.start()])))

    text = YOUR_PLACE.sub(place, YOUR_PART.sub(part_place, text))
    return PART_GENERAL.sub(lambda match: capital(match.group(2), match.group(1) == "З"), text)


def by_number(numbers, items: list) -> list:
    """The items the model named by number from 1 ([2, "1", "[3]"]), each once; other values are skipped.
    isdecimal(), as int() takes no "²" that isdigit() lets through; type() is int, as JSON true is a bool,
    and a bool is an int in Python."""
    chosen = []
    for number in numbers if isinstance(numbers, list) else []:
        if isinstance(number, str) and number.strip(" []").isdecimal():
            number = int(number.strip(" []"))
        if type(number) is int and 1 <= number <= len(items) and items[number - 1] not in chosen:
            chosen.append(items[number - 1])
    return chosen


def best_per_level(hits: list[Hit]) -> tuple[list[Document], list[Document]]:
    """The best keyword matches of each place separately: the mitten, each of its forest parts, the whole forest.
    Each place's best match comes apart from the others, so it can be given a seat in the prompt first."""
    groups: dict[str, list[Hit]] = {}
    for hit in hits:
        groups.setdefault(hit.doc.scope.encode(), []).append(hit)
    firsts: list[Hit] = []
    others: list[Hit] = []
    for group in groups.values():
        kind, best = group[0].doc.scope.kind, group[0].score
        matching = [hit for hit in group if hit.score >= LEVEL_CUTOFF[kind] * best]
        kept = matching[:LEVEL_LIMIT[kind]]
        # Between animals' documents for one place the newer advice wins: with an animal's document kept, the newest
        # animal's match comes too, however it scores, in a seat of its own, so no archive document gives way to it.
        # Archive files stay as they are: one edited later is another document, not a newer version of the others.
        animals = [hit for hit in matching if not hit.doc.from_archive]
        newest = max(animals, key=lambda hit: hit.doc.created_at, default=None)
        if newest and newest not in kept and any(not hit.doc.from_archive for hit in kept):
            kept.append(newest)
        firsts.append(kept[0])
        others += kept[1:]
    # The mitten's documents first, then the forest parts', then the whole forest's.
    by_level = lambda chosen: [hit.doc for hit in sorted(chosen, key=lambda hit: (-hit.doc.scope.level, -hit.score))]
    return by_level(firsts), by_level(others)


def merge(first: list[Document], second: list[Document], limit: int) -> list[Document]:
    result = []
    for doc in first + second:
        if doc not in result:
            result.append(doc)
    return result[:limit]


class Answerer:
    def __init__(self, store: Store, forest: Forest, llm: LLM, tz: ZoneInfo):
        self.store, self.forest, self.llm, self.tz = store, forest, llm, tz
        self._index: Index | None = None
        self._revision = -1
        self._residents: dict[str, str] = {}

    @property
    def index(self) -> Index:
        if self._index is None or self._revision != self.store.revision:
            self._index = Index(self.store.active_documents())
            self._revision = self.store.revision
            self._residents = {
                name: doc.scope.value
                for doc in self._index.documents
                if doc.from_archive and doc.scope.kind == "mitten" and doc.norm_title.startswith("статут")
                for name in residents_from_statute(doc.text)
            }
        return self._index

    @property
    def residents(self) -> dict[str, str]:
        """Who lives where, from the mittens' statutes: {"Ведмідь": "У ліщині", ...}."""
        _ = self.index
        return self._residents

    async def ask(self, question: str, now: datetime, mitten: str | None = None) -> Reply:
        deadline = time.monotonic() + QUESTION_SECONDS
        mittens = [mitten] if mitten else self.forest.detect_mittens(question)
        parts = [] if mittens else self.forest.detect_parts(question)
        resident = None
        if not mittens and not parts:
            # "Що робить Сова вночі?": the animal's own mitten from the statutes is the place.
            homes = resident_homes(question, self.residents)
            if len(set(homes.values())) == 1:
                resident, mittens = ", ".join(homes), [next(iter(homes.values()))]

        if mittens:
            allowed = lambda doc: any(self.forest.applies(doc.scope, mitten=m) for m in mittens)
            docs = await self._docs_for_place(question, allowed, deadline)
        elif parts:
            # Mishka: a forest part asked gets the part's own advice and the whole forest's. A mitten's
            # advice does not fit all its neighbours, so it comes only when the mitten itself is named.
            allowed = lambda doc: any(self.forest.applies(doc.scope, part=p) for p in parts)
            docs = await self._docs_for_place(question, allowed, deadline)
        else:
            docs = await self._docs_without_place(question, deadline)
        if docs:
            reply = await self._answer(question, docs, mittens, parts, now, resident, deadline)
        else:
            reply = Reply("not_found", texts.NOT_FOUND)
        if parts and reply.kind == "not_found" and self._mittens_know(question, parts):
            # "Скільки внесок біля річки?": the part has no such advice, its mittens have their own dues.
            return Reply("need_place", texts.ASK_MITTEN_IN_PART.format(part=self._place_note([], parts)),
                         self.forest.mittens_in(parts))
        return reply

    async def _docs_for_place(self, question: str, allowed, deadline: float) -> list[Document]:
        """Keyword matches level by level, plus what the model picks by title: "що їсти" finds
        "Чим поживитись поруч із рукавичкою" that shares no word with the question."""
        # All matches: a mitten's own documents count however low they score, so none is cut before grouping.
        hits = self.index.search(self.forest.strip_places(question), allowed, limit=None, cutoff=0)
        candidates = [doc for doc in self.index.documents if allowed(doc)]
        picked = await self._pick_by_title(question, candidates, deadline)
        # Each place's best match before the picks: four deciduous and whole-forest picks once left "Під вербою"
        # without its river part's guide. The mitten's own other matches too, so they survive when the prompt is full.
        firsts, others = best_per_level(hits)
        own = [doc for doc in others if doc.scope.kind == "mitten"]
        return merge(firsts + own + picked, others, limit=MAX_DOCS)

    def _varies_by_place(self, question: str) -> bool:
        """Whether particular places have advice of their own on the topic, as strong as the whole forest's. Then the
        answer says whose advice is whose, and that the asked place has none of its own ("Окремих порад про утеплення
        для узлісся немає"); a topic the whole forest shares is answered plainly, as Mishka answered about hunters."""
        query = self.forest.strip_places(question)
        topic = expand(set(tokens(query)))
        best = lambda allowed: max((hit.score for hit in self.index.search(query, allowed, limit=1, cutoff=0)), default=0.0)
        # A place's own advice on the topic says so in its title ("Як утеплити рукавичку в хвойному лісі"); hunters
        # only mentioned among a mitten's "Небезпеки поруч із рукавичкою" are no advice of its own on what to do.
        places = best(lambda doc: doc.scope.kind != "forest" and bool(topic & set(tokens(doc.title))))
        return places > 0 and places >= PLACE_VARIES * best(lambda doc: doc.scope.kind == "forest")

    def _mittens_know(self, question: str, parts: list[str]) -> bool:
        """Whether the advice of the asked part's mittens matches the question, so asking which mitten helps."""
        mittens = set(self.forest.mittens_in(parts))
        theirs = lambda doc: doc.scope.kind == "mitten" and doc.scope.value in mittens
        return bool(self.index.search(self.forest.strip_places(question), theirs, limit=1))

    async def _docs_without_place(self, question: str, deadline: float) -> list[Document]:
        """No place named: the best whole-forest advice plus the best advice for particular places.
        The model then decides whether the answer differs by mitten ("утеплювати": October in the
        conifer forest, November by the calendar) or is the same for everyone ("сорока кричить тричі")."""
        hits = self.index.search(question, limit=12)
        forest_docs = [hit.doc for hit in hits if hit.doc.scope.kind == "forest"][:3]
        place_docs = [hit.doc for hit in hits if hit.doc.scope.kind != "forest"][:4]
        picked = await self._pick_by_title(question, self.index.documents, deadline)
        return merge(picked, forest_docs + place_docs, limit=MAX_DOCS + 1)

    async def _pick_by_title(self, question: str, candidates: list[Document], deadline: float) -> list[Document]:
        """A light model chooses documents by their titles; keyword search alone misses synonyms."""
        archive = [doc for doc in candidates if doc.from_archive]
        animals = sorted((doc for doc in candidates if not doc.from_archive), key=lambda d: d.created_at, reverse=True)
        head = f"Питання: {question}\n\nДокументи:\n"
        # Every archive title, then the newest animals' titles while the prompt still fits Groq's minute.
        room = prompt_chars(PICK_TOKENS) - len(PICK_SYSTEM) - len(head)
        candidates, lines = [], []
        for doc in archive + animals[:PICK_ANIMAL_TITLES]:
            line = f"{len(lines) + 1}. {self._title_line(doc)}"
            room -= len(line) + 1
            if room < 0 and not doc.from_archive:
                break
            candidates.append(doc)
            lines.append(line)
        if not candidates:
            return []
        try:
            result = await self.llm.json(PICK_SYSTEM, head + "\n".join(lines), max_tokens=PICK_TOKENS, fast=True,
                                         deadline=min(deadline, time.monotonic() + PICK_SECONDS))
        except LLMUnavailable:
            return []
        return by_number(result.get("ids"), candidates)[:PICK_LIMIT]

    def _title_line(self, doc: Document) -> str:
        if doc.from_archive:
            return f"{doc.title} ({doc.scope.describe()})"
        return f"{doc.title} ({doc.scope.describe()}, надіслав звір, {doc.created_at.astimezone(self.tz):%d.%m %H:%M})"

    async def _answer(self, question: str, docs: list[Document], mittens: list[str], parts: list[str], now: datetime,
                      resident: str | None, deadline: float) -> Reply:
        # The documents were chosen for one place: named in the question, picked by button or a named resident's home.
        named = bool(mittens or parts)
        place_rule = PLACE_UNKNOWN_RULE if not named and any(doc.scope.kind != "forest" for doc in docs) else ""
        # Most specific first, archive before animals, newer before older.
        docs = sorted(docs, key=lambda d: (-d.scope.level, not d.from_archive, -d.modified_at.timestamp()))
        place = self._place_note(mittens, parts) + (RESIDENT_NOTE.format(name=resident) if resident else "")
        # Thinking longer and the calendar cost Groq tokens (the free tier allows about 8000 a minute),
        # so only questions that count dates from today get longer thinking and the days of the months.
        need = calendar_need(question)
        reasoning, budget = ("medium", 2000) if need == "days" else ("low", 1500)
        # "Де шукати гриби біля верби?" is about mushrooms for the mitten, not about willow trees.
        topic = re.sub(r"\s+", " ", self.forest.strip_places(question)).strip(" ?.!,") if named else ""
        # A place found by a resident ("коли Ведмідь лягає спати?") was not asked about: no labels for it.
        varies = ("так" if self._varies_by_place(question) else "ні") if named and not resident else ""
        system = ANSWER_SYSTEM + place_rule
        head = (
            f"{calendar_note(now, self.tz, need)}\n\n"
            f"Питання звіра: {question}\n"
            f"Місце з питання: {place}\n"
            + (f"Тема питання без місця: {topic}\n" if topic else "")
            + (f"Поради на цю тему різні в різних місцях: {varies}\n" if varies else "")
            + "\nДокументи:\n\n"
        )
        headers = [self._doc_header(i, doc) for i, doc in enumerate(docs, 1)]
        # The documents' text gets what Groq's minute leaves after the rules, the calendar, the question and the reply.
        separator = "\n\n"
        room = (prompt_chars(budget) - len(system) - len(head) - sum(map(len, headers))
                - len(separator) * (len(docs) - 1))
        size = self._excerpt_size(docs, room)
        prompt = head + separator.join(header + self._excerpt(doc, question, mittens, parts, size)
                                       for header, doc in zip(headers, docs, strict=True))
        asked_sources = False
        kept: Draft | None = None  # the answer so far, kept while the model is asked once to put it right
        # At most three calls: the answer, once more when it forgot its sources, and once more to put it right.
        for _ in range(3):
            try:
                result = await self.llm.json(system, prompt, max_tokens=budget, reasoning=reasoning, deadline=deadline)
            except LLMUnavailable:
                if kept:  # no time or no model left for the second try: the first answer still holds
                    break
                return Reply("unavailable", texts.LLM_UNAVAILABLE)
            answer = clean_text(result.get("answer"), len(docs))
            # Advice that says otherwise and gives way here (the whole forest's, or an animal's under the archive).
            other = clean_other(result.get("other"), answer, len(docs))
            sources = by_number(result.get("sources"), docs)
            if not sources and len(docs) == 1:
                sources = docs  # one document given, so it is the source
            # A flag written as a string: "false" is no, as for "found".
            needs_place = bool(place_rule) and result.get("needs_place") in (True, "true")
            gave_up = needs_place or result.get("found") in (False, "false") or not answer
            if kept and (gave_up or not sources):
                break  # asked to put it right, the model lost the answer: keep the first
            if needs_place:
                return Reply("need_place")
            if gave_up:
                return Reply("not_found", texts.NOT_FOUND)
            if not sources:
                if asked_sources:
                    return Reply("not_found", texts.NOT_FOUND)
                asked_sources = True  # the model now and then forgets the source numbers; ask once more
                prompt += SOURCES_REMINDER  # the very same request would mostly bring the very same reply
                continue
            facts = result.get("facts") if isinstance(result.get("facts"), list) else []
            reminders, faults, general, points = self._review(question, docs, mittens, named, answer, other, sources,
                                                              facts, varies)
            draft = Draft(answer, other, sources, faults, general, facts, points)
            if kept:
                # The one second try is taken unless it came out worse: more broken rules, the advice that gives way
                # still without its document, or more of the points both answers listed, or its documents give as the
                # answers to the question, left out (a weaker fallback model may answer it worse than the first). A new
                # "other" without its document is dropped anyway. The whole forest's advice weighs last: adding it is
                # no reason to lose a place's own point ("після першого морозу" for the acorns under the oak).
                every = kept.facts + facts
                unsourced = bool(kept.other) and len(kept.sources) == 1 and len(docs) > 1
                before = (kept.faults, unsourced,
                          len(missing_facts(every, docs, kept.sources, f"{kept.answer} {kept.other}", question,
                                            kept.points)), kept.general)
                after = (faults, unsourced and bool(other) and len(sources) == 1,
                         len(missing_facts(every, docs, sources, f"{answer} {other}", question, points)), general)
                if after <= before:
                    kept = draft
                break
            kept = draft
            if not reminders:
                break
            # Everything to put right goes in one second try: each try costs a Groq request, and the free tier
            # allows only so many tokens a minute.
            prompt += "".join(reminders)

        answer, other, sources = kept.answer, kept.other, kept.sources
        # No "у вашій рукавичці" (who asks does not matter) and no "загальна" for a forest part's advice, though the
        # rules say so: the model still wrote them in two answers of six on 2 November, and a try more costs tokens.
        mitten = mittens[0] if len(mittens) == 1 else None
        places = self.forest.mitten_parts.get(mitten, frozenset()) if mitten else frozenset(parts)
        part = next(iter(places)) if len(places) == 1 else None
        answer, other = clean_wording(answer, mitten, part), clean_wording(other, mitten, part)
        # A document deleted, replaced or revoked while the model was answering is no source any more, nor is what it
        # says. The answer comes from the first source; the advice that gives way retells a later one.
        gone = [doc for doc in sources if (current := self.store.get(doc.id)) is None or current.status != "active"]
        if gone:
            if sources[0] in gone:
                return Reply("not_found", texts.NOT_FOUND)
            sources, other = [doc for doc in sources if doc not in gone], ""
        if len(sources) == 1:
            cite = "Джерело: " + source_line(sources[0], self.tz)
            # The document the advice that gives way retells is not among the sources: the animal would not see where
            # that advice comes from, nor who sent it and when (Mishka, 20:46).
            other = ""
        else:
            cite = "Джерела:\n" + "\n".join("- " + source_line(doc, self.tz) for doc in sources)
        # Mishka: «Окремих порад про утеплення для узлісся немає. Загальна порада для всього лісу — …». Where the
        # topic differs by place and only the whole forest answers, the bot says so if the model did not.
        own = self._place_genitive(mittens, parts)
        if varies == "так" and own and all(doc.scope.kind == "forest" for doc in sources) \
                and not re.search(r"\bнема|загальн", answer.lower()):
            answer = texts.NO_OWN_ADVICE.format(place=own) + " " + answer
        # The advice that gives way stands in a paragraph of its own, so it never reads as part of the answer.
        text = f"{answer}\n\n{other[0].upper()}{other[1:]}" if other else answer
        return Reply("answer", f"{text}\n\n{cite}")

    def _topic_terms(self, question: str) -> set[str]:
        """What the question is about, for telling whether an archive document is on its topic: the word stems
        without the place, short words, numbers and the names of residents ("Ведмідь" is in his mitten's statute)."""
        names = {stem for name in self.residents for stem in fact_terms(name)}
        return {term for term in fact_terms(self.forest.strip_places(question))
                if len(term) >= 4 and not term.isdigit()} - names

    def _review(self, question: str, docs: list[Document], mittens: list[str], named: bool, answer: str, other: str,
                sources: list[Document], facts: list, varies: str = "") -> tuple[list[str], int, bool, list[str]]:
        """What to put right in an answer: the reminders for the model, how many of the rules (levels, the archive,
        both forest parts, Ukrainian) the answer breaks, whether it leaves out the whole forest's advice on the
        topic of a place asked, and the points its documents give as the answers to the question. `varies` is "так" or "ні" when the question itself names the place, else ""."""
        reminders = []
        asked = re.sub(r"\s+", " ", self.forest.strip_places(question)).strip(" ?.!,")
        # A section about the asked mitten inside a whole-forest guide tempts the model to put the
        # guide first, though a cited document for the mitten's forest part is more specific.
        # With no place asked, PLACE_UNKNOWN_RULE alone decides between the whole forest and asking.
        better = next((doc for doc in sources[1:] if doc.scope.level > sources[0].scope.level), None)
        if better and named:
            reminders.append(LEVEL_REMINDER.format(
                better=docs.index(better) + 1, better_level=LEVEL_NAMES[better.scope.kind],
                weaker=docs.index(sources[0]) + 1, weaker_level=LEVEL_NAMES[sources[0].scope.kind]))
        # The decision told to Mishka (20:46): for the same place the archive beats an animal's document,
        # though the animal's is newer. A weaker fallback model now and then answers by the newer one, and
        # does not even cite the archive document it was given on the same topic.
        topic = self._topic_terms(question)
        archived = [] if not named or sources[0].from_archive else [
            doc for doc in docs if doc.from_archive and doc.scope == sources[0].scope and topic & fact_terms(doc.text)]
        if archived:
            reminders.append(ARCHIVE_REMINDER.format(archive=", ".join(f"[{docs.index(doc) + 1}]" for doc in archived),
                                                     animal=docs.index(sources[0]) + 1))
        # Mishka: "для «Під вербою» підходять поради і листяного лісу, і річки". When the answer took one
        # part's advice on the topic and left the other part's document out, or took neither part's guide on the
        # topic, the model is reminded of it.
        missing = self._missing_parts(mittens, docs, sources, answer, question)
        if missing:
            cited = ", ".join(PART_NAMES[p] for p in sorted(
                {doc.scope.value for doc in sources if doc.scope.kind == "part"} - {doc.scope.value for doc in missing}))
            names = [f'"{PART_NAMES[doc.scope.value]}"' for doc in missing]
            if len(missing) == 1:
                absent, label = f"частини {names[0]}", f"що це порада для частини лісу {names[0]}"
            else:  # each part with its own document, or the model takes them for one part (rule 7)
                absent = "частин " + ", ".join(f"{name} (документ [{docs.index(doc) + 1}])"
                                               for name, doc in zip(names, missing, strict=True))
                label = "до якої частини лісу належить кожна порада"
            reminders.append(PARTS_REMINDER.format(
                mitten=mittens[0], cited=f'є порада для частини лісу "{cited}", але ' if cited else "",
                missing=absent, label=label, docs=", ".join(f"[{docs.index(doc) + 1}]" for doc in missing),
                topic=asked, quotes="".join(QUOTE_NOTE.format(number=docs.index(doc) + 1, quote=quote)
                                            for doc in missing if (quote := self._topic_sentences(doc, question)))))
        # Decided on 2 November: with a place asked, the whole forest's advice on the topic is a must, as in Mishka's
        # pines answer («Загальна порада для всього лісу — шукати після дощу, коли земля волога, але не хлюпає»), which
        # a live answer left out. Only the sentences sharing the question's own words count (strict): a second try
        # costs Groq tokens, so "що їсти" is not reminded of "Ліс годує щедро".
        general = [] if not varies or any(doc.scope.kind == "forest" for doc in sources) else [
            (doc, quote) for doc in docs
            if doc.scope.kind == "forest" and (quote := self._topic_sentences(doc, question, strict=True))]
        # Rule 6: Ukrainian only. Russian letters are a slip a second try mends.
        russian = sorted(set(re.findall(r"[ыэъёЫЭЪЁ]", f"{answer} {other}")))
        if russian:
            reminders.append(UKRAINIAN_REMINDER.format(letters=", ".join(russian)))
        faults = len(reminders)
        if general:
            reminders.append(FOREST_REMINDER.format(
                docs=", ".join(f"[{docs.index(doc) + 1}]" for doc, _ in general), topic=asked,
                label=' словами "Загальна порада для всього лісу — …"' if varies == "так" else "",
                quotes="".join(QUOTE_NOTE.format(number=docs.index(doc) + 1, quote=quote) for doc, quote in general)))
        # Mishka (20:46): the advice that gives way is mentioned with who sent it and when, so its document must be
        # among the sources; with the answer's alone there, it is not. With one document given there is no other to
        # add, and the paragraph goes anyway.
        if other and len(sources) == 1 and len(docs) > 1:
            reminders.append(OTHER_REMINDER)
        # The model lists the points of the cited documents in "facts", then now and then drops some of them
        # when it writes the answer ("Кабанові в кошик біля входу" in the customer's own example). What a cited
        # document, or a forest part's document the answer left out, gives itself as the answer to the very question
        # is a must too: live, the birch boletes went missing from about half the willow answers, never in "facts".
        points = [point for doc in sources + [doc for doc in missing if doc not in sources]
                  for point in self._asked_points(doc, question)]
        left_out = missing_facts(facts, docs, sources, f"{answer} {other}", question, points)
        if left_out:
            reminders.append(FACTS_REMINDER.format(facts="; ".join(f'"{fact}"' for fact in left_out)))
        return reminders, faults, bool(general), points

    def _topic_sentences(self, doc: Document, question: str, strict: bool = False) -> str:
        """The document's sentences on the topic of the question, the ones sharing most words first, up to QUOTE_CHARS
        and in the document's order: "Білі гриби тримаються старих дубів." for "де шукати гриби біля верби?".
        Strict: only sentences sharing two of the question's own words (its only one, if it has one), synonyms aside,
        so "Ліс годує щедро" is not on "що їсти", nor "Кошик можна знайти потім" on "де знайти їжу". There "шукати"
        and "йти" do not count, so "Гриби шукайте після дощу" is not on "де шукати гриби" (Mishka's willow answer has
        no general advice), while "коли" in the question counts for a sentence with "коли" ("коли йти по гриби"), and
        the opening paragraph saying whom the document is for is skipped."""
        query = self.forest.strip_places(question)
        topic = expand(set(tokens(query)))
        words = {frozenset(tokens(word)) for word in query.split()} - {frozenset()}
        least = min(2, len(words)) if strict else 0
        own = [word for word in words if not word & SEARCH_STEMS]
        when = strict and bool(WHEN.search(query))
        # A sentence counts with its heading and with the question it answers: "Чим? Сухим дубовим листям.",
        # "Де шукати білі гриби?" over "Білі гриби тримаються старих дубів.".
        scored = []
        for heading, asked, sentence in answered_sentences(doc, skip_intro=strict):
            quote = f"{asked} {sentence}".strip()
            said = set(tokens(f"{heading} {quote}"))
            shared = sum(1 for word in own if word & said) + bool(when and WHEN.search(f"{heading} {quote}"))
            if shared >= least:
                scored.append((len(topic & said), len(scored), quote))
        chosen, used = [], 0
        for score, i, quote in sorted(scored, key=lambda item: (-item[0], item[1])):
            if score and used + len(quote) <= QUOTE_CHARS:
                chosen.append((i, quote))
                used += len(quote) + 1
        return " ".join(quote for _, quote in sorted(chosen))

    def _asked_points(self, doc: Document, question: str) -> list[str]:
        """What the document itself gives as the answer to the very question asked, where it asks that question: for
        "Де шукати гриби біля верби?" the deciduous guide's "Де шукати білі гриби? Білі гриби тримаються старих дубів."
        and "А де підберезники? Підберезники ростуть на галявинах біля беріз.", the sentence after each. Its question
        asks with the same question word (question_word) and has a word of the topic ("шукати" and "йти" do not count,
        synonyms do, but one kind of mushroom stands only for itself: KINDS), so "Коли йти по гриби?" is no answer to
        "де", and the opening paragraph, which no question asks, is none either. A question naming a place other than
        the document's own, or under a heading that does, is that place's."""
        query = self.forest.strip_places(question)
        word = question_word(query)
        topic = set()
        for part in QUESTION_WORDS.sub(" ", query).split():
            terms = set(tokens(part))
            if not terms & SEARCH_STEMS:
                topic |= terms if terms & KINDS else expand(terms)
        own = {doc.scope.value} - {None}
        points = []
        for heading, asked, sentence in answered_sentences(doc):
            if not (word and question_word(asked) == word and topic & set(tokens(asked))):
                continue
            named = f"{heading} {asked}"
            if set(self.forest.detect_mittens(named) + self.forest.detect_parts(named)) <= own:
                points.append(sentence)
        return points

    def _missing_parts(self, mittens: list[str], docs: list[Document], sources: list[Document], answer: str,
                       question: str) -> list[Document]:
        """For a mitten in several forest parts: the best document of each of its parts that was given to the model
        but left out of the answer. A part counts only when its advice is in the answer itself, not just in the
        paragraph of advice that gives way. With no part's advice in the answer, only a part's document whose title
        is on the topic is left out: "Грибний путівник листяного лісу" for mushrooms, none for the mitten's dues."""
        if len(mittens) != 1:
            return []
        parts = self.forest.mitten_parts.get(mittens[0], frozenset())
        if len(parts) < 2:
            return []  # a mitten in one forest part needs no advice of its part beside its own
        said = fact_terms(answer) - fact_terms(question)
        answered = {doc.scope.value for doc in sources
                    if doc.scope.kind == "part" and len(fact_terms(doc.text) & said) >= 2}
        topic = expand(set(tokens(self.forest.strip_places(question))))
        fits = lambda doc: bool(answered) or bool(topic & set(tokens(doc.title)))
        missing = []
        for part in sorted(parts - answered):
            doc = next((doc for doc in sources + docs if doc.scope == Scope("part", part) and fits(doc)), None)
            if doc:
                missing.append(doc)
        return missing

    @staticmethod
    def _place_genitive(mittens: list[str], parts: list[str]) -> str | None:
        """ "для рукавички "На узліссі"", "для хвойного лісу": one place asked, else None."""
        if len(mittens) == 1:
            return f'рукавички "{mittens[0]}"'
        if not mittens and len(parts) == 1:
            return {"conifer": "хвойного лісу", "deciduous": "листяного лісу", "river": "частини лісу біля річки"}[parts[0]]
        return None

    def _place_note(self, mittens: list[str], parts: list[str]) -> str:
        if mittens:
            notes = []
            for mitten in mittens:
                mitten_parts = self.forest.mitten_parts.get(mitten, frozenset())
                where = ", ".join(PART_NAMES[p] for p in sorted(mitten_parts)) or "не належить до жодної частини лісу"
                notes.append(f'рукавичка "{mitten}" (частина лісу: {where})')
            return "; ".join(notes)
        if parts:
            return ", ".join(PART_NAMES[p] for p in parts)
        return "не вказано"

    def _doc_header(self, number: int, doc: Document) -> str:
        """The lines over a document's text in the prompt: its number, place, level and origin."""
        if doc.from_archive:
            origin = f"Лісовий архів (стандартний документ лісу), версія від {human_date(doc.modified_at, self.tz)}"
        else:
            # With the time, two contradicting animals' documents of one day still have a newer one.
            origin = f"надіслав звір {doc.author_name}, {human_date(doc.created_at, self.tz)} о {doc.created_at.astimezone(self.tz):%H:%M}"
        return (
            f"[{number}] \"{doc.title}\"\n"
            f"Для кого: {doc.scope.describe()}\n"
            f"Рівень: {LEVEL_NAMES[doc.scope.kind]}\n"
            f"Звідки: {origin}\n"
            f"Текст:\n"
        )

    @staticmethod
    def _excerpt_size(docs: list[Document], room: int) -> int:
        """One cut for all documents: the shorter ones go whole, the longer ones are cut to an equal share,
        and together they stay within `room` characters. Archive documents (under 1100 characters) go whole
        in the usual case, eight of 650-750 characters even with no place named and the whole calendar."""
        left, count = max(room, 0), len(docs)
        for length in sorted(len(doc.text) for doc in docs):
            share = left // count
            if length > share:
                return min(DOC_CHARS, share)
            left, count = left - length, count - 1
        return DOC_CHARS

    def _excerpt(self, doc: Document, question: str, mittens: list[str], parts: list[str], size: int) -> str:
        """A long guide is cut to its parts about the question: the sections about the asked place first,
        then the chunks sharing most words with the question, all in the document's order."""
        if len(doc.text) <= size:
            return doc.text
        # Words of the topic only: "під дубом" in another section's sentence is no match for the question.
        query = expand(set(tokens(self.forest.strip_places(question))))
        asked_parts = set(parts) | {p for m in mittens for p in self.forest.mitten_parts.get(m, ())}

        def place_of(text: str) -> int | None:
            """2 for the asked mitten, 1 for its forest part, 0 for another place, None for no place."""
            named_mittens, named_parts = set(self.forest.detect_mittens(text)), set(self.forest.detect_parts(text))
            if not named_mittens and not named_parts:
                return None
            # With a forest part asked, a section about one of its mittens is not the part's advice.
            return 2 if named_mittens & set(mittens) else 1 if named_parts & asked_parts else 0

        def opens_section(line: str, previous: str) -> bool:
            """A short heading with a place ("Під дубом", "Біля річки:"), or a paragraph that begins with one
            ("Біля річки рогіз..."). Not a sentence that only starts with a place word ("До ліщини по горіхи...",
            "Ліщиновими гілками...") or a line of a PDF paragraph wrapped mid-sentence."""
            words = line.split()
            if looks_like_heading(line) and len(words) <= 8:
                return True
            new_paragraph = not previous or previous.endswith((".", "!", "?", "…", ":")) or looks_like_heading(previous)
            return new_paragraph and words[0].lower() in LOCATIVE and place_of(" ".join(words[:3])) is not None

        chunks = split_chunks(doc.text)
        scores, section, previous = [], 0, ""
        for chunk in chunks:
            # Line by line, as a chunk may hold the end of one section and the start of the next.
            best = 0
            for line in (line.strip() for line in chunk.split("\n")):
                place = place_of(line) if line else None
                if place is not None and opens_section(line, previous):
                    section = place
                # The asked place's sections first, then a place named in passing, then its forest part.
                best = max(best, 2 * section, 2 * place - 1 if place else 0)
                previous = line or previous
            scores.append((best, len(query & set(tokens(chunk)))))
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        gap = "\n...\n"
        chosen, used = [], -len(gap)
        for i in ranked:
            if used + len(gap) + len(chunks[i]) <= size:
                chosen.append(i)
                used += len(gap) + len(chunks[i])
        if not chosen:  # a single chunk longer than the share: its start is better than nothing
            return chunks[ranked[0]][:size]
        return gap.join(chunks[i] for i in sorted(chosen))
