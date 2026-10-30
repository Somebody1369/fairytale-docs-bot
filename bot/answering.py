"""From a question to an answer: pick the place, find the documents, ask Groq, cite sources."""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from .llm import LLM, LLMUnavailable
from .places import PART_NAMES, Forest, Scope, resident_homes, residents_from_statute
from .search import COMMON_STEMS, STOPWORDS, Hit, Index, expand, looks_like_heading, split_chunks, tokens
from .storage import Document, Store
from . import texts

log = logging.getLogger(__name__)

MAX_DOCS = 7  # documents in one answer prompt; archive documents are short (under 1100 characters)
DOC_CHARS = 2500  # a long guide from an animal is cut to the parts that match the question
# All documents' text in one prompt. Groq's free tier takes about 8000 tokens a minute, which is
# about 20000 characters of Ukrainian, and the rules, the calendar and the question need their share.
PROMPT_DOC_CHARS = 12000
# Keyword search keeps the best documents of every level (mitten, each forest part, whole forest)
# separately, so a strong whole-forest match cannot push the mitten's own advice out. A mitten has
# only a handful of documents and they matter most, so any of them that matches at all may go.
LEVEL_LIMIT = {"mitten": 3, "part": 2, "forest": 2}
LEVEL_CUTOFF = {"mitten": 0.0, "part": 0.3, "forest": 0.3}
PICK_LIMIT = 4
# Seconds an animal waits for an answer at most, rate-limit pauses and slower fallback models included;
# picking documents by title is a help, not a must, so it gets a short part of them.
QUESTION_SECONDS = 45
PICK_SECONDS = 10
LOCATIVE = {"під", "біля", "коло", "поблизу", "у", "в", "на", "серед", "над", "за", "між"}  # a place at a line's start
LEVEL_NAMES = {"mitten": "рукавичка", "part": "частина лісу", "forest": "увесь ліс"}
PICK_ANIMAL_TITLES = 150  # newest animals' documents shown to the model when it picks by title
# Questions about dates and time need the model to think longer over the calendar.
TIME_WORDS = re.compile(
    r"коли|дат[аиуі]|числ|день|дня|дні|тижн|тиждень|місяц|сьогодні|завтра|вчора|вже|досі|пор[аиіу]\b|сезон|"
    r"зим|весн|літ[оа]|осін|осен|понеділ|вівтор|серед[уаи]|четвер|п.ятниц|субот|неділ|годин|термін|до якого")
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
5. Якщо питання про час, пору року, день тижня чи "чи вже можна", враховуй сьогоднішню дату. Дні тижня й дати бери лише з наданого календаря (там уже пораховано, наприклад, яка дата перша субота місяця), не обчислюй їх сам. Порівнюй дату події із сьогоднішньою: якщо вона раніше за сьогодні, пиши в минулому часі, що подія вже була, і не вигадуй дату наступної, якщо її немає в документах.
6. Пиши грамотною українською, ввічливо, просто, доброзичливо й по суті, без казковості та без звертань на кшталт "любий звіре". Якщо звертаєшся до звіра, то на "Ви" (з великої літери): "Вам", "здайте", "зверніться". Відповідь коротка (1-5 речень), але переказуй усі пункти документа, що стосуються питання, нічого з них не пропускай: скільки, коли, кому й куди віддавати чи приносити, чим можна замінити, хто звільнений.
7. Якщо рукавичка належить до кількох частин лісу (як "Під вербою": і листяний ліс, і біля річки), їй підходять поради кожної з цих частин: наведи в "answer" поради всіх її частин лісу, що стосуються питання, навіть якщо вони не суперечать одна одній, і скажи, до якої частини лісу належить кожна. Поради різних частин лісу однієї рукавички одна одній не поступаються, тож жодна з них не йде в "other". Порада самої рукавички, як і завжди, іде першою.
8. Не згадуй в "answer" та "other" номери й назви документів: джерела бот додасть сам.
9. Якщо в документі є окремі розділи для різних місць, бери розділ для місця з питання. Але рівень розділу той самий, що й у всього документа (рядок "Рівень"): розділ "Під дубом" у путівнику для всього лісу є порадою рівня "увесь ліс", тож документ для рукавички Під дубом або для її частини лісу важливіший за нього.
10. Місце в питанні ("під дубом", "біля сосон", "біля верби") означає рукавичку з рядка "Місце з питання", а не саме дерево: відповідай усіма порадами, що підходять цій рукавичці, навіть якщо в них це дерево не згадано.

Поверни JSON: {"facts": ["кожен пункт документів, з яких відповідаєш, на тему питання, окремим рядком, навіть якщо про нього прямо не питали (на \"скільки внесок\" це й кому та куди віддавати, і чим можна замінити): скільки, коли, кому, куди, чим замінити, хто звільнений; на початку рядка номер документа і його рівень, наприклад [2, частина лісу]"], "found": true або false, "answer": "текст відповіді для місця з питання, у якому є всі ці пункти, крім тих, що пішли в other", "other": "порада, що радить інакше й тут поступається відповіді, з поясненням чому, або \"\"", "sources": [номери документів, з яких узято відповідь, найважливіший перший]}"""

PLACE_UNKNOWN_RULE = """

Місце в питанні не вказано, а серед документів є поради для окремих рукавичок чи частин лісу. Якщо відповідь залежить від того, про яку рукавичку питає звір (для різних рукавичок чи частин лісу поради різні, або відповідь є лише в порадах для окремих місць і для кожного своя), не відповідай, а поверни "needs_place": true. Якщо ж відповідь однакова для всього лісу, або питання про конкретного звіра, який у документах живе в певній рукавичці (наприклад, "коли Ведмідь лягає спати?" чи "де живе Зайчик?"), відповідай як звичайно, назвавши цю рукавичку, і поверни "needs_place": false. Порада для рукавички цього звіра важливіша за загальну. Якщо відповідь є лише для одного місця, відповідай і прямо скажи, для якого місця ця порада."""

PICK_SYSTEM = """Тобі дають питання звіра і перелік назв документів із порадами. Обери до чотирьох документів, у яких найімовірніше є відповідь. Зважай на зміст, а не лише на однакові слова: "що їсти" і "чим поживитись" про те саме, "хто живе" є у статуті. Якщо на тему питання є кілька документів для різних місць або від різних звірів, обери їх усі. Якщо жоден не підходить, поверни порожній список.
Поверни JSON: {"ids": [номери документів]}"""

LEVEL_REMINDER = """

Зверни увагу: документ [{better}] (рівень "{better_level}") конкретніший за документ [{weaker}] (рівень "{weaker_level}"), навіть якщо в [{weaker}] є розділ про місце з питання. Якщо в [{better}] є відповідь на питання, відповідай за ним і постав його першим у "sources", а пораду [{weaker}], якщо вона радить інакше, перекажи в полі "other"."""

PARTS_REMINDER = """

Зверни увагу: рукавичка "{mitten}" належить до кількох частин лісу, і їй підходять поради кожної з них. У відповіді є порада для частини лісу "{cited}", але немає поради для частини "{missing}". Якщо в документі {docs} є щось на тему питання (навіть без дерева, названого в питанні: воно лише вказує на рукавичку), додай це у відповідь, сказавши, що це порада для частини лісу "{missing}", і додай номер документа в "sources"."""

FACTS_REMINDER = """

Зверни увагу: ти виписав у "facts" із документів, на які посилаєшся, пункти, яких немає ні в "answer", ні в "other": {facts}. Додай їх у "answer" (можна коротко), нічого не вигадуючи, а пункт поради, що тут поступається конкретнішій чи архіву, — коротко в "other"."""

ARCHIVE_REMINDER = """

Зверни увагу: перша порада у відповіді — з документа звіра [{animal}], а документ Лісового архіву {archive} написаний для того самого місця. Якщо в архіві теж є відповідь на питання, за правилом 4 важливіший архів, навіть якщо документ звіра новіший: відповідай за архівом і постав його першим у "sources", а пораду [{animal}], якщо вона радить інакше, перекажи в полі "other". Якщо відповіді в архіві немає, відповідай як раніше."""

RESIDENT_NOTE = " (місце визначено за звіром {name}, який тут живе; якщо порада однакова для всього лісу, на рукавичці не наголошуй)"


@dataclass
class Reply:
    kind: str  # "answer" | "not_found" | "need_place" | "unavailable"
    text: str = ""
    mittens: list[str] = field(default_factory=list)  # need_place: the mittens to choose from, empty means all


def format_date(moment: datetime, tz: ZoneInfo) -> str:
    local = moment.astimezone(tz)
    return f"{local:%d.%m.%Y} о {local:%H:%M}"


def human_date(moment: datetime, tz: ZoneInfo) -> str:
    local = moment.astimezone(tz)
    return f"{local.day} {MONTHS[local.month - 1]} {local.year}"


def calendar_note(moment: datetime, tz: ZoneInfo, full: bool = True) -> str:
    """Today with its weekday and the days of this and next month, so the model reads
    "перша субота жовтня" off the calendar instead of counting weekdays itself.
    Without `full` only today: the month tables cost tokens a question without dates does not need."""
    local = moment.astimezone(tz).date()
    lines = [f"Сьогодні: {WEEKDAYS[local.weekday()]}, {local.day} {MONTHS[local.month - 1]} {local.year}."]
    if not full:
        return lines[0]
    year, month = local.year, local.month
    for _ in range(2):
        first = date(year, month, 1)
        days = []
        day = first
        while day.month == month:
            days.append(f"{day.day} {WEEKDAYS_SHORT[day.weekday()]}")
            day += timedelta(days=1)
        lines.append(f"{MONTHS_NOMINATIVE[month - 1].capitalize()} {year}: " + ", ".join(days))
        firsts = []
        for offset in range(7):
            day = first + timedelta(days=offset)
            ordinal = "перша" if day.weekday() in (2, 4, 5, 6) else "перший"
            firsts.append(f"{ordinal} {WEEKDAYS[day.weekday()]} {MONTHS[month - 1]} = {day.day} {MONTHS[month - 1]}")
        lines.append("Перші дні тижня: " + ", ".join(sorted(firsts, key=lambda line: int(line.split("= ")[1].split()[0]))))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
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


def missing_facts(facts, docs: list[Document], sources: list[Document], text: str, question: str = "") -> list[str]:
    """The facts ("[1, рукавичка] ...") the model listed from the documents it cites that the text leaves out:
    checked sentence by sentence (and line by line for lists), a sentence is left out when the text lacks one of
    its numbers or half of its word stems. Words of the question need not be repeated ("у хвойному лісі" in
    "Коли утеплювати у соснах?" -> "У жовтні"). Facts from documents the answer does not cite are not required."""
    cited = {docs.index(doc) + 1 for doc in sources}
    have, asked = fact_terms(text), fact_terms(question)
    left_out = []
    for fact in facts if isinstance(facts, list) else []:
        match = FACT.match(str(fact))
        if not match or int(match.group(1)) not in cited:
            continue
        for sentence in re.split(r"(?<=[.!?;])\s+|\n+", match.group(2).strip()):
            terms = fact_terms(sentence)
            numbers = {term for term in terms if term.isdigit()}
            terms = (terms - asked) or terms
            if numbers - have or (terms and len(terms & have) * 2 < len(terms)):
                left_out.append(sentence.strip())
    return left_out


def clean_text(value, documents: int) -> str:
    """The model's text without the document numbers it now and then puts in ("у документі [2]", "[1, рукавичка]");
    other numbers in brackets ("[2026]") stay."""
    text = " ".join(str(part) for part in value) if isinstance(value, list) else str(value or "")
    numbered = lambda match: "" if 1 <= int(match.group(1)) <= documents else match.group(0)
    text = re.sub(r"(?:\s+(?:у\s+|в\s+)?документ\w*)?\s*\[(\d+)(?:,[^\]]*)?\]", numbered, text)
    return re.sub(r"\s{2,}", " ", text).strip()


PLACEHOLDER = re.compile(r"(?:немає|нема|-+|—|–|null|none|n/a|порожній рядок|\"\")\.?"
                         r"|(?:немає|нема)\s+ін\w*.*|інших\s+порад\s+немає.*", re.I)


# The model talking about its own JSON ("тому її не включаємо в \"other\"") is not advice for an animal.
SERVICE_WORDS = re.compile(r"\b(?:other|answer|sources|facts|json)\b", re.I)


def clean_other(value, answer: str, documents: int) -> str:
    """The advice that gives way, or "" for the stubs the model writes instead of nothing ("Немає", "-", "null"),
    for a repeat of the answer and for the model's words about its own reply."""
    if value is not None and not isinstance(value, (str, list)):
        return ""
    other = clean_text(value, documents)
    if not other or PLACEHOLDER.fullmatch(other) or SERVICE_WORDS.search(other) or fact_terms(other) <= fact_terms(answer):
        return ""
    return other


def best_per_level(hits: list[Hit]) -> list[Document]:
    """The best keyword matches of each place separately: the mitten, each of its forest parts, the whole forest."""
    groups: dict[str, list[Hit]] = {}
    for hit in hits:
        groups.setdefault(hit.doc.scope.encode(), []).append(hit)
    chosen: list[Hit] = []
    for group in groups.values():
        kind, best = group[0].doc.scope.kind, group[0].score
        chosen += [hit for hit in group if hit.score >= LEVEL_CUTOFF[kind] * best][:LEVEL_LIMIT[kind]]
    # The mitten's own documents first, so they survive when the prompt is full.
    return [hit.doc for hit in sorted(chosen, key=lambda hit: (-hit.doc.scope.level, -hit.score))]


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
            allowed = None
            docs = await self._docs_without_place(question, deadline)
        if docs:
            place_rule = PLACE_UNKNOWN_RULE if not allowed and any(doc.scope.kind != "forest" for doc in docs) else ""
            reply = await self._answer(question, docs, mittens, parts, now, place_rule, resident, allowed, deadline)
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
        hits = self.index.search(self.forest.strip_places(question), allowed, limit=60, cutoff=0)
        candidates = [doc for doc in self.index.documents if allowed(doc)]
        picked = await self._pick_by_title(question, candidates, deadline)
        return merge(picked, best_per_level(hits), limit=MAX_DOCS)

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
        candidates = archive + animals[:PICK_ANIMAL_TITLES]
        if not candidates:
            return []
        listing = "\n".join(f"{i}. {self._title_line(doc)}" for i, doc in enumerate(candidates, 1))
        try:
            result = await self.llm.json(PICK_SYSTEM, f"Питання: {question}\n\nДокументи:\n{listing}",
                                         max_tokens=700, fast=True,
                                         deadline=min(deadline, time.monotonic() + PICK_SECONDS))
        except LLMUnavailable:
            return []
        ids = []
        for i in result.get("ids", []) if isinstance(result.get("ids"), list) else []:
            if isinstance(i, str) and i.strip().isdigit():
                i = int(i.strip())
            if isinstance(i, int) and 1 <= i <= len(candidates) and i not in ids:
                ids.append(i)
        return [candidates[i - 1] for i in ids[:PICK_LIMIT]]

    def _title_line(self, doc: Document) -> str:
        if doc.from_archive:
            return f"{doc.title} ({doc.scope.describe()})"
        return f"{doc.title} ({doc.scope.describe()}, надіслав звір, {doc.created_at.astimezone(self.tz):%d.%m %H:%M})"

    async def _answer(
        self, question: str, docs: list[Document], mittens: list[str], parts: list[str], now: datetime,
        place_rule: str = "", resident: str | None = None, allowed=None, deadline: float | None = None,
    ) -> Reply:
        """`allowed`: the documents were chosen for a place named in the question."""
        # Most specific first, archive before animals, newer before older.
        docs = sorted(docs, key=lambda d: (-d.scope.level, not d.from_archive, -d.modified_at.timestamp()))
        place = self._place_note(mittens, parts) + (RESIDENT_NOTE.format(name=resident) if resident else "")
        # Thinking longer and the calendar cost Groq tokens (the free tier allows about 8000 a minute),
        # so only questions about dates and time get them.
        about_time = bool(TIME_WORDS.search(question.lower()))
        reasoning, budget = ("medium", 2000) if about_time else ("low", 1500)
        size = self._excerpt_size(docs)
        blocks = [self._doc_block(i, doc, question, mittens, parts, size) for i, doc in enumerate(docs, 1)]
        # "Де шукати гриби біля верби?" is about mushrooms for the mitten, not about willow trees.
        topic = re.sub(r"\s+", " ", self.forest.strip_places(question)).strip(" ?.!,") if mittens or parts else ""
        prompt = (
            f"{calendar_note(now, self.tz, full=about_time)}\n\n"
            f"Питання звіра: {question}\n"
            f"Місце з питання: {place}\n"
            + (f"Тема питання без місця: {topic}\n" if topic else "")
            + "\nДокументи:\n\n" + "\n\n".join(blocks)
        )
        system = ANSWER_SYSTEM + place_rule
        asked_sources = False
        # The first answer with sources, (answer, other, sources, faults, facts), while the model is asked once
        # to put it right.
        first = None
        while True:
            try:
                result = await self.llm.json(system, prompt, max_tokens=budget, reasoning=reasoning, deadline=deadline)
            except LLMUnavailable:
                if first:  # no time or no model left for the second try: the first answer still holds
                    answer, other, sources = first[:3]
                    break
                return Reply("unavailable", texts.LLM_UNAVAILABLE)
            answer = clean_text(result.get("answer"), len(docs))
            # Advice that says otherwise and gives way here (the whole forest's, or an animal's under the archive).
            other = clean_other(result.get("other"), answer, len(docs))
            sources = self._sources(result.get("sources"), docs)
            if not sources and len(docs) == 1:
                sources = docs  # one document given, so it is the source
            gave_up = (place_rule and result.get("needs_place")) or result.get("found") in (False, "false") or not answer
            if first and (gave_up or not sources):
                answer, other, sources = first[:3]  # asked to put it right, the model lost the answer: keep the first
                break
            if place_rule and result.get("needs_place"):
                return Reply("need_place")
            if gave_up:
                return Reply("not_found", texts.NOT_FOUND)
            if not sources:
                if asked_sources:
                    return Reply("not_found", texts.NOT_FOUND)
                asked_sources = True  # the model now and then forgets the source numbers; ask once more
                continue
            facts = result.get("facts") if isinstance(result.get("facts"), list) else []
            reminders, faults = self._review(question, docs, mittens, allowed, answer, other, sources, facts)
            if first:
                # The one second try is taken unless it came out worse: more broken rules, or more of the points
                # both answers listed left out (a weaker fallback model may answer it worse than the first).
                every = first[4] + facts
                before = (first[3], len(missing_facts(every, docs, first[2], f"{first[0]} {first[1]}", question)))
                after = (faults, len(missing_facts(every, docs, sources, f"{answer} {other}", question)))
                if after > before:
                    answer, other, sources = first[:3]
                break
            if not reminders:
                break
            # Everything to put right goes in one second try: each try costs a Groq request, and the free tier
            # allows only so many tokens a minute.
            first = (answer, other, sources, faults, facts)
            prompt += "".join(reminders)

        if len(sources) == 1:
            cite = "Джерело: " + source_line(sources[0], self.tz)
        else:
            cite = "Джерела:\n" + "\n".join("- " + source_line(doc, self.tz) for doc in sources)
        # The advice that gives way stands in a paragraph of its own, so it never reads as part of the answer.
        text = f"{answer}\n\n{other[0].upper()}{other[1:]}" if other else answer
        return Reply("answer", f"{text}\n\n{cite}")

    def _topic_terms(self, question: str) -> set[str]:
        """What the question is about, for telling whether an archive document is on its topic: the word stems
        without the place, short words, numbers and the names of residents ("Ведмідь" is in his mitten's statute)."""
        names = {stem for name in self.residents for stem in fact_terms(name)}
        return {term for term in fact_terms(self.forest.strip_places(question))
                if len(term) >= 4 and not term.isdigit()} - names

    def _review(self, question: str, docs: list[Document], mittens: list[str], allowed, answer: str, other: str,
                sources: list[Document], facts: list) -> tuple[list[str], int]:
        """What to put right in an answer: the reminders for the model, and how many of the places' rules
        (levels, the archive, both forest parts) the answer breaks."""
        reminders = []
        # A section about the asked mitten inside a whole-forest guide tempts the model to put the
        # guide first, though a cited document for the mitten's forest part is more specific.
        # With no place asked, PLACE_UNKNOWN_RULE alone decides between the whole forest and asking.
        better = next((doc for doc in sources[1:] if doc.scope.level > sources[0].scope.level), None)
        if better and allowed:
            reminders.append(LEVEL_REMINDER.format(
                better=docs.index(better) + 1, better_level=LEVEL_NAMES[better.scope.kind],
                weaker=docs.index(sources[0]) + 1, weaker_level=LEVEL_NAMES[sources[0].scope.kind]))
        # The decision told to Mishka (20:46): for the same place the archive beats an animal's document,
        # though the animal's is newer. A weaker fallback model now and then answers by the newer one, and
        # does not even cite the archive document it was given on the same topic.
        topic = self._topic_terms(question)
        archived = [] if not allowed or sources[0].from_archive else [
            doc for doc in docs if doc.from_archive and doc.scope == sources[0].scope and topic & fact_terms(doc.text)]
        if archived:
            reminders.append(ARCHIVE_REMINDER.format(archive=", ".join(f"[{docs.index(doc) + 1}]" for doc in archived),
                                                     animal=docs.index(sources[0]) + 1))
        # Mishka: "для «Під вербою» підходять поради і листяного лісу, і річки". When the answer took one
        # part's advice on the topic and left the other part's document out, the model is reminded of it.
        missing = self._missing_parts(mittens, docs, sources, answer, question)
        if missing:
            cited = {doc.scope.value for doc in sources if doc.scope.kind == "part"} - {doc.scope.value for doc in missing}
            reminders.append(PARTS_REMINDER.format(
                mitten=mittens[0], cited=", ".join(PART_NAMES[p] for p in sorted(cited)),
                missing=", ".join(PART_NAMES[doc.scope.value] for doc in missing),
                docs=", ".join(f"[{docs.index(doc) + 1}]" for doc in missing)))
        faults = len(reminders)
        # The model lists the points of the cited documents in "facts", then now and then drops some of them
        # when it writes the answer ("Кабанові в кошик біля входу" in the customer's own example).
        left_out = missing_facts(facts, docs, sources, f"{answer} {other}", question)
        if left_out:
            reminders.append(FACTS_REMINDER.format(facts="; ".join(f'"{fact}"' for fact in left_out)))
        return reminders, faults

    def _missing_parts(self, mittens: list[str], docs: list[Document], sources: list[Document], answer: str,
                       question: str) -> list[Document]:
        """For a mitten in several forest parts, with one part's advice in the answer: the best document of
        each of its other parts that was given to the model but left out of the answer. A part counts only
        when its advice is in the answer itself, not just in the paragraph of advice that gives way."""
        if len(mittens) != 1:
            return []
        said = fact_terms(answer) - fact_terms(question)
        answered = {doc.scope.value for doc in sources
                    if doc.scope.kind == "part" and len(fact_terms(doc.text) & said) >= 2}
        if not answered:
            return []  # the answer needs no forest part's advice at all, e.g. the mitten's own dues
        missing = []
        for part in sorted(self.forest.mitten_parts.get(mittens[0], frozenset()) - answered):
            doc = next((doc for doc in sources if doc.scope == Scope("part", part)), None) or \
                next((doc for doc in docs if doc.scope == Scope("part", part)), None)
            if doc:
                missing.append(doc)
        return missing

    @staticmethod
    def _sources(numbers, docs: list[Document]) -> list[Document]:
        sources = []
        for number in numbers if isinstance(numbers, list) else []:
            if isinstance(number, str) and number.strip(" []").isdigit():
                number = int(number.strip(" []"))
            if isinstance(number, int) and 1 <= number <= len(docs) and docs[number - 1] not in sources:
                sources.append(docs[number - 1])
        return sources

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

    def _doc_block(self, number: int, doc: Document, question: str, mittens: list[str], parts: list[str],
                   size: int) -> str:
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
            f"Текст:\n{self._excerpt(doc, question, mittens, parts, size)}"
        )

    @staticmethod
    def _excerpt_size(docs: list[Document]) -> int:
        """One cut for all documents: the shorter ones go whole, the longer ones are cut to an equal share,
        and together they stay within PROMPT_DOC_CHARS. Archive documents (under 1100 characters) are
        never cut: with at most eight documents a share is at least 1500."""
        left, count = PROMPT_DOC_CHARS, len(docs)
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
