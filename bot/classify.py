"""Which place an animal's document is written for. Unclear means: ask the sender, never guess."""

import asyncio
import logging
import re
import time
from collections.abc import Iterator
from typing import NamedTuple

from .llm import LLM, LLMUnavailable
from .places import (FOREST_SCOPE, PART_NAMES, PART_PATTERNS, RIVER, WHOLE_FOREST_PATTERN, Forest, Scope,
                     mitten_pattern, normalize)

log = logging.getLogger(__name__)

EXCERPT_CHARS = 3000
# Seconds the model has to name the place, rate-limit pauses included. Then the sender is asked with buttons,
# as when Groq is unavailable: the sender's next files, and a question that needs the mitten, wait for this.
CLASSIFY_SECONDS = 30

CLASSIFY_SYSTEM = """Ти визначаєш, для якого місця в лісі написано документ із порадами.

Рукавички: {mittens}.
Частини лісу: хвойний ліс, листяний ліс, біля річки.

Правила:
- Вирішує те, для кого порада написана (адресат: "порада для листяного лісу", "для мешканців хвойного лісу", "шановні мешканці рукавички Під дубом"), а не місця, згадані всередині поради. "Білі гриби шукайте під дубом" у пораді для листяного лісу не робить її порадою для рукавички Під дубом.
- Якщо документ прямо стосується однієї рукавички (її названо, наприклад "у рукавичці Під дубом"), це "mitten". Слова "під дубом", "під ялиною", "під вербою", "у соснах" без слова "рукавичка" можуть означати просто дерево чи місцевість: це рукавичка, лише якщо видно, що порада саме для її мешканців.
- Якщо порада загалом для типу місцевості ("у хвойному лісі", "під ялинами", "серед сосен", "у листяному лісі", "біля річки", "біля води"), це "part".
- Якщо прямо сказано, що порада для всіх мешканців лісу або для всього лісу, це "forest".
- Документ стосується одного місця. Якщо з тексту не зрозуміло, для якого саме, або в ньому поради для кількох різних місць без пояснення, для кого він увесь, це "unclear". Не вгадуй.

Поверни JSON: {{"kind": "mitten" | "part" | "forest" | "unclear", "value": "точна назва рукавички або частини лісу, або порожньо"}}"""

PART_BY_NAME = {normalize(name): key for key, name in PART_NAMES.items()} | {
    "хвойний": "conifer", "листяний": "deciduous", "річка": "river",
} | {key: key for key in PART_NAMES}  # the model may answer with the key itself: "conifer"


WHOLE_FOREST_CAPTION = r"(?:весь|увесь|всього|усього|для)\s+(?:ліс|лісу)\b|всіх\s+звір|усіх\s+звір"

ADDRESSEE_HEAD = 300
QUOTE = r"[«\"'“„‘]"
QUOTES = " \"'«»“”„‘’"
PLACE_PREPOSITION = r"(?:(?:у|в|на|під|біля|коло|поблизу|серед)\s+)?"
# Words right before a place in a caption that negate it or name it as somewhere else: "це не для Під дубом",
# "для всіх, крім Під дубом", "дорога до Під дубом", "від сусідів з Під вербою". Such a caption decides nothing.
CAPTION_ELSEWHERE = re.compile(
    rf"\b(?:не|крім|окрім|без|від|до|повз|з|із|зі)\s+(?:для\s+)?(?:рукавич\w*\s+)?{QUOTE}?{PLACE_PREPOSITION}$")
# ... but "(я) з Під дубом" is where the sender lives, as typed in answer to "Про яку рукавичку?",
SENDER_FROM = re.compile(rf"(?:^|\b(?:я|ми)\s+)(?:з|із|зі)\s+(?:рукавич\w*\s+)?{QUOTE}?{PLACE_PREPOSITION}$")
# ... and "для звірів з Під дубом", "для мешканців з листяного лісу" say who the advice is for, unless negated,
# the next place too: "для звірів з Під дубом і з Під вербою".
FOR_FROM = re.compile(rf"(?<!\bне\s)(?<!крім\s)(?<!без\s)\bдля\s+(?:(?!(?:не|крім|окрім|без|від|до|повз)\b)[\w']+\s+){{1,3}}"
                      rf"(?:(?:з|із|зі)\s+[^,;.]+?\s(?:і|й|та)\s+)?(?:з|із|зі)\s+(?:рукавич\w*\s+)?{QUOTE}?{PLACE_PREPOSITION}$")
# Who the advice is for, at the opening: "Порада для листяного лісу:", "Мешканцям рукавички Під вербою:",
# and, as a greeting only, "Шановні мешканці лісу!" or "Мешканці лісу під ялинами!" at the start of a line.
# "Звірі під ялиною ховаються від дощу" is a sentence about animals, not an address.
ADDRESSEE = re.compile(r"\bдля\s+|\b(?:мешканц|звір)(?:ям|ам)\s+|(?:^|\b(?:шановні|дорогі|любі|усі|всі)\s+)(?:мешканц|звір)\w*\s+")
# The words between it and the place: "для [всіх мешканців рукавички «]Під дубом»", "мешканці [лісу ]біля річки".
ADDRESSEE_LEAD = re.compile(
    rf"(?:(?:всіх|усіх)\s+)?(?:(?:мешканц|звір)\w*\s+)?(?:(?:нашого\s+)?(?:(?:всього|усього)\s+)?лісу\s*)?"
    rf"(?:(?:у|в)\s+)?(?:рукавич\w*\s+)?(?:під\s+назвою\s+)?{QUOTE}?")
# After "для", the whole forest: "для всього лісу", "для всіх мешканців лісу", "для всіх:", "для кожного звіра".
WHOLE_FOR = re.compile(r"(?:(?:всіх|усіх)\s+(?:(?:мешканц|звір)\w*\s+)?(?:нашого\s+)?|(?:всього|усього)\s+)лісу\b"
                       r"|(?:всіх|усіх)(?:\s+(?:мешканц|звір)\w*)?$|кожного\s+звіра\b|(?:всіх|усіх)\s+рукавичок\b|лісу$")
# "для всіх", "для (всього) лісу", "для річки" are said of anything: "це небезпечно для лісу", "вогонь страшний
# для всіх", "Сміття шкідливе для річки". They address the advice only where they open the sentence ...
GENERAL_FOR = re.compile(r"(?:(?:всього|усього)\s+)?лісу|(?:всіх|усіх|кожного)\b|річки")
# ... after a heading at most: "Порада для всіх:", "Ця пам'ятка для всіх мешканців.", "Пишу для всіх",
# "Ця карта для всіх мешканців лісу." ("Це небезпечно для лісу" is no heading).
ADDRESS_OPENING = re.compile(
    r"(?:(?:[\w']+\s+){0,2}(?:(?:порад|пам'ятк|нагадуванн|правил|оголошенн|звернен)\w*|пишу)|(?:ця|цей)\s+[\w']+)?")
# In a greeting: "Шановні мешканці лісу!", "Дорогі звірі нашого лісу!", "Мешканці всіх рукавичок, увага!".
WHOLE_OF = re.compile(r"(?:(?:нашого\s+)?(?:(?:всього|усього)\s+)?лісу|(?:всіх|усіх)\s+рукавичок)$")
# "Для всіх, хто живе біля річки": only where they live says who.
WHO_LIVES = re.compile(r"\s*(?:хто|які|котрі)\s+(?:живе|живуть|мешка\w*)\s+(.*)")
# The whole forest in the text, clear enough to decide: "Ця пам'ятка стосується всього лісу".
WHOLE_FOREST_FOR = (WHOLE_FOREST_PATTERN + r"|(?:для|стосу\w*)\s+(?:(?:всіх|усіх)\s+(?:мешканц|звір)\w*\s+(?:нашого\s+)?"
                    r"|(?:всього|усього)\s+)лісу\b")
# ... and words that only make the whole forest a possible answer: "буди мешканців усього лісу",
# "Усім звірам!" (a mitten's note may begin so too).
WHOLE_FOREST_TEXT = WHOLE_FOREST_FOR + r"|(?:мешканц|звір)\w*\s+(?:нашого\s+)?(?:(?:всього|усього)\s+)?лісу\b"
EVERY_MITTEN = r"(?:кожн\w*|всіх|усіх)\s+рукавич"  # "Осінні справи в кожній рукавичці": for the whole forest
WHOLE_GREETING = re.compile(r"(?:шановні\s+|дорогі\s+)?(?:всім|усім|всі|усі)\s+(?:мешканц|звір)\w*\s*[!:]")
# A mitten named as somewhere else, not as the advice's own place: "до (старої) рукавички Під дубом іти 200 кроків",
# "як і в рукавичці Під дубом", "сусідів з рукавички Під вербою".
OTHER_MITTEN = r"(?:\b(?:до|від|з|із|зі|повз|сусід\w*)|як\s+(?:і\s+)?(?:у|в))\s+(?:(?!наш|сво)\w+(?:ої|ою|у|ю|ій|ого)\s+)?"
# A mitten as a landmark: "збираємося біля рукавички Під вербою". Only before a mitten named as a mitten,
# since "біля" also begins a mitten's own name: "Мешканцям Під ялиною і Біля мурашника".
NEAR_MITTEN = r"\b(?:біля|коло|поблизу|навпроти)\s+(?:\w+(?:ої|ою|у|ю|ій|ого)\s+)?"
# A river place counts by itself only where a line opens with it ("Біля річки навесні буває повінь"):
# "не залишайте малечу біля води" or "хто носить воду з річки" say nothing about who the advice is for.
OPENING_WORDS = 3
ASK = Scope("ask")  # several places addressed: the sender is asked
# Advice for several mittens: Mishka wants one document per place, so the sender is asked to split it.
# The scope's value lists the mittens.
SPLIT = "split"


class ModelChoice(NamedTuple):
    """The words did not settle the place: the question for the model and the answers that may be taken."""
    prompt: str
    accepted: set[Scope]


def _split(mittens: list[str], low: str) -> Scope:
    """The mittens in the order the text names them, by name or by the words for it: "Для мешканців Під дубом
    і біля сосон" is Під дубом, У соснах. Each one was found in `low`, so its pattern matches there."""
    def first_named(mitten: str) -> int:
        name = normalize(mitten)
        return low.find(name) if name in low else re.search(mitten_pattern(mitten), low).start()
    return Scope(SPLIT, ", ".join(sorted(mittens, key=first_named)))


def scope_from_caption(caption: str | None, forest: Forest) -> Scope | None:
    """A caption on the file like "це для Під вербою" or "для хвойного лісу" names the place.
    "для Під дубом і Під вербою" is two documents in one: a scope of kind SPLIT, as for such a text."""
    if not caption:
        return None
    low = normalize(caption)
    mittens, parts = forest.detect_mittens(low), forest.detect_parts(low)
    whole_patterns = [WHOLE_FOREST_PATTERN, WHOLE_FOREST_CAPTION]
    patterns = [mitten_pattern(m) for m in mittens] + [PART_PATTERNS[p] for p in parts] + whole_patterns
    befores = [low[:found.start()] for pattern in patterns for found in re.finditer(pattern, low)]
    if any(CAPTION_ELSEWHERE.search(before) and not SENDER_FROM.search(before) and not FOR_FROM.search(before)
           for before in befores):
        return None
    whole = any(re.search(pattern, low) for pattern in whole_patterns)
    if len(mittens) == 1 and not parts and not whole:
        return Scope("mitten", mittens[0])
    # Two mittens, not counting where the sender lives: "я з Під вербою, це для Під дубом" is one place.
    targets = [m for m in mittens
               if any(not SENDER_FROM.search(low[:found.start()]) for found in re.finditer(mitten_pattern(m), low))]
    if len(targets) >= 2 and not parts and not whole:
        return _split(targets, low)
    if len(parts) == 1 and not mittens and not whole:
        return Scope("part", parts[0])
    if whole and not mittens and not parts:
        return FOREST_SCOPE
    return None


async def classify_document(title: str, text: str, forest: Forest, llm: LLM) -> Scope | None:
    """Clear signals decide by themselves: who the advice is addressed to, a file name with one place,
    a mitten named as the advice's own mitten, words for the whole forest. A place only mentioned in the
    advice ("білі гриби шукайте під дубом") is not clear: the model chooses between the places that cover
    the whole document, and when it cannot tell, or the text names no place, the sender is asked.
    Signals that disagree, or a document addressed to several places, also mean asking. Advice for two
    or more mittens, each named as a mitten or heading a section ("Під дубом: ..."), gives a scope of kind
    SPLIT: the sender splits the document."""
    # The words are looked for in the whole text, seconds for a file of megabytes: in a thread, so the bot keeps
    # answering everyone else. On a copy of the map, which the archive sync may re-read meanwhile.
    found = await asyncio.to_thread(_classify_by_words, title, text, Forest(dict(forest.mitten_parts)))
    if not isinstance(found, ModelChoice):
        return found
    try:
        result = await llm.json(
            CLASSIFY_SYSTEM.format(mittens=", ".join(forest.mittens)), found.prompt,
            max_tokens=400, fast=True,  # a short {"kind", "value"}: the budget counts against the minute
            deadline=time.monotonic() + CLASSIFY_SECONDS,
        )
    except LLMUnavailable:
        log.warning("Groq unavailable, asking the sender about %r", title)
        return None
    scope = _validate(result, forest)
    return scope if scope in found.accepted else None


def _classify_by_words(title: str, text: str, forest: Forest) -> Scope | ModelChoice | None:
    """The place when the words settle it, None when the sender must be asked, else the choice left to the model."""
    addressee, title_scope = _addressee(text, forest), _from_title(title, forest)
    if addressee == ASK:
        return None
    if addressee and addressee.kind == SPLIT:
        return addressee
    if addressee and addressee.kind == "mitten":
        # "Мешканцям рукавички Під дубом: ..." and further on "У рукавичці У соснах внески 3 шишки" or
        # "Під вербою: внесок 3 горіхи": two documents.
        low, sections = normalize(text), _sections(text, forest)
        others = [m for m in forest.mittens
                  if m != addressee.value and (m in sections or _named_as_mitten(m, low, own=True))]
        if others:
            return _split([addressee.value, *others], low)
    if addressee and title_scope:
        # "Внески біля річки" with "Шановні мешканці рукавички Під вербою!": the more specific one.
        return addressee if _within(addressee, title_scope, forest) else \
            title_scope if _within(title_scope, addressee, forest) else None
    if addressee:
        return addressee  # "Порада для листяного лісу: ...": the rest of the text cannot change it
    low, head = normalize(text), normalize(text[:EXCERPT_CHARS])
    own_anywhere = [m for m in forest.mittens if _named_as_mitten(m, low, own=True)]
    sections = _sections(text, forest)
    advised = [m for m in forest.mittens if m in own_anywhere or m in sections]
    # "У рукавичці Під дубом внесок 4 жолуді, у рукавичці Під вербою 3 горіхи", or the lines "Під дубом: 4 жолуді."
    # and "Під вербою: 3 горіхи.": two documents in one, unless the file name or the opening says it is for a forest
    # part or the whole forest (Mishka, 22:05: a guide's section about a mitten weighs as the whole guide).
    whole_doc = bool(re.search(WHOLE_FOREST_FOR, head) or re.search(WHOLE_FOREST_TEXT, head)
                     or WHOLE_GREETING.match(low) or re.search(EVERY_MITTEN, f"{normalize(title)} {head}"))
    if len(advised) >= 2 and not whole_doc and not (title_scope and title_scope.kind != "mitten"):
        return _split(advised, low)
    if title_scope:
        # "Внески Під дубом" with "У рукавичці У соснах внески 3 шишки": the text names another mitten.
        clash = title_scope.kind == "mitten" and any(m != title_scope.value for m in advised)
        return None if clash else title_scope

    # The whole text rules a decision out (a guide may get to other places on its third page),
    # but a decision without the model needs its evidence in the part the model would read.
    # Not Forest.detect_mittens and detect_parts: they would normalize the whole text once more.
    mentioned = [m for m in forest.mittens if re.search(mitten_pattern(m), low)]  # "під дубом", "серед сосен"
    parts = {key for key, pattern in PART_PATTERNS.items() if re.search(pattern, low)}  # "у хвойному лісі"
    whole = bool(re.search(WHOLE_FOREST_FOR, head))
    decided = None
    if whole and not mentioned and not parts:
        decided = FOREST_SCOPE
    # "У рукавичці Під дубом ..." on the first pages, no other mitten named as a mitten and no other forest part;
    # a tree word elsewhere ("гілки ялинки біля входу") does not change whose advice it is.
    own = [m for m in own_anywhere if _named_as_mitten(m, head, own=True)]
    if len(own) == 1 and own_anywhere == own and not whole and parts <= forest.mitten_parts.get(own[0], frozenset()):
        decided = Scope("mitten", own[0])
    # "під ялинами" is the conifer forest; a river place only where a line opens with it.
    if len(parts) == 1 and not mentioned and not whole:
        part = next(iter(parts))
        if part in forest.detect_parts(head) and (part != RIVER or _opens_with_part(text, part, forest)):
            decided = Scope("part", part)
    if decided:
        return decided if _title_agrees(title, decided, forest) else None

    # The model may name a single mentioned mitten, but it counts only when the text names it as its own
    # mitten or opens with it ("Під дубом при пожежі збираємося під дубом"): "рижики ростуть під ялиною" may be
    # about any spruce, and then the sender is asked rather than the advice filed for its forest part.
    offered = {Scope("part", p) for p in parts}
    if mentioned:
        offered |= {Scope("part", p) for p in frozenset.intersection(*(forest.mitten_parts[m] for m in mentioned))}
    if len(mentioned) == 1:
        offered.add(Scope("mitten", mentioned[0]))
    # One document is for one place: a choice must cover every place the text mentions.
    offered = {scope for scope in offered if _covers(scope, mentioned, parts, forest)}
    if whole or re.search(WHOLE_FOREST_TEXT, low) or WHOLE_GREETING.match(low):
        offered.add(FOREST_SCOPE)
    offered = {scope for scope in offered if _title_agrees(title, scope, forest)}
    if not offered:
        return None
    accepted = {scope for scope in offered
                if scope.kind != "mitten" or scope.value in own_anywhere or _opens_with(text, scope.value)}
    if not accepted:
        return None  # "Звіробій ... збирайте на узліссі": no answer of the model could be taken

    explicit = [m for m in mentioned if _named_as_mitten(m, low)]
    places = [f'рукавичка "{m}"' if m in explicit else f'"{m.lower()}" (рукавичка або просто дерево чи місцевість)'
              for m in mentioned] + [PART_NAMES[p] for p in sorted(parts)]
    choices = [_choice(scope) for scope in sorted(offered, key=lambda s: (-s.level, s.value or ""))]
    return ModelChoice(
        f"Назва файлу: {title}\n\nМісця, згадані в документі: {', '.join(places) or 'жодного'}\n"
        f"Можливі відповіді: {'; '.join(choices)}; або \"unclear\", якщо з тексту це не зрозуміло."
        f"\n\nТекст:\n{text[:EXCERPT_CHARS]}",
        accepted,
    )


def _named_as_mitten(mitten: str, low: str, own: bool = False) -> bool:
    """ "у рукавичці «Під дубом»", "рукавичка під назвою Під дубом", "«Під дубом»". With `own`, not as somewhere
    else: "до рукавички Під дубом іти 200 кроків" and "збираємося біля рукавички Під дубом" are not advice for it."""
    named = rf"(?:рукавич\w*\s+(?:під\s+назвою\s+)?{QUOTE}?|{QUOTE}){re.escape(normalize(mitten))}"
    starts = [found.start() for found in re.finditer(named, low)]
    if own:
        starts = [start for start in starts
                  if not _elsewhere(low, start) and not re.search(NEAR_MITTEN + "$", low[max(0, start - 40):start])]
    return bool(starts)


def _sections(text: str, forest: Forest) -> list[str]:
    """The mittens whose own name heads a line: "Під дубом: внесок 4 жолуді.", "• «У ліщині» — 3 горіхи",
    "1) Під вербою -- 3 горіхи". Not a tree, "Під ялинами: рижики", nor a longer word, "Під дубом-велетнем ростуть
    гриби", nor a line a PDF broke off a sentence, "під дубом — після дощу": a heading starts with a capital.
    A dash also says "is", "Біля бобрової греблі — глибоко", so a dash heads a section only beside another mitten's
    heading. One search of the text as it is: a file of megabytes is not cut into lines, and the lead before the name
    stays within its line."""
    names = "|".join(r"[^\S\n]+".join(map(re.escape, normalize(m).split())) for m in forest.mittens)
    # Before the name any bullet or a list number, after it a colon or a dash, "--" in plain text too.
    found = re.finditer(rf"^(?:[^\w\n]|\d{{1,3}}[.)])*({names})(?:[^\S\n]|[{QUOTES}])*(:|--?\s|[—–])",
                        text, re.MULTILINE | re.IGNORECASE)
    colon, dash = set(), set()
    for match in found:
        if match.group(1)[0].isupper():
            (colon if match.group(2) == ":" else dash).add(normalize(match.group(1)))
    headings = colon | dash if len(colon | dash) >= 2 else colon
    return [m for m in forest.mittens if normalize(m) in headings]


def _elsewhere(text: str, start: int) -> bool:
    """The words before `start` name the place as somewhere else: "до (старої) рукавички", "сусідів з"."""
    return bool(re.search(OTHER_MITTEN + "$", text[max(0, start - 40):start]))


def _covers(scope: Scope, mentioned: list[str], parts: set[str], forest: Forest) -> bool:
    if scope.kind == "forest":
        return True
    if scope.kind == "mitten":
        return set(mentioned) <= {scope.value} and parts <= forest.mitten_parts.get(scope.value, frozenset())
    return parts <= {scope.value} and all(scope.value in forest.mitten_parts[m] for m in mentioned)


def _within(inner: Scope, outer: Scope, forest: Forest) -> bool:
    """Advice for `inner` is also for a place inside `outer`: a mitten in its forest part, anything in the forest."""
    return inner == outer or outer.kind == "forest" or (
        outer.kind == "part" and inner.kind == "mitten" and outer.value in forest.mitten_parts.get(inner.value, ()))


def _title_agrees(title: str, scope: Scope, forest: Forest) -> bool:
    """A file name that names other places ("Звірі лісу та їхні рукавички", "Гриби під ялиною") than the
    place the text gives makes the place unclear."""
    low = normalize(title)
    mentioned, parts = forest.detect_mittens(low), set(forest.detect_parts(low))
    whole = re.search(WHOLE_FOREST_TEXT, low)
    if scope.kind == "forest":
        return not mentioned and not parts
    return _covers(scope, mentioned, parts, forest) and not whole


def _lines(text: str) -> list[str]:
    return [normalize(line).strip(QUOTES + "-–— ") for line in text.splitlines() if line.strip()]


def _opens_with(text: str, mitten: str) -> bool:
    """The first or second line (after a title line) starts with the mitten's name."""
    return any(line.startswith(normalize(mitten)) for line in _lines(text[:EXCERPT_CHARS])[:2])


def _opens_with_part(text: str, part: str, forest: Forest) -> bool:
    """A line on the first pages starts with the place: "Біля річки навесні буває повінь"."""
    return any(part in forest.detect_parts(" ".join(line.split()[:OPENING_WORDS]))
               for line in _lines(text[:EXCERPT_CHARS]))


def _choice(scope: Scope) -> str:
    """A possible answer in the JSON the model returns."""
    value = scope.value if scope.kind == "mitten" else PART_NAMES[scope.value] if scope.kind == "part" else ""
    return f'{{"kind": "{scope.kind}", "value": "{value}"}}'


def _addressee(text: str, forest: Forest) -> Scope | None:
    """Who the opening of the text is addressed to. A document is for one place, so addressees not one inside
    the other ("Для мешканців рукавички Під дубом: ...", then "Для мешканців рукавички Під вербою: ...", on one
    line or on two) are several documents in one: SPLIT when all are mittens, ASK otherwise."""
    # A PDF breaks its lines inside a sentence, "Порада для\nхвойного лісу": a line in lower case goes on the last one.
    head = re.sub(r"\n[ \t]*(?=[а-яіїєґa-z])", " ", text[:ADDRESSEE_HEAD])
    found: list[Scope] = []
    for line in _lines(head):
        for scope in _line_addressees(line, forest):
            if scope == ASK:
                return ASK
            if not any(_within(scope, known, forest) or _within(known, scope, forest) for known in found):
                found.append(scope)
    if len(found) <= 1:
        return found[0] if found else None
    mittens = []
    for scope in found:
        if scope.kind == "mitten":
            mittens.append(scope.value)
        elif scope.kind == SPLIT:
            mittens += scope.value.split(", ")
        else:
            return ASK  # "Порада для листяного лісу: ...", then "Порада для хвойного лісу: ..."
    return Scope(SPLIT, ", ".join(dict.fromkeys(mittens)))


def _line_addressees(line: str, forest: Forest) -> Iterator[Scope]:
    """Each place right after "для", "мешканцям" or a greeting in the line. A place further on in the clause
    is not the addressee: "Для зручності внески здавайте біля бобрової греблі". A sentence that names another
    place too ("для хвойного та листяного лісу") is addressed to several places: ASK."""
    for match in ADDRESSEE.finditer(line):
        if re.search(rf"{QUOTE}$", line[:match.start()]):
            continue  # «для хвойного лісу» in quotes: words cited, not who the advice is for
        kind = "for" if match.group(0).startswith("для") else \
            "to" if re.search(r"(?:ям|ам)\s+$", match.group(0)) else "greeting"
        # Up to the end of the sentence for the places it names, up to a comma for the addressee itself.
        sentence = re.split(r"[.:;!?…]", line[match.end():], maxsplit=1)[0]
        clause, _, after = sentence.partition(",")
        clause, who = clause.strip(), after.strip()
        opening = re.split(r"[.!?…]", line[:match.start()])[-1].strip(QUOTES)
        if kind == "for" and GENERAL_FOR.match(clause) and not ADDRESS_OPENING.fullmatch(opening):
            continue  # "це небезпечно для лісу", "бо вогонь страшний для всіх", "Сміття шкідливе для річки"
        lead = ADDRESSEE_LEAD.match(clause).group(0)
        place = clause[len(lead):].strip(QUOTES)
        scope = _place_at_start(place, forest)
        if scope and kind != "for" and not _just_the_place(place, forest):
            # "Звірі під ялиною ховаються від дощу", "Звірам під дубом у грозу стояти небезпечно":
            # a sentence about animals, not an address
            scope = None
        if scope and scope.kind == "mitten" and "лісу" in lead and "рукавич" not in lead:
            scope = None  # "мешканців лісу біля бобрової греблі" may be the area by the dam
        if scope is None and kind == "for":
            if re.fullmatch(r"(?:всіх|усіх)(?:\s+(?:мешканц|звір)\w*)?", clause) and \
                    who.startswith(("хто", "які", "котрі")):
                # "Для всіх, хто живе біля річки": only where they live says who.
                lives = WHO_LIVES.match(who)
                scope = _place_at_start(lives.group(1).strip(QUOTES), forest) if lives else None
                if not scope and not (forest.detect_mittens(who) or forest.detect_parts(who)):
                    scope = FOREST_SCOPE  # "для всіх, хто любить гриби"
            elif WHOLE_FOR.match(clause):
                scope = FOREST_SCOPE  # "Нагадування для всього лісу про збори": the rest is the topic
        elif scope is None and WHOLE_OF.fullmatch(clause):
            scope = FOREST_SCOPE  # "Шановні мешканці лісу!"
        if scope is None:
            continue
        if scope != FOREST_SCOPE:
            # A mitten named as somewhere to go ("які ходять по горіхи до ліщини") is not one the advice is for.
            named = [mitten for mitten in forest.detect_mittens(sentence) if _addressed(mitten, sentence)]
            if not _covers(scope, named, set(forest.detect_parts(sentence)), forest):
                # "для Під дубом і Під вербою" is two documents in one; "для хвойного та листяного лісу" is asked about
                scope = _split(named, sentence) if len(named) >= 2 else ASK
        yield scope


def _addressed(mitten: str, sentence: str) -> bool:
    """The mitten is named in the sentence other than as a destination or a neighbour: "для Під дубом і Під вербою",
    but not "... які ходять по горіхи до ліщини" or "... що поруч із вербою"."""
    return any(not _elsewhere(sentence, found.start()) for found in re.finditer(mitten_pattern(mitten), sentence))


def _just_the_place(place: str, forest: Forest) -> bool:
    """The greeting or the address ends with the place, or with places joined by "і", "й", "та": "Шановні мешканці
    рукавички «Під дубом»!", "Мешканці лісу під ялинами!", "Мешканцям Під ялиною і Біля мурашника:"."""
    return _is_place(place, forest) or all(
        _is_place(item.strip(QUOTES), forest) for item in re.split(r"\s+(?:і|й|та)\s+", place))


def _is_place(words: str, forest: Forest) -> bool:
    """A mitten or a forest part, and nothing more: "під дубом", "біля сосон", "під ялинами", "хвойного лісу"."""
    patterns = [mitten_pattern(mitten) for mitten in forest.mittens] + list(PART_PATTERNS.values())
    return any(re.fullmatch(rf"{PLACE_PREPOSITION}(?:{pattern})\w*(?:\s+лісу|\s+ліс)?", words) for pattern in patterns)


def _place_at_start(place: str, forest: Forest) -> Scope | None:
    named = [m for m in forest.mittens if place.startswith(normalize(m))]
    parts = [key for key, pattern in PART_PATTERNS.items() if re.match(rf"{PLACE_PREPOSITION}(?:{pattern})", place)]
    if len(named) == 1 and not parts:
        return Scope("mitten", named[0])
    if len(parts) == 1 and not named:
        return Scope("part", parts[0])
    return None


def _from_title(title: str, forest: Forest) -> Scope | None:
    """Only a title with one place settles it. "Гриби Під дубом і в соснах" names two, so the text decides
    or the sender is asked; a mitten with its own forest part ("Під дубом, листяний ліс") is still one place.
    A mitten counts as named in its own spelling, "Внески Під дубом" or "ВНЕСКИ ПІД ДУБОМ", or as a mitten,
    "рукавичка під дубом", but not as a destination, "Дорога до Під дубом": "Білі гриби під дубом" may be any oak."""
    spaced = " ".join(title.split())
    low = normalize(spaced)
    named = []
    for mitten in forest.mittens:
        for found in re.finditer(re.escape(normalize(mitten)), low):
            before = low[:found.start()]
            if re.search(rf"\b(?:до|повз)\s+(?:рукавич\w*\s+)?{QUOTE}?$", before):
                continue
            if spaced[found.start()].isupper() or re.search(rf"рукавич\w*\s+(?:під\s+назвою\s+)?{QUOTE}?$", before):
                named.append(mitten)
                break
    mentioned = set(named) | set(forest.detect_mittens(title))
    parts = set(forest.detect_parts(title))
    if len(named) == 1 and mentioned == set(named) and parts <= forest.mitten_parts.get(named[0], frozenset()):
        return Scope("mitten", named[0])
    if not mentioned and len(parts) == 1:
        return Scope("part", parts.pop())
    if not mentioned and not parts and re.search(WHOLE_FOREST_FOR, low):  # "Путівник для всього лісу"
        return FOREST_SCOPE
    return None


def _validate(result: dict, forest: Forest) -> Scope | None:
    kind = str(result.get("kind", "")).strip().lower()
    value = normalize(str(result.get("value") or "")).strip(' ".«»')
    if kind == "forest":
        return FOREST_SCOPE
    if kind == "mitten":
        for mitten in forest.mittens:
            if normalize(mitten) == value:
                return Scope("mitten", mitten)
    if kind == "part":
        key = PART_BY_NAME.get(value)
        if key:
            return Scope("part", key)
    return None
