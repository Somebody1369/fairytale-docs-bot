"""Which place an animal's document is written for. Unclear means: ask the sender, never guess."""

import logging
import re

from .llm import LLM, LLMUnavailable
from .places import FOREST_SCOPE, PART_NAMES, WHOLE_FOREST_PATTERN, Forest, Scope, normalize

log = logging.getLogger(__name__)

EXCERPT_CHARS = 3000

CLASSIFY_SYSTEM = """Ти визначаєш, для якого місця в лісі написано документ із порадами.

Рукавички: {mittens}.
Частини лісу: хвойний ліс, листяний ліс, біля річки.

Правила:
- Вирішує те, для кого порада написана (адресат: "порада для листяного лісу", "для мешканців хвойного лісу", "шановні мешканці рукавички Під дубом"), а не місця, згадані всередині поради. "Білі гриби шукайте під дубом" у пораді для листяного лісу не робить її порадою для рукавички Під дубом.
- Якщо документ прямо стосується однієї рукавички (її названо, наприклад "у рукавичці Під дубом"), це "mitten".
- Якщо порада загалом для типу місцевості ("у хвойному лісі", "під ялинами", "серед сосен", "у листяному лісі", "біля річки", "біля води"), це "part".
- Якщо прямо сказано, що порада для всіх мешканців лісу або для всього лісу, це "forest".
- Якщо з тексту не зрозуміло, для якого місця порада, або згадано кілька різних місць без пояснення, це "unclear". Не вгадуй.

Поверни JSON: {{"kind": "mitten" | "part" | "forest" | "unclear", "value": "точна назва рукавички або частини лісу, або порожньо"}}"""

PART_BY_NAME = {normalize(name): key for key, name in PART_NAMES.items()} | {
    "хвойний": "conifer", "листяний": "deciduous", "річка": "river", "біля річки": "river",
}


WHOLE_FOREST_CAPTION = r"(?:весь|увесь|всього|усього|для)\s+(?:ліс|лісу)\b|всіх\s+звір|усіх\s+звір"

# "Порада для листяного лісу: ...", "Ця пам'ятка для мешканців біля річки", "Шановні мешканці рукавички Під дубом!"
ADDRESSEE = re.compile(r"(?:\bдля|мешканц\w*|звір\w*)\s+([^.:;!?,…\n—–]{3,60})")
ADDRESSEE_HEAD = 300
WHOLE_ADDRESSEE = r"(?:всього|усього)\s+лісу|(?:всіх|усіх)\s+(?:мешканців|звірів)"


def scope_from_caption(caption: str | None, forest: Forest) -> Scope | None:
    """A caption on the file like "це для Під вербою" or "для хвойного лісу" names the place."""
    if not caption:
        return None
    low = normalize(caption)
    mittens, parts = forest.detect_mittens(low), forest.detect_parts(low)
    whole = bool(re.search(WHOLE_FOREST_PATTERN, low) or re.search(WHOLE_FOREST_CAPTION, low))
    if len(mittens) == 1 and not parts and not whole:
        return Scope("mitten", mittens[0])
    if len(parts) == 1 and not mittens and not whole:
        return Scope("part", parts[0])
    if whole and not mittens and not parts:
        return FOREST_SCOPE
    return None


async def classify_document(title: str, text: str, forest: Forest, llm: LLM) -> Scope | None:
    """Clear signals decide by themselves; the model only chooses between places the text
    actually mentions. No signal at all means the sender is asked."""
    # A title that names the place settles it: "Внески Під дубом.pdf", "Гриби хвойного лісу.docx".
    title_scope = _from_title(title, forest)
    if title_scope:
        return title_scope

    low = normalize(text[:EXCERPT_CHARS])
    # Who the advice is addressed to settles it, whatever places the advice itself mentions:
    # "Порада для листяного лісу: білі гриби шукайте під дубом" is for the deciduous forest.
    addressee = _addressee(low, forest)
    if addressee:
        return addressee
    named = [m for m in forest.mittens if normalize(m) in low]  # "рукавичка Під дубом"
    whole = bool(re.search(WHOLE_FOREST_PATTERN, low))
    parts = forest.detect_parts(low)  # "у хвойному лісі", "під ялинами", "біля річки"
    if len(named) == 1 and not whole and not parts:
        return Scope("mitten", named[0])
    if whole and not named:
        return FOREST_SCOPE
    loose = [m for m in forest.detect_mittens(low) if m not in named]  # "серед сосен", "біля мурашника"
    if len(parts) == 1 and not named and not loose:
        return Scope("part", parts[0])

    candidates = {Scope("mitten", m) for m in named + loose} | {Scope("part", p) for p in parts}
    for mitten in loose:
        candidates |= {Scope("part", p) for p in forest.mitten_parts.get(mitten, ())}
    if whole:
        candidates.add(FOREST_SCOPE)
    if not candidates:
        return None

    try:
        result = await llm.json(
            CLASSIFY_SYSTEM.format(mittens=", ".join(forest.mittens)),
            f"Назва файлу: {title}\n\nТекст:\n{text[:EXCERPT_CHARS]}",
            max_tokens=700, fast=True,
        )
    except LLMUnavailable:
        log.warning("Groq unavailable, asking the sender about %r", title)
        return None
    scope = _validate(result, forest)
    return scope if scope in candidates else None


def _addressee(low: str, forest: Forest) -> Scope | None:
    for match in ADDRESSEE.finditer(low[:ADDRESSEE_HEAD]):
        span = match.group(1)
        named = [m for m in forest.mittens if normalize(m) in span]
        parts = forest.detect_parts(span)
        if len(named) == 1 and not parts:
            return Scope("mitten", named[0])
        if len(parts) == 1 and not named:
            return Scope("part", parts[0])
        if not named and not parts and re.search(WHOLE_ADDRESSEE, match.group(0)):
            return FOREST_SCOPE
    return None


def _from_title(title: str, forest: Forest) -> Scope | None:
    """Only a title with one place settles it. "Гриби Під дубом і в соснах" names two, so the text decides
    or the sender is asked; a mitten with its own forest part ("Під дубом, листяний ліс") is still one place."""
    low = normalize(title)
    named = [m for m in forest.mittens if normalize(m) in low]
    mentioned = set(named) | set(forest.detect_mittens(title))
    parts = set(forest.detect_parts(title))
    if len(named) == 1 and mentioned == set(named) and parts <= forest.mitten_parts.get(named[0], frozenset()):
        return Scope("mitten", named[0])
    if not mentioned and len(parts) == 1:
        return Scope("part", parts.pop())
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
        key = PART_BY_NAME.get(value) or next((k for k in PART_NAMES if k == value), None)
        if key:
            return Scope("part", key)
    return None
