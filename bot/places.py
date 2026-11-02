"""Mittens, forest parts and the rules that decide which advice applies where.

The mapping of mittens to forest parts comes from the archive document
"Карта лісу". Built-in defaults mirror it, and `Forest.update_from_map`
re-reads the document whenever the archive brings a new version.
"""

import re
from dataclasses import dataclass, field

CONIFER, DECIDUOUS, RIVER = "conifer", "deciduous", "river"

PART_NAMES = {
    CONIFER: "хвойний ліс",
    DECIDUOUS: "листяний ліс",
    RIVER: "біля річки",
}

DEFAULT_MAP: dict[str, frozenset[str]] = {
    "Під ялиною": frozenset({CONIFER}),
    "У соснах": frozenset({CONIFER}),
    "Біля мурашника": frozenset({CONIFER}),
    "Під дубом": frozenset({DECIDUOUS}),
    "У ліщині": frozenset({DECIDUOUS}),
    "Під вербою": frozenset({DECIDUOUS, RIVER}),
    "Біля бобрової греблі": frozenset({RIVER}),
    "На узліссі": frozenset(),
}

# How animals refer to each mitten in questions: "під дубом", "біля сосон", "в ліщині"...
# Only the tree itself names a mitten: "соснові шишки", "вербові котики" and "ліщинові горіхи" are things,
# "у сосновому лісі" and "у сосняку" are the conifer forest (Mishka, 30 Oct, 14:18: a mitten only when it is named).
MITTEN_PATTERNS = {
    "Біля бобрової греблі": r"бобров\w*\s+гребл|\b(?:біля|коло|поблизу|від|до|під|на|у|в)\s+гребл",
    "Біля мурашника": r"мурашник",
    "На узліссі": r"узлісс",
    "Під вербою": r"\bверб(?:а|и|і|у|ою)\b",
    "Під дубом": r"\bдуб(?:ом|а|і|у)?\b",
    "Під ялиною": r"\bялин(?:а|и|і|у|ою|ка|ки|ці|ку|кою)\b",
    "У ліщині": r"\bліщин(?:а|и|і|у|ою|ам|ами|ах|ка|ки|ці|ку|кою|ник\w*)?\b",
    "У соснах": r"\bсосн(?:а|и|і|у|ою|ам|ами|ах)\b|\bсосон\b|\bсосен\b",
}

# "хвойний ліс", "у хвойному", "хвойного та листяного лісу", but not "хвойні гілки" or "листяна підстилка";
# "біля річки", "річкою", but not "порічки".
# "хвойно-листяний ліс" is both parts: the first half of a mixed forest is followed by its second, not a space.
AS_PLACE = r"\b{0}\w*(?:-\w+)?\s+(?:\w+\s+){{0,2}}(?:ліс|частин)|\b(?:у|в|на|серед|для)\s+{0}"
# A tree's adjective is a place only with its forest: "у сосновому лісі", "соснового й дубового лісу",
# "у дубовому гаю", but "для соснових шишок", "дубовий стіл", "соснове борошно" or "дубовий лісник" is a thing.
# A mixed forest names both trees: "сосново-дубовий ліс" is asked about like "соснового й дубового лісу".
TREE_FOREST = (r"\b{0}\w*(?:-\w+|\s+(?:і|й|та|чи|або)\s+\w+)?\s+"
               r"(?:ліс(?:у|і|ом|и|ах|ами|ів|ам|ок|ку|ком)?\b|бір\b|бор(?:у|і|ові|ом|и|ів|ам|ами|ах)\b|га(?:й|ю|єві|єм|ї|їв|ям|ями|ях)\b|частин)")
PART_PATTERNS = {
    # "серед ялин", "між ялинок", "у ялиннику", "у сосняку", "у бору", but not "вибір"; one "ялинка" is a mitten.
    # "ялин" and "ялинок" alone are trees ("гілки ялин", "не рубайте ялинок", "чи у ялин є шишки?"): a place only
    # after "серед", "між", "біля", "під"; "ялинками" only after "під", "між"... ("прикрасити ялинками" is a thing).
    # "борів" alone may be a hog: only "соснових борів" is the forest.
    CONIFER: AS_PLACE.format("хвойн") + "|" + TREE_FOREST.format("(?:соснов|ялинов)")
    + r"|\bялин(?:ами|ах|ам)\b|\b(?:серед|між|біля|коло|поблизу|під)\s+ялин(?:ок)?\b"
    + r"|\b(?:серед|між|під|у|в|на)\s+ялинк(?:ами|ах)\b"
    + r"|\bялинник|\bсосняк|\bбір\b|\bбор(?:у|і|ом|и|ах|ами)\b",
    DECIDUOUS: AS_PLACE.format("листян") + "|" + TREE_FOREST.format("(?:дубов|березов)")
    + r"|\bдуб(?:и|ів|ами|ах|ам)\b|\bдубняк|\bдібров|\bберезняк",
    RIVER: r"\b(?:при)?річк|\bрічц|прибереж|\b(?:біля|коло|поблизу)\s+вод(?:и|і)\b",
}

WHOLE_FOREST_PATTERN = r"(?:всіх|усіх)\s+мешканців\s+лісу|для\s+(?:всього|усього)\s+лісу|незалежно\s+від\s+того,\s+де"

LEVEL_MITTEN, LEVEL_PART, LEVEL_FOREST = 3, 2, 1

# Word writes "пам’ятка", keyboards "пам'ятка", "памʼятка" or "пам`ятка": texts are compared with one apostrophe.
APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "`": "'"})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(APOSTROPHES))


def mitten_pattern(name: str) -> str:
    """How texts name the mitten: "біля сосон" is У соснах. One the map adds without words of its own
    is named by its name."""
    return MITTEN_PATTERNS.get(name) or re.escape(normalize(name))


@dataclass(frozen=True)
class Scope:
    kind: str  # "mitten" | "part" | "forest"
    value: str | None = None

    @property
    def level(self) -> int:
        return {"mitten": LEVEL_MITTEN, "part": LEVEL_PART, "forest": LEVEL_FOREST}[self.kind]

    def describe(self) -> str:
        if self.kind == "mitten":
            return f'рукавичка "{self.value}"'
        if self.kind == "part":
            return PART_NAMES[self.value] if self.value != RIVER else "частина лісу біля річки"
        return "увесь ліс"

    def encode(self) -> str:
        return f"{self.kind}:{self.value or ''}"

    @staticmethod
    def decode(raw: str) -> "Scope":
        kind, _, value = raw.partition(":")
        return Scope(kind, value or None)


FOREST_SCOPE = Scope("forest")


@dataclass
class Forest:
    mitten_parts: dict[str, frozenset[str]] = field(default_factory=lambda: dict(DEFAULT_MAP))

    @property
    def mittens(self) -> list[str]:
        return list(self.mitten_parts)

    def mittens_in(self, parts: list[str]) -> list[str]:
        """The mittens of these forest parts: "Під вербою" is in both the deciduous forest and by the river."""
        return [m for m in self.mittens if self.mitten_parts[m] & set(parts)]

    def applies(self, scope: Scope, mitten: str | None = None, part: str | None = None) -> bool:
        """Whether advice written for `scope` is valid for the asked place."""
        if scope.kind == "forest":
            return True
        if mitten:
            if scope.kind == "mitten":
                return scope.value == mitten
            return scope.value in self.mitten_parts.get(mitten, frozenset())
        if part:
            return scope.kind == "part" and scope.value == part
        return False

    def detect_mittens(self, text: str) -> list[str]:
        low = normalize(text)
        found = []
        for name in self.mittens:
            if re.search(mitten_pattern(name), low):
                found.append(name)
        return found

    def strip_places(self, text: str) -> str:
        """The question without place words, so search looks for the topic, not the address."""
        low = normalize(text)
        # All places are found in the question as it is, then cut out together: "у дубовому й сосновому лісі"
        # is the deciduous forest's match whole, though the conifer one alone would leave "у дубовому й" behind.
        # "у дубово-сосновому лісі" goes whole too, though the conifer forest is found from its second half.
        found = sorted(match.span() for pattern in [*MITTEN_PATTERNS.values(), *PART_PATTERNS.values()]
                       for match in re.finditer(rf"(?:\b(?:під|біля|коло|у|в|на|серед|між)\s+)?(?:\b\w+-)?(?:{pattern})\w*", low))
        kept, done = [], 0
        for start, end in found:
            if end > done:  # overlapping matches are cut as one
                kept.append(low[done:max(start, done)])
                done = end
        return " ".join(kept + [low[done:]])

    def detect_parts(self, text: str) -> list[str]:
        low = normalize(text)
        return [key for key, pattern in PART_PATTERNS.items() if re.search(pattern, low)]

    def archive_scope(self, title: str, text: str) -> Scope:
        """Scope of a standard archive document.

        Titles name the mitten or forest part ("... Під дубом", "... хвойного лісу");
        otherwise the opening sentence says who the advice is for.
        """
        low_title = normalize(title)
        for name in self.mittens:
            if normalize(name) in low_title:
                return Scope("mitten", name)
        title_parts = self.detect_parts(title)
        if len(title_parts) == 1:
            return Scope("part", title_parts[0])

        head = normalize(text[:400])
        if re.search(WHOLE_FOREST_PATTERN, head):
            return FOREST_SCOPE
        head_parts = self.detect_parts(head)
        if len(head_parts) == 1:
            return Scope("part", head_parts[0])
        return FOREST_SCOPE

    def update_from_map(self, text: str) -> bool:
        """Re-read the "Карта лісу" legend: lines like "Під вербою — і листяний ліс, і біля річки"."""
        flat = re.sub(r"\s+", " ", text)
        parsed: dict[str, frozenset[str]] = {}
        legend_line = r"((?:Під|Біля|На|У|В|Над|За|Між)\s[^—–.«»:]{2,40}?)\s+[—–]\s+([^.]+)\."
        for match in re.finditer(legend_line, flat):
            name, description = match.group(1).strip(), normalize(match.group(2))
            if name in parsed:
                continue
            if description.startswith("ні "):
                parsed[name] = frozenset()
            else:
                parsed[name] = frozenset(self.detect_parts(description))
        if len(parsed) < 3:
            return False
        self.mitten_parts = parsed
        return True


RESIDENTS_TRIGGER = re.compile(r"(?:живуть|Мешканц\w*(?:\s+рукавички)?(?:\s+є)?)\s*[:–-]*\s*(.+)$")
NAME_WORD = re.compile(r"[А-ЯІЇЄҐ][а-яіїєґ'’]+(?:-[а-яіїєґ'’]+)?")


def residents_from_statute(text: str) -> list[str]:
    """Names from statute lines like "У рукавичці живуть Ведмідь, Сорока і Вивірка." or, after a
    heading, "Мешканці та обрані – Мешканці: Лисичка, Ворона і Дрізд."."""
    for sentence in re.split(r"(?<=\.)\s+", re.sub(r"\s+", " ", text)):
        sentence = sentence.strip(" .–-")
        match = RESIDENTS_TRIGGER.search(sentence)
        if not match or "«" in sentence:
            continue
        listing = sentence.rsplit(":", 1)[1] if ":" in sentence else match.group(1)
        if " і " not in listing:
            continue
        names = [n.strip(" .") for n in re.split(r",\s*|\s+і\s+", listing)]
        names = [n for n in names if NAME_WORD.fullmatch(n)]
        if len(names) >= 2:
            return names
    return []


# Names whose stem changes when declined: "Ведмідь" / "Ведмедя", "Кріт" / "Крота".
IRREGULAR_FORMS = {
    "ведмідь": r"ведмідь|ведмед(?:я|ю|еві|ем|ику|ик|ика|иком)",
    "кріт": r"кріт|крот(?:а|у|ові|ом|е)",
    "горобець": r"горобець|горобц(?:я|ю|еві|ем)",
    "бобер": r"бобер|бобр(?:а|у|ові|ом|е)",
    "дятел": r"дятел|дятл(?:а|у|ові|ом|е)",
    "дрізд": r"дрізд|дрозд(?:а|у|ові|ом|е)",
}


def name_pattern(name: str) -> str:
    """Singular forms of a resident's name. Plural and other words are left out on purpose:
    "кабани" are any boars, "жолуді" are acorns, not Жолудівка-сойка, "їжа" is not Їжак."""
    if "-" in name:  # "Жолудівка-сойка": either part names her, "де живе сойка?"
        return "|".join(name_pattern(part.capitalize()) for part in name.split("-"))
    low = name.lower().translate(APOSTROPHES)  # the statute's "Солов’їха" as the question's "Солов'їха"
    if low in IRREGULAR_FORMS:
        return rf"\b(?:{IRREGULAR_FORMS[low]})\b"
    if low.endswith("а"):  # Сова, Білочка: сови, сові, сову, совою; Білочці
        stem = low[:-1]
        alternation = rf"|{stem[:-1]}ц(?:і)" if stem.endswith("к") else ""
        return rf"\b(?:{stem}(?:а|и|і|у|ою|о){alternation})\b"
    if low.endswith("я"):  # Синиця: синиці, синицю, синицею
        return rf"\b{low[:-1]}(?:я|і|ю|ею|е)\b"
    return rf"\b{low}(?:а|у|ові|ом|е)?\b"  # Кабан, Зайчик, Їжак, Борсук


def plural_forms(name: str) -> set[str]:
    """Forms of a name that are also the plural of a common word: "лисички" are chanterelles,
    "сови" any owls, "синиці" any tits. They name the resident only with a capital letter ("у Лисички")."""
    forms = set()
    for part in name.lower().translate(APOSTROPHES).split("-"):
        if part.endswith("а"):
            forms.add(part[:-1] + "и")
        elif part.endswith("я"):
            forms.add(part[:-1] + "і")
    return forms


def resident_homes(question: str, residents: dict[str, str]) -> dict[str, str]:
    """The residents named in the question and their mittens: {"Сова": "Під дубом"}."""
    text = question.translate(APOSTROPHES)
    homes = {}
    for name, mitten in residents.items():
        plural = plural_forms(name)
        for match in re.finditer(name_pattern(name), text, re.IGNORECASE):
            word = match.group(0)
            if word.lower() not in plural or word[0].isupper():
                homes[name] = mitten
                break
    return homes
