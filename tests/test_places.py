import time

import pytest

from bot.places import CONIFER, DECIDUOUS, RIVER, FOREST_SCOPE, Forest, Scope, resident_homes, residents_from_statute

from .conftest import FOREST_MAP_TEXT


@pytest.mark.parametrize(
    "question, mitten",
    [
        ("скільки внесок під дубом?", "Під дубом"),
        ("які гриби біля сосон?", "У соснах"),
        ("Хто чергує в ліщині?", "У ліщині"),
        ("правила рукавички під ялиною", "Під ялиною"),
        ("коли збори біля бобрової греблі?", "Біля бобрової греблі"),
        ("Що можна їсти біля мурашника?", "Біля мурашника"),
        ("внески на узліссі", "На узліссі"),
        ("хто живе під вербою?", "Під вербою"),
    ],
)
def test_detects_mitten_in_question(forest, question, mitten):
    assert forest.detect_mittens(question) == [mitten]


def test_general_questions_name_no_mitten(forest):
    assert forest.detect_mittens("що робити при пожежі?") == []
    assert forest.detect_mittens("коли бобри відкривають греблю?") == []


def test_plural_spruces_mean_conifer_forest(forest):
    assert forest.detect_mittens("гриби під ялинами") == []
    assert forest.detect_parts("гриби під ялинами") == [CONIFER]


def test_advice_applies_by_map(forest):
    conifer, river, deciduous = Scope("part", CONIFER), Scope("part", RIVER), Scope("part", DECIDUOUS)
    assert forest.applies(conifer, mitten="У соснах")
    assert not forest.applies(conifer, mitten="Під дубом")
    # "Під вербою" belongs to two parts, "На узліссі" to none.
    assert forest.applies(river, mitten="Під вербою") and forest.applies(deciduous, mitten="Під вербою")
    assert not any(forest.applies(s, mitten="На узліссі") for s in (conifer, river, deciduous))
    assert forest.applies(FOREST_SCOPE, mitten="На узліссі")
    assert forest.applies(Scope("mitten", "Під дубом"), mitten="Під дубом")
    assert not forest.applies(Scope("mitten", "У ліщині"), mitten="Під дубом")


@pytest.mark.parametrize(
    "title, head, expected",
    [
        ("Членські внески рукавички Під дубом", "Шановні мешканці...", Scope("mitten", "Під дубом")),
        ("Небезпеки хвойного лісу", "Хвойний ліс гостинний", Scope("part", CONIFER)),
        ("Грибний путівник біля річки", "", Scope("part", RIVER)),
        ("Броди і коли бобри відкривають греблю", "Ця пам'ятка для мешканців біля річки.", Scope("part", RIVER)),
        ("Коли падають жолуді й горіхи", "Для мешканців листяного лісу.", Scope("part", DECIDUOUS)),
        ("Як не заблукати серед однакових сосен", "Ця порада для мешканців хвойного лісу.", Scope("part", CONIFER)),
        ("Карта лісу", FOREST_MAP_TEXT, FOREST_SCOPE),
        ("Що робити при пожежі", "Ця пам'ятка для всіх мешканців лісу, незалежно від того, де", FOREST_SCOPE),
        ("Календар сезонів", "Цей календар для всіх мешканців лісу.", FOREST_SCOPE),
    ],
)
def test_archive_scope(forest, title, head, expected):
    assert forest.archive_scope(title, head) == expected


def test_map_is_read_from_document():
    forest = Forest(mitten_parts={})
    assert forest.update_from_map(FOREST_MAP_TEXT)
    assert forest.mitten_parts["Під вербою"] == {DECIDUOUS, RIVER}
    assert forest.mitten_parts["На узліссі"] == frozenset()
    assert forest.mitten_parts["Біля мурашника"] == {CONIFER}
    assert len(forest.mittens) == 8


def test_updated_map_changes_rules():
    forest = Forest()
    forest.update_from_map(FOREST_MAP_TEXT.replace("Біля мурашника — хвойний ліс.", "Біля мурашника — листяний ліс."))
    assert forest.applies(Scope("part", DECIDUOUS), mitten="Біля мурашника")
    assert not forest.applies(Scope("part", CONIFER), mitten="Біля мурашника")


def test_plural_oaks_and_water_mean_forest_parts(forest):
    assert forest.detect_mittens("білі гриби під дубами") == []
    assert forest.detect_parts("білі гриби під дубами") == [DECIDUOUS]
    assert forest.detect_parts("що їсти біля води") == [RIVER]


@pytest.mark.parametrize("question", [
    "які гриби біля сосон?", "що росте серед сосен?", "через скільки днів йти по гриби в соснах?", "гриби між соснами",
    "чи можна лазити на сосну?", "що росте під сосною?", "біля якої сосни нора?", "сосна біля входу",
])
def test_pine_tree_names_the_mitten(forest, question):
    assert forest.detect_mittens(question) == ["У соснах"]
    assert forest.detect_parts(question) == []


@pytest.mark.parametrize("question, part", [
    # Mishka (30 Oct, 14:18): a forest part asked gets the part's advice, not its mittens'.
    ("Що їсти в сосновому лісі?", CONIFER),
    ("гриби соснового лісу", CONIFER),
    ("сосновий ліс восени", CONIFER),
    ("що робити в ялиновому лісі?", CONIFER),
    ("у сосновому бору багато маслюків", CONIFER),
    ("Що їсти серед ялин?", CONIFER),
    ("гриби між ялинок", CONIFER),
    ("рижики під ялинками", CONIFER),
    ("які гриби в ялиннику?", CONIFER),
    ("що росте в сосняку?", CONIFER),
    ("гриби у бору", CONIFER),
    ("сосновий бір", CONIFER),
    ("що їсти в дубовому лісі?", DECIDUOUS),
    ("дубовий ліс", DECIDUOUS),
    ("що росте в дубняку?", DECIDUOUS),
    ("гриби в діброві", DECIDUOUS),
    ("що їсти в березняку?", DECIDUOUS),
    ("у дубовому гаю", DECIDUOUS),
    ("соснові бори", CONIFER),
    ("гриби соснових борів", CONIFER),
    ("у березових гаях", DECIDUOUS),
    ("за дубовим гаєм", DECIDUOUS),
    ("дубово-грабовий ліс", DECIDUOUS),
])
def test_forest_words_name_the_part_not_a_mitten(forest, question, part):
    assert forest.detect_mittens(question) == []
    assert forest.detect_parts(question) == [part]


@pytest.mark.parametrize("question", [
    "порада для соснового й дубового лісу", "що їсти в сосново-дубовому лісі?", "у дубово-сосновому лісі",
    "мішаний сосново-березовий ліс",
    # The usual name of a mixed forest, and mixed forests that start with "хвойно-" or "листяно-".
    "хвойно-листяний ліс", "у хвойно-дубовому лісі", "для листяно-соснового лісу", "хвойно-березового лісу",
])
def test_mixed_forest_names_both_parts(forest, question):
    assert forest.detect_parts(question) == [CONIFER, DECIDUOUS]


@pytest.mark.parametrize("question", [
    "де збирати соснові шишки?", "кошик для соснових шишок", "соснові шишки в лісі", "дубовий стіл",
    "де шукати дубові жолуді?", "вербові котики", "вибір шишок", "збір грибів", "варення з порічки",
    "що робить Борсук?", "соснове борошно", "мед із дубової борті", "дубовий борщ", "дубова гайка",
    "сосново-дубовий стіл", "що їсть борів?",
    # A forest word must be the forest itself, and spruces as things are no place.
    "дубовий лісник приходить щосуботи", "соснова лісопилка", "гілки ялин для утеплення", "не рубайте ялинок",
    "шишки ялин", "Що робити з ялинками після свята?", "Як прикрасити рукавичку ялинками?", "Чи у ялин є шишки?",
    "Ялинкам потрібен сніг?",
])
def test_things_made_of_trees_are_no_place(forest, question):
    assert forest.detect_mittens(question) == []
    assert forest.detect_parts(question) == []


def test_one_spruce_is_still_the_mitten(forest):
    assert forest.detect_mittens("гриби біля ялинки") == ["Під ялиною"]
    assert forest.detect_parts("гриби біля ялинки") == []


@pytest.mark.parametrize("question, rest", [
    ("Що їсти в сосновому лісі?", "що їсти ?"),
    ("що їсти серед ялин?", "що їсти ?"),
    ("гриби між ялинок", "гриби"),
    ("які гриби в ялиннику?", "які гриби ?"),
    ("що росте у бору?", "що росте ?"),
    ("що росте в дубняку?", "що росте ?"),
    ("жолуді в діброві", "жолуді"),
    ("що їсти в дубово-сосновому лісі?", "що їсти ?"),
    # Both forests of a conjoined name go, whichever comes first.
    ("що їсти у дубовому й сосновому лісі?", "що їсти ?"),
    ("що їсти у сосновому й дубовому лісі?", "що їсти ?"),
    ("коли утеплювати під дубом у хвойному лісі?", "коли утеплювати ?"),
    ("де збирати соснові шишки?", "де збирати соснові шишки?"),
])
def test_strip_places_keeps_the_topic(forest, question, rest):
    assert " ".join(forest.strip_places(question).split()) == rest


def test_residents_from_statutes():
    assert residents_from_statute("Мешканці. У рукавичці живуть Ведмідь, Сорока і Вивірка.") == ["Ведмідь", "Сорока", "Вивірка"]
    # The real "Біля мурашника" statute has a heading right before the list.
    assert residents_from_statute("Мешканці та обрані – Мешканці: Лисичка, Ворона і Дрізд.") == ["Лисичка", "Ворона", "Дрізд"]
    assert residents_from_statute("– Мешканці рукавички: Кабан, Сова і Жолудівка-сойка.") == ["Кабан", "Сова", "Жолудівка-сойка"]
    assert residents_from_statute("Мешканці рукавички -- Їжак, Кріт і Перепілка.") == ["Їжак", "Кріт", "Перепілка"]
    assert residents_from_statute("Мешканці рукавички «Під дубом» приймають цей статут.") == []


HOMES = {
    "Ведмідь": "У ліщині", "Зайчик": "У соснах", "Жолудівка-сойка": "Під дубом", "Горобець": "Під вербою",
    "Кріт": "На узліссі", "Їжак": "На узліссі", "Перепілка": "На узліссі", "Білочка": "Під ялиною",
    "Вовчик": "У соснах", "Кабан": "Під дубом", "Сова": "Під дубом", "Лисичка": "Біля мурашника",
}


@pytest.mark.parametrize("question, home", [
    ("коли Ведмідь лягає спати?", "У ліщині"),
    ("що їсть ведмедик?", "У ліщині"),
    ("де нора Крота?", "На узліссі"),
    ("що робить Горобець?", "Під вербою"),
    ("кому віддати Білочці горіхи?", "Під ялиною"),
    ("де живе Зайчик?", "У соснах"),
    ("коли Сова спить?", "Під дубом"),
    ("де живе сойка?", "Під дубом"),
    ("що робить Жолудівка-сойка?", "Під дубом"),
])
def test_resident_named_in_question(question, home):
    assert set(resident_homes(question, HOMES).values()) == {home}


@pytest.mark.parametrize("question", [
    "коли падають жолуді?", "де росте горобина?", "хто платить, крім дорослих?", "яка їжа є взимку?",
    "перед ким звітує скарбник?", "де шукати білого гриба?", "чи можна їсти вовчі ягоди?", "де ходять кабани?",
    "коли толока?", "що таке совість?",
])
def test_words_that_only_look_like_names(question):
    assert resident_homes(question, HOMES) == {}


@pytest.mark.parametrize("question, expected", [
    ("Де шукати лисички?", {}),  # chanterelles
    ("де живуть сови?", {}),  # any owls
    ("Де нора Лисички?", {"Лисичка": "Біля бобрової греблі"}),
    ("що їсть лисичка?", {"Лисичка": "Біля бобрової греблі"}),
])
def test_plural_of_a_common_word_is_not_the_resident(question, expected):
    homes = {"Лисичка": "Біля бобрової греблі", "Сова": "Під дубом"}
    assert resident_homes(question, homes) == expected


def test_long_words_do_not_stall_the_place_search(forest):
    # One long run of letters once took seconds: the hyphenated first half was looked for at every letter.
    started = time.monotonic()
    forest.strip_places("а" * 5000)
    forest.strip_places("Що їсти під дубом? " + "а" * 4070)
    assert time.monotonic() - started < 0.5
