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
