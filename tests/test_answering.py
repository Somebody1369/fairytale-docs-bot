import io
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import docx
import pytest

from bot import texts
from bot.answering import (ANSWER_SYSTEM, DOC_CHARS, PICK_SECONDS, PICK_SYSTEM, PICK_TOKENS, PLACE_UNKNOWN_RULE,
                           QUESTION_SECONDS, SPARE_TOKENS, Answerer, by_number, calendar_need, calendar_note, clean_text,
                           prompt_chars)
from bot.extract import extract_text
from bot.llm import LLM, LLMUnavailable
from bot.places import CONIFER, DECIDUOUS, Scope
from bot.storage import Store

from .conftest import NOW, FakeLLM, add_animal, add_archive

TZ = ZoneInfo("Europe/Kyiv")


@pytest.fixture
def forest_docs(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом",
                "Внесок 4 жолуді з дорослого звіра на місяць. Здавати до 10 числа.")
    add_archive(store, forest, "Членські внески рукавички У соснах", "Внесок 3 шишки на місяць.")
    add_archive(store, forest, "Як утеплити рукавичку в хвойному лісі",
                "Порада для тих, чия рукавичка стоїть у хвойному лісі. Утеплюйте рукавичку в жовтні.")
    add_archive(store, forest, "Календар сезонів",
                "Цей календар для всіх мешканців лісу. Рукавички утеплюють у листопаді.")
    add_archive(store, forest, "Що робити при пожежі",
                "Ця пам'ятка для всіх мешканців лісу. Довгий безперервний крик означає пожежу. Біжіть до річки.")
    return store


async def test_place_dependent_question_without_place_asks(forest_docs, forest):
    llm = FakeLLM({"needs_place": True, "found": False})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок?", NOW)
    assert reply.kind == "need_place"
    system, prompt = llm.prompts[0]
    assert "Місце в питанні не вказано" in system
    # The model sees the dues of different mittens side by side and can tell they differ.
    assert "Під дубом" in prompt and "У соснах" in prompt


async def test_model_may_answer_without_place_when_it_says_so(forest_docs, forest):
    llm = FakeLLM({"needs_place": False, "found": True, "answer": "4 жолуді.", "sources": [1]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок?", NOW)
    assert reply.kind == "answer"


async def test_whole_forest_question_is_answered_without_place(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": "Біжіть до річки.", "sources": [1]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("що робити при пожежі?", NOW)
    assert reply.kind == "answer"
    assert 'Джерело: "Що робити при пожежі" (Лісовий архів, місце: увесь ліс)' in reply.text


async def test_only_documents_for_the_named_place_reach_the_model(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": "Утеплюйте в жовтні.", "sources": [1, 2]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("коли утеплювати рукавичку у соснах?", NOW)
    prompt = llm.prompts[0][1]
    assert "хвойному лісі" in prompt and "Календар сезонів" in prompt
    assert "Під дубом" not in prompt
    # The more specific (forest part) document goes first.
    assert prompt.index("Як утеплити рукавичку в хвойному лісі") < prompt.index("Календар сезонів")
    assert "Сьогодні: середа, 28 жовтня 2026." in prompt
    # "Коли" asks for a time of year: the first weekdays of the months, not every day of them.
    assert "Перші дні тижня, жовтень 2026: перший четвер жовтня = 1 жовтня" in prompt and "Жовтень 2026: 1 чт" not in prompt
    assert reply.text.startswith("Утеплюйте в жовтні.")
    assert "Джерела:" in reply.text


async def test_mitten_chosen_by_button_is_used(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок?", NOW, mitten="Під дубом")
    assert "Членські внески рукавички Під дубом" in llm.prompts[0][1]
    assert "У соснах" not in llm.prompts[0][1].split("Документи:")[1]
    assert 'Джерело: "Членські внески рукавички Під дубом" (Лісовий архів, місце: рукавичка "Під дубом")' in reply.text


async def test_model_without_answer_gives_honest_not_found(forest_docs, forest):
    llm = FakeLLM({"found": False, "answer": "", "sources": []})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок під дубом за малечу?", NOW)
    assert reply.kind == "not_found" and reply.text == texts.NOT_FOUND


async def test_answer_without_sources_is_not_trusted(forest_docs, forest):
    # Two documents could be the source: the bot asks once more, then gives up rather than guess.
    no_sources = {"found": True, "answer": "Щось вигадане.", "sources": []}
    llm = FakeLLM(no_sources, dict(no_sources))
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("коли утеплювати рукавичку у соснах?", NOW)
    assert reply.kind == "not_found" and len(llm.prompts) == 2


async def test_forgotten_source_is_asked_again(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": "У жовтні.", "sources": []},
                  {"found": True, "answer": "У жовтні.", "sources": ["1"]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("коли утеплювати рукавичку у соснах?", NOW)
    assert reply.kind == "answer" and 'Джерело: "Як утеплити рукавичку в хвойному лісі"' in reply.text
    # The second try says what was missing: the very same request would mostly bring the very same reply.
    assert 'у "sources" немає номерів документів' in llm.prompts[1][1]


async def test_flags_and_numbers_written_as_strings_or_odd_values(forest_docs, forest):
    # "false" as a string is no, as for "found".
    llm = FakeLLM({"needs_place": "false", "found": True, "answer": "Для всіх однаково.", "sources": [1]})
    assert (await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок?", NOW)).kind == "answer"
    # A superscript is no number (int() refuses "²"), and JSON true is no document number.
    llm = FakeLLM({"found": True, "answer": "4 жолуді.", "sources": ["²", True, "[1]"]}, picks=[{"ids": ["²", " 1 "]}])
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок під дубом?", NOW)
    assert reply.kind == "answer" and reply.text.endswith('місце: рукавичка "Під дубом")')
    assert by_number([True, "²", 0, 3, 1.0, None, " 2 ", "[1]", 1], ["a", "b"]) == ["b", "a"]


async def test_an_object_instead_of_the_answer_text_is_no_answer(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": {"text": "4 жолуді."}, "sources": [1]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок під дубом?", NOW)
    assert reply.kind == "not_found" and reply.text == texts.NOT_FOUND
    # Of a list, only its texts are the answer.
    llm = FakeLLM({"found": True, "answer": ["4 жолуді.", {"note": "x"}], "sources": [1]})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок під дубом?", NOW)
    assert reply.text.startswith("4 жолуді.\n\nДжерело:")


async def test_answers_follow_documents_sent_and_deleted_after_the_first_question(store, forest):
    # The search index is kept between questions: a document sent since must reach the prompt, and one deleted
    # since must leave it.
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    llm = FakeLLM(*({"found": True, "answer": "4 жолуді.", "sources": [1]} for _ in range(3)))
    answerer = Answerer(store, forest, llm, TZ)
    await answerer.ask("Скільки внесок під дубом?", NOW)
    doc, _ = add_animal(store, "Нові внески", "Внесок під дубом тепер 5 жолудів.", Scope("mitten", "Під дубом"))
    await answerer.ask("Скільки внесок під дубом?", NOW)
    assert "Нові внески" in llm.prompts[1][1]
    store.delete_animal_document(doc.id, doc.author_id)
    await answerer.ask("Скільки внесок під дубом?", NOW)
    assert "Нові внески" not in llm.prompts[2][1]


async def test_document_deleted_while_the_model_answers_is_not_cited(store, forest):
    # The documents are chosen once, and the model may take up to 45 seconds to answer.
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    doc, _ = add_animal(store, "Нові внески", "Внесок під дубом тепер 5 жолудів.", Scope("mitten", "Під дубом"))

    class DeletingLLM(FakeLLM):
        async def json(self, system, user, *args, **options):
            if system != PICK_SYSTEM:
                store.delete_animal_document(doc.id, doc.author_id)
            return await super().json(system, user, *args, **options)

    llm = DeletingLLM({"found": True, "answer": "4 жолуді на місяць.",
                       "other": "Звір Зайчик пише, що внесок тепер 5 жолудів.", "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert "Нові внески" in llm.prompts[0][1] and reply.kind == "answer"
    # The advice that gives way retold the deleted document, so it goes with it.
    assert reply.text == ('4 жолуді на місяць.\n\nДжерело: "Членські внески рукавички Під дубом" (Лісовий архів, '
                          'місце: рукавичка "Під дубом")')
    # The answer comes from its first source: with that one gone, it is not given, though the archive's stays.
    doc, _ = add_animal(store, "Внески з листопада", "Внесок під дубом з листопада 5 жолудів.",
                        Scope("mitten", "Під дубом"))
    llm = DeletingLLM({"found": True, "answer": "З листопада 5 жолудів.", "sources": [2, 1]})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert '"Внески з листопада"' in llm.prompts[0][1] and reply.kind == "not_found" and reply.text == texts.NOT_FOUND


async def test_animal_source_shows_author_and_time(store, forest):
    add_animal(store, "Гриби під ялинами", "Під ялинами ростуть рижики після дощу.", Scope("part", CONIFER),
               name="Білочка", when=NOW - timedelta(hours=3))
    llm = FakeLLM({"found": True, "answer": "Рижики після дощу.", "sources": [1]})
    reply = await Answerer(store, forest, llm, TZ).ask("які гриби ростуть під ялиною?", NOW)
    assert ('Джерело: "Гриби під ялинами", надіслано 28.10.2026 о 11:00, автор: Білочка, місце: хвойний ліс'
            in reply.text)
    assert "надіслав звір Білочка" in llm.prompts[0][1]


async def test_no_matching_document_asks_model_to_pick_by_title(forest_docs, forest):
    llm = FakeLLM(picks=[{"ids": []}])
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("як навчитися літати під дубом?", NOW)
    assert reply.kind == "not_found"
    assert len(llm.pick_prompts) == 1 and llm.prompts == []  # title picking only, no answer call


async def test_general_advice_is_not_given_when_places_differ(forest_docs, forest):
    # The forest calendar says November, the conifer forest says October: ask which mitten.
    llm = FakeLLM({"needs_place": True})
    reply = await Answerer(forest_docs, forest, llm, TZ).ask("коли утеплювати рукавичку?", NOW)
    assert reply.kind == "need_place"
    prompt = llm.prompts[0][1]
    assert "Календар сезонів" in prompt and "Як утеплити рукавичку в хвойному лісі" in prompt


async def test_whole_forest_question_does_not_mention_place_rule(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": "Біжіть до річки.", "sources": [1]})
    await Answerer(forest_docs, forest, llm, TZ).ask("що робити при пожежі?", NOW)
    assert "Місце в питанні не вказано" not in llm.prompts[0][0]


async def test_named_resident_gives_the_place(store, forest):
    add_archive(store, forest, "Статут ОСМД У ліщині", "Мешканці. У рукавичці живуть Ведмідь, Сорока і Вивірка.")
    add_archive(store, forest, "Правила співжиття в рукавичці У ліщині", "З листопада Ведмідь лягає в сплячку.")
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Хто спить узимку, лягає в грудні.")
    llm = FakeLLM({"found": True, "answer": "З листопада.", "sources": [1]})
    reply = await Answerer(store, forest, llm, TZ).ask("коли Ведмідь лягає спати?", NOW)
    assert reply.kind == "answer"
    # The place comes from the statute before the search, so the model never has to ask for it.
    system, prompt = llm.prompts[0]
    assert 'рукавичка "У ліщині"' in prompt and "за звіром Ведмідь" in prompt
    assert "Місце в питанні не вказано" not in system
    assert "Правила співжиття в рукавичці У ліщині" in prompt


async def test_resident_question_finds_the_mittens_own_documents(store, forest):
    add_archive(store, forest, "Статут ОСМД Під дубом", "У рукавичці живуть Кабан, Сова і Їжак.")
    add_archive(store, forest, "Чергування в рукавичці Під дубом", "Вночі чергує Сова: вона стежить за входом.")
    add_archive(store, forest, "Чергування в рукавичці У соснах", "Вночі вартує Пугач, а вдень Дятел.")
    llm = FakeLLM({"found": True, "answer": "Сова чергує вночі.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Що робить Сова вночі?", NOW)
    prompt = llm.prompts[0][1].split("Документи:")[1]
    assert "Чергування в рукавичці Під дубом" in prompt and "У соснах" not in prompt


async def test_mitten_advice_is_not_pushed_out_by_stronger_general_matches(store, forest):
    # "Коли збирати жолуді під дубом?": forest and part documents match the words better,
    # yet the mitten's own documents still reach the model.
    add_archive(store, forest, "Коли падають жолуді й горіхи",
                "Порада для листяного лісу. Жолуді збирати у вересні, жолуді сушити, жолуді зберігати.")
    add_archive(store, forest, "Що їсти в листяному лісі", "Порада для листяного лісу. Жолуді, горіхи, жолуді.")
    add_archive(store, forest, "Календар сезонів", "Для всіх мешканців лісу. Жолуді збирати восени, коли жолуді падають.")
    add_archive(store, forest, "Що робити при пожежі", "Для всіх мешканців лісу. Не збирати нічого, біжіть до річки.")
    add_archive(store, forest, "Небезпеки листяного лісу", "Порада для листяного лісу. Жолуді на стежці слизькі.")
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    add_archive(store, forest, "Чергування в рукавичці Під дубом", "Хто чергує, той і збирати жолуді до комори йде.")
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під дубом",
                "Жолуді під нашим дубом достигають у другій половині вересня.")
    llm = FakeLLM({"found": True, "answer": "У другій половині вересня.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Коли збирати жолуді під дубом?", NOW)
    prompt = llm.prompts[0][1]
    assert "Чим поживитись поруч із рукавичкою Під дубом" in prompt
    # The mitten's documents go first: they are the most specific.
    assert prompt.index("Чим поживитись") < prompt.index("Коли падають жолуді")


async def test_mitten_advice_is_found_among_many_better_matches(store, forest):
    # Search kept only the 60 best matches before grouping them by place, and the mitten's own rules, which
    # mention acorns once, fell out behind the deciduous forest's guides.
    for i in range(61):
        add_animal(store, f"Жолуді {i}", "Жолуді збирайте, жолуді сушіть, жолуді зберігайте.", Scope("part", DECIDUOUS),
                   author_id=i)
    add_archive(store, forest, "Правила рукавички Під дубом", "Жолуді до комори носить Кабан. " + FILLER * 3)
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Коли збирати жолуді під дубом?", NOW)
    assert "Правила рукавички Під дубом" in llm.prompts[0][1]


async def test_every_place_keeps_its_best_match_beside_the_title_pick(store, forest):
    # "Коли під вербою збирають гриби і горіхи?": four deciduous and whole-forest picks and the mitten's own
    # documents filled the prompt, and the river part's guide was left out.
    add_archive(store, forest, "Грибний путівник біля річки", "Цей путівник для тих, хто живе біля річки. "
                                                               "Опеньки ростуть на старих вербах.")
    add_archive(store, forest, "Грибний путівник листяного лісу", "Любі сусіди з листяного лісу! "
                                                                   "Білі гриби тримаються старих дубів.")
    add_archive(store, forest, "Коли падають жолуді й горіхи", "Для мешканців листяного лісу. "
                                                                "Горіхи збирайте у вересні.")
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Гриби збирають у вересні.")
    add_archive(store, forest, "Їстівне й отруйне", "Ця пам'ятка для всіх мешканців лісу. Збирайте лише знайомі гриби.")
    for title in ("Чим поживитись поруч із рукавичкою Під вербою", "Статут ОСМД Під вербою",
                  "Стежки до сусідів від рукавички Під вербою"):
        add_archive(store, forest, title, "Горіхи й гриби шукайте з сусідами.")
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": [2, 3, 4, 5]}])
    await Answerer(store, forest, llm, TZ).ask("Коли під вербою збирають гриби і горіхи?", NOW)
    assert "Грибний путівник біля річки" in llm.prompts[0][1]


async def test_newest_advice_for_the_place_is_not_cut(store, forest):
    # Four animals' documents on the dues of one mitten: for one place the newer advice wins, so the newest one
    # reaches the model, though it mentions the dues only once.
    for i in range(3):
        add_animal(store, f"Внески {i}", "Внесок 4 жолуді. Внесок здають Кабанові. Внески до 10 числа.",
                   Scope("mitten", "Під дубом"), author_id=i, when=NOW - timedelta(days=3 - i))
    add_animal(store, "Що вирішили на зборах", "На зборах біля дупла Сови говорили про зиму, комору й нічні "
               "чергування. Тепер внесок 6 жолудів.", Scope("mitten", "Під дубом"), author_id=9)
    llm = FakeLLM({"found": True, "answer": "6 жолудів.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert "Що вирішили на зборах" in llm.prompts[0][1]


async def test_newest_animal_advice_takes_no_archive_documents_seat(store, forest):
    # The archive beats animals for one place: the newest animal's weak match comes beside it, not instead of it.
    for i in range(2):
        add_animal(store, f"Жолуді {i}", "Жолуді збирайте. Жолуді сушіть. Жолуді зберігайте.",
                   Scope("mitten", "Під дубом"), author_id=i, when=NOW - timedelta(days=3))
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під дубом", "Жолуді збирайте після першого морозу.")
    add_animal(store, "Що вирішили на зборах", "На зборах біля дупла Сови говорили про зиму, комору й нічні "
               "чергування. Жолуді теж згадали.", Scope("mitten", "Під дубом"), author_id=9)
    llm = FakeLLM({"found": True, "answer": "Після першого морозу.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Коли збирати жолуді під дубом?", NOW)
    assert "Чим поживитись поруч із рукавичкою Під дубом" in llm.prompts[0][1]
    assert "Що вирішили на зборах" in llm.prompts[0][1]


async def test_archive_document_edited_last_is_no_newer_advice(store, forest):
    # An archive file edited on Drive later is another document, not a newer version of the others: the mitten's
    # rules, matching once, took the seat of "Чим поживитись поруч із рукавичкою Під дубом".
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді. Жолуді здають Кабанові.")
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під дубом", "Жолуді збирайте після першого морозу.")
    add_archive(store, forest, "Чергування в рукавичці Під дубом", "Черговий стежить, щоб жолуді в коморі не намокли.")
    add_archive(store, forest, "Правила співжиття в рукавичці Під дубом", "Не несіть у рукавичку зайвого, жолуді теж. "
                + FILLER * 3, modified=NOW - timedelta(days=1))
    llm = FakeLLM({"found": True, "answer": "Після першого морозу.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Коли збирати жолуді під дубом?", NOW)
    assert "Чим поживитись поруч із рукавичкою Під дубом" in llm.prompts[0][1]
    assert "Правила співжиття" not in llm.prompts[0][1]


async def test_mitten_matches_keep_their_seats_beside_the_title_pick(store, forest):
    # Four picks from the forest part and the whole forest, with each place's best match, filled the prompt, and
    # the mitten's own second and third matches were left out.
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді. Жолуді здають Кабанові.")
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під дубом", "Жолуді збирайте після першого морозу.")
    add_archive(store, forest, "Чергування в рукавичці Під дубом", "Черговий стежить, щоб жолуді в коморі не намокли.")
    add_archive(store, forest, "Коли падають жолуді й горіхи", "Для мешканців листяного лісу. Жолуді падають у вересні.")
    add_archive(store, forest, "Що їсти в листяному лісі", "Для мешканців листяного лісу. Горіхи, ягоди й гриби.")
    add_archive(store, forest, "Як утеплити рукавичку в листяному лісі", "Для мешканців листяного лісу. Сухим листям.")
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Жолуді збирають у вересні.")
    add_archive(store, forest, "Осінній путівник лісу", "Цей путівник для всіх мешканців лісу. Готуйтеся до зими.")
    add_archive(store, forest, "Їстівне й отруйне", "Ця пам'ятка для всіх мешканців лісу. Збирайте знайомі гриби.")
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": [5, 6, 8, 9]}])
    await Answerer(store, forest, llm, TZ).ask("Коли збирати жолуді під дубом?", NOW)
    prompt = llm.prompts[0][1]
    assert "Чим поживитись поруч із рукавичкою Під дубом" in prompt and "Чергування в рукавичці Під дубом" in prompt
    assert "Членські внески рукавички Під дубом" in prompt and "Коли падають жолуді" in prompt


async def test_title_pick_finds_advice_with_other_words(store, forest):
    # "Що їсти" shares no word with "Чим поживитись": the model picks it by the title.
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під вербою", "Верболіз, кора, ягоди калини.")
    add_archive(store, forest, "Що їсти біля річки", "Ця пам'ятка для мешканців біля річки. Що їсти: рогіз, очерет.")
    llm = FakeLLM({"found": True, "answer": "Кора й ягоди калини.", "sources": [1]}, picks=[{"ids": [1, 2]}])
    await Answerer(store, forest, llm, TZ).ask("Що їсти під вербою?", NOW)
    assert "1. Чим поживитись поруч із рукавичкою Під вербою (рукавичка \"Під вербою\")" in llm.pick_prompts[0]
    assert "Чим поживитись поруч із рукавичкою Під вербою" in llm.prompts[0][1]


async def test_two_animal_documents_for_the_same_place_both_reach_the_model(store, forest):
    add_archive(store, forest, "Що робити при пожежі",
                "Для всіх мешканців лісу. При пожежі збираємося біля бобрової греблі. Пожежа: кричати, пожежа.")
    add_animal(store, "Пожежа", "При пожежі збираємося під дубом, пожежа.", Scope("mitten", "Під дубом"),
               when=NOW - timedelta(days=5))
    add_animal(store, "Новий збір", "Тепер при пожежі збираємося на галявині.", Scope("mitten", "Під дубом"),
               author_id=2, name="Сова", when=NOW - timedelta(days=1))
    llm = FakeLLM({"found": True, "answer": "На галявині.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Де збиратися при пожежі під дубом?", NOW)
    prompt = llm.prompts[0][1]
    assert "Новий збір" in prompt and '"Пожежа"' in prompt
    assert prompt.index("Новий збір") < prompt.index('"Пожежа"')  # newer first


async def test_only_date_questions_get_longer_thinking(forest_docs, forest):
    llm = FakeLLM({"found": True, "answer": "4 жолуді.", "sources": [1]},
                  {"found": True, "answer": "До 10 числа.", "sources": [1]})
    answerer = Answerer(forest_docs, forest, llm, TZ)
    await answerer.ask("скільки внесок під дубом?", NOW)
    await answerer.ask("до якого числа здавати внесок під дубом?", NOW)
    assert llm.reasoning == ["low", "medium"]


class BudgetLLM(FakeLLM):
    """FakeLLM that also records each call's token budget: Groq counts it against the minute's limit."""

    def __init__(self, *answers, **options):
        super().__init__(*answers, **options)
        self.budgets: list[tuple[str, int]] = []

    async def json(self, system, user, max_tokens=700, reasoning="low", fast=False, deadline=None):
        self.budgets.append(("pick" if system == PICK_SYSTEM else "answer", max_tokens))
        return await super().json(system, user, max_tokens, reasoning, fast, deadline)


@pytest.mark.parametrize("question, reasoning, budget, firsts, days", [
    # A plain fact gets only today: counting days after rain is no date.
    ("Через скільки днів після дощу йти по гриби в соснах?", "low", 1500, False, False),
    ("скільки внесок під дубом?", "low", 1500, False, False),
    # A time of year gets the first weekdays of last, this and next month.
    ("коли утеплювати рукавичку у соснах?", "low", 1500, True, False),
    # Counting from today gets every day of this and next month and longer thinking.
    ("до якого числа здавати внесок під дубом?", "medium", 2000, True, True),
    ("Чи вже час утеплювати рукавичку у соснах?", "medium", 2000, True, True),
    ("Чи можна вже утеплювати рукавичку у соснах?", "medium", 2000, True, True),
])
async def test_calendar_and_thinking_only_as_much_as_the_question_needs(forest_docs, forest, question, reasoning,
                                                                        budget, firsts, days):
    llm = BudgetLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": [1]}])  # a document for any question
    await Answerer(forest_docs, forest, llm, TZ).ask(question, NOW)
    calendar = llm.prompts[0][1].split("\n\nПитання звіра:")[0]
    assert llm.reasoning == [reasoning] and llm.budgets[-1] == ("answer", budget)
    assert calendar.startswith("Сьогодні: середа, 28 жовтня 2026.")
    assert [line.split(":")[0] for line in calendar.splitlines() if line.startswith("Перші дні тижня")] == (
        ["Перші дні тижня, вересень 2026", "Перші дні тижня, жовтень 2026", "Перші дні тижня, листопад 2026"]
        if firsts else [])
    assert ("Жовтень 2026: 1 чт, 2 пт, 3 сб" in calendar and "Листопад 2026: 1 нд" in calendar) == days
    assert "Вересень 2026: 1" not in calendar  # last month only says which date its first weekdays are


@pytest.mark.parametrize("need, questions", [
    ("today", ["Через скільки днів після дощу йти по гриби в соснах?", "Де шукати гриби біля верби?",
               "Куди здати внесок?", "Що дати Кабанові?", "До якої частини лісу належить Під вербою?",
               "Що ще можна їсти під дубом?", "Що їсти серед дубів?", "Скільки внесок на місяць?",
               "як навчитися літати під дубом?", "Що робити терміново при пожежі?",
               # Words that only look like dates: they would cost the longest thinking for nothing.
               "Як не заразитися від Кабана?", "Скільки звірів у численних родинах?", "Чи ще хтось живе під дубом?",
               "Хто звільнений від внеску, в тому числі зайці?", "Як дати знати Сові про пожежу?",
               "Скільки днів після дощу ростуть гриби?", "Скільки залишилось жолудів у коморі?"]),
    ("months", ["Коли осіння толока?", "Коли утеплювати рукавичку на узліссі?", "В якому місяці збирати жолуді?",
                "Що робити взимку під ялиною?", "Яка пора для горіхів?", "Що робити в середині жовтня?"]),
    ("days", ["Чи вже час збирати гриби?", "Чи ще можна збирати гриби?", "Що робити завтра?", "Яка дата толоки?",
              "До якого часу можна збирати гриби?", "Чи працює комора в середу?", "Що робити наступного тижня?",
              "Що було минулої осені?", "Цього року толока вже була?", "Чи не пізно утеплювати рукавичку?",
              "Якого дня толока?", "Комору відкривають у четвер?",
              # Counted from today too, though in other words: "у першу суботу жовтня" needs the calendar's date.
              "Чи можна вже утеплювати рукавичку?", "Можна вже йти на толоку?", "Толока вже відбулася?",
              "Чи толока вже минула?", "Зима вже почалася?", "Чи скоро толока?", "Чи ще не пізно утеплювати рукавичку?",
              "Скільки днів до толоки?", "Через скільки днів толока?", "Скільки ще чекати до толоки?",
              "Скільки ще чекати толоки?", "Скільки лишилося до толоки?", "Скільки часу до зими?", "Які дати толок?",
              "З якої дати збирають жолуді?", "Що робити зараз?", "Яке сьогодні число?"]),
])
def test_calendar_need_follows_the_words_of_the_question(need, questions):
    assert {question: calendar_need(question) for question in questions} == dict.fromkeys(questions, need)


@pytest.mark.parametrize("moment, today, months", [
    (datetime(2026, 12, 10, 12, tzinfo=timezone.utc), "четвер, 10 грудня 2026",
     ["листопад 2026", "грудень 2026", "січень 2027"]),
    # Already 1 January in Kyiv: last month is December of the year before.
    (datetime(2026, 12, 31, 23, tzinfo=timezone.utc), "пʼятниця, 1 січня 2027",
     ["грудень 2026", "січень 2027", "лютий 2027"]),
])
def test_calendar_names_last_month_with_its_year(moment, today, months):
    last, this, following = months
    lines = calendar_note(moment, TZ, "days").splitlines()
    assert lines[0] == f"Сьогодні: {today}."
    assert [line.split(":")[0] for line in lines[1:]] == [
        f"Перші дні тижня, {last}", this.capitalize(), f"Перші дні тижня, {this}", following.capitalize(),
        f"Перші дні тижня, {following}"]
    assert [line.split(":")[0] for line in calendar_note(moment, TZ, "months").splitlines()[1:]] == [
        f"Перші дні тижня, {month}" for month in months]
    assert calendar_note(moment, TZ, "today") == lines[0]
    assert "перший вівторок грудня = 1 грудня" in calendar_note(moment, TZ, "months")


async def test_rule_of_a_date_already_past_this_year_meets_the_calendar(store, forest):
    # On 1 November "Коли осіння толока?" was answered "відбудеться у першу суботу жовтня": the calendar had only
    # November and December, so 3 October could not be seen as past.
    add_archive(store, forest, "Осіння толока", "Для всіх мешканців лісу. Осіння толока буває у першу суботу жовтня.")
    llm = FakeLLM({"found": True, "answer": "Цього року осіння толока вже була, 3 жовтня.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Коли осіння толока?", datetime(2026, 11, 1, 9, tzinfo=timezone.utc))
    system, prompt = llm.prompts[0]
    assert "Сьогодні: неділя, 1 листопада 2026." in prompt and "перша субота жовтня = 3 жовтня" in prompt
    assert llm.reasoning == ["low"]
    assert "пиши, що цього року вона вже була" in system and "за тим самим правилом наступного року" in system
    assert "Про минулу подію ніколи не пиши в майбутньому часі." in system
    assert "якщо вона раніше за сьогодні, пиши в минулому часі, що подія вже була" in system  # rule 5 as it was


async def test_title_pick_asks_for_a_small_budget(forest_docs, forest):
    # The pick is only {"ids": [...]}: a smaller budget leaves more of Groq's minute to the answer.
    llm = BudgetLLM({"found": True, "answer": "4 жолуді.", "sources": [1]})
    await Answerer(forest_docs, forest, llm, TZ).ask("скільки внесок під дубом?", NOW)
    assert PICK_TOKENS == 400 and llm.budgets == [("pick", 400), ("answer", 1500)]
    # No model is skipped as capped below the smallest reply, qwen with its 900 neither; the first model's minute
    # is left to the answer.
    real, asked = LLM("x", ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]), []

    async def create(model, **options):
        asked.append((model, options["max_completion_tokens"]))
        content = '{"ids": [1]}' if len(asked) == 2 else "не JSON"
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    real.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assert await real.json(PICK_SYSTEM, "x", max_tokens=PICK_TOKENS, fast=True) == {"ids": [1]}
    assert asked == [("openai/gpt-oss-20b", 400), ("qwen/qwen3.8-27b", 400)]


# Measured on Groq: these prompts are 2.55 characters a token, and a request is admitted only when the prompt and
# max_completion_tokens together fit the model's 8000 tokens a minute.
def groq_tokens(*texts: str) -> float:
    return sum(map(len, texts)) / 2.55


async def test_title_pick_lists_all_archive_titles_and_the_newest_animals_that_fit(store, forest):
    # 150 animals' titles made a listing of about 20000 characters, and every model refused it.
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    for i in range(150):
        add_animal(store, f"Поради сусідам про запаси горіхів, жолудів і сушених грибів на довгу зиму, частина {i}",
                   "Запаси робіть восени.", Scope("forest"), author_id=i, when=NOW - timedelta(hours=i))
    llm = FakeLLM()
    await Answerer(store, forest, llm, TZ).ask("Що робити взимку?", NOW)
    listing = llm.pick_prompts[0]
    assert groq_tokens(PICK_SYSTEM, listing) + PICK_TOKENS < 8000
    assert groq_tokens(PICK_SYSTEM, listing) * 1.01 < 7000  # qwen counts 1% more and takes 7000 for the prompt alone
    assert "Членські внески рукавички Під дубом" in listing
    assert "частина 0 (" in listing and "частина 149 (" not in listing  # the newest are listed, the oldest left out


def test_prompt_labels_the_level_of_the_advice():
    from bot.answering import ANSWER_SYSTEM
    # Mishka's reference answers (30 October): the more general advice says it is the whole forest's.
    assert '"Загальна порада для всього лісу — …"' in ANSWER_SYSTEM
    assert '"Окремої поради про … для … немає, тож діє загальна порада для всього лісу: …"' in ANSWER_SYSTEM
    # Only where the topic differs by place: whose advice is whose ("У хвойному лісі …").
    assert ('стоїть "так", скажи, для якого місця кожна порада ("У хвойному лісі …")') in ANSWER_SYSTEM
    assert 'Якщо там "ні", цих слів не пиши: порада для всього лісу однакова для всіх' in ANSWER_SYSTEM


async def test_the_bot_tells_the_model_whether_the_topic_differs_by_place(store, forest):
    # Mishka's узлісся answer says it has no warming advice of its own, as other parts have theirs; her hunters'
    # answer goes straight to the cry, which only the whole forest's memo has, though "біля греблі" names a mitten.
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Рукавички утеплюють у листопаді.")
    add_archive(store, forest, "Як утеплити рукавичку в хвойному лісі",
                "Порада для тих, чия рукавичка стоїть у хвойному лісі. Утеплюйте рукавичку в жовтні.")
    add_archive(store, forest, "Як кричати, коли бачиш мисливців", "Це звернення до всіх мешканців лісу. Побачив "
                "мисливців — крикни тричі коротко, як Сорока. Почув постріл — ховайся в гущавину.")
    add_archive(store, forest, "Небезпеки поруч із рукавичкою Під дубом", "Восени біля галявини бувають мисливці.")
    llm = FakeLLM({"found": True, "answer": "У листопаді.", "sources": [1]},
                  {"found": True, "answer": "Крикни тричі коротко.", "sources": [1]},
                  {"found": True, "answer": "Крикни тричі коротко.", "sources": [1]})
    answerer = Answerer(store, forest, llm, TZ)
    await answerer.ask("Коли утеплювати рукавичку на узліссі?", NOW)
    await answerer.ask("Біля греблі мисливці! Як кричати?", NOW)
    await answerer.ask("Як кричати, коли бачиш мисливців?", NOW)
    assert "Поради на цю тему різні в різних місцях: так" in llm.prompts[0][1]
    assert "Поради на цю тему різні в різних місцях: ні" in llm.prompts[1][1]
    assert "Поради на цю тему різні в різних місцях" not in llm.prompts[2][1]  # no place asked, rule 11 is not for it
    # Reworded, the question keeps only "мисливці", which the mitten's dangers mention: still no advice of its own,
    # as the dangers' title is not about hunters.
    llm.answers.append({"found": True, "answer": "Крикни тричі коротко.", "sources": [1]})
    await answerer.ask("Що робити, коли біля греблі мисливці?", NOW)
    assert "Поради на цю тему різні в різних місцях: ні" in llm.prompts[3][1]


async def test_answer_only_from_the_whole_forest_says_so_and_is_taken(store, forest):
    # Mishka: «Окремих порад про утеплення для узлісся немає. Загальна порада для всього лісу — утеплювати в
    # листопаді…». The labelled sentence holds the point, so it is not asked about again.
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Рукавички утеплюють у листопаді.")
    add_archive(store, forest, "Як утеплити рукавичку в хвойному лісі",
                "Порада для тих, чия рукавичка стоїть у хвойному лісі. Утеплюйте рукавичку в жовтні.")
    answer = "Окремої поради про утеплення для узлісся немає, тож діє загальна порада для всього лісу: утеплювати в листопаді."
    llm = FakeLLM({"facts": ["[1, увесь ліс] Рукавички утеплюють у листопаді."], "found": True, "answer": answer,
                   "other": "", "sources": [1]})
    reply = await Answerer(store, forest, llm, TZ).ask("Коли утеплювати рукавичку на узліссі?", NOW)
    assert len(llm.prompts) == 1
    assert reply.text == f'{answer}\n\nДжерело: "Календар сезонів" (Лісовий архів, місце: увесь ліс)'


async def test_mitten_answered_from_its_forest_part_says_whose_advice_it_is(store, forest):
    # "У соснах" has no warming advice of its own: the conifer forest's is named as such, and the calendar's
    # November that says otherwise keeps its own paragraph.
    add_archive(store, forest, "Як утеплити рукавичку в хвойному лісі",
                "Порада для тих, чия рукавичка стоїть у хвойному лісі. Утеплюйте рукавичку в жовтні.")
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Рукавички утеплюють у листопаді.")
    answer = "У хвойному лісі рукавичку утеплюють у жовтні."
    other = "Загальний календар для всього лісу радить утеплювати в листопаді, але для хвойного лісу діє конкретніша порада."
    llm = FakeLLM({"facts": ["[1, частина лісу] Утеплюйте рукавичку в жовтні.",
                             "[2, увесь ліс] Рукавички утеплюють у листопаді."],
                   "found": True, "answer": answer, "other": other, "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Коли утеплювати рукавичку у соснах?", NOW)
    assert len(llm.prompts) == 1 and reply.text.split("\n\n")[:2] == [answer, other]


async def test_general_advice_labelled_in_the_answer_is_no_missing_point_and_no_other(store, forest):
    # Mishka: «У хвойному лісі гриби з’являються на 2–3 день після дощу. Загальна порада для всього лісу — шукати
    # після дощу, коли земля волога, але не хлюпає.»
    add_archive(store, forest, "Грибний путівник хвойного лісу",
                "Порада для тих, чия рукавичка стоїть у хвойному лісі. Гриби з'являються на 2–3 день після дощу.")
    add_archive(store, forest, "Їстівне й отруйне",
                "Ця пам'ятка для всіх мешканців лісу. Шукайте гриби після дощу, коли земля волога, але не хлюпає.")
    facts = ["[1, частина лісу] Гриби з'являються на 2–3 день після дощу.",
             "[2, увесь ліс] Шукайте гриби після дощу, коли земля волога, але не хлюпає."]
    answer = ("У хвойному лісі гриби з’являються на 2–3 день після дощу. Загальна порада для всього лісу — шукати "
              "після дощу, коли земля волога, але не хлюпає.")
    question = "Через скільки днів після дощу йти по гриби в соснах?"
    # The general sentence put in "other" as well is a repeat and goes.
    llm = FakeLLM({"facts": facts, "found": True, "answer": answer, "sources": [1, 2],
                   "other": "Загальна порада для всього лісу — шукати після дощу, коли земля волога."})
    reply = await Answerer(store, forest, llm, TZ).ask(question, NOW)
    assert len(llm.prompts) == 1 and reply.text.startswith(answer + "\n\nДжерела:")
    assert "Грибний путівник хвойного лісу" in reply.text and "Їстівне й отруйне" in reply.text
    # Advice that says otherwise keeps its own paragraph, though it shares the label's words with the answer.
    other = "Загальна порада для всього лісу від Сороки — чекати тиждень, але в хвойному лісі діє конкретніша порада."
    llm = FakeLLM({"facts": facts, "found": True, "answer": answer, "other": other, "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask(question, NOW)
    assert len(llm.prompts) == 1 and reply.text.split("\n\n")[:2] == [answer, other]


async def test_willow_answer_saying_whose_advice_is_whose_is_taken(store, forest):
    # Mishka's "Де шукати гриби біля верби?": each piece of advice says which forest part or the mitten it is for.
    add_archive(store, forest, "Грибний путівник біля річки", "Цей путівник для тих, хто живе біля річки. "
                                                               "Опеньки ростуть на старих вербах і пеньках біля води.")
    add_archive(store, forest, "Грибний путівник листяного лісу", "Любі сусіди з листяного лісу! Білі гриби "
                                                                   "тримаються старих дубів, підберезники галявин біля беріз.")
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під вербою", "Гриби опеньки ростуть на корінні верби.")
    answer = ("Біля самої рукавички опеньки ростуть на корінні верби. У листяному лісі білі гриби — під старими дубами, "
              "підберезники — на галявинах біля беріз. Біля річки опеньки — на старих вербах і пеньках біля води.")
    llm = FakeLLM({"found": True, "answer": answer, "sources": [1, 2, 3]})
    reply = await Answerer(store, forest, llm, TZ).ask("Де шукати гриби біля верби?", NOW)
    assert len(llm.prompts) == 1 and reply.text.startswith(answer + "\n\nДжерела:")


def test_newer_archive_document_wins_among_equals():
    from bot.answering import ANSWER_SYSTEM
    assert "Серед документів архіву, як і серед документів від звірів, важливіший новіший" in ANSWER_SYSTEM


async def test_animal_documents_of_one_day_carry_the_time(store, forest):
    add_animal(store, "Збір", "При пожежі збираємося під дубом.", Scope("mitten", "Під дубом"),
               when=NOW - timedelta(hours=3))
    add_animal(store, "Новий збір", "При пожежі тепер збираємося на галявині.", Scope("mitten", "Під дубом"),
               author_id=2, name="Сова", when=NOW - timedelta(hours=1))
    llm = FakeLLM({"found": True, "answer": "На галявині.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Де збиратися при пожежі під дубом?", NOW)
    prompt = llm.prompts[0][1]
    assert "надіслав звір Сова, 28 жовтня 2026 о 13:00" in prompt and "надіслав звір Зайчик, 28 жовтня 2026 о 11:00" in prompt


async def test_guide_section_weighs_as_the_whole_guide(store, forest):
    # Mishka: a section about "Під дубом" in a whole-forest guide is still whole-forest advice,
    # so the deciduous forest's own advice comes first. A model that puts the guide first is asked again.
    add_archive(store, forest, "Жолуді листяного лісу", "Порада для листяного лісу. Жолуді збирайте в жовтні.")
    add_animal(store, "Путівник лісом", "Путівник для всіх звірів.\n\nПід дубом. Жолуді збирайте у вересні.",
               Scope("forest"), name="Сорока")
    llm = FakeLLM({"found": True, "answer": "У вересні.", "sources": [2, 1]},
                  {"found": True, "answer": "У жовтні.", "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("коли збирати жолуді під дубом?", NOW)
    first, second = llm.prompts[0][1], llm.prompts[1][1]
    assert "Рівень: частина лісу" in first and "Рівень: увесь ліс" in first
    assert 'документ [1] (рівень "частина лісу") конкретніший за документ [2] (рівень "увесь ліс")' in second
    assert reply.text.startswith("У жовтні.") and reply.text.index("Жолуді листяного") < reply.text.index("Путівник")


async def test_second_try_out_of_time_keeps_the_first_answer(store, forest):
    # Asked again to put the levels right, but no model answers before the question's deadline.
    add_archive(store, forest, "Жолуді листяного лісу", "Порада для листяного лісу. Жолуді збирайте в жовтні.")
    add_animal(store, "Путівник лісом", "Путівник для всіх звірів.\n\nПід дубом. Жолуді збирайте у вересні.",
               Scope("forest"), name="Сорока")
    llm = FakeLLM({"found": True, "answer": "У жовтні.", "sources": [2, 1]}, LLMUnavailable("no time left"))
    started = time.monotonic()
    reply = await Answerer(store, forest, llm, TZ).ask("коли збирати жолуді під дубом?", NOW)
    assert len(llm.prompts) == 2 and reply.kind == "answer" and reply.text.startswith("У жовтні.")
    # Every call to the model carries the question's deadline; picking by title gets a shorter one.
    assert all(started < deadline <= time.monotonic() + QUESTION_SECONDS for deadline in llm.deadlines)
    assert llm.deadlines[0] <= started + PICK_SECONDS + 1


async def test_willow_answer_without_one_of_its_forest_parts_is_asked_again(store, forest):
    # Mishka's example: "Де шукати гриби біля верби?" takes the deciduous forest's and the river's guides.
    add_archive(store, forest, "Грибний путівник біля річки", "Цей путівник для тих, хто живе біля річки. "
                                                               "Опеньки ростуть на старих вербах.")
    add_archive(store, forest, "Грибний путівник листяного лісу", "Любі сусіди з листяного лісу! "
                                                                   "Білі гриби тримаються старих дубів.")
    llm = FakeLLM({"found": True, "answer": "Опеньки на старих вербах.", "sources": [1]},
                  {"found": True, "answer": "Біля річки опеньки; у листяному лісі білі під дубами.", "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Де шукати гриби біля верби?", NOW)
    order = [line for line in llm.prompts[0][1].splitlines() if line.startswith("[")]
    deciduous = next(i for i, line in enumerate(order, 1) if "листяного" in line)
    assert 'немає поради для частини "листяний ліс". Якщо в документі [%d]' % deciduous in llm.prompts[1][1]
    # The topic and its sentences from the guide: reminded without them, the live model still kept to the willow.
    assert 'на тему питання "де шукати гриби"' in llm.prompts[1][1]
    assert 'На цю тему в документі [%d] сказано: «Білі гриби тримаються старих дубів.»' % deciduous in llm.prompts[1][1]
    assert "Грибний путівник листяного лісу" in reply.text and "Грибний путівник біля річки" in reply.text
    # No forest part's advice in the answer (the mitten's own dues): nothing to ask about.
    add_archive(store, forest, "Членські внески рукавички Під вербою", "Внесок 3 горіхи на місяць.")
    llm = FakeLLM({"found": True, "answer": "3 горіхи.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Скільки внесок під вербою?", NOW)
    assert len(llm.prompts) == 1


DUES_FACTS = ["[1, рукавичка] Внесок 4 жолуді з дорослого звіра на місяць, здавати до 10 числа.",
              "[1, рукавичка] Внески віддають Кабанові в кошик біля входу.",
              "[1, рукавичка] Замість 4 жолудів можна віддати 2 лісові горіхи."]


async def test_answer_without_its_own_facts_is_asked_again(store, forest):
    # The customer's example: the model listed "Кабанові в кошик" and the nuts, then left them out of the answer.
    add_archive(store, forest, "Членські внески рукавички Під дубом",
                "Внесок 4 жолуді з дорослого звіра на місяць, до 10 числа. Віддавати Кабанові в кошик біля входу. "
                "Замість 4 жолудів можна віддати 2 лісові горіхи.")
    short = {"facts": DUES_FACTS, "found": True, "sources": [1],
             "answer": "Внесок 4 жолуді з дорослого звіра на місяць, до 10 числа."}
    full = dict(short, answer="Внесок 4 жолуді з дорослого звіра на місяць, до 10 числа, Кабанові в кошик біля входу; "
                              "замість жолудів можна 2 лісові горіхи.")
    llm = FakeLLM(short, full)
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом і до якого числа?", NOW)
    reminder = llm.prompts[1][1]
    assert '"Внески віддають Кабанові в кошик біля входу."' in reminder and "2 лісові горіхи" in reminder
    assert "4 жолуді з дорослого звіра на місяць, здавати" not in reminder.split("пункти, яких немає")[1]
    assert reply.text.startswith(full["answer"])
    # A full answer in other words is not asked again; with no time for the second try the first one holds.
    llm = FakeLLM(dict(short, answer="Дорослий звір платить 4 жолуді щомісяця до 10 числа, у кошик Кабанові біля "
                                     "входу; можна віддати 2 лісові горіхи замість жолудів."))
    await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert len(llm.prompts) == 1
    llm = FakeLLM(short, LLMUnavailable("no time left"))
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert reply.kind == "answer" and reply.text.startswith(short["answer"])


async def test_a_fact_of_several_sentences_is_checked_sentence_by_sentence(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць до 10 числа. "
                                                                       "Віддавати Кабанові в кошик біля входу.")
    merged = ["[1, рукавичка] Внесок 4 жолуді на місяць, здавати до 10 числа. Внески віддають Кабанові в кошик біля входу."]
    llm = FakeLLM({"facts": merged, "found": True, "sources": [1], "answer": "4 жолуді на місяць до 10 числа."},
                  {"facts": merged, "found": True, "sources": [1],
                   "answer": "4 жолуді на місяць до 10 числа, Кабанові в кошик біля входу."})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert '"Внески віддають Кабанові в кошик біля входу."' in llm.prompts[1][1] and "Кабанові" in reply.text


async def test_animal_advice_put_before_the_archive_for_the_same_place_is_asked_again(store, forest):
    # Decided with the customer (20:46): for the same place the archive wins, though the animal's advice is newer.
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    add_animal(store, "Нові внески", "Для рукавички Під дубом: внесок тепер 5 жолудів.", Scope("mitten", "Під дубом"),
               name="Білочка")
    llm = FakeLLM({"found": True, "answer": "5 жолудів на місяць.", "sources": [2, 1]},
                  {"found": True, "answer": "4 жолуді на місяць.", "other": "Білочка пише про 5 жолудів, але тут діє "
                                                                            "архів.", "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert "з документа звіра [2], а документ Лісового архіву [1]" in llm.prompts[1][1]
    assert reply.text.startswith("4 жолуді на місяць.\n\nБілочка пише")
    assert reply.text.index("Лісовий архів") < reply.text.index("автор: Білочка")
    # The archive document not even cited, and document numbers in the text: still asked, and the numbers go.
    llm = FakeLLM({"found": True, "answer": "5 жолудів на місяць [2].", "sources": [2]},
                  {"found": True, "answer": "4 жолуді на місяць [1].", "other": "Документ [2] пише про 5.",
                   "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert len(llm.prompts) == 2 and reply.text.startswith("4 жолуді на місяць.\n\nДокумент пише про 5.")


async def test_animal_advice_first_with_an_archive_document_on_another_topic_is_not_asked_again(store, forest):
    add_archive(store, forest, "Небезпеки поруч із рукавичкою Під дубом", "Яма за 30 кроків на північ.")
    add_animal(store, "Пожежа", "Пожежа: збираємося під дубом.", Scope("mitten", "Під дубом"))
    # The model picking by title brings the mitten's dangers document into the prompt too.
    llm = FakeLLM({"found": True, "answer": "Збирайтеся під дубом.", "sources": [2]}, picks=[{"ids": [1]}])
    reply = await Answerer(store, forest, llm, TZ).ask("Що робити при пожежі під дубом?", NOW)
    assert '"Пожежа"' in llm.prompts[0][1].split("[2] ")[1].splitlines()[0]
    assert len(llm.prompts) == 1 and reply.text.startswith("Збирайтеся під дубом.") and "автор: Зайчик" in reply.text


async def test_numbers_decide_and_may_be_written_as_words(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом",
                "Внесок 4 жолуді з дорослого звіра на місяць, до 10 числа. Віддавати Кабанові в кошик біля входу. "
                "Замість 4 жолудів можна віддати 2 лісові горіхи.")
    question = "Скільки внесок під дубом і до якого числа?"
    # The nuts dropped, though the words of that point are all in the answer: still asked about.
    no_nuts = {"facts": DUES_FACTS, "found": True, "sources": [1], "answer": "Внесок 4 жолуді з дорослого звіра на "
               "місяць, до 10 числа; віддавайте його Кабанові в кошик біля входу, можна й жолудями замість горіхів."}
    llm = FakeLLM(no_nuts, dict(no_nuts, answer=no_nuts["answer"] + " Замість 4 жолудів — 2 лісові горіхи."))
    await Answerer(store, forest, llm, TZ).ask(question, NOW)
    assert len(llm.prompts) == 2 and '"Замість 4 жолудів можна віддати 2 лісові горіхи."' in llm.prompts[1][1]
    # Numbers in words, the day as "десятого", the question's words not repeated: complete, no second call.
    llm = FakeLLM({"facts": DUES_FACTS, "found": True, "sources": [1], "answer": "Чотири жолуді з дорослого звіра "
                   "щомісяця, до десятого числа, Кабанові в кошик біля входу; або два лісові горіхи замість жолудів."})
    await Answerer(store, forest, llm, TZ).ask(question, NOW)
    assert len(llm.prompts) == 1


async def test_second_try_that_loses_the_answer_keeps_the_first(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць, Кабанові в кошик.")
    short = {"facts": DUES_FACTS, "found": True, "sources": [1], "answer": "4 жолуді на місяць."}
    for lost in ({"found": False}, {"found": True, "answer": "", "sources": [1]}):
        llm = FakeLLM(short, lost)
        reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
        assert len(llm.prompts) == 2 and reply.kind == "answer" and reply.text.startswith("4 жолуді на місяць.")
    assert 'а пункт поради, що тут поступається конкретнішій чи архіву, — коротко в "other"' in llm.prompts[1][1]


async def test_stub_or_repeated_other_is_left_out_and_document_numbers_go(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    for stub in ("Немає.", "-", "null", "Немає іншої поради.", "4 жолуді на місяць.", ["", ""],
                 'Загальна пам\'ятка про інше, тому її не включаємо в "other".'):
        llm = FakeLLM({"found": True, "answer": "4 жолуді на місяць.", "other": stub, "sources": [1]})
        reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
        assert reply.text.split("\n\n")[1].startswith("Джерело:"), stub
    llm = FakeLLM({"found": True, "answer": "Як сказано у документі [1], 4 жолуді на місяць [1, рукавичка] з 2026 [2026].",
                   "sources": [1]})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert reply.text.startswith("Як сказано, 4 жолуді на місяць з 2026 [2026].")


async def test_everything_to_put_right_goes_in_one_second_try(store, forest):
    # A guide section put before the part's advice, and a point left out: one request more, not two.
    add_archive(store, forest, "Жолуді листяного лісу", "Порада для листяного лісу. Жолуді збирайте в жовтні, "
                                                       "після першого морозу.")
    add_animal(store, "Путівник лісом", "Путівник для всіх звірів.\n\nПід дубом. Жолуді збирайте у вересні.",
               Scope("forest"), name="Сорока")
    facts = ["[1, частина лісу] Жолуді збирайте в жовтні, після першого морозу."]
    llm = FakeLLM({"facts": facts, "found": True, "answer": "У вересні.", "sources": [2, 1]},
                  {"facts": facts, "found": True, "answer": "У жовтні, після першого морозу.", "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("коли збирати жолуді під дубом?", NOW)
    second = llm.prompts[1][1]
    assert len(llm.prompts) == 2 and "конкретніший за документ [2]" in second and "Жолуді збирайте в жовтні" in second
    assert reply.text.startswith("У жовтні, після першого морозу.")


async def test_second_try_that_came_out_worse_is_not_taken(store, forest):
    # Asked to add the nuts, a weaker fallback model dropped the basket and the 10th as well.
    add_archive(store, forest, "Членські внески рукавички Під дубом",
                "Внесок 4 жолуді з дорослого звіра на місяць, до 10 числа. Віддавати Кабанові в кошик біля входу. "
                "Замість 4 жолудів можна віддати 2 лісові горіхи.")
    first = {"facts": DUES_FACTS, "found": True, "sources": [1],
             "answer": "Внесок 4 жолуді з дорослого звіра на місяць, до 10 числа, Кабанові в кошик біля входу."}
    worse = {"facts": DUES_FACTS, "found": True, "sources": [1], "answer": "Можна віддати 2 лісові горіхи."}
    llm = FakeLLM(first, worse)
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert len(llm.prompts) == 2 and reply.text.startswith(first["answer"])


async def test_willow_answer_with_the_river_only_in_the_other_paragraph_is_asked_again(store, forest):
    # Both parts' advice fits "Під вербою"; the river's must not be shown as advice that gives way.
    add_archive(store, forest, "Як утеплити рукавичку в листяному лісі", "Порада для листяного лісу. Утеплюйте "
                                                                         "сухим дубовим листям.")
    add_archive(store, forest, "Як утеплити рукавичку біля річки", "Ця порада для тих, хто живе біля річки. "
                                                                   "Утеплюйте очеретом, мох біля води гниє.")
    llm = FakeLLM({"found": True, "answer": "Утеплюйте сухим дубовим листям.", "sources": [1, 2],
                   "other": "Порада для річки радить очерет, але тут діє листяний ліс."},
                  {"found": True, "answer": "Листяний ліс: сухим дубовим листям; біля річки: очеретом, мох гниє.",
                   "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Чим утеплити рукавичку під вербою?", NOW)
    assert len(llm.prompts) == 2 and 'немає поради для частини "' in llm.prompts[1][1]
    assert "очеретом" in reply.text.split("\n\n")[0]


async def test_facts_of_documents_the_answer_does_not_cite_are_not_required(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Рукавички утеплюють у листопаді.")
    llm = FakeLLM({"facts": ["[1, рукавичка] Внесок 4 жолуді на місяць.",
                             "[2, увесь ліс] Рукавички утеплюють у листопаді."],
                   "found": True, "answer": "4 жолуді на місяць.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Скільки внесок під дубом?", NOW)
    assert len(llm.prompts) == 1


async def test_advice_that_gives_way_stands_in_its_own_paragraph(store, forest):
    # The fire example agreed with the customer: the mitten's own gathering place, the memo's only as a note.
    add_archive(store, forest, "Що робити при пожежі",
                "Ця пам'ятка для всіх мешканців лісу. Довгий безперервний крик означає пожежу. "
                "Усі збираються біля бобрової греблі.")
    add_animal(store, "Пожежа Під дубом", "При пожежі мешканці рукавички Під дубом збираються під дубом, біля дупла Сови.",
               Scope("mitten", "Під дубом"), name="Їжачок")
    other = ("загальна пам'ятка для всього лісу радить збиратися біля бобрової греблі, але для рукавички Під дубом "
             "діє конкретніша порада.")
    llm = FakeLLM({"found": True, "answer": "При пожежі збирайтеся під дубом, біля дупла Сови. Сигнал пожежі — довгий "
                                            "безперервний крик.", "other": other, "sources": [1, 2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Що робити при пожежі під дубом?", NOW)
    answer, note, cite = reply.text.split("\n\n")
    assert "бобрової" not in answer and note == other[0].upper() + other[1:]
    assert cite.startswith("Джерела:") and "автор: Їжачок" in cite and '"Що робити при пожежі"' in cite
    system = llm.prompts[0][0]
    assert 'коротко перекажи її в полі "other"' in system and '"other":' in system


def word_text(paragraphs: list[str]) -> str:
    """Text as Word gives it: paragraphs joined by single line breaks, no blank lines."""
    document = docx.Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return extract_text(buffer.getvalue(), "Путівник.docx")


def word_guide(sections: list[tuple[str, str]], intro: str = "") -> str:
    """A guide of sections, each a heading over three paragraphs."""
    return word_text(([intro] if intro else []) + [line for heading, body in sections for line in (heading, body, body, body)])


FILLER = "Восени дні коротшають, тож запаси робіть заздалегідь і перевіряйте їх щотижня після дощу. "


async def test_long_word_guide_reaches_the_model(store, forest):
    # Word separates paragraphs with a single line break: the whole guide used to be one chunk
    # over the size limit, and the model got the guide's title with an empty text.
    text = word_guide([("Запаси на зиму", FILLER * 4), ("Гриби", FILLER * 4),
                       ("Жолуді", "Під дубом запасайте п'ять кошиків жолудів. " + FILLER)])
    assert len(text) > DOC_CHARS and "\n\n" not in text
    add_animal(store, "Зимовий путівник", text, Scope("mitten", "Під дубом"), name="Сова")
    llm = FakeLLM({"found": True, "answer": "П'ять кошиків.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("скільки кошиків жолудів запасати під дубом?", NOW)
    excerpt = llm.prompts[0][1].split("Текст:\n", 1)[1]
    assert "п'ять кошиків жолудів" in excerpt and len(excerpt) <= DOC_CHARS + 50


async def test_long_guide_gives_the_section_of_the_asked_place(store, forest):
    # "біля сосон" shares no word with the heading "У соснах"; the forest map still finds the section.
    sections = [(f"{place}", f"{fact} {FILLER * 2}") for place, fact in [
        ("Під ялиною", "Під ялиною ростуть рижики."), ("Під дубом", "Біля дуба шукайте білі гриби."),
        ("У ліщині", "У ліщині ростуть лисички."), ("У соснах", "Маслюки ростуть на галявинах."),
        ("Біля бобрової греблі", "Біля греблі ростуть опеньки."), ("На узліссі", "На узліссі ростуть печериці.")]]
    add_animal(store, "Грибний путівник", word_guide(sections, "Путівник для всіх мешканців лісу."), Scope("forest"))
    llm = FakeLLM({"found": True, "answer": "Маслюки.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("які гриби біля сосон?", NOW)
    excerpt = llm.prompts[0][1].split("Текст:\n", 1)[1]
    assert "У соснах\nМаслюки ростуть на галявинах." in excerpt  # the heading stays with its section


def document_texts(prompt: str) -> list[str]:
    return [block.split("Текст:\n", 1)[1] for block in prompt.split("\n\n[")[1:]]


@pytest.mark.parametrize("lengths", [[40 * len(FILLER)] * 6, [2400] * 7, [2400] * 5 + [6000] * 2])
async def test_documents_together_fit_one_prompt(store, forest, lengths):
    # Groq's free tier rejects a prompt over about 8000 tokens, so long and middling documents share a cut.
    for i, length in enumerate(lengths):
        add_animal(store, f"Поради про жолуді {i}", ("Під дубом жолуді. " + FILLER * 70)[:length],
                   Scope("mitten", "Під дубом"), author_id=i, when=NOW - timedelta(hours=i))
    picks = list(range(len(lengths) - 3, len(lengths) + 1))  # the title pick and keyword search take different ones
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": picks}])
    await Answerer(store, forest, llm, TZ).ask("Коли запасати жолуді під дубом?", NOW)
    texts = document_texts(llm.prompts[0][1])
    system, prompt = llm.prompts[0]
    assert len(texts) >= 5 and all(texts) and len(system) + len(prompt) <= prompt_chars(1500)


@pytest.mark.parametrize("question, budget", [("Коли запасати жолуді під дубом?", 1500),
                                              ("Чи вже час запасати жолуді під дубом?", 2000)])
async def test_long_guides_leave_the_minute_room_for_the_reply(store, forest, question, budget):
    # Seven long guides took 12000 characters, and with the rules and the reply's budget every model answered
    # "Request too large". A little stays free for a count a little off.
    for i in range(7):
        add_animal(store, f"Поради про жолуді {i}", ("Під дубом жолуді. " + FILLER * 70)[:6000],
                   Scope("mitten", "Під дубом"), author_id=i, when=NOW - timedelta(hours=i))
    llm = BudgetLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": [4, 5, 6, 7]}])
    await Answerer(store, forest, llm, TZ).ask(question, NOW)
    system, prompt = llm.prompts[0]
    assert len(document_texts(prompt)) == 7 and llm.budgets[-1] == ("answer", budget)
    assert groq_tokens(system, prompt) + budget <= 8000 - SPARE_TOKENS


async def test_archive_documents_go_whole_in_the_usual_case(store, forest):
    # Seven documents as long as the archive's longest ones, which take 5722 characters together.
    texts = [f"Внесок {i} жолуді на місяць. " + FILLER * 9 for i in range(7)]
    for i, text in enumerate(texts):
        add_archive(store, forest, f"Членські внески рукавички Під дубом {i}", text)
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": [4, 5, 6, 7]}])
    await Answerer(store, forest, llm, TZ).ask("Коли здавати внесок під дубом?", NOW)
    assert sorted(document_texts(llm.prompts[0][1])) == sorted(texts) and sum(map(len, texts)) > 5722


async def test_archive_documents_go_whole_with_no_place_and_the_whole_calendar(store, forest):
    # The biggest prompt: the rule for no place named, every day of two months and eight documents. With 600 tokens
    # kept free for a second try, all eight archive documents were cut, and the date of the acorns with them.
    titles = [f"Чим поживитись поруч із рукавичкою {mitten}" for mitten in ("Під дубом", "У ліщині", "Під вербою",
                                                                          "На узліссі")]
    titles += ["Календар сезонів", "Осінній путівник лісу", "Коли падають жолуді", "Правила толоки"]
    texts = [(f"Жолуді збирають восени, порада {i}. " + FILLER * 8)[:650 + 14 * i] for i in range(8)]
    for title, text in zip(titles, texts, strict=True):
        add_archive(store, forest, title, text)
    llm = BudgetLLM({"found": True, "answer": "x", "sources": [1]}, picks=[{"ids": [5, 6, 7, 8]}])
    await Answerer(store, forest, llm, TZ).ask("Чи вже можна збирати жолуді?", NOW)
    system, prompt = llm.prompts[0]
    assert system == ANSWER_SYSTEM + PLACE_UNKNOWN_RULE and llm.budgets[-1] == ("answer", 2000)
    assert sorted(document_texts(prompt)) == sorted(texts)
    assert groq_tokens(system, prompt) + 2000 <= 8000 - SPARE_TOKENS


LATE = " Це важливо, бо зима довга, а весна приходить пізно."


WINTER_GUIDE = [
    "Зимовий путівник для всіх мешканців лісу",
    "Тут зібрано поради, як перезимувати в кожній рукавичці. Знайдіть розділ своєї рукавички й прочитайте його уважно.",
    "Під ялиною",
    "Під ялиною тепло навіть у мороз, бо гілки низько. Вхід утеплюйте мохом, а не листям: листя мокне." + LATE,
    "Запаси тримайте в сухій хвої біля коріння, шишки перевіряйте щотижня, бо волога їх псує, а миші люблять сухі." + LATE,
    "Вечорами не розпалюйте вогню: хвоя легко займається. Світіть світлячками, їх приносить Сорока." + LATE,
    "У соснах",
    "У соснах вітряно, тож рукавичку прив'язуйте до коріння лозою. Перевіряйте вузли після кожної бурі." + LATE,
    "Смолу не чіпайте: вона потрібна для латання рукавички, і латає її лише Дятел." + LATE,
    "Маслюки сушіть на гілках до першого снігу, потім знімайте, бо замерзнуть." + LATE,
    "Біля мурашника",
    "Біля мурашника не копайте ям для запасів: мурахи сердяться й кусають навіть узимку, коли прокидаються у відлигу." + LATE,
    "Запаси тримайте на високому пеньку, накривайте корою від снігу." + LATE,
    "Стежкою повз мурашник ходіть лише вдень, щоб не наступити на мурашині ходи." + LATE,
    "Під дубом",
    "Під дубом взимку найбільше гостей, бо тут найбільша комора лісу. Гостей пускайте лише вдень." + LATE,
    "Жолуді зберігайте в коморі під корінням, а не в дуплі, бо там їх знаходять сойки й розносять по лісу." + LATE,
    "Кожен мешканець приносить до комори п'ять кошиків жолудів до першого снігу, а Кабан приносить сім. "
    "Хто не приніс, той чергує біля комори тиждень." + LATE,
    "У ліщині",
    "У ліщині горіхи сушіть три дні на сонці, тоді вони не гниють у коморі й не гірчать." + LATE,
    "Шкаралупу не викидайте біля входу, бо її бачать лисиці й приходять до рукавички вночі." + LATE,
    "Ведмідь лягає спати в листопаді, тож його частку горіхів відкладайте заздалегідь." + LATE,
    "Під вербою",
    "Під вербою вода восени піднімається, тож запаси тримайте на гілках, а не на землі." + LATE,
    "Утеплюйте рукавичку очеретом, його носять Жабка й Качка з річки щосуботи." + LATE,
    "Кору верби не обдирайте взимку: дерево замерзне й навесні не дасть листя." + LATE,
    "Біля бобрової греблі",
    "Біля греблі не ходіть по тонкому льоду, особливо в грудні, коли лід ще не встиг стати." + LATE,
    "Бобри відкривають греблю в березні, до того всі запаси мають бути на високому березі." + LATE,
    "Рибу морозьте в ополонці, яку рубає Бобер щоранку." + LATE,
    "На узліссі",
    "На узліссі вітер холодний, тож заносьте запаси глибше до лісу, до старого пня." + LATE,
    "Мисливці взимку ходять узліссям, тож вхід ховайте за кущем шипшини." + LATE,
    "Шипшину сушіть і зберігайте в сухій норі, вона знадобиться від застуди." + LATE,
]


async def guide_excerpt(store, forest, paragraphs: list[str], question: str) -> str:
    add_animal(store, "Путівник", word_text(paragraphs), Scope("forest"), name="Сова")
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask(question, NOW)
    return document_texts(llm.prompts[0][1])[0]


async def test_end_of_the_asked_section_is_kept(store, forest):
    # A chunk may hold the end of the asked section and then the next heading: that end still belongs to it.
    excerpt = await guide_excerpt(store, forest, WINTER_GUIDE, "Що треба робити взимку під дубом?")
    assert "п'ять кошиків жолудів" in excerpt


async def test_section_of_the_asked_forest_part_is_kept(store, forest):
    more = LATE * 2
    paragraphs = ["Як утеплити рукавичку", "Поради для всіх мешканців лісу на зиму." + more,
                  "Під ялинами", "Утеплюйте рукавичку хвоєю й мохом, шар у три лапи." + more,
                  "Вхід закривайте ялиновою гілкою." + more, "Хвою міняйте раз на місяць." + more,
                  "Серед дубів і ліщини", "Утеплюйте рукавичку сухим дубовим листям, шар у дві лапи." + more,
                  "Листя збирайте лише сухе, після трьох сонячних днів." + more, "Мокре листя гниє й холодить." + more,
                  "Біля річки", "Утеплюйте рукавичку очеретом, а не мохом, і піднімайте її на палі." + more,
                  "Очерет ріжте в жовтні, поки він сухий." + more, "Палі ставте до першого льоду." + more,
                  "На узліссі", "Утеплюйте рукавичку сіном і ставте заслін від вітру." + more,
                  "Сіно беріть у полі після покосу." + more, "Заслін плетіть із лози." + more]
    excerpt = await guide_excerpt(store, forest, paragraphs, "Як утеплити рукавичку біля річки?")
    assert "Палі ставте до першого льоду" in excerpt


async def test_section_after_the_asked_one_does_not_take_its_place(store, forest):
    # The hazel section after a short oak section is not oak advice; the deciduous forest's advice is.
    more = LATE + " І запаси мають дожити до квітня."
    hazel = ["Шкаралупу не викидайте біля входу, бо її бачать лисиці й приходять уночі.",
             "Ведмідь лягає спати в листопаді, тож його частку горіхів відкладайте раніше.",
             "Кошики для горіхів плетіть із лози заздалегідь, великі кошики не беріть.",
             "Вхід закривайте гілками щовечора, бо вітер холодний.",
             "Гостей із сусідніх рукавичок пускайте лише вдень і записуйте на корі.",
             "Старі кошики латайте лозою, нові плетіть лише в морозні дні.",
             "Мішечки підвішуйте до стелі, щоб миші не дісталися.", "Сушені ягоди тримайте в глечиках із кришкою.",
             "Лозу для кошиків ріжте восени біля струмка.", "Горіхи для Білочки складайте окремо, у малий кошик.",
             "Глечики перевіряйте раз на тиждень, чи не відсиріли.", "Підлогу застеляйте сухим мохом."]
    deciduous = ["Узимку в листяному лісі стежки розчищає Кабан.", "Сніг біля входу відкидайте вранці, поки він м'який.",
                 "Листя під деревами не згрібайте: воно гріє коріння.",
                 "Комору в листяному лісі відкривають лише по суботах, ключ від комори тримає Сова.",
                 "Хворим звірам видають без черги.", "Сухі гілки збирайте лише з землі.",
                 "Воду беріть лише з джерела, не з калюж.", "Дітей самих далеко не відпускайте."]
    paragraphs = (["Зимовий путівник", "Поради на зиму для мешканців рукавичок." + more,
                   "Під дубом", "Під дубом запасайте п'ять кошиків жолудів на кожного мешканця до першого снігу." + more,
                   "У ліщині", "У ліщині горіхи сушіть три дні на сонці, тоді вони не гниють у коморі." + more]
                  + [line + more for line in hazel] + ["Листяний ліс загалом"] + [line + more for line in deciduous])
    excerpt = await guide_excerpt(store, forest, paragraphs, "Коли відкривають комору під дубом?")
    assert "п'ять кошиків жолудів" in excerpt and "по суботах" in excerpt


def excerpt_of(store, forest, lines: list[str], question: str, mittens: list[str], parts: list[str]) -> str:
    add_animal(store, "Путівник", "\n".join(lines), Scope("forest"))
    doc = store.active_documents()[0]
    assert len(doc.text) > DOC_CHARS
    return Answerer(store, forest, FakeLLM(), TZ)._excerpt(doc, question, mittens, parts, DOC_CHARS)


NOISY = "Гриби ростуть після дощу, гриби ростуть у затінку, гриби ростуть рядами. "


def test_asked_place_beats_word_overlap(store, forest):
    sections = [("У листяному лісі", NOISY * 3)] * 3 + [("Під ялиною", NOISY * 3)] * 2 + \
               [("Під дубом", "Білі шукайте біля коріння старого дерева. " + FILLER * 2)]
    text = excerpt_of(store, forest, [line for heading, body in sections for line in (heading, body, body, body)],
                      "які гриби ростуть під дубом?", ["Під дубом"], [])
    assert "Білі шукайте біля коріння" in text


def test_section_running_over_chunks_keeps_its_place(store, forest):
    lines = []
    for heading in ["Під ялиною", "Під дубом", "У ліщині"]:
        lines += [heading] + [NOISY * 3] * 2
    lines += ["У соснах", FILLER * 7, FILLER * 7, "Маслюки збирайте на галявинах."]
    for heading in ["На узліссі", "Під вербою", "Біля бобрової греблі"]:
        lines += [heading] + [NOISY * 3] * 2
    text = excerpt_of(store, forest, lines, "які гриби ростуть біля сосон?", ["У соснах"], [])
    assert "Маслюки збирайте" in text


def test_part_question_takes_the_parts_own_section_not_its_mittens(store, forest):
    # Mishka: a mitten's advice does not fit all its neighbours, so it is not the part's advice.
    lines = []
    for mitten in ["Під дубом", "У ліщині", "Під ялиною", "У соснах", "Біля мурашника"]:
        lines += [mitten] + [f"Тут живуть звірі рукавички {mitten}. {FILLER * 3}"] * 3
    lines += ["У хвойному лісі"] + [f"Порада для всього хвойного лісу. {FILLER * 3}"] * 3
    text = excerpt_of(store, forest, lines, "що робити у хвойному лісі?", [], ["conifer"])
    assert "Порада для всього хвойного лісу" in text
    assert text.count("рукавички Під ялиною") < 3 and text.count("рукавички У соснах") < 3


def test_excerpt_finds_synonyms(store, forest):
    lines = []
    for topic in ["Запаси", "Прибирання", "Сусіди", "Свята", "Нічна тиша", "Чим поживитись"]:
        lines += [topic] + [FILLER * 4] * 3 + (["Кора і бруньки дуже поживні."] if topic == "Чим поживитись" else [])
    assert "Кора і бруньки" in excerpt_of(store, forest, lines, "що їсти?", [], [])


async def test_part_question_about_mitten_advice_asks_which_mitten(store, forest):
    # Dues are written per mitten and the river has none of its own: the river's mittens are offered,
    # instead of "no such advice", and no mitten's dues are given out as the river's.
    add_archive(store, forest, "Членські внески рукавички Під вербою", "Внесок 3 горіхи на місяць.")
    add_archive(store, forest, "Членські внески рукавички Біля бобрової греблі", "Внесок 2 рибини на місяць.")
    add_archive(store, forest, "Членські внески рукавички Під дубом", "Внесок 4 жолуді на місяць.")
    llm = FakeLLM()
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок біля річки?", NOW)
    assert reply.kind == "need_place" and reply.mittens == ["Під вербою", "Біля бобрової греблі"]
    assert reply.text == texts.ASK_MITTEN_IN_PART.format(part="біля річки") and llm.prompts == []


async def test_part_question_answers_only_from_the_parts_own_advice(store, forest):
    # Mishka: "Поради окремих рукавичок він додає лише коли названо конкретну рукавичку".
    add_archive(store, forest, "Що їсти біля річки", "Ця пам'ятка для мешканців біля річки. Їжте рогіз і очерет.")
    add_archive(store, forest, "Чим поживитись поруч із рукавичкою Під вербою", "Їжте кору верби.")
    llm = FakeLLM({"found": True, "answer": "Рогіз і очерет.", "sources": [1]})
    reply = await Answerer(store, forest, llm, TZ).ask("Що їсти біля річки?", NOW)
    prompt = llm.prompts[0][1]
    assert "Що їсти біля річки" in prompt and "Чим поживитись" not in prompt and "кору верби" not in prompt
    assert reply.text.startswith("Рогіз і очерет.") and "Чим поживитись" not in reply.text
    # With the mitten named, its own advice comes first.
    llm = FakeLLM({"found": True, "answer": "Кору верби.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Що їсти під вербою?", NOW)
    assert llm.prompts[0][1].index("Чим поживитись") < llm.prompts[0][1].index("Що їсти біля річки")


async def test_part_advice_that_does_not_answer_leads_to_its_mittens(store, forest):
    add_archive(store, forest, "Небезпеки біля річки", "Ця пам'ятка для мешканців біля річки. Внески тут ні до чого: "
                                                        "не ходіть по тонкому льоду.")
    add_archive(store, forest, "Членські внески рукавички Під вербою", "Внесок 3 горіхи на місяць.")
    llm = FakeLLM({"found": False})
    reply = await Answerer(store, forest, llm, TZ).ask("Скільки внесок біля річки?", NOW)
    assert len(llm.prompts) == 1 and reply.kind == "need_place" and reply.mittens == ["Під вербою", "Біля бобрової греблі"]
    # Nothing in the mittens either: the honest "not found".
    llm = FakeLLM({"found": False})
    reply = await Answerer(store, forest, llm, TZ).ask("Коли біля річки зимують раки?", NOW)
    assert reply.kind == "not_found"


async def test_terminal_ask_works_with_empty_settings(monkeypatch, capsys):
    # docker-compose passes unset settings as empty strings: TZ_NAME="" is Kyiv, not a crash.
    from bot import ask
    from bot.answering import Reply

    used = {}

    class Answering:
        def __init__(self, store, forest, llm, tz):
            used["tz"] = tz

        async def ask(self, question, now, mitten=None):
            return Reply("answer", "4 жолуді.")

    async def no_sync(*args):
        used["ocr"] = args[2].ocr_langs

    for name in ("TZ_NAME", "OCR_LANGS"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("DB_PATH", ":memory:")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setattr(ask, "sync_archive", no_sync)
    monkeypatch.setattr(ask, "Answerer", Answering)
    stores = []
    monkeypatch.setattr(ask, "Store", lambda path: stores.append(Store(path)) or stores[-1])
    await ask.main("скільки внесок під дубом?")
    stores[0].db.close()
    assert str(used["tz"]) == "Europe/Kyiv" and used["ocr"] == "ukr"
    assert capsys.readouterr().out.strip() == "4 жолуді."


async def test_terminal_tools_read_the_settings_as_the_bot_does(monkeypatch):
    # The bot reads GROQ_MODELS="a, b" as ["a", "b"] and DB_PATH="" as data/bot.db. bot.ask sent Groq the model
    # " b", and bot.check opened "" as an empty temporary database, so every reference question failed.
    from bot import ask, check
    from bot.answering import Reply

    used = []

    class Answering:
        def __init__(self, store, forest, llm, tz):
            used.append(llm.models)

        async def ask(self, question, now, mitten=None):
            return Reply("need_place")

    async def nothing(*args):
        pass

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GROQ_MODELS", "openai/gpt-oss-120b, openai/gpt-oss-20b")
    monkeypatch.setenv("DB_PATH", "")
    monkeypatch.setattr(ask, "sync_archive", nothing)
    monkeypatch.setattr(LLM, "check_models", nothing)
    stores = []

    def open_store(path):
        used.append(path)
        stores.append(Store(":memory:"))
        return stores[-1]

    for module in (ask, check):
        monkeypatch.setattr(module, "Answerer", Answering)
        monkeypatch.setattr(module, "Store", open_store)
    await ask.main("скільки внесок?")
    await check.main()
    for store in stores:
        store.db.close()
    models = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    assert used == ["data/bot.db", models, "data/bot.db", models]


def test_text_without_blank_lines_is_cut_into_chunks():
    from bot.search import CHUNK_CHARS, split_chunks
    one_line = FILLER * 30  # a TXT guide written as one long line
    lines = "\n".join(["Під дубом"] + [FILLER * 2] * 20)  # Word: one line per paragraph
    ending = lines + "\nЖолуді\nЗбирайте до 10 жовтня"  # last lines without a full stop are kept too
    for text in (one_line, lines, ending):
        chunks = split_chunks(text)
        assert len(chunks) > 1 and all(len(chunk) <= CHUNK_CHARS + 81 for chunk in chunks)
        assert "".join(chunks).replace("\n", "").replace(" ", "") == text.replace("\n", "").replace(" ", "")
    assert split_chunks("Календар сезонів\nУтеплюйте в листопаді.") == ["Календар сезонів\nУтеплюйте в листопаді."]


def test_long_guide_does_not_make_its_words_look_common(store, forest):
    # Counted per chunk, a guide of 16 chunks about mushrooms made "гриби" a common word, and the deciduous
    # forest's mushroom guide fell under the search cutoff for "Де шукати гриби?".
    from bot.search import Index, split_chunks
    add_archive(store, forest, "Грибний путівник листяного лісу", "Любі сусіди з листяного лісу! "
                                                                   "Білі гриби тримаються старих дубів.")
    add_archive(store, forest, "Їстівне й отруйне", "Ця пам'ятка для всіх мешканців лісу. Шукайте лише знайомі гриби.")
    add_archive(store, forest, "Що робити при пожежі", "Ця пам'ятка для всіх мешканців лісу. Біжіть до річки.")
    places = ["Під ялиною", "У соснах", "Під дубом", "У ліщині", "Під вербою", "На узліссі", "Біля греблі",
              "Біля мурашника"]
    guide = "\n".join(f"{place}: гриби ростуть після дощу. " + FILLER * 6 for place in places * 2)
    add_animal(store, "Поради Сови на осінь", guide, Scope("forest"), name="Сова")
    assert len(split_chunks(guide)) == 16
    hits = Index(store.active_documents()).search("Де шукати гриби?")
    assert "Грибний путівник листяного лісу" in [hit.doc.title for hit in hits]


async def test_part_question_keeps_the_whole_forest_advice(store, forest):
    # The part's and the whole forest's advice go in; the mittens' own documents stay out of a part question.
    add_archive(store, forest, "Що їсти біля річки", "Ця пам'ятка для мешканців біля річки. Що їсти: рогіз.")
    add_archive(store, forest, "Що їсти взимку", "Для всіх мешканців лісу. Що їсти взимку: жолуді й горіхи.")
    for mitten in ("Під вербою", "Біля бобрової греблі"):
        for i in range(3):
            add_animal(store, f"Що їсти {mitten} {i}", f"Що їсти: кору, ягоди, їжу {i}.", Scope("mitten", mitten),
                       author_id=i + 1)
    llm = FakeLLM({"found": True, "answer": "Рогіз.", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Що їсти біля річки?", NOW)
    prompt = llm.prompts[0][1]
    assert "Що їсти біля річки" in prompt and "Що їсти взимку" in prompt and "Що їсти Під вербою" not in prompt


OAK = WINTER_GUIDE.index("Під дубом")


@pytest.mark.parametrize("start", ["До ліщини по горіхи ходіть лише вдень, а жолуді",
                                   "Ліщиновими гілками закривайте вхід щовечора, а жолуді"])
async def test_paragraph_naming_another_place_does_not_end_the_section(store, forest, start):
    guide = list(WINTER_GUIDE)
    guide[OAK + 2] = start + " зберігайте в коморі під корінням, а не в дуплі." + LATE
    excerpt = await guide_excerpt(store, forest, guide, "Що треба робити взимку під дубом?")
    assert "п'ять кошиків жолудів" in excerpt


async def test_wrapped_pdf_line_naming_another_place_does_not_end_the_section(store, forest):
    import textwrap
    guide = list(WINTER_GUIDE)
    guide[OAK + 1] = ("Під дубом взимку найбільше гостей, бо тут найбільша комора лісу. Своїх гостей пускайте лише "
                      "вдень, а сусідів із ліщини та з-під верби можна пускати й увечері." + LATE)
    add_animal(store, "Путівник", "\n".join(line for p in guide for line in textwrap.wrap(p, 76)), Scope("forest"))
    llm = FakeLLM({"found": True, "answer": "x", "sources": [1]})
    await Answerer(store, forest, llm, TZ).ask("Що треба робити взимку під дубом?", NOW)
    assert "Хто не приніс, той чергує біля комори тиждень" in " ".join(document_texts(llm.prompts[0][1])[0].split())


async def test_place_named_in_passing_does_not_outrank_the_asked_section(store, forest):
    guide = list(WINTER_GUIDE)
    for heading, line in [("Під ялиною", "Шишки міняйте на жолуді в коморі під дубом, але не частіше разу на місяць."),
                          ("Під вербою", "Жолуді для малечі просіть у сусідів під дубом заздалегідь."),
                          ("На узліссі", "Шипшину носіть до комори під дубом, там її сушать для всіх.")]:
        guide[guide.index(heading) + 2] = line + LATE
    excerpt = await guide_excerpt(store, forest, guide, "Що треба робити взимку під дубом?")
    assert "п'ять кошиків жолудів" in excerpt and "Жолуді зберігайте в коморі" in excerpt


async def test_no_place_asked_is_not_turned_into_one_mittens_answer(store, forest):
    # The model answers a whole-forest question from the whole-forest memo [2] and mentions the mitten's
    # exception [1] after it: with no place asked, nothing makes it put the mitten first.
    add_archive(store, forest, "Що робити при пожежі", "Ця пам'ятка для всіх мешканців лісу. При пожежі біжіть до річки.")
    add_archive(store, forest, "Небезпеки поруч із рукавичкою Під дубом", "При пожежі біжіть до дуба.")
    llm = FakeLLM({"found": True, "answer": "Біжіть до річки; під дубом до дуба.", "sources": [2, 1]})
    await Answerer(store, forest, llm, TZ).ask("що робити при пожежі?", NOW)
    assert "Небезпеки поруч" in llm.prompts[0][1].split("[1]")[1].split("[2]")[0] and len(llm.prompts) == 1


async def test_the_bot_says_the_place_has_no_advice_of_its_own_when_the_model_does_not(store, forest):
    # Live, the model answered "У листопаді … утеплювати рукавичку «На узліссі»." without Mishka's
    # «Окремих порад про утеплення для узлісся немає»; the bot adds it, as the conifer forest has its own.
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Рукавички утеплюють у листопаді.")
    add_archive(store, forest, "Як утеплити рукавичку в хвойному лісі",
                "Порада для тих, чия рукавичка стоїть у хвойному лісі. Утеплюйте рукавичку в жовтні.")
    add_archive(store, forest, "Як кричати, коли бачиш мисливців",
                "Це звернення до всіх мешканців лісу. Побачив мисливців — крикни тричі коротко, як Сорока.")
    llm = FakeLLM({"found": True, "answer": "Утеплюйте рукавичку в листопаді.", "sources": [1]},
                  {"found": True, "answer": "Крикни тричі коротко, як Сорока.", "sources": [1]})
    answerer = Answerer(store, forest, llm, TZ)
    reply = await answerer.ask("Коли утеплювати рукавичку на узліссі?", NOW)
    assert reply.text.startswith('Окремої поради на цю тему для рукавички "На узліссі" немає, тож діє загальна порада '
                                 "для всього лісу. Утеплюйте рукавичку в листопаді.\n\nДжерело:")
    # The hunters' cry is the same for all: straight to it, as in Mishka's own answer.
    reply = await answerer.ask("Біля греблі мисливці! Як кричати?", NOW)
    assert reply.text.startswith("Крикни тричі коротко, як Сорока.\n\nДжерело:")


async def test_a_place_found_by_its_resident_gets_no_note_about_its_own_advice(store, forest):
    add_archive(store, forest, "Статут ОСМД У ліщині", "Мешканці. У рукавичці живуть Ведмідь, Сорока і Вивірка.")
    add_archive(store, forest, "Календар сезонів", "Цей календар для всіх мешканців лісу. Хто спить узимку, лягає в грудні. "
                "Рукавички утеплюють у листопаді.")
    add_archive(store, forest, "Як утеплити рукавичку в листяному лісі",
                "Порада для тих, чия рукавичка стоїть у листяному лісі. Утеплюйте рукавичку в жовтні, спати лягайте в грудні.")
    llm = FakeLLM({"found": True, "answer": "Ведмідь лягає спати в грудні.", "sources": [2]})
    reply = await Answerer(store, forest, llm, TZ).ask("Коли Ведмідь лягає спати?", NOW)
    assert reply.text.startswith("Ведмідь лягає спати в грудні.")
    assert "Поради на цю тему різні в різних місцях" not in llm.prompts[0][1]


def test_bare_document_mentions_are_cleaned():
    assert clean_text("Шукайте гриби після дощу (документ).", 3) == "Шукайте гриби після дощу."
    assert clean_text("Збір біля греблі (з документа «Що робити при пожежі»).", 3) == "Збір біля греблі."
    assert clean_text("Толока була 3 жовтня (у 2026 році).", 3) == "Толока була 3 жовтня (у 2026 році)."
    assert clean_text("Утеплюйте в жовтні (у документі [2]).", 3) == "Утеплюйте в жовтні."
    # Any one bracket after "(документ…" goes, as before the runs of numbers came in.
    for cited in ("(у документі [№2])", "(документ [Членські внески])", "(у документі [ 2 ])"):
        assert clean_text(f"Внесок 4 жолуді {cited}.", 3) == "Внесок 4 жолуді.", cited
    # A parenthesis with a fact in it stays, though it names a document.
    assert clean_text("Внесок 4 жолуді (у документі сказано: до 10 числа).", 3) == \
        "Внесок 4 жолуді (у документі сказано: до 10 числа)."
    assert clean_text("Здайте внесок (документи приймає Сова до 15 числа).", 3) == \
        "Здайте внесок (документи приймає Сова до 15 числа)."
    # Several numbers go together, with nothing like "(,)", "()" or "жолуді і." left behind.
    for cited in ("([1], [2])", "([1])", "([1, рукавичка])", "(у документах [1] і [2])", "у документах [1] і [2]",
                  "[1], [2]", "[1] та [2]"):
        assert clean_text(f"Внесок 4 жолуді {cited}.", 3) == "Внесок 4 жолуді.", cited
    assert clean_text("Толока [2026].", 3) == "Толока [2026]."


def test_quoted_sentences_answer_the_topic(store, forest):
    # A sentence counts with the question over it ("Чим? Сухим дубовим листям."), and the title is no heading.
    add_archive(store, forest, "Як утеплити рукавичку в листяному лісі",
                "Як утеплити рукавичку в листяному лісі\nЦя порада для тих, чия рукавичка лежить у листяному лісі.\n"
                "Коли? Утеплюйте рукавичку тоді, коли облетить листя.\nЧим? Сухим дубовим листям.\n"
                "Тоді зимові вечори минатимуть у теплі.")
    add_archive(store, forest, "Грибний путівник листяного лісу",
                "Грибний путівник листяного лісу\nЛюбі сусіди з листяного лісу! Цей путівник для вас.\n"
                "Де шукати білі гриби?\nБілі гриби тримаються старих дубів.\nКоли йти по\nгриби?\n"
                "У листяному лісі гриби йдуть на 4–5 день після дощу.")
    answerer = Answerer(store, forest, FakeLLM(), TZ)
    docs = {doc.title: doc for doc in store.active_documents()}
    warming = answerer._topic_sentences(docs["Як утеплити рукавичку в листяному лісі"], "Чим утеплити рукавичку під вербою?")
    assert "Чим? Сухим дубовим листям." in warming and "Як утеплити рукавичку в листяному лісі" not in warming
    mushrooms = answerer._topic_sentences(docs["Грибний путівник листяного лісу"], "Де шукати гриби біля верби?")
    assert mushrooms.startswith("Де шукати білі гриби? Білі гриби тримаються старих дубів.")
    assert "Коли йти по гриби? У листяному лісі гриби йдуть на 4–5 день після дощу." in mushrooms
