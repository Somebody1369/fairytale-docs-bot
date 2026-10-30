import io

import pytest
from datetime import datetime, timezone

import docx
from pathlib import Path

from bot import archive as archive_module
from bot.archive import DriveFile, parse_folder_page, parse_shown_date, sync_archive
from bot.classify import classify_document, scope_from_caption
from bot.extract import _clean_ocr, extract_text, old_word_supported, title_from_filename
from bot.places import CONIFER, DECIDUOUS, FOREST_SCOPE, Scope

from .conftest import FOREST_MAP_TEXT, FakeLLM

NOW = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)

FOLDER_PAGE = """
<div class="flip-entry" id="entry-abc123" tabindex="0" role="link"><div class="flip-entry-info">
<a href="https://drive.google.com/file/d/abc123/view"><div class="flip-entry-title">Календар сезонів.txt</div></a>
</div><div class="flip-entry-last-modified"><div>Oct 13</div></div></div>
<div class="flip-entry" id="entry-def456" tabindex="0" role="link"><div class="flip-entry-info">
<a href="https://drive.google.com/file/d/def456/view"><div class="flip-entry-title">Карта лісу.pdf</div></a>
</div><div class="flip-entry-last-modified"><div>3:45 PM</div></div></div>
"""


def test_folder_page_is_parsed():
    files = parse_folder_page(FOLDER_PAGE, NOW)
    assert [(f.id, f.name) for f in files] == [("abc123", "Календар сезонів.txt"), ("def456", "Карта лісу.pdf")]
    assert files[0].modified == datetime(2026, 10, 13, tzinfo=timezone.utc)
    assert files[1].modified == NOW


def test_google_docs_document_in_public_folder_is_exported_as_text():
    page = FOLDER_PAGE + """
<div class="flip-entry" id="entry-gdoc1" tabindex="0" role="link"><div class="flip-entry-info">
<a href="https://docs.google.com/document/d/gdoc1/edit?usp=drive_web"><div class="flip-entry-title">Нові правила</div></a>
</div><div class="flip-entry-last-modified"><div>Oct 20</div></div></div>
"""
    files = {f.id: f for f in parse_folder_page(page, NOW)}
    assert files["gdoc1"].name == "Нові правила.txt" and files["gdoc1"].export_mime == "text/plain"
    assert files["abc123"].export_mime is None


def test_shown_dates():
    assert parse_shown_date("Oct 13, 2025", NOW) == datetime(2025, 10, 13, tzinfo=timezone.utc)
    assert parse_shown_date("10/13/25", NOW) == datetime(2025, 10, 13, tzinfo=timezone.utc)


class FakeArchive:
    api_key = None
    ocr_langs = "ukr"

    def __init__(self, files: dict[str, tuple[str, str, str]]):
        self.files = files  # id -> (name, marker, text)

    async def list_files(self, session):
        return [DriveFile(id=i, name=n, marker=m, modified=NOW) for i, (n, m, _) in self.files.items()]

    async def download(self, session, file):
        return self.files[file.id][2].encode("utf-8")


async def test_sync_adds_updates_and_removes(store, forest):
    archive = FakeArchive({
        "1": ("Календар сезонів.txt", "Oct 13", "Цей календар для всіх мешканців лісу. Гриби з серпня."),
        "2": ("Небезпеки хвойного лісу.txt", "Oct 13", "Живиця липне до шерсті."),
    })
    report = await sync_archive(store, forest, archive)
    assert (report.added, report.updated) == (2, 0)
    scopes = {d.title: d.scope for d in store.active_documents()}
    assert scopes == {"Календар сезонів": FOREST_SCOPE, "Небезпеки хвойного лісу": Scope("part", CONIFER)}

    archive.files["2"] = ("Небезпеки хвойного лісу.txt", "4:10 PM", "Живиця липне. Відмивайте олією.")
    del archive.files["1"]
    report = await sync_archive(store, forest, archive)
    assert (report.updated, report.removed) == (1, 1)
    assert [d.text for d in store.active_documents()] == ["Живиця липне. Відмивайте олією."]


async def test_renamed_archive_file_gets_its_new_title(store, forest):
    text = "Ця пам'ятка для мешканців біля річки. Броди мілкі в серпні."
    archive = FakeArchive({"1": ("Броди.txt", "Oct 13", text)})
    await sync_archive(store, forest, archive)
    archive.files["1"] = ("Броди і коли бобри відкривають греблю.txt", "Oct 14", text)
    await sync_archive(store, forest, archive)
    [doc] = store.active_documents()
    assert doc.title == "Броди і коли бобри відкривають греблю"
    assert doc.modified_at == NOW  # the advice itself did not change


async def test_sync_reads_forest_map(store, forest):
    changed = FOREST_MAP_TEXT.replace("Біля мурашника — хвойний ліс.", "Біля мурашника — листяний ліс.")
    await sync_archive(store, forest, FakeArchive({"m": ("Карта лісу.txt", "Oct 13", changed)}))
    assert forest.mitten_parts["Біля мурашника"] == {"deciduous"}


def test_txt_in_windows_encoding_and_docx():
    assert extract_text("Порада для лісу".encode("cp1251"), "a.txt") == "Порада для лісу"
    document = docx.Document()
    document.add_paragraph("Гриби під ялинами")
    buffer = io.BytesIO()
    document.save(buffer)
    assert extract_text(buffer.getvalue(), "a.docx") == "Гриби під ялинами"


def test_title_from_filename():
    assert title_from_filename("Гриби_під_ялинами (1).pdf") == "Гриби під ялинами"


async def test_document_place_from_title_skips_model(forest):
    llm = FakeLLM()
    scope = await classify_document("Чергування Під дубом", "текст", forest, llm)
    assert scope == Scope("mitten", "Під дубом") and llm.prompts == []


async def test_document_place_from_text_without_model(forest):
    llm = FakeLLM()
    assert await classify_document("Порада", "Під ялинами після дощу ростуть рижики.", forest, llm) \
        == Scope("part", CONIFER)
    assert await classify_document("Нагадування", "Шановні мешканці рукавички Під дубом!", forest, llm) \
        == Scope("mitten", "Під дубом")
    assert await classify_document("Вогонь", "Ця порада для всіх мешканців лісу.", forest, llm) == FOREST_SCOPE
    assert llm.prompts == []


async def test_document_without_any_place_is_not_guessed(forest):
    llm = FakeLLM({"kind": "forest", "value": ""})
    assert await classify_document("Нотатка", "Збирайте гриби після дощу.", forest, llm) is None
    assert llm.prompts == []


async def test_model_chooses_only_between_mentioned_places(forest):
    text = "Гриби збирайте у хвойному лісі серед сосен, на галявинах після дощу."
    llm = FakeLLM({"kind": "part", "value": "хвойний ліс"})
    assert await classify_document("Порада", text, forest, llm) == Scope("part", CONIFER)
    llm = FakeLLM({"kind": "mitten", "value": "Під дубом"})  # not mentioned in the text
    assert await classify_document("Порада", text, forest, llm) is None
    # Two forest parts: no single place covers the document, so the sender is asked, not the model.
    llm = FakeLLM({"kind": "part", "value": "біля річки"})
    assert await classify_document("Порада", "Гриби ростуть серед сосен, а ягоди біля річки.", forest, llm) is None
    assert llm.prompts == []


def test_caption_names_the_place(forest):
    assert scope_from_caption("це для Під вербою", forest) == Scope("mitten", "Під вербою")
    assert scope_from_caption("для хвойного лісу", forest) == Scope("part", CONIFER)
    assert scope_from_caption("для всього лісу", forest) == FOREST_SCOPE
    assert scope_from_caption("ось моя порада", forest) is None
    assert scope_from_caption(None, forest) is None


async def test_empty_drive_listing_keeps_the_archive(store, forest):
    archive = FakeArchive({"1": ("Календар сезонів.txt", "Oct 13", "Цей календар для всіх мешканців лісу.")})
    await sync_archive(store, forest, archive)
    archive.files.clear()  # a consent page or a new page layout looks like an empty folder
    with pytest.raises(RuntimeError):
        await sync_archive(store, forest, archive)
    assert [d.title for d in store.active_documents()] == ["Календар сезонів"]


def test_ocr_digit_three_read_as_letter():
    assert _clean_ocr("Внесок становить З шишки з дорослого звіра.") == "Внесок становить 3 шишки з дорослого звіра."
    assert _clean_ocr("Видра може платити рибою: 1 рибина за З гілки.") == "Видра може платити рибою: 1 рибина за 3 гілки."
    assert _clean_ocr("З берега не стрибайте.") == "З берега не стрибайте."


async def test_better_reading_keeps_drive_date(store, forest, monkeypatch):
    archive = FakeArchive({"1": ("Внески Під ялиною.txt", "Oct 13", "Внесок З шишки.")})
    await sync_archive(store, forest, archive)
    first = store.active_documents()[0]

    # The bot learns to read the file better; the file on Drive did not change.
    monkeypatch.setattr(archive_module, "EXTRACT_VERSION", "test")
    archive.files["1"] = ("Внески Під ялиною.txt", "Oct 13", "Внесок 3 шишки.")
    await sync_archive(store, forest, archive)
    [current] = store.active_documents()
    assert current.text == "Внесок 3 шишки." and current.modified_at == first.modified_at


async def test_addressee_decides_over_places_inside_the_advice(forest):
    llm = FakeLLM()
    assert await classify_document("Гриби", "Порада для листяного лісу: білі гриби шукайте під дубом.", forest, llm) \
        == Scope("part", DECIDUOUS)
    assert await classify_document(
        "Шишки", "Ця порада для мешканців хвойного лісу. Найкращі шишки у соснах біля стежки.", forest, llm) \
        == Scope("part", CONIFER)
    assert await classify_document("Збори", "Шановні мешканці рукавички Під вербою! Збори в суботу.", forest, llm) \
        == Scope("mitten", "Під вербою")
    assert llm.prompts == []


async def test_mitten_and_part_without_addressee_are_not_decided_by_keywords(forest):
    # A mitten mentioned next to a forest part may be just a place inside general advice: ask the model.
    text = "У хвойному лісі восени багато маслюків, особливо у соснах."
    llm = FakeLLM({"kind": "part", "value": "хвойний ліс"})
    assert await classify_document("Маслюки", text, forest, llm) == Scope("part", CONIFER)
    assert len(llm.prompts) == 1


@pytest.mark.skipif(not old_word_supported(), reason="antiword is installed in the Docker image")
def test_old_word_document():
    data = (Path(__file__).parent / "fixtures" / "old_word.doc").read_bytes()
    assert extract_text(data, "Гриби.doc").startswith("Порада для листяного лісу.\nБілі гриби шукайте під дубом")


async def test_title_with_two_places_is_not_decided_by_the_title(forest):
    llm = FakeLLM()
    assert await classify_document("Гриби Під дубом і в соснах", "Збирайте після дощу.", forest, llm) is None
    assert await classify_document("Внески Під дубом, листяний ліс", "текст", forest, llm) == Scope("mitten", "Під дубом")
    assert llm.prompts == []


def test_long_scan_is_refused_instead_of_cut(monkeypatch):
    import pypdfium2 as pdfium
    from bot import extract

    pdf = pdfium.PdfDocument.new()
    for _ in range(extract.MAX_OCR_PAGES + 1):
        pdf.new_page(200, 200)
    data = io.BytesIO()
    pdf.save(data)
    monkeypatch.setattr(extract, "_ocr_page", lambda page, langs: pytest.fail("no page is read before the check"))
    with pytest.raises(extract.TooManyScannedPages):
        extract_text(data.getvalue(), "scan.pdf")


async def test_whole_forest_greeting_is_the_addressee(forest):
    # The place named inside the advice does not decide: the memo is for the whole forest.
    llm = FakeLLM()
    text = "Шановні мешканці лісу! При пожежі всі збираються біля бобрової греблі."
    assert await classify_document("Пам'ятка", text, forest, llm) == FOREST_SCOPE
    assert llm.prompts == []


async def test_addressee_decides_over_the_file_name(forest):
    llm = FakeLLM()
    text = "Порада для листяного лісу: білі гриби шукайте під дубом."
    assert await classify_document("Білі гриби під дубом", text, forest, llm) == Scope("part", DECIDUOUS)


async def test_tree_named_in_advice_is_not_a_mitten_by_itself(forest):
    # "під ялиною" may be any spruce: the model decides, and when it cannot tell, the sender is asked.
    llm = FakeLLM({"kind": "unclear", "value": ""})
    assert await classify_document("Рижики", "Рижики ростуть під ялиною, де багато моху.", forest, llm) is None
    prompt = llm.prompts[0][1]
    assert 'Місця, згадані в документі: "під ялиною" (рукавичка або просто дерево чи місцевість)' in prompt
    assert ('Можливі відповіді: {"kind": "mitten", "value": "Під ялиною"}; {"kind": "part", "value": "хвойний ліс"}; '
            'або "unclear"') in prompt
    # The model may see that the advice is about the mitten, but a tree word alone does not prove it: ask.
    llm = FakeLLM({"kind": "mitten", "value": "Під ялиною"})
    assert await classify_document("Рижики", "Рижики ростуть під ялиною, де багато моху.", forest, llm) is None
    # A note that opens with the mitten is about it (the fire example agreed with the customer at 23:53).
    llm = FakeLLM({"kind": "mitten", "value": "Під дубом"})
    assert await classify_document("Пожежа", "Під дубом при пожежі збираємося під дубом.", forest, llm) \
        == Scope("mitten", "Під дубом")
    llm = FakeLLM()
    assert await classify_document("Чергування", "У рукавичці Під дубом чергують уночі.", forest, llm) \
        == Scope("mitten", "Під дубом")
    assert llm.prompts == []  # named as a mitten, so no doubt


async def test_advice_for_several_mittens_is_not_filed_under_one(forest):
    llm = FakeLLM({"kind": "mitten", "value": "Під дубом"})
    assert await classify_document("Гриби", "Білі гриби шукайте під дубом, а опеньки в ліщині.", forest, llm) is None
    llm = FakeLLM({"kind": "part", "value": "листяний ліс"})  # both are in the deciduous forest
    assert await classify_document("Гриби", "Білі гриби шукайте під дубом, а опеньки в ліщині.", forest, llm) \
        == Scope("part", DECIDUOUS)
    # Mittens of different forest parts and no word on who the guide is for: the sender is asked.
    llm = FakeLLM({"kind": "part", "value": "хвойний ліс"})
    guide = "Осінній путівник.\nПід дубом: жолуді.\nУ ліщині: горіхи.\nБіля сосон: шишки."
    assert await classify_document("Путівник", guide, forest, llm) is None and llm.prompts == []


async def test_places_after_the_first_pages_count(forest):
    filler = "Восени дні коротшають, тож запаси робіть заздалегідь. " * 70
    text = f"У рукавичці Під дубом запасайте жолуді.\n{filler}\nА в рукавичці У соснах запасайте шишки."
    assert len(text) > 3000
    llm = FakeLLM({"kind": "mitten", "value": "Під дубом"})
    assert await classify_document("Запаси", text, forest, llm) is None


@pytest.mark.parametrize("title, text, expected", [
    # The whole forest, however the memo opens; the gathering place inside does not decide.
    ("Пам'ятка", "Шановні мешканці лісу\nПри пожежі всі збираються біля бобрової греблі.", FOREST_SCOPE),
    ("Нагадування", "Нагадування для всього лісу про збори під дубом щосуботи.", FOREST_SCOPE),
    ("Тиша", "Порада для всіх: взимку не шуміть після заходу сонця.", FOREST_SCOPE),
    ("Путівник для всього лісу", "Під дубом: жолуді.\nУ ліщині: горіхи.", FOREST_SCOPE),
    # A mitten named as a mitten, in any quotes.
    ("Чергування", "У рукавичці “Під дубом” чергують уночі.", Scope("mitten", "Під дубом")),
    ("Чергування", "Чергування у рукавичці під назвою Під дубом.", Scope("mitten", "Під дубом")),
    # The file name with one place, when the text has no addressee of its own.
    ("Внески Під вербою", "Для зручності внески можна здавати Боброві біля бобрової греблі.", Scope("mitten", "Під вербою")),
    ("Гриби хвойного лісу", "Для засолювання рижики шукайте під ялиною.", Scope("part", CONIFER)),
])
async def test_clear_place_is_decided_without_the_model(forest, title, text, expected):
    llm = FakeLLM({"kind": "unclear", "value": ""})
    assert await classify_document(title, text, forest, llm) == expected
    assert llm.prompts == []


@pytest.mark.parametrize("title, text", [
    # A place inside the advice is not who it is for.
    ("Порада", "Для сушіння рижики шукайте під ялиною."),
    ("Порада", "Звірі збирають жолуді під дубом після першого морозу."),
    ("Гриби", "Під ялиною ростуть рижики, а для сушіння краще брати гриби під дубом."),
    # "усіх звірів" or "весь ліс" inside a mitten's note does not make it the whole forest's.
    ("Внески", "Внески збирає Кабан з усіх звірів до 10 числа: 4 жолуді з дорослого звіра на місяць."),
    ("Збори", "Усім звірам нашої рукавички: чергування вночі по черзі."),
    ("Чергування", "Кожної ночі черговий обходить рукавичку. Якщо побачиш дим, буди всіх: вогонь може знищити весь ліс."),
    # Another mitten named as a destination or a comparison.
    ("Стежки до сусідів", "До рукавички Під дубом іти 200 кроків на схід, повз старий пень."),
    ("Внески", "Як і в рукавичці Під дубом, внески здаємо до 10 числа: 3 шишки з дорослого звіра."),
    # A tree phrase in the file name is no more a mitten than in the text.
    ("Рижики під ялиною", "Рижики збирайте після дощу."),
    # The file name and the addressee disagree.
    ("Чергування Під дубом", "Порада для мешканців хвойного лісу: чергуйте по двоє."),
    # Sections for different places: no single place covers the document.
    ("Путівник", "Осінній путівник.\nУ хвойному лісі: шишки.\nУ листяному лісі: жолуді.\nБіля річки: очерет."),
    ("Путівник", "Осінній путівник.\nПід дубом: жолуді.\nБіля річки: очерет."),
])
async def test_unclear_place_is_asked(forest, title, text):
    llm = FakeLLM({"kind": "mitten", "value": "Під дубом"}, {"kind": "part", "value": "хвойний ліс"})
    assert await classify_document(title, text, forest, llm) is None


async def test_place_word_deep_in_a_general_guide_does_not_decide(forest):
    filler = "Взимку тримайте запаси сухими й провітрюйте комору в ясні дні. " * 55
    for ending in ("І не ходіть по тонкій кризі на річці.", "У рукавичці Під дубом теж є сухе листя."):
        llm = FakeLLM({"kind": "unclear", "value": ""})
        assert await classify_document("Зимові поради", f"{filler}\n{ending}", forest, llm) is None


async def test_forest_part_and_whole_forest_together_are_not_decided_by_words(forest):
    text = "Рижики ростуть під ялинами після дощу. " + "Збирайте їх зранку. " * 10 + "Ця порада стосується всього лісу."
    llm = FakeLLM({"kind": "unclear", "value": ""})
    assert await classify_document("Рижики", text, forest, llm) is None and len(llm.prompts) == 1
    assert '{"kind": "forest", "value": ""}' in llm.prompts[0][1]


async def test_everyone_who_lives_somewhere_is_that_place(forest):
    llm = FakeLLM()
    assert await classify_document("Небезпеки", "Для всіх, хто живе біля річки. Вода дає життя.", forest, llm) \
        == Scope("part", "river")
    assert await classify_document("Небезпеки", "Ця пам'ятка для всіх, хто живе в хвойному лісі.", forest, llm) \
        == Scope("part", CONIFER)
    assert await classify_document("Гриби", "Для всіх, хто любить гриби: збирайте зранку.", forest, llm) == FOREST_SCOPE
    assert llm.prompts == []


@pytest.mark.parametrize("text", [
    # A sentence about the forest's animals or a greeting to all: a mitten's note may read so too.
    "Мешканці лісу при пожежі збираються біля бобрової греблі.",
    "Усім звірам! Взимку не шуміть після заходу сонця.",
])
async def test_whole_forest_by_the_model_when_not_addressed_so(forest, text):
    llm = FakeLLM({"kind": "forest", "value": ""})
    assert await classify_document("Порада", text, forest, llm) == FOREST_SCOPE
    assert '{"kind": "forest", "value": ""}' in llm.prompts[0][1]
    llm = FakeLLM({"kind": "mitten", "value": "Біля бобрової греблі"})
    assert await classify_document("Порада", text, forest, llm) is None


@pytest.mark.parametrize("title, text, expected", [
    ("Небезпеки", "Порада для хвойного та листяного лісу: утеплюйте рукавички в жовтні.", None),
    ("Збори", "Шановні мешканці рукавичок «Під дубом» та «У соснах»! Спільні збори в суботу.", None),
    ("Мурахи", "Для всіх, хто проходить повз мурашник: не наступайте на мурах.", None),
    ("Жолуді", "Для всіх, хто живе під дубом: жолуді збираємо разом.", Scope("mitten", "Під дубом")),
    ("Гриби", "Мешканці лісу під ялинами! Рижики вже пішли.", Scope("part", CONIFER)),
    ("Лід", "Порада для всіх мешканців лісу біля річки: не ходіть по тонкому льоду.", Scope("part", "river")),
    ("Порада", "Мешканці рукавички Під дубом попросили мене написати пораду для хвойного лісу: шишки в жовтні.",
     Scope("part", CONIFER)),
    ("Збори", "Мешканці всіх рукавичок, увага! Збори в суботу.", FOREST_SCOPE),
    ("Варення", "Порічки збирайте в липні, коли вони зовсім червоні.", None),
    ("Утеплення", "Перед зимою утепліть рукавичку хвойними гілками і мохом.", None),
    ("Вода", "Не залишайте малечу біля води без нагляду.", None),
    ("Повінь", "Біля річки навесні буває повінь, тримайте запаси вище.", Scope("part", "river")),
    ("Правила", "Правила рукавички Під дубом. Не ламайте гілки ялинки біля входу.", Scope("mitten", "Під дубом")),
    ("ВНЕСКИ ПІД ДУБОМ", "Внесок 4 жолуді з дорослого звіра.", Scope("mitten", "Під дубом")),
    ("Як дійти до Під дубом", "Від нашої рукавички йдіть на схід 200 кроків.", None),
    ("Звірі лісу та їхні рукавички", "У рукавичці Під дубом живуть Кабан, Сова і Їжак.", None),
    ("Внески біля річки", "Шановні мешканці рукавички Під вербою! Внески до 10 числа.", Scope("mitten", "Під вербою")),
])
async def test_addressee_file_name_and_text_together(forest, title, text, expected):
    # Several places addressed, places merely passed, words that only look like places, and how the file
    # name and the text agree. Nothing here needs the model; "unclear" stands for asking the sender.
    assert await classify_document(title, text, forest, FakeLLM({"kind": "unclear", "value": ""})) == expected
