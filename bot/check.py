"""Reference questions on the live model, to see that answers still hold after a change:

    docker compose run --rm bot python -m bot.check

The first one is the customer's own example. Every answer must mention the expected words
(facts or the source document); "need_place" means the bot must ask which mitten.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .answering import Answerer
from .archive import refresh_map
from .config import DEFAULT_GROQ_MODELS
from .llm import LLM
from .places import Forest
from .storage import Store

CHECKS = [
    ("Скільки внесок під дубом і до якого числа?",
     ["4 жолуд", "10", "Кабан", "2 лісов", "Членські внески рукавички Під дубом"]),
    ("Що їсти під вербою?", ["Чим поживитись поруч із рукавичкою Під вербою"]),
    ("Хто живе під вербою?", ["Жабка", "Качка", "Горобець"]),
    ("Де знайти їжу біля мурашника?", ["Чим поживитись поруч із рукавичкою Біля мурашника"]),
    ("Коли збирати жолуді під дубом?", ["мороз", "Чим поживитись поруч із рукавичкою Під дубом"]),
    ("Що робить Сова вночі?", ["Під дубом"]),
    ("Коли Ведмідь лягає спати?", ["У ліщині"]),
    ("Що робити при пожежі?", ["Що робити при пожежі"]),
    ("Скільки внесок?", "need_place"),
    ("Скільки внесок біля річки?", "need_place"),  # dues differ by mitten: the river's mittens are offered
    # The customer's other examples (30 October), with the facts and sources of her answers.
    ("Біля греблі мисливці! Як кричати?", ["три", "Сорок", "гущавин", "Як кричати, коли бачиш мисливців"]),
    ("Через скільки днів після дощу йти по гриби в соснах?", ["2", "волог", "Грибний путівник хвойного лісу"]),
    ("Де шукати гриби біля верби?", ["дуб", "берез", "опеньк", "Грибний путівник листяного лісу",
                                     "Грибний путівник біля річки", "Чим поживитись поруч із рукавичкою Під вербою"]),
    ("Коли утеплювати рукавичку на узліссі?", ["листопад", "Календар сезонів"]),
    ("Що їсти в хвойному лісі?", ["Що їсти в хвойному лісі"]),  # the part's own advice, no mitten's
]


async def main() -> int:
    store = Store(os.getenv("DB_PATH", "data/bot.db"))
    forest = Forest()
    refresh_map(store, forest)
    models = (os.getenv("GROQ_MODELS") or DEFAULT_GROQ_MODELS).split(",")
    llm = LLM(os.environ["GROQ_API_KEY"], models)
    await llm.check_models()
    answerer = Answerer(store, forest, llm, ZoneInfo(os.getenv("TZ_NAME") or "Europe/Kyiv"))
    failed = 0
    for question, expected in CHECKS:
        reply = await answerer.ask(question, datetime.now(timezone.utc))
        if expected == "need_place":
            ok, shown = reply.kind == "need_place", f"[{reply.kind}]"
        else:
            missing = [word for word in expected if word.lower() not in reply.text.lower()]
            ok, shown = reply.kind == "answer" and not missing, reply.text + (f"\n  missing: {missing}" if missing else "")
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {question}\n  " + shown.replace("\n", "\n  ") + "\n")
    print(f"{len(CHECKS) - failed} of {len(CHECKS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
