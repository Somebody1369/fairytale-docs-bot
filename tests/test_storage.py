from datetime import timedelta

from aiogram.fsm.storage.base import StorageKey

from bot.fsm import SQLiteStorage
from bot.places import CONIFER, Scope
from bot.storage import Store

from .conftest import NOW, add_animal, add_archive


def test_same_animal_same_title_replaces_previous_version(store):
    scope = Scope("mitten", "Під дубом")
    first, _ = add_animal(store, "Гриби", "старий текст", scope)
    second, replaced = add_animal(store, "Гриби", "новий текст", scope, when=NOW + timedelta(days=1))
    assert replaced.id == first.id
    assert [d.text for d in store.active_documents()] == ["новий текст"]


def test_other_animal_with_same_title_does_not_replace(store):
    scope = Scope("part", CONIFER)
    add_animal(store, "Гриби", "від Зайчика", scope, author_id=1)
    _, replaced = add_animal(store, "Гриби", "від Білочки", scope, author_id=2, name="Білочка")
    assert replaced is None
    assert len(store.active_documents()) == 2


def test_only_author_can_delete(store):
    doc, _ = add_animal(store, "Порада", "текст", Scope("forest"), author_id=1)
    assert store.delete_animal_document(doc.id, author_id=2) is None
    assert store.delete_animal_document(doc.id, author_id=1).id == doc.id
    assert store.active_documents() == []


def test_archive_keeps_only_newest_version_of_a_title(store, forest):
    add_archive(store, forest, "Членські внески рукавички Під дубом", "4 жолуді", file_id="a",
                modified=NOW - timedelta(days=20))
    add_archive(store, forest, "Членські внески рукавички Під дубом", "5 жолудів", file_id="b",
                modified=NOW - timedelta(days=1))
    assert [d.text for d in store.active_documents()] == ["5 жолудів"]

    # The newer file disappears from Drive: the previous version is valid again.
    store.mark_missing_archive_files({"a"})
    store.recompute_archive_status()
    assert [d.text for d in store.active_documents()] == ["4 жолуді"]


def test_archive_file_content_update(store, forest):
    add_archive(store, forest, "Правила толоки", "Толока в першу суботу жовтня", file_id="f")
    store.add_archive_version(drive_file_id="f", marker="m2", filename="Правила толоки.txt", title="Правила толоки",
                              scope=Scope("forest"), text="Толока в другу суботу жовтня", modified_at=NOW, now=NOW)
    store.recompute_archive_status()
    assert [d.text for d in store.active_documents()] == ["Толока в другу суботу жовтня"]


DUES = "Членські внески рукавички Під дубом"


def add_dues_version(store, forest, text, marker="m2"):
    """A new version of the dues file as sync_archive stores it, before recompute_archive_status() runs."""
    store.add_archive_version(drive_file_id="f", marker=marker, filename=f"{DUES}.txt", title=DUES,
                              scope=forest.archive_scope(DUES, text), text=text, modified_at=NOW, now=NOW)


def test_new_archive_version_waits_for_the_end_of_the_sync(store, forest):
    add_archive(store, forest, DUES, "Внесок 4 жолуді.", file_id="f")
    add_dues_version(store, forest, "Внесок 5 жолудів.")
    # The sync still reads the next files (the scans take about two minutes to read) or a restart cuts it short:
    # answers keep the old version alone instead of "4 жолуді" and "5 жолудів" both.
    assert [d.text for d in store.active_documents()] == ["Внесок 4 жолуді."]
    store.recompute_archive_status()
    assert [d.text for d in store.active_documents()] == ["Внесок 5 жолудів."]


def test_first_load_cut_short_is_not_a_loaded_archive(store, forest):
    add_dues_version(store, forest, "Внесок 4 жолуді.", marker="m1")
    # On the next start __main__ answers at once when has_archive() is true; half a folder must not count.
    assert not store.has_archive()
    store.recompute_archive_status()
    assert store.has_archive()


def test_files_gone_from_drive_are_counted_once(store, forest):
    add_archive(store, forest, DUES, "Внесок 4 жолуді.", file_id="f")
    add_dues_version(store, forest, "Внесок 5 жолудів.")
    add_archive(store, forest, "Правила толоки", "Толока в першу суботу жовтня", file_id="t")
    assert store.mark_missing_archive_files({"t"}) == 1  # one file, two versions
    assert store.mark_missing_archive_files({"t"}) == 0
    store.recompute_archive_status()
    assert [d.title for d in store.active_documents()] == ["Правила толоки"]


def test_archive_status_is_recomputed_without_the_texts(store, forest):
    # It runs at every sync over every version of every archive file: the texts are not needed for it.
    add_archive(store, forest, DUES, "Внесок 4 жолуді.", file_id="f")
    statements = []
    store.db.set_trace_callback(statements.append)
    store.recompute_archive_status()
    selects = [statement for statement in statements if statement.startswith("SELECT")]
    assert selects and not [select for select in selects if "*" in select or "text" in select]


def test_search_index_follows_a_sync_made_by_another_process(tmp_path, forest):
    # `python -m bot.ask` syncs the bot's database from another process. The bot's own next sync then finds every
    # status right already, and its search index must still be rebuilt.
    bot_store, terminal = Store(str(tmp_path / "bot.db")), Store(str(tmp_path / "bot.db"))
    add_archive(bot_store, forest, DUES, "Внесок 4 жолуді.", file_id="f")
    revision = bot_store.revision
    add_dues_version(terminal, forest, "Внесок 5 жолудів.")
    terminal.recompute_archive_status()
    bot_store.recompute_archive_status()
    assert bot_store.revision != revision
    assert [doc.text for doc in bot_store.active_documents()] == ["Внесок 5 жолудів."]
    bot_store.db.close()
    terminal.db.close()


def test_user_name_is_remembered(store):
    assert store.user_name(7) is None
    store.set_user_name(7, "Ведмідь", NOW)
    assert store.user_name(7) == "Ведмідь"


def test_first_invite_is_reused_until_someone_joins(store):
    assert not store.has_members()
    token = store.first_invite(NOW)
    assert store.first_invite(NOW + timedelta(hours=1)) == token  # a restart does not make a new one
    assert store.first_invite(NOW + timedelta(days=30)) == token  # nor a restart weeks later: it never expires
    assert store.use_invite(token, 7, NOW + timedelta(days=30))
    assert store.has_members() and store.first_invite(NOW) != token
    assert not store.use_invite(store.create_invite(7, NOW), 8, NOW + timedelta(days=8))  # expired


def test_operators_invite_never_expires_but_a_members_does(store):
    # Mishka's invite from the start log of 29 October must still work after 5 November.
    mishka = store.create_invite(None, NOW)                                    # the start log or python -m bot.invite
    assert store.use_invite(mishka, 1, NOW + timedelta(days=60))
    assert not store.use_invite(mishka, 2, NOW + timedelta(days=60))           # still one use only
    assert store.use_invite(store.create_invite(1, NOW), 2, NOW + timedelta(days=7))   # /invite: 7 days
    assert not store.use_invite(store.create_invite(1, NOW), 3, NOW + timedelta(days=8))
    store.revoke_member(2, by_id=1)
    assert not store.use_invite(store.create_invite(2, NOW), 4, NOW)           # the inviter must still be a member


def test_moving_a_document_to_a_place_with_its_older_version_replaces_it(store):
    old, _ = add_animal(store, "Гриби", "старий текст", Scope("part", CONIFER), when=NOW - timedelta(days=2))
    new, _ = add_animal(store, "Гриби", "новий текст", Scope("mitten", "Під ялиною"))
    store.change_scope(new.id, 1, Scope("part", CONIFER))
    assert [d.text for d in store.active_documents()] == ["новий текст"]


def test_inviter_closes_access_for_a_stranger_and_whoever_they_invited(store):
    store.use_invite(store.create_invite(None, NOW), 1, NOW)    # Мишка, the first animal
    store.use_invite(store.create_invite(1, NOW), 2, NOW)       # Мишка invites Ведмідь
    store.use_invite(store.create_invite(2, NOW), 3, NOW)       # Ведмідь invites a fox who turns out a stranger
    store.use_invite(store.create_invite(3, NOW), 4, NOW)       # the fox invites another stranger
    spare = store.create_invite(3, NOW)                         # and hides a spare link
    doc, _ = add_animal(store, "Пожежа", "Під дубом при пожежі збираємося під дубом.", Scope("mitten", "Під дубом"),
                        author_id=4, name="Лис")

    assert [member[0] for member in store.invited_by_chain(1)] == [2, 3, 4]
    assert store.in_invite_chain(4, 1) and not store.in_invite_chain(1, 4) and not store.in_invite_chain(2, 3)
    assert store.revoke_member(3, by_id=4) is None              # nobody below in the chain can do it
    # Ведмідь sleeps till spring, so Мишка, higher in the chain, closes the fox's access.
    assert sorted(store.revoke_member(3, by_id=1)) == [3, 4]
    assert store.is_member(2) and not store.is_member(3) and not store.is_member(4)
    assert store.get(doc.id).status == "revoked" and store.active_documents() == []

    assert not store.use_invite(spare, 5, NOW)                  # the fox's hidden link no longer works

    # Closed by mistake: a neighbour vouches for the animal again and its documents count again.
    assert store.use_invite(store.create_invite(1, NOW), 4, NOW)
    assert store.get(doc.id).status == "active"


def test_document_read_while_access_was_closed_comes_back_with_a_new_invite(store):
    store.use_invite(store.create_invite(None, NOW), 1, NOW)
    store.use_invite(store.create_invite(1, NOW), 2, NOW)
    old, _ = add_animal(store, "Гриби", "старий текст", Scope("forest"), author_id=2)
    store.revoke_member(2, by_id=1)
    # The new version was still being read: it is kept, not used, and stands in for the older one.
    new, replaced = store.add_animal_document(title="Гриби", filename="Гриби.txt", scope=Scope("forest"), author_id=2,
                                              author_name="Зайчик", text="новий текст", now=NOW, status="revoked")
    assert replaced.id == old.id and new.status == "revoked" and store.active_documents() == []
    assert store.use_invite(store.create_invite(1, NOW), 2, NOW)
    assert [d.text for d in store.active_documents()] == ["новий текст"]


def test_own_spare_link_does_not_bring_one_back(store):
    store.use_invite(store.create_invite(None, NOW), 1, NOW)
    store.use_invite(store.create_invite(1, NOW), 2, NOW)
    own = store.create_invite(2, NOW)
    store.db.execute("DELETE FROM members WHERE tg_id = 2")   # as if access were closed without clearing invites
    assert not store.use_invite(own, 2, NOW)


async def test_dialog_state_survives_a_new_connection(tmp_path):
    # A real restart opens the file again; a new Dispatcher on the same connection would also see
    # writes that were never committed.
    key = StorageKey(bot_id=1, chat_id=10, user_id=10)
    data = {"queue": [{"id": "ab12", "title": "Гриби"}], "saved": {"cd34": 7}}
    store = Store(str(tmp_path / "bot.db"))
    fsm = SQLiteStorage(store.db)
    await fsm.set_state(key, "Flow:waiting_doc_place")
    await fsm.update_data(key, data)
    store.db.close()

    store = Store(str(tmp_path / "bot.db"))
    fsm = SQLiteStorage(store.db)
    assert await fsm.get_state(key) == "Flow:waiting_doc_place"
    assert await fsm.get_data(key) == data
    assert await fsm.update_data(key, {"epoch": 1}) == {**data, "epoch": 1}
    await fsm.set_state(key, None)
    await fsm.set_data(key, {})
    assert await fsm.get_state(key) is None and await fsm.get_data(key) == {}
    store.db.close()
