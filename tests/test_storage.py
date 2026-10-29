from datetime import timedelta

from bot.places import CONIFER, Scope

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


def test_user_name_is_remembered(store):
    assert store.user_name(7) is None
    store.set_user_name(7, "Ведмідь", NOW)
    assert store.user_name(7) == "Ведмідь"


def test_first_invite_is_reused_until_someone_joins(store):
    assert not store.has_members()
    token = store.first_invite(NOW)
    assert store.first_invite(NOW + timedelta(hours=1)) == token  # a restart does not make a new one
    assert store.use_invite(token, 7, NOW)
    assert store.has_members() and store.first_invite(NOW) != token
    assert not store.use_invite(store.create_invite(7, NOW), 8, NOW + timedelta(days=8))  # expired


def test_bot_is_closed_unless_switched_off(monkeypatch):
    from bot.config import load_settings
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.delenv("INVITES_ONLY", raising=False)
    assert load_settings().invites_only
    monkeypatch.setenv("INVITES_ONLY", "false")
    assert not load_settings().invites_only


def test_inviter_closes_access_for_a_stranger_and_whoever_they_invited(store):
    store.use_invite(store.create_invite(None, NOW), 1, NOW)    # Мишка, the first animal
    store.use_invite(store.create_invite(1, NOW), 2, NOW)       # Мишка invites Ведмідь
    store.use_invite(store.create_invite(2, NOW), 3, NOW)       # Ведмідь invites a fox who turns out a stranger
    store.use_invite(store.create_invite(3, NOW), 4, NOW)       # the fox invites another stranger
    spare = store.create_invite(3, NOW)                         # and hides a spare link
    doc, _ = add_animal(store, "Пожежа", "Під дубом при пожежі збираємося під дубом.", Scope("mitten", "Під дубом"),
                        author_id=4, name="Лис")

    assert store.revoke_member(3, by_id=4) is None              # nobody below in the chain can do it
    # Ведмідь sleeps till spring, so Мишка, higher in the chain, closes the fox's access.
    assert sorted(store.revoke_member(3, by_id=1)) == [3, 4]
    assert store.is_member(2) and not store.is_member(3) and not store.is_member(4)
    assert store.get(doc.id).status == "revoked" and store.active_documents() == []

    assert not store.use_invite(spare, 5, NOW)                  # the fox's hidden link no longer works

    # Closed by mistake: a neighbour vouches for the animal again and its documents count again.
    assert store.use_invite(store.create_invite(1, NOW), 4, NOW)
    assert store.get(doc.id).status == "active"


def test_own_spare_link_does_not_bring_one_back(store):
    store.use_invite(store.create_invite(None, NOW), 1, NOW)
    store.use_invite(store.create_invite(1, NOW), 2, NOW)
    own = store.create_invite(2, NOW)
    store.db.execute("DELETE FROM members WHERE tg_id = 2")   # as if access were closed without clearing invites
    assert not store.use_invite(own, 2, NOW)
