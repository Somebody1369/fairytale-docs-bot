import pytest

from bot.config import DEFAULT_DRIVE_FOLDER_ID, load_settings


@pytest.fixture(autouse=True)
def tokens(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "1:x")
    monkeypatch.setenv("GROQ_API_KEY", "x")


def test_optional_settings_are_stripped_and_blank_means_the_default(monkeypatch):
    monkeypatch.setenv("GROQ_MODELS", " a , b ")
    monkeypatch.setenv("DRIVE_FOLDER_ID", " ")
    monkeypatch.setenv("GOOGLE_API_KEY", " ")
    settings = load_settings()
    assert settings.groq_models == ["a", "b"]
    assert settings.drive_folder_id == DEFAULT_DRIVE_FOLDER_ID and settings.google_api_key is None


@pytest.mark.parametrize("minutes", ["0", "-5"])
def test_archive_sync_needs_at_least_a_minute(monkeypatch, minutes):
    # With 0 the periodic sync would list Google Drive back to back, without a pause.
    monkeypatch.setenv("ARCHIVE_SYNC_MINUTES", minutes)
    with pytest.raises(SystemExit, match="ARCHIVE_SYNC_MINUTES"):
        load_settings()


def test_bot_is_closed_unless_switched_off(monkeypatch):
    monkeypatch.delenv("INVITES_ONLY", raising=False)
    assert load_settings().invites_only
    monkeypatch.setenv("INVITES_ONLY", "false")
    assert not load_settings().invites_only


def test_empty_settings_from_docker_compose_mean_defaults(monkeypatch):
    for name in ("GROQ_MODELS", "DRIVE_FOLDER_ID", "GOOGLE_API_KEY", "ARCHIVE_SYNC_MINUTES", "TZ_NAME", "INVITES_ONLY"):
        monkeypatch.setenv(name, "")
    settings = load_settings()
    assert settings.sync_minutes == 10 and settings.timezone == "Europe/Kyiv" and settings.invites_only
    assert settings.google_api_key is None and settings.groq_models[0] == "openai/gpt-oss-120b"
