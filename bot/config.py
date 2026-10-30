import os
from dataclasses import dataclass, field

DEFAULT_DRIVE_FOLDER_ID = "1b_hEjnkQfP8ZVP-OEbnYWqoGSFwXMAP3"
# Answers go down the list; qwen comes last, as the free tier lets it write only 1000 tokens a minute.
DEFAULT_GROQ_MODELS = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    groq_api_key: str
    groq_models: list[str] = field(default_factory=list)
    drive_folder_id: str = DEFAULT_DRIVE_FOLDER_ID
    google_api_key: str | None = None
    sync_minutes: int = 10
    db_path: str = "data/bot.db"
    timezone: str = "Europe/Kyiv"
    max_file_mb: int = 20
    ocr_langs: str = "ukr"
    invites_only: bool = False  # True: only animals invited by a neighbour may use the bot
    set_profile: bool = True  # the bot sets its own name, description and avatar


def env(name: str, default: str) -> str:
    """docker-compose passes unset settings as empty strings; empty means the default."""
    return os.getenv(name, "").strip() or default


def load_settings() -> Settings:
    missing = [name for name in ("TELEGRAM_BOT_TOKEN", "GROQ_API_KEY") if not os.getenv(name)]
    if missing:
        raise SystemExit(f"Missing environment variables: {', '.join(missing)}. See .env.example.")

    models = os.getenv("GROQ_MODELS") or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODELS
    return Settings(
        telegram_token=os.environ["TELEGRAM_BOT_TOKEN"],
        groq_api_key=os.environ["GROQ_API_KEY"],
        groq_models=[m.strip() for m in models.split(",") if m.strip()],
        drive_folder_id=os.getenv("DRIVE_FOLDER_ID") or DEFAULT_DRIVE_FOLDER_ID,
        google_api_key=os.getenv("GOOGLE_API_KEY") or None,
        sync_minutes=int(env("ARCHIVE_SYNC_MINUTES", "10")),
        db_path=env("DB_PATH", "data/bot.db"),
        timezone=env("TZ_NAME", "Europe/Kyiv"),
        max_file_mb=int(env("MAX_FILE_MB", "20")),
        ocr_langs=env("OCR_LANGS", "ukr"),
        invites_only=env("INVITES_ONLY", "false").lower() in ("1", "true", "yes", "так"),
        set_profile=env("SET_BOT_PROFILE", "true").lower() not in ("0", "false", "no", "ні"),
    )
