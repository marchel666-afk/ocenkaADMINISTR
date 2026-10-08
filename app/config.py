"""Настройки сервиса. Читаются из переменных окружения и файла .env в корне проекта."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "да"}


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


@dataclass
class Settings:
    # Общие
    clinic_name: str = "Варикоза нет"
    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    checklists_path: Path = field(default_factory=lambda: BASE_DIR / "config" / "checklists.yaml")
    host: str = "0.0.0.0"
    port: int = 8000
    app_password: str = ""  # если задан — вход по логину admin и этому паролю

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "anthropic/claude-haiku-5.5"
    llm_reasoning_effort: str = ""  # пусто — не передавать; иначе low / medium / high
    llm_max_tokens: int = 12000
    llm_prompt_cache: bool = True
    llm_timeout_sec: int = 240

    # Расшифровка (faster-whisper)
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_threads: int = 0  # 0 — автоматически
    whisper_initial_prompt: str = (
        "Клиника лазерной хирургии «Варикоза нет», администратор. "
        "Флеболог, УЗИ вен нижних конечностей, варикоз, склеротерапия, КЛАКС, СНИЛС."
    )
    stereo_admin_channel: str = "auto"  # auto / left / right

    # Обработка
    min_call_seconds: int = 20
    worker_enabled: bool = True
    worker_poll_sec: float = 3.0

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'ocenka.db').as_posix()}"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or BASE_DIR / ".env", override=False)
        e = os.environ.get
        s = cls()
        s.clinic_name = e("CLINIC_NAME", s.clinic_name)
        if e("DATA_DIR"):
            s.data_dir = Path(e("DATA_DIR")).expanduser()
        if e("CHECKLISTS_PATH"):
            s.checklists_path = Path(e("CHECKLISTS_PATH")).expanduser()
        s.host = e("HOST", s.host)
        s.port = _int(e("PORT"), s.port)
        s.app_password = e("APP_PASSWORD", s.app_password)

        s.openrouter_api_key = e("OPENROUTER_API_KEY", s.openrouter_api_key)
        s.openrouter_base_url = e("OPENROUTER_BASE_URL", s.openrouter_base_url).rstrip("/")
        s.llm_model = e("LLM_MODEL", s.llm_model)
        s.llm_reasoning_effort = e("LLM_REASONING_EFFORT", s.llm_reasoning_effort).strip().lower()
        s.llm_max_tokens = _int(e("LLM_MAX_TOKENS"), s.llm_max_tokens)
        s.llm_prompt_cache = _bool(e("LLM_PROMPT_CACHE"), s.llm_prompt_cache)
        s.llm_timeout_sec = _int(e("LLM_TIMEOUT_SEC"), s.llm_timeout_sec)

        s.whisper_model = e("WHISPER_MODEL", s.whisper_model)
        s.whisper_device = e("WHISPER_DEVICE", s.whisper_device)
        s.whisper_compute_type = e("WHISPER_COMPUTE_TYPE", s.whisper_compute_type)
        s.whisper_threads = _int(e("WHISPER_THREADS"), s.whisper_threads)
        s.whisper_initial_prompt = e("WHISPER_INITIAL_PROMPT", s.whisper_initial_prompt)
        s.stereo_admin_channel = e("STEREO_ADMIN_CHANNEL", s.stereo_admin_channel).strip().lower()

        s.min_call_seconds = _int(e("MIN_CALL_SECONDS"), s.min_call_seconds)
        s.worker_enabled = _bool(e("WORKER_ENABLED"), s.worker_enabled)
        return s

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
