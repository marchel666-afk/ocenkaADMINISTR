import json
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import LLMResponse
from app.transcription import ADMIN, PATIENT, Segment, Transcript

ROOT = Path(__file__).resolve().parent.parent

SAMPLE_DIALOG = [
    (ADMIN, "Добрый день. Клиника лазерной хирургии «Варикоза нет», администратор Анна. Слушаю Вас."),
    (PATIENT, "Здравствуйте, хочу записаться к флебологу, у меня вены на ногах выступают."),
    (ADMIN, "Как я могу к Вам обращаться?"),
    (PATIENT, "Ирина Петровна."),
    (ADMIN, "Ирина Петровна, Вы у нас уже были или впервые?"),
    (PATIENT, "Впервые."),
    (ADMIN, "Давайте запишемся на консультацию. Вам удобнее утро или вечер?"),
    (PATIENT, "Вечер."),
    (ADMIN, "Ирина Петровна, записала Вас на 15 октября в 18:00 к доктору Иванову. Стоимость приёма 2000 рублей."),
    (PATIENT, "Спасибо, до свидания."),
    (ADMIN, "Спасибо, что доверяете нам. Всего доброго!"),
]


def sample_transcript(duration: float = 95.0) -> Transcript:
    segments = [
        Segment(start=i * 8.0, end=i * 8.0 + 7.0, text=text, speaker=speaker)
        for i, (speaker, text) in enumerate(SAMPLE_DIALOG)
    ]
    return Transcript(segments=segments, duration=duration, stereo=True)


def sample_text() -> str:
    labels = {ADMIN: "Администратор", PATIENT: "Пациент"}
    return "\n".join(f"{labels[s]}: {t}" for s, t in SAMPLE_DIALOG)


class FakeTranscriber:
    def __init__(self, transcript: Transcript | None = None):
        self.transcript = transcript or sample_transcript()
        self.calls = 0

    def transcribe(self, path: Path) -> Transcript:
        assert path.is_file(), path
        self.calls += 1
        return self.transcript


_CODE_RE = re.compile(r"^([A-Z]{1,2}\d{2}(?:_\d)?) \[", re.MULTILINE)


class FakeLLM:
    """Отвечает как модель: тип звонка или оценка «да» по всем пунктам, кроме заданных."""

    def __init__(self, call_type="incoming", admin_name="Анна", failed=("IN11", "G04"), na=("G26",), raw=None):
        self.call_type = call_type
        self.admin_name = admin_name
        self.failed = set(failed)
        self.na = set(na)
        self.raw = list(raw or [])  # заранее заданные ответы (строки) — отдаются первыми
        self.requests: list[tuple[list[str], str]] = []
        self.models: list[str | None] = []

    def complete(self, system_parts, user, max_tokens=None, model=None):
        self.requests.append((system_parts, user))
        self.models.append(model)
        if self.raw:
            return LLMResponse(text=self.raw.pop(0), model="fake", prompt_tokens=10, completion_tokens=5)
        system = "\n".join(system_parts)
        if "определи его тип" in system:
            body = {"call_type": self.call_type, "admin_name": self.admin_name, "reason": "тест"}
        else:
            codes = _CODE_RE.findall(system)
            body = {
                "criteria": [
                    {
                        "id": c,
                        "result": "no" if c in self.failed else "na" if c in self.na else "yes",
                        "evidence": "Добрый день" if c not in self.failed else "",
                        "comment": "проверка",
                    }
                    for c in codes
                ],
                "summary": "Хороший звонок.",
                "strengths": ["Вежливость"],
                "improvements": [
                    {"criterion_id": "IN11", "issue": "Нет регалий врача", "recommendation": "Назовите регалии",
                     "example": "Доктор Иванов — флеболог высшей категории"}
                ],
            }
        return LLMResponse(
            text="```json\n" + json.dumps(body, ensure_ascii=False) + "\n```",
            model="fake-model",
            prompt_tokens=1000,
            completion_tokens=500,
            cost_usd=0.001,
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.data_dir = tmp_path / "data"
    s.checklists_path = ROOT / "config" / "checklists.yaml"
    s.openrouter_api_key = "test-key"
    s.worker_enabled = False
    s.ensure_dirs()
    return s
