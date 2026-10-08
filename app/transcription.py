"""Расшифровка записей звонков (faster-whisper, работает локально на процессоре)."""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .config import Settings

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
ADMIN, PATIENT, SPEAKER_1, SPEAKER_2 = "admin", "patient", "s1", "s2"
SPEAKER_LABELS = {
    ADMIN: "Администратор",
    PATIENT: "Пациент",
    SPEAKER_1: "Собеседник 1",
    SPEAKER_2: "Собеседник 2",
}
# Слова, по которым в стерео-записи узнаём канал администратора.
_ADMIN_MARKERS = ("варикоз", "клиник", "администратор", "лазерной хирург")


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    duration: float = 0.0
    stereo: bool = False

    @property
    def roles_known(self) -> bool:
        return any(s.speaker in (ADMIN, PATIENT) for s in self.segments)

    @property
    def has_speakers(self) -> bool:
        return any(s.speaker for s in self.segments)

    @property
    def is_empty(self) -> bool:
        return not any(s.text.strip() for s in self.segments)

    def to_text(self) -> str:
        lines = []
        for s in self.segments:
            stamp = f"[{int(s.start) // 60:02d}:{int(s.start) % 60:02d}]"
            label = SPEAKER_LABELS.get(s.speaker or "", "")
            lines.append(f"{stamp} {label}: {s.text}" if label else f"{stamp} {s.text}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {"duration": self.duration, "stereo": self.stereo, "segments": [asdict(s) for s in self.segments]},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "Transcript":
        data = json.loads(raw)
        return cls(
            segments=[Segment(**s) for s in data.get("segments", [])],
            duration=float(data.get("duration", 0.0)),
            stereo=bool(data.get("stereo", False)),
        )


_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<mm>\d{1,3}):(?P<ss>\d{2})\]\s*)?"
    r"(?:(?P<who>администратор|админ|сотрудник|оператор|пациент|клиент|собеседник\s*1|собеседник\s*2)\s*:\s*)?"
    r"(?P<text>.*)$",
    re.IGNORECASE,
)


def transcript_from_text(text: str) -> Transcript:
    """Готовая расшифровка из текстового файла.

    Поддерживаются строки вида «Администратор: ...», «Пациент: ...», с меткой времени «[01:23]» или без.
    """
    segments: list[Segment] = []
    t = 0.0
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        m = _LINE_RE.match(raw_line)
        assert m is not None  # шаблон совпадает с любой строкой
        if m.group("mm") is not None:
            t = int(m.group("mm")) * 60 + int(m.group("ss"))
        who = (m.group("who") or "").lower().replace(" ", "")
        speaker = None
        if who in ("администратор", "админ", "сотрудник", "оператор"):
            speaker = ADMIN
        elif who in ("пациент", "клиент"):
            speaker = PATIENT
        elif who == "собеседник1":
            speaker = SPEAKER_1
        elif who == "собеседник2":
            speaker = SPEAKER_2
        body = m.group("text").strip()
        if not body:
            continue
        segments.append(Segment(start=t, end=t, text=body, speaker=speaker))
        if m.group("mm") is None:
            t += 1.0
    duration = segments[-1].end if segments else 0.0
    return Transcript(segments=segments, duration=duration, stereo=False)


class Transcriber(Protocol):
    def transcribe(self, path: Path) -> Transcript: ...


def channels_differ(left, right, threshold: float = 0.05) -> bool:
    """True, если в каналах разный звук (настоящее стерео, а не продублированное моно)."""
    import numpy as np

    level = float(np.mean(np.abs(left)) + np.mean(np.abs(right)))
    if level < 1e-6:
        return False
    diff = float(np.mean(np.abs(left - right)))
    return diff / level > threshold


def _admin_marker_score(segments: list[Segment], first_seconds: float = 60.0) -> int:
    text = " ".join(s.text.lower() for s in segments if s.start <= first_seconds)
    return sum(text.count(marker) for marker in _ADMIN_MARKERS)


def assign_stereo_roles(left: list[Segment], right: list[Segment], mode: str = "auto") -> list[Segment]:
    """Проставляет роли сегментам двух каналов и объединяет их по времени."""
    if mode == "left":
        admin_left = True
    elif mode == "right":
        admin_left = False
    else:
        score_l, score_r = _admin_marker_score(left), _admin_marker_score(right)
        admin_left = score_l > score_r if score_l != score_r else None

    if admin_left is None:
        roles = (SPEAKER_1, SPEAKER_2)
    elif admin_left:
        roles = (ADMIN, PATIENT)
    else:
        roles = (PATIENT, ADMIN)

    for s in left:
        s.speaker = roles[0]
    for s in right:
        s.speaker = roles[1]
    return sorted(left + right, key=lambda s: (s.start, s.end))


def decode_audio(path: Path, sampling_rate: int = SAMPLE_RATE):
    """Читает аудиофайл любого распространённого формата и возвращает (левый, правый) каналы float32.

    Моно-запись дублируется в оба канала.
    """
    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="stereo", rate=sampling_rate)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            frame.pts = None
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    if not chunks:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty
    interleaved = np.concatenate(chunks).astype(np.float32) / 32768.0
    return interleaved[0::2].copy(), interleaved[1::2].copy()


class WhisperTranscriber:
    """Локальная расшифровка через faster-whisper. Модель скачивается при первом запуске."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                log.info("Загрузка модели расшифровки %s ...", self.settings.whisper_model)
                try:
                    self._model = WhisperModel(
                        self.settings.whisper_model,
                        device=self.settings.whisper_device,
                        compute_type=self.settings.whisper_compute_type,
                        cpu_threads=self.settings.whisper_threads,
                        download_root=str(self.settings.data_dir / "models"),
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Не удалось загрузить модель расшифровки «{self.settings.whisper_model}» ({e}). "
                        "При первом запуске модель скачивается из интернета (около 1,5 ГБ). "
                        "Если сайт huggingface.co недоступен, впишите в .env строку HF_ENDPOINT=https://hf-mirror.com "
                        "и перезапустите сервис."
                    ) from e
            return self._model

    def _run(self, audio) -> list[Segment]:
        model = self._get_model()
        segments, _info = model.transcribe(
            audio,
            language="ru",
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
            initial_prompt=self.settings.whisper_initial_prompt or None,
        )
        return [
            Segment(start=round(s.start, 2), end=round(s.end, 2), text=s.text.strip())
            for s in segments
            if s.text.strip()
        ]

    def transcribe(self, path: Path) -> Transcript:
        left, right = decode_audio(path)
        duration = round(len(left) / SAMPLE_RATE, 2)

        if channels_differ(left, right):
            segments = assign_stereo_roles(self._run(left), self._run(right), self.settings.stereo_admin_channel)
            return Transcript(segments=segments, duration=duration, stereo=True)

        return Transcript(segments=self._run(left), duration=duration, stereo=False)
