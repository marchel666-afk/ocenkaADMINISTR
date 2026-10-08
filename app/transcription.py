"""Расшифровка записей звонков локально на процессоре.

Распознавание речи — GigaAM v2 (Сбер), разбивка по паузам — Silero VAD, разделение собеседников
в моно-записи — по голосу (TitaNet). Все модели работают через sherpa-onnx и скачиваются с GitHub
при первом запуске (~270 МБ).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tarfile
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
# Слова, по которым узнаём реплики администратора (название клиники распознаётся и как «ворикоза»).
_ADMIN_MARKERS = ("варикоз", "ворикоз", "клиник", "администратор", "регистратур", "лазерной хирург")


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


# ---------------------------------------------------------------- работа со звуком


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


def channels_differ(left, right, threshold: float = 0.05) -> bool:
    """True, если в каналах разный звук (настоящее стерео, а не продублированное моно)."""
    import numpy as np

    level = float(np.mean(np.abs(left)) + np.mean(np.abs(right)))
    if level < 1e-6:
        return False
    diff = float(np.mean(np.abs(left - right)))
    return diff / level > threshold


# ---------------------------------------------------------------- кто говорит


def _admin_marker_score(segments: list[Segment], first_seconds: float | None = None) -> int:
    text = " ".join(s.text.lower() for s in segments if first_seconds is None or s.start <= first_seconds)
    return sum(text.count(marker) for marker in _ADMIN_MARKERS)


def _admin_index(groups: list[list[Segment]]) -> int | None:
    """Номер группы реплик, принадлежащей администратору, или None, если по словам не понять."""
    scores = [_admin_marker_score(g) for g in groups]
    best = max(scores, default=0)
    if best == 0 or scores.count(best) > 1:
        return None
    return scores.index(best)


def assign_stereo_roles(left: list[Segment], right: list[Segment], mode: str = "auto") -> list[Segment]:
    """Проставляет роли сегментам двух каналов и объединяет их по времени."""
    if mode == "left":
        admin = 0
    elif mode == "right":
        admin = 1
    else:
        admin = _admin_index([[s for s in left if s.start <= 60], [s for s in right if s.start <= 60]])
        if admin is None:
            admin = _admin_index([left, right])

    roles = (SPEAKER_1, SPEAKER_2) if admin is None else ((ADMIN, PATIENT) if admin == 0 else (PATIENT, ADMIN))
    for s in left:
        s.speaker = roles[0]
    for s in right:
        s.speaker = roles[1]
    return sorted(left + right, key=lambda s: (s.start, s.end))


def assign_cluster_roles(segments: list[Segment]) -> None:
    """Сегменты с метками s1/s2 переименовывает в администратора и пациента, если это видно по словам."""
    groups = [[s for s in segments if s.speaker == SPEAKER_1], [s for s in segments if s.speaker == SPEAKER_2]]
    admin = _admin_index(groups)
    if admin is None:
        return
    mapping = {SPEAKER_1: ADMIN if admin == 0 else PATIENT, SPEAKER_2: PATIENT if admin == 0 else ADMIN}
    for s in segments:
        if s.speaker in mapping:
            s.speaker = mapping[s.speaker]


def apply_admin_speaker(transcript: Transcript, admin_speaker: str | None) -> bool:
    """Переименовывает «Собеседник 1/2» в администратора и пациента по ответу ИИ. True, если что-то изменилось."""
    if admin_speaker not in (SPEAKER_1, SPEAKER_2) or transcript.roles_known:
        return False
    other = SPEAKER_2 if admin_speaker == SPEAKER_1 else SPEAKER_1
    changed = False
    for s in transcript.segments:
        if s.speaker == admin_speaker:
            s.speaker, changed = ADMIN, True
        elif s.speaker == other:
            s.speaker, changed = PATIENT, True
    return changed


def cluster_speakers(embeddings: list, durations: list[float], min_fit: float = 0.8, same_speaker: float = 0.6):
    """Делит куски записи на двух собеседников по «отпечаткам голоса».

    Центры кластеров считаются по кускам длиннее min_fit секунд, остальные куски относятся к ближайшему.
    Возвращает (метки 0/1 по кускам — None, если нет отпечатка; центры кластеров)
    или None, если похоже, что говорит один человек.
    """
    import numpy as np

    fit = [i for i, (e, d) in enumerate(zip(embeddings, durations)) if e is not None and d >= min_fit]
    if len(fit) < 2:
        return None
    X = np.stack([embeddings[i] for i in fit])
    w = np.array([durations[i] for i in fit])
    sims = X @ X.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    centers = np.stack([X[i], X[j]])
    for _ in range(50):
        labels = np.argmax(X @ centers.T, axis=1)
        new = []
        for k in range(2):
            m = (X[labels == k] * w[labels == k, None]).sum(0) if (labels == k).any() else centers[k]
            new.append(m / (np.linalg.norm(m) + 1e-9))
        new = np.stack(new)
        if np.allclose(new, centers):
            break
        centers = new
    if float(centers[0] @ centers[1]) > same_speaker:
        return None
    return [None if e is None else int(np.argmax(centers @ e)) for e in embeddings], centers


def smooth_window_labels(labels: list[int], min_run: int = 3) -> list[int]:
    """Сглаживает метки окон: убирает одиночные выбросы и слишком короткие смены собеседника."""
    if not labels:
        return []
    sm = [sorted(labels[max(0, i - 1) : i + 2])[len(labels[max(0, i - 1) : i + 2]) // 2] for i in range(len(labels))]
    runs: list[list[int]] = []  # [метка, длина]
    for lab in sm:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([lab, 1])
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (lab, n) in enumerate(runs):
            if n < min_run:
                neighbour = runs[i - 1] if i > 0 else runs[i + 1]
                neighbour[1] += n
                runs.pop(i)
                changed = True
                break
        merged: list[list[int]] = []
        for lab, n in runs:
            if merged and merged[-1][0] == lab:
                merged[-1][1] += n
            else:
                merged.append([lab, n])
        runs = merged
    return [lab for lab, n in runs for _ in range(n)]


def piece_words(tokens: list[str], times: list[float]) -> list[tuple[str, float]]:
    """Слова распознанного куска и время начала каждого (по буквам/токенам модели)."""
    words: list[tuple[str, float]] = []
    current, start = "", None
    for tok, t in zip(tokens, times):
        boundary = tok.strip() == "" or tok.startswith("▁")
        piece = tok.replace("▁", "").strip()
        if boundary and current:
            words.append((current, start))
            current, start = "", None
        if piece:
            if start is None:
                start = t
            current += piece
    if current:
        words.append((current, start))
    return words


def split_words_by_windows(
    words: list[tuple[str, float]], window_centers: list[float], window_labels: list[int]
) -> list[tuple[int, str, float]]:
    """Делит слова куска на части по меткам окон. Возвращает [(метка, текст, время начала части)]."""
    parts: list[tuple[int, list[str], float]] = []
    for word, t in words:
        k = min(range(len(window_centers)), key=lambda i: abs(window_centers[i] - t))
        label = window_labels[k]
        if parts and parts[-1][0] == label:
            parts[-1][1].append(word)
        else:
            parts.append((label, [word], t))
    return [(label, " ".join(ws), t) for label, ws, t in parts]


def merge_segments(segments: list[Segment], max_gap: float = 1.0) -> list[Segment]:
    """Склеивает подряд идущие куски одного собеседника в одну реплику."""
    merged: list[Segment] = []
    for s in segments:
        prev = merged[-1] if merged else None
        if prev and s.speaker and prev.speaker == s.speaker and s.start - prev.end <= max_gap:
            prev.text = f"{prev.text} {s.text}"
            prev.end = s.end
        else:
            merged.append(Segment(start=s.start, end=s.end, text=s.text, speaker=s.speaker))
    for s in merged:
        s.text = s.text[:1].upper() + s.text[1:]
    return merged


# ---------------------------------------------------------------- модели


MODEL_FILES = {
    # ключ: (путь в релизах sherpa-onnx на GitHub, имя файла или папки после распаковки)
    "asr": (
        "asr-models/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2",
        "sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19",
    ),
    "vad": ("asr-models/silero_vad.onnx", "silero_vad.onnx"),
    "speaker": ("speaker-recongition-models/nemo_en_titanet_small.onnx", "nemo_en_titanet_small.onnx"),
}


def ensure_models(models_dir: Path, base_url: str) -> dict[str, Path]:
    """Скачивает недостающие модели. Возвращает пути к ним."""
    import httpx

    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, (remote, local) in MODEL_FILES.items():
        target = models_dir / local
        paths[key] = target
        if target.exists():
            continue
        url = f"{base_url.rstrip('/')}/{remote}"
        archive = models_dir / (Path(remote).name + ".part")
        log.info("Скачиваю модель %s ...", url)
        with httpx.stream("GET", url, follow_redirects=True, timeout=httpx.Timeout(60, read=300)) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length") or 0)
            done, next_report = 0, 0.1
            with open(archive, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if total and done / total >= next_report:
                        log.info("  %s: %d%%", Path(remote).name, int(100 * done / total))
                        next_report += 0.1
        if remote.endswith(".tar.bz2"):
            with tarfile.open(archive, "r:bz2") as tar:
                try:
                    tar.extractall(models_dir, filter="data")
                except TypeError:  # Python без поддержки filter
                    tar.extractall(models_dir)
            archive.unlink()
        else:
            archive.replace(target)
        if not target.exists():
            raise RuntimeError(f"После загрузки не найден файл модели {target}")
    return paths


# ---------------------------------------------------------------- расшифровка


class SherpaTranscriber:
    """Расшифровка через sherpa-onnx: GigaAM v2 + Silero VAD + разделение собеседников по голосу."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = threading.Lock()
        self._asr = None
        self._speaker = None
        self._paths: dict[str, Path] = {}

    @property
    def threads(self) -> int:
        return self.settings.asr_threads or max(1, min(4, os.cpu_count() or 1))

    def _load(self) -> None:
        with self._lock:
            if self._asr is not None:
                return
            import sherpa_onnx

            try:
                self._paths = ensure_models(self.settings.data_dir / "models", self.settings.models_base_url)
            except Exception as e:
                raise RuntimeError(
                    f"Не удалось скачать модели расшифровки ({e}). При первом запуске модели (около 270 МБ) "
                    "скачиваются с github.com — проверьте подключение к интернету."
                ) from e
            asr_dir = self._paths["asr"]
            self._asr = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(asr_dir / "encoder.int8.onnx"),
                decoder=str(asr_dir / "decoder.onnx"),
                joiner=str(asr_dir / "joiner.onnx"),
                tokens=str(asr_dir / "tokens.txt"),
                model_type="nemo_transducer",
                num_threads=self.threads,
            )
            self._speaker = sherpa_onnx.SpeakerEmbeddingExtractor(
                sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(self._paths["speaker"]), num_threads=self.threads)
            )

    def _vad(self, samples) -> list[tuple[float, object]]:
        """Режет запись на куски речи по паузам (не длиннее 12 с — ограничение модели распознавания)."""
        import numpy as np
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(self._paths["vad"])
        cfg.silero_vad.min_silence_duration = 0.2
        cfg.silero_vad.min_speech_duration = 0.2
        cfg.silero_vad.max_speech_duration = 12.0
        cfg.sample_rate = SAMPLE_RATE
        cfg.num_threads = 1
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(samples) / SAMPLE_RATE + 30)
        pieces = []

        def drain():
            while not vad.empty():
                pieces.append((vad.front.start / SAMPLE_RATE, np.array(vad.front.samples, dtype=np.float32)))
                vad.pop()

        window = cfg.silero_vad.window_size
        for i in range(0, len(samples), window):
            vad.accept_waveform(samples[i : i + window])
            drain()
        vad.flush()
        drain()
        return pieces

    def _recognize(self, audio) -> tuple[str, list[str], list[float]]:
        """Текст куска, его токены и время начала каждого токена (с начала куска)."""
        stream = self._asr.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        self._asr.decode_stream(stream)
        r = stream.result
        return r.text.strip(), list(r.tokens), list(r.timestamps)

    def _embed(self, audio):
        import numpy as np

        if len(audio) < 0.4 * SAMPLE_RATE:
            return None
        stream = self._speaker.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        stream.input_finished()
        v = np.array(self._speaker.compute(stream), dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def _channel(self, samples, with_embeddings: bool):
        """Куски речи одного канала: сегмент, звук, слова со временем и «отпечаток голоса»."""
        pieces = []
        for start, audio in self._vad(samples):
            text, tokens, times = self._recognize(audio)
            if not text:
                continue
            pieces.append(
                {
                    "segment": Segment(start=round(start, 2), end=round(start + len(audio) / SAMPLE_RATE, 2), text=text),
                    "audio": audio,
                    "words": piece_words(tokens, times),
                    "embedding": self._embed(audio) if with_embeddings else None,
                }
            )
        return pieces

    def _split_long_piece(self, piece, centers, window: float = 1.5, hop: float = 0.5) -> list[Segment] | None:
        """Если в длинном куске говорят оба собеседника без паузы — режет его по смене голоса."""
        import numpy as np

        audio, seg = piece["audio"], piece["segment"]
        duration = len(audio) / SAMPLE_RATE
        if duration < 3.0 or len(piece["words"]) < 4:
            return None
        window_centers, window_labels = [], []
        t = 0.0
        while t + window <= duration + 1e-6:
            e = self._embed(audio[int(t * SAMPLE_RATE) : int((t + window) * SAMPLE_RATE)])
            if e is not None:
                window_centers.append(t + window / 2)
                window_labels.append(int(np.argmax(centers @ e)))
            t += hop
        labels = smooth_window_labels(window_labels)
        if len(set(labels)) < 2:
            return None
        parts = split_words_by_windows(piece["words"], window_centers, labels)
        segments = []
        for i, (label, text, start) in enumerate(parts):
            end = parts[i + 1][2] if i + 1 < len(parts) else duration
            segments.append(
                Segment(
                    start=round(seg.start + start, 2),
                    end=round(seg.start + end, 2),
                    text=text,
                    speaker=(SPEAKER_1, SPEAKER_2)[label],
                )
            )
        return segments

    def transcribe(self, path: Path) -> Transcript:
        self._load()
        left, right = decode_audio(path)
        duration = round(len(left) / SAMPLE_RATE, 2)

        if channels_differ(left, right):
            seg_l = [p["segment"] for p in self._channel(left, with_embeddings=False)]
            seg_r = [p["segment"] for p in self._channel(right, with_embeddings=False)]
            segments = assign_stereo_roles(seg_l, seg_r, self.settings.stereo_admin_channel)
            return Transcript(segments=merge_segments(segments), duration=duration, stereo=True)

        pieces = self._channel(left, with_embeddings=self.settings.diarization)
        clusters = None
        if self.settings.diarization:
            clusters = cluster_speakers(
                [p["embedding"] for p in pieces], [p["segment"].end - p["segment"].start for p in pieces]
            )
        if not clusters:
            return Transcript(segments=merge_segments([p["segment"] for p in pieces]), duration=duration, stereo=False)

        labels, centers = clusters
        segments: list[Segment] = []
        for piece, label in zip(pieces, labels):
            split = self._split_long_piece(piece, centers)
            if split:
                segments.extend(split)
            else:
                piece["segment"].speaker = None if label is None else (SPEAKER_1, SPEAKER_2)[label]
                segments.append(piece["segment"])
        assign_cluster_roles(segments)
        return Transcript(segments=merge_segments(segments), duration=duration, stereo=False)
