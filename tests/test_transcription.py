import math
import struct
import wave

import numpy as np

from app.config import Settings
from app.transcription import (
    ADMIN,
    PATIENT,
    SPEAKER_1,
    Segment,
    WhisperTranscriber,
    assign_stereo_roles,
    channels_differ,
    transcript_from_text,
)


def write_wav(path, left, right=None, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1 if right is None else 2)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i, value in enumerate(left):
            frames += struct.pack("<h", int(value))
            if right is not None:
                frames += struct.pack("<h", int(right[i]))
        w.writeframes(bytes(frames))


def tone(freq, seconds, rate=8000, amp=8000, start=0.0):
    n = int(seconds * rate)
    return [amp * math.sin(2 * math.pi * freq * i / rate) if i >= start * rate else 0 for i in range(n)]


def test_transcript_from_text():
    t = transcript_from_text("[00:01] Администратор: Добрый день\nПациент: Здравствуйте\n\nпросто строка")
    assert [s.speaker for s in t.segments] == [ADMIN, PATIENT, None]
    assert t.segments[0].start == 1.0
    assert t.roles_known
    assert "Администратор: Добрый день" in t.to_text()


def test_assign_stereo_roles_by_markers():
    left = [Segment(0, 2, "Алло, да"), Segment(5, 6, "Спасибо")]
    right = [Segment(1, 3, "Клиника «Варикоза нет», администратор Анна")]
    merged = assign_stereo_roles(left, right)
    assert [s.speaker for s in merged] == [PATIENT, ADMIN, PATIENT]
    assert [s.start for s in merged] == [0, 1, 5]


def test_assign_stereo_roles_unknown_and_forced():
    left, right = [Segment(0, 1, "алло")], [Segment(1, 2, "да")]
    assert assign_stereo_roles(left, right)[0].speaker == SPEAKER_1
    left, right = [Segment(0, 1, "алло")], [Segment(1, 2, "да")]
    assert assign_stereo_roles(left, right, "left")[0].speaker == ADMIN


def test_channels_differ():
    a = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    assert not channels_differ(a, a.copy())
    assert channels_differ(a, np.zeros_like(a))
    assert not channels_differ(np.zeros(100), np.zeros(100))


def test_whisper_transcriber_stereo_and_mono(tmp_path, monkeypatch):
    """Декодирование настоящих файлов через faster-whisper; сама модель подменена."""
    stereo = tmp_path / "stereo.wav"
    write_wav(stereo, tone(440, 3), [0] * 24000)
    mono = tmp_path / "mono.wav"
    write_wav(mono, tone(440, 2))

    s = Settings()
    s.data_dir = tmp_path
    tr = WhisperTranscriber(s)
    texts = iter(["Клиника Варикоза нет, администратор Анна", "Здравствуйте"])

    def fake_run(audio):
        assert audio.dtype == np.float32
        return [Segment(0.0, 1.0, next(texts))]

    monkeypatch.setattr(tr, "_run", fake_run)
    t = tr.transcribe(stereo)
    assert t.stereo and t.duration == 3.0
    assert {s.speaker for s in t.segments} == {ADMIN, PATIENT}

    monkeypatch.setattr(tr, "_run", lambda audio: [Segment(0.0, 1.0, "текст")])
    t = tr.transcribe(mono)
    assert not t.stereo and t.duration == 2.0 and t.segments[0].speaker is None
