import math
import struct
import wave

import numpy as np
import pytest

from app.config import Settings
from app.transcription import (
    piece_words,
    smooth_window_labels,
    split_words_by_windows,
    ADMIN,
    PATIENT,
    SPEAKER_1,
    SPEAKER_2,
    Segment,
    SherpaTranscriber,
    Transcript,
    apply_admin_speaker,
    assign_cluster_roles,
    assign_stereo_roles,
    channels_differ,
    cluster_speakers,
    merge_segments,
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


def test_cluster_speakers():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=16), rng.normal(size=16)
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)

    def noisy(v):
        x = v + 0.05 * rng.normal(size=16)
        return x / np.linalg.norm(x)

    emb = [noisy(a), noisy(b), noisy(a), None, noisy(b), noisy(a)]
    dur = [3.0, 2.0, 1.5, 0.2, 0.5, 2.5]  # пятый кусок короткий — относится к ближайшему центру
    labels, centers = cluster_speakers(emb, dur)
    assert centers.shape == (2, 16)
    assert labels[0] == labels[2] == labels[5] != labels[1] == labels[4]
    assert labels[3] is None
    # один и тот же голос — не делим
    assert cluster_speakers([noisy(a) for _ in range(5)], [2.0] * 5) is None
    assert cluster_speakers([noisy(a)], [2.0]) is None


def test_merge_and_roles():
    segs = [
        Segment(0, 2, "добрый день клиника ворикоза нет", SPEAKER_2),
        Segment(2.3, 4, "администратор анна", SPEAKER_2),
        Segment(5, 6, "здравствуйте", SPEAKER_1),
        Segment(9, 10, "запишите меня", SPEAKER_1),
    ]
    assign_cluster_roles(segs)
    assert [s.speaker for s in segs] == [ADMIN, ADMIN, PATIENT, PATIENT]
    merged = merge_segments(segs)
    assert [m.text for m in merged] == [
        "Добрый день клиника ворикоза нет администратор анна",
        "Здравствуйте",
        "Запишите меня",  # пауза больше секунды — отдельная реплика
    ]


def test_apply_admin_speaker():
    t = Transcript(segments=[Segment(0, 1, "алло", SPEAKER_1), Segment(1, 2, "да", SPEAKER_2)])
    assert apply_admin_speaker(t, SPEAKER_2)
    assert [s.speaker for s in t.segments] == [PATIENT, ADMIN]
    assert not apply_admin_speaker(t, SPEAKER_1)  # роли уже известны
    assert not apply_admin_speaker(Transcript(segments=[Segment(0, 1, "x")]), None)


def _fake_sherpa(tmp_path, monkeypatch, texts):
    s = Settings()
    s.data_dir = tmp_path
    tr = SherpaTranscriber(s)
    monkeypatch.setattr(tr, "_load", lambda: None)
    monkeypatch.setattr(tr, "_vad", lambda samples: [(0.5, samples[:16000]), (2.0, samples[16000:40000])])
    it = iter(texts)

    def recognize(audio):
        text = next(it)
        return text, list(text), [i * 0.05 for i in range(len(text))]

    monkeypatch.setattr(tr, "_recognize", recognize)
    return tr


def test_sherpa_transcriber_stereo(tmp_path, monkeypatch):
    stereo = tmp_path / "stereo.wav"
    write_wav(stereo, tone(440, 3), [0] * 24000)
    tr = _fake_sherpa(tmp_path, monkeypatch, ["клиника варикоза нет", "администратор анна", "алло", ""])
    t = tr.transcribe(stereo)
    assert t.stereo and t.duration == 3.0
    # реплика пациента по времени между репликами администратора — склеивать нельзя
    assert [(s.speaker, s.text) for s in t.segments] == [
        (ADMIN, "Клиника варикоза нет"),
        (PATIENT, "Алло"),
        (ADMIN, "Администратор анна"),
    ]


def test_sherpa_transcriber_mono_diarization(tmp_path, monkeypatch):
    mono = tmp_path / "mono.wav"
    write_wav(mono, tone(440, 3))
    tr = _fake_sherpa(tmp_path, monkeypatch, ["клиника ворикоза нет слушаю", "хочу записаться"])
    a, b = np.eye(4)[0], np.eye(4)[1]
    vectors = iter([a, b])
    monkeypatch.setattr(tr, "_embed", lambda audio: next(vectors))  # куски короче 3 с — без нарезки на окна
    t = tr.transcribe(mono)
    assert not t.stereo and t.duration == 3.0
    assert [s.speaker for s in t.segments] == [ADMIN, PATIENT]


def test_smooth_window_labels():
    assert smooth_window_labels([0, 0, 0, 1, 0, 0, 0]) == [0] * 7  # одиночный выброс
    assert smooth_window_labels([0, 0, 0, 0, 1, 1, 0, 0]) == [0] * 8  # смена короче 3 окон
    assert smooth_window_labels([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]) == [0] * 4 + [1] * 4 + [0] * 3
    assert smooth_window_labels([]) == []


def test_piece_words_and_split():
    tokens = list("да я понимаю")
    times = [i * 0.1 for i in range(len(tokens))]
    words = piece_words(tokens, times)
    assert [w for w, _ in words] == ["да", "я", "понимаю"]
    assert words[2][1] == pytest.approx(0.5)
    assert [w for w, _ in piece_words(["▁ну", "▁да", "вай", "те"], [0, 0.2, 0.3, 0.4])] == ["ну", "давайте"]
    parts = split_words_by_windows(words, [0.1, 0.6], [0, 1])
    assert parts == [(0, "да я", 0.0), (1, "понимаю", pytest.approx(0.5))]


def test_split_long_piece(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    tr = SherpaTranscriber(s)
    a, b = np.eye(4)[0], np.eye(4)[1]
    # 6 секунд, окна по 1,5 с с шагом 0,5 с: первые 5 окон — голос a, остальные — голос b

    def embed(audio, _state={"t": 0.0}):
        v = a if _state["t"] < 2.5 else b
        _state["t"] += 0.5
        return v

    tr._embed = embed
    text = "ну я потерплю да я понимаю ваши переживания"
    words = [(w, i * 0.8) for i, w in enumerate(text.split())]
    piece = {"segment": Segment(10.0, 16.0, text), "audio": np.zeros(6 * 16000, dtype=np.float32), "words": words}
    segs = tr._split_long_piece(piece, np.stack([a, b]))
    assert [(x.speaker, x.text) for x in segs] == [
        (SPEAKER_1, "ну я потерплю да"),
        (SPEAKER_2, "я понимаю ваши переживания"),
    ]
    assert segs[0].start == 10.0 and segs[1].start == pytest.approx(13.2) and segs[1].end == pytest.approx(16.0)
