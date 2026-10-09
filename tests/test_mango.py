"""Автозагрузка звонков из Mango Office: подпись, статистика, скачивание записей, планировщик."""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs

import httpx
import pytest
from sqlalchemy import select

from app import db
from app.mango_import import SETTLE_SECONDS, MangoClient, MangoError, MangoScheduler, parse_stats_csv

from .conftest import FakeLLM
from .test_web import make

NOW = 1_791_500_000  # фиксированное «сейчас» (unix)
FIELDS_ROW = "records;start;finish;answer;from_extension;from_number;to_extension;to_number;disconnect_reason;line_number;location;entry_id"


def row(records, start, talk, from_ext="", from_num="", to_ext="", to_num="", line="74232000000"):
    return f"{records};{start};{start + talk + 5};{start + 5};{from_ext};{from_num};{to_ext};{to_num};1110;{line};abonent;e{start}"


class FakeMango:
    """Поддельный API ВАТС: отвечает на stats/request, stats/result и queries/recording/post."""

    def __init__(self, rows=None, key="KEY", salt="SALT", pending=1):
        self.rows = rows or []
        self.key, self.salt = key, salt
        self.pending = pending  # сколько раз stats/result отвечает «ещё не готово»
        self.requests = []
        self.downloads = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.startswith("/files/"):
            return httpx.Response(200, content=b"ID3fake-mp3-" + path.encode(), headers={"content-type": "audio/mpeg"})
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        expected = hashlib.sha256((form["vpbx_api_key"] + form["json"] + self.salt).encode()).hexdigest()
        if form["vpbx_api_key"] != self.key or form["sign"] != expected:
            return httpx.Response(401, json={"result": 3102})
        payload = json.loads(form["json"])
        self.requests.append((path, payload))
        if path.endswith("/stats/request"):
            self.window = (int(payload["date_from"]), int(payload["date_to"]))
            return httpx.Response(200, json={"key": "stat-key"})
        if path.endswith("/stats/result"):
            if self.pending:
                self.pending -= 1
                return httpx.Response(204)
            lo, hi = self.window
            text = "\n".join(r for r in self.rows if lo <= int(r.split(";")[1]) <= hi)
            return httpx.Response(200, text=text, headers={"content-type": "text/plain"})
        if path.endswith("/queries/recording/post"):
            self.downloads.append(payload["recording_id"])
            return httpx.Response(302, headers={"location": f"https://files.mango.test/files/{payload['recording_id']}.mp3"})
        return httpx.Response(404)


def http_for(fake):
    return httpx.Client(transport=httpx.MockTransport(fake), follow_redirects=True)


def client_for(fake, **kw):
    return MangoClient("KEY", "SALT", http=http_for(fake), poll_interval=0, **kw)


# ---------------------------------------------------------------- разбор и клиент


def test_sign():
    c = MangoClient("abc", "salt", http=httpx.Client())
    assert c.sign('{"a":1}') == hashlib.sha256(b'abc{"a":1}salt').hexdigest()


def test_parse_stats_csv_and_direction():
    text = "\n".join([
        row("[rec1,rec2]", 1000, 120, to_ext="101", from_num="79001112233"),  # входящий
        row("[rec3]", 2000, 60, from_ext="102", to_num="79004445566"),  # исходящий
        row("[]", 3000, 30, from_ext="101", to_ext="102"),  # внутренний, без записи
        row("rec4", 4000, 0, from_num="79007778899"),  # не дождался сотрудника (IVR)
        "",
    ])
    calls = parse_stats_csv(text)
    assert [c.recording_ids for c in calls] == [["rec1", "rec2"], ["rec3"], [], ["rec4"]]
    assert [c.direction for c in calls] == ["in", "out", None, "in"]
    assert calls[0].phone == "79001112233" and calls[0].extension == "101" and calls[0].talk_seconds == 120
    assert calls[1].phone == "79004445566" and calls[1].extension == "102"
    assert parse_stats_csv(FIELDS_ROW) == []  # строка заголовка пропускается


def test_stats_polls_until_ready():
    fake = FakeMango(rows=[row("[r1]", NOW - 600, 90, to_ext="101", from_num="79001112233")], pending=2)
    calls = client_for(fake).stats(NOW - 3600, NOW)
    assert len(calls) == 1 and calls[0].recording_ids == ["r1"]
    paths = [p for p, _ in fake.requests]
    assert paths == ["/vpbx/stats/request", "/vpbx/stats/result", "/vpbx/stats/result", "/vpbx/stats/result"]
    req = fake.requests[0][1]
    assert req["date_from"] == str(NOW - 3600) and "records" in req["fields"].split(",")


def test_bad_keys_and_errors(tmp_path):
    fake = FakeMango()
    bad = MangoClient("WRONG", "SALT", http=http_for(fake), poll_interval=0)
    with pytest.raises(MangoError, match="отклонил ключи"):
        bad.stats(NOW - 60, NOW)
    with pytest.raises(MangoError, match="Не указаны"):
        MangoClient("", "", http=httpx.Client())

    def no_key(request):
        return httpx.Response(200, json={"result": 3101, "message": "bad request"})

    with pytest.raises(MangoError, match="код 3101"):
        MangoClient("K", "S", http=httpx.Client(transport=httpx.MockTransport(no_key))).stats(0, 1)

    def json_instead_of_file(request):
        return httpx.Response(200, json={"result": 3300}, headers={"content-type": "application/json"})

    c = MangoClient("K", "S", http=httpx.Client(transport=httpx.MockTransport(json_instead_of_file)))
    with pytest.raises(MangoError, match="не отдал запись"):
        c.download_recording("r1", tmp_path / "a.mp3")
    assert not (tmp_path / "a.mp3").exists() and not (tmp_path / "a.mp3.part").exists()


def test_download_follows_redirect(tmp_path):
    fake = FakeMango()
    target = tmp_path / "x" / "r9.mp3"
    client_for(fake).download_recording("r9", target)
    assert target.read_bytes().startswith(b"ID3fake-mp3-") and fake.downloads == ["r9"]


# ---------------------------------------------------------------- загрузка в клинику


def enable_mango(app, clinic_id, **extra):
    with app.state.session_factory() as s:
        c = s.get(db.Clinic, clinic_id)
        c.mango_enabled, c.mango_api_key, c.mango_api_salt, c.utc_offset = True, "KEY", "SALT", 10
        for k, v in extra.items():
            setattr(c, k, v)
        s.commit()


class DirectionLLM(FakeLLM):
    """Как FakeLLM, но тип звонка выбирает по направлению из данных звонка."""

    def complete(self, system_parts, user, max_tokens=None, model=None):
        self.call_type = "out_request" if "Направление звонка: исходящий" in user else "incoming"
        return super().complete(system_parts, user, max_tokens, model)


def test_scheduler_imports_calls(settings):
    app, client, _, _ = make(settings, llm=DirectionLLM())
    clinic = app.state.first_clinic_id
    client.post("/employees", data={"name": "Наталья", "mango_id": "101"})
    enable_mango(app, clinic)
    fake = FakeMango(rows=[
        row("[in1]", NOW - 3000, 180, to_ext="101", from_num="79001112233"),  # входящий → Наталья
        row("[out1]", NOW - 2500, 240, from_ext="102", to_num="79004445566"),  # исходящий
        row("[short]", NOW - 2400, 5, to_ext="101", from_num="79000000000"),  # короче 20 с — пропуск
        row("[]", NOW - 2300, 100, to_ext="101", from_num="79000000001"),  # без записи
        row("[int1]", NOW - 2200, 100, from_ext="101", to_ext="102"),  # внутренний
        row("[late]", NOW - SETTLE_SECONDS + 60, 100, to_ext="101", from_num="79000000002"),  # ещё «свежий»
    ])
    scheduler = MangoScheduler(settings, app.state.session_factory, http=http_for(fake))
    scheduler.client_for = lambda c: client_for(fake)
    assert scheduler.run_clinic(clinic, now=NOW) == 2
    assert sorted(fake.downloads) == ["in1", "out1"]

    with app.state.session_factory() as s:
        calls = {c.external_id: c for c in s.scalars(select(db.Call).where(db.Call.source == "mango"))}
        c_in, c_out = calls["in1"], calls["out1"]
        assert c_in.direction == "in" and c_in.phone == "79001112233" and c_in.mango_line == "101"
        assert c_in.employee.name == "Наталья" and c_in.status == db.QUEUED
        utc = datetime.fromtimestamp(NOW - 3000, tz=timezone.utc).replace(tzinfo=None)
        assert c_in.started_at == utc + timedelta(hours=10)  # местное время клиники (UTC+10)
        assert (settings.audio_dir / c_in.audio_path).read_bytes().startswith(b"ID3")
        assert c_out.direction == "out" and c_out.phone == "79004445566" and c_out.employee_id is None
        clinic_row = s.get(db.Clinic, clinic)
        assert clinic_row.mango_synced_until == NOW - SETTLE_SECONDS
        assert clinic_row.mango_last_count == 2 and clinic_row.mango_last_error == ""

    # повторный запуск: окно перекрывается, но дублей нет; «свежий» звонок забирается позже
    assert scheduler.run_clinic(clinic, now=NOW + 600) == 1
    assert sorted(fake.downloads) == ["in1", "late", "out1"]

    # загруженные звонки обрабатываются как обычные
    from .test_web import process_all

    process_all(app)
    with app.state.session_factory() as s:
        assert {c.status for c in s.scalars(select(db.Call))} == {db.DONE}
    assert "79001112233" in client.get("/calls/1").text


def test_scheduler_lines_filter_and_errors(settings):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    enable_mango(app, clinic, mango_lines="102")
    fake = FakeMango(rows=[
        row("[a]", NOW - 3000, 100, to_ext="101", from_num="79001112233"),
        row("[b]", NOW - 2900, 100, from_ext="102", to_num="79004445566"),
    ])
    scheduler = MangoScheduler(settings, app.state.session_factory)
    scheduler.client_for = lambda c: client_for(fake)
    assert scheduler.run_clinic(clinic, now=NOW) == 1 and fake.downloads == ["b"]

    # ошибка ключей: окно не сдвигается, ошибка видна в настройках клиники
    scheduler.client_for = lambda c: MangoClient("WRONG", "SALT", http=http_for(fake), poll_interval=0)
    with app.state.session_factory() as s:
        synced = s.get(db.Clinic, clinic).mango_synced_until
    assert scheduler.run_clinic(clinic, now=NOW + 3600) == 0
    with app.state.session_factory() as s:
        c = s.get(db.Clinic, clinic)
        assert "отклонил ключи" in c.mango_last_error and c.mango_synced_until == synced
    assert "отклонил ключи" in client.get("/settings/clinic").text


def test_due_clinics_and_test_connection(settings):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    scheduler = MangoScheduler(settings, app.state.session_factory)
    assert scheduler.due_clinics(datetime.now()) == []
    enable_mango(app, clinic, mango_interval_min=30)
    assert scheduler.due_clinics(datetime.now()) == [clinic]
    with app.state.session_factory() as s:
        s.get(db.Clinic, clinic).mango_last_run = datetime.now()
        s.commit()
    assert scheduler.due_clinics(datetime.now()) == []

    fake = FakeMango(rows=[row("[r1]", 0, 100, to_ext="101", from_num="7900")])
    scheduler.client_for = lambda c: client_for(fake)
    ok, message = scheduler.test_connection(clinic)
    assert ok and "Подключение работает" in message
    scheduler.client_for = lambda c: MangoClient("WRONG", "SALT", http=http_for(fake), poll_interval=0)
    ok, message = scheduler.test_connection(clinic)
    assert not ok and "отклонил ключи" in message
