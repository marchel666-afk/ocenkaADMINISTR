"""Очередь обработки и карточка звонка: блокировки базы, гонки с пользователем, справедливая очередь, формы."""

import sqlite3
import threading
from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from app import db
from app.pipeline import Worker

from .conftest import FakeLLM, sample_text
from .test_clinics import make_two_clinics, upload_text
from .test_web import get_call, make, process_all


class HookLLM(FakeLLM):
    """FakeLLM, который во время запроса оценки выполняет действие «пользователя»."""

    def __init__(self, on_evaluate=None, **kw):
        super().__init__(**kw)
        self.on_evaluate = on_evaluate

    def complete(self, system_parts, user, max_tokens=None, model=None):
        if self.on_evaluate and "определи его тип" not in "\n".join(system_parts):
            hook, self.on_evaluate = self.on_evaluate, None
            hook()
        return super().complete(system_parts, user, max_tokens, model)


def employee_name(app, call_id=1):
    with app.state.session_factory() as s:
        call = s.get(db.Call, call_id)
        return call.employee.name if call.employee else None


def test_database_is_writable_while_ai_evaluates(settings):
    errors = []

    def write_from_other_connection():
        con = sqlite3.connect(settings.data_dir / "ocenka.db", timeout=0.5)
        try:
            con.execute("UPDATE clinics SET ai_context = 'правка во время оценки'")
            con.commit()
        except sqlite3.OperationalError as e:
            errors.append(str(e))
        finally:
            con.close()

    llm = HookLLM(on_evaluate=write_from_other_connection, admin_name="Анна")
    app, client, _, _ = make(settings, llm=llm)
    client.post("/employees", data={"name": "Анна"})  # сотрудник подбирается по имени — раньше это держало блокировку
    upload_text(client)
    process_all(app)
    assert errors == [] and get_call(app).status == db.DONE
    assert employee_name(app) == "Анна"


def test_user_change_during_evaluation_wins(settings):
    holder = {}
    llm = HookLLM(on_evaluate=lambda: holder["client"].post("/calls/1/update", data={"call_type": "out_request"}))
    app, client, _, _ = make(settings, llm=llm)
    holder["client"] = client
    upload_text(client)
    process_all(app)  # оценка по «входящему» отброшена: пока она шла, пользователь выбрал тип
    call = get_call(app)
    assert call.status == db.QUEUED and call.call_type == "out_request" and call.call_type_manual
    process_all(app)
    call = get_call(app)
    assert call.status == db.DONE and call.call_type == "out_request"
    with app.state.session_factory() as s:
        assert {r.checklist for r in s.get(db.Call, 1).results} == {"general", "out_request"}

    # удаление во время оценки тоже не ломает обработку
    llm.on_evaluate = lambda: client.post("/calls/1/delete")
    client.post("/calls/1/reevaluate")
    process_all(app)
    with app.state.session_factory() as s:
        assert s.get(db.Call, 1) is None


def test_worker_survives_errors_and_skips_disabled_clinic(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    upload_text(admin, "b.txt")  # клиника Б (админ переключён на неё)
    with app.state.session_factory() as s:
        s.get(db.Clinic, clinic_b).active = False
        s.commit()
    worker = Worker(app.state.processor)
    assert worker._next_call_id() == 1  # звонок клиники А; звонок отключённой Б ждёт
    with app.state.session_factory() as s:
        s.get(db.Call, 1).status = db.DONE
        s.commit()
    assert worker._next_call_id() is None

    calls = []

    class Broken:
        session_factory = app.state.session_factory

        def process(self, call_id):
            calls.append(call_id)
            if len(calls) == 1:
                raise RuntimeError("сбой записи статуса")
            with app.state.session_factory() as s:
                s.get(db.Call, call_id).status = db.DONE
                s.commit()

    with app.state.session_factory() as s:
        s.get(db.Call, 1).status = db.QUEUED
        s.commit()
    worker = Worker(Broken(), poll_sec=0.01)
    worker.start()
    for _ in range(300):
        if len(calls) >= 2:
            break
        threading.Event().wait(0.01)
    worker.stop()
    assert calls[:2] == [1, 1] and a  # после сбоя поток жив и берёт звонок снова


def test_queue_is_fair_between_clinics(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    with app.state.session_factory() as s:
        s.get(db.Call, 1).source = "mango"
        for i in range(3):
            s.add(db.Call(clinic_id=a, source="mango", status=db.QUEUED, transcript_json="{}", original_filename=f"a{i}"))
        s.add(db.Call(clinic_id=clinic_b, source="mango", status=db.QUEUED, transcript_json="{}", original_filename="b0"))
        s.commit()
    worker = Worker(app.state.processor)
    order = []
    for _ in range(5):
        call_id = worker._next_call_id()
        order.append(call_id)
        with app.state.session_factory() as s:
            s.get(db.Call, call_id).status = db.DONE
            s.commit()
    with app.state.session_factory() as s:
        clinics = [s.get(db.Call, i).clinic_id for i in order]
    assert clinics[:2] == [a, clinic_b]  # большая очередь клиники А не задерживает Б
    # загруженное вручную идёт раньше импорта из Mango
    upload_text(admin, "manual.txt")
    with app.state.session_factory() as s:
        s.add(db.Call(clinic_id=a, source="mango", status=db.QUEUED, transcript_json="{}", original_filename="late"))
        s.commit()
        manual = s.scalars(select(db.Call.id).where(db.Call.original_filename == "manual.txt")).one()
    assert worker._next_call_id() == manual


def test_manual_type_still_matches_employee(settings):
    app, client, _, llm = make(settings, llm=FakeLLM(admin_name="Анна"))
    client.post("/employees", data={"name": "Анна Смирнова"})
    upload_text(client, call_type="incoming")
    process_all(app)
    call = get_call(app)
    assert call.status == db.DONE and call.call_type == "incoming" and employee_name(app) == "Анна Смирнова"


# ---------------------------------------------------------------- карточка звонка


def test_card_form_keeps_data(settings):
    app, client, _, _ = make(settings)
    client.post("/employees", data={"name": "Анна"})
    with app.state.session_factory() as s:
        emp = s.scalars(select(db.Employee)).one()
    client.post(
        "/calls/upload",
        files=[("files", ("2026-10-08__14-32-05__79001112233__101.txt", sample_text().encode(), "text/plain"))],
        data={"employee_id": str(emp.id)},
    )
    process_all(app)
    client.post(f"/employees/{emp.id}/toggle")  # сотрудник уволился
    page = client.get("/calls/1").text
    assert f'<option value="{emp.id}" selected>Анна (не работает)</option>' in page
    call = get_call(app)
    assert call.started_at.second == 5 and f'value="{call.started_at:%Y-%m-%dT%H:%M:%S}"' in page  # секунды не теряются

    form = {"employee_id": str(emp.id), "call_type": "auto", "direction": call.direction or "",
            "started_at": call.started_at.strftime("%Y-%m-%dT%H:%M:%S")}
    client.post("/calls/1/update", data=form)
    after = get_call(app)
    assert after.employee_id == emp.id and after.started_at == call.started_at and after.status == db.DONE

    # ошибочно выбранный исходящий тип, затем возврат к автоопределению с верным направлением
    client.post("/calls/1/update", data={**form, "call_type": "out_request"})
    assert get_call(app).direction == "out"
    client.post("/calls/1/update", data={**form, "call_type": "auto", "direction": "in"})
    call = get_call(app)
    assert not call.call_type_manual and call.direction == "in" and call.status == db.QUEUED


def test_deleted_manual_type_not_reset_on_save(settings):
    app, client, _, _ = make(settings)
    upload_text(client, call_type="out_noshow")
    process_all(app)
    with app.state.session_factory() as s:
        sc = s.scalars(select(db.Scenario).where(db.Scenario.key == "out_noshow")).one()
    client.post(f"/settings/checklist/sections/{sc.id}/delete")
    page = client.get("/calls/1").text
    assert "(удалён из критериев)" in page
    call = get_call(app)
    client.post("/calls/1/update", data={"call_type": "out_noshow", "direction": call.direction or "",
                                          "started_at": "2026-10-01T10:00:00"})
    call = get_call(app)
    assert call.status == db.DONE and call.call_type == "out_noshow" and call.call_type_manual
    assert call.started_at == datetime(2026, 10, 1, 10, 0, 0)


def test_upload_rejects_unknown_formats(settings):
    app, client, _, _ = make(settings)
    r = client.post("/calls/upload", files=[
        ("files", ("договор.pdf", b"%PDF", "application/pdf")),
        ("files", ("a." + "x" * 300, b"data", "application/octet-stream")),
        ("files", ("call.mp3", b"ID3data", "audio/mpeg")),
    ])
    assert r.status_code == 200 and "Загружено файлов: 1" in r.text and "договор.pdf" in r.text
    r = client.post("/calls/upload", files=[("files", ("x.docx", b"x", "application/octet-stream"))])
    assert "Нет файлов для загрузки" in r.text


# ---------------------------------------------------------------- ошибки и мелочи


def test_friendly_errors(settings):
    app, client, _, _ = make(settings)
    html = {"accept": "text/html"}
    r = client.get("/calls/999", headers=html)
    assert r.status_code == 404 and "Не найдено" in r.text and "<html" in r.text
    assert client.get("/calls/99999999999999999999", headers=html).status_code == 404
    assert client.get("/calls?days=99999999").status_code == 200
    assert client.get("/calls?page=99999999999999").status_code == 200
    assert client.get("/stats?days=99999999").status_code == 200
    assert client.get("/calls?days=abc", headers=html).status_code == 422
    upload_text(client)
    client.post("/calls/1/delete")
    r = client.post("/calls/1/delete", headers={"referer": "http://testserver/calls"})  # двойное нажатие
    assert r.status_code == 200 and "Звонок не найден" in r.text and str(r.url).startswith("http://testserver/calls")
    r = client.post("/settings/checklist/weights", data={"high": "", "medium": "2", "low": "1"})
    assert "Баллы — числа от 0 до 1000" in r.text
    r = client.post("/settings/checklist/weights", data={"high": "2,5", "medium": "2", "low": "1"})
    assert "сохранены" in r.text


def test_login_redirect_keeps_query_and_disabled_clinic_message(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    guest = TestClient(app)
    r = guest.get("/calls?employee=5&days=0", follow_redirects=False)
    assert r.headers["location"] == "/login?next=/calls%3Femployee%3D5%26days%3D0"
    admin.post(f"/admin/clinics/{clinic_b}/toggle")
    manager = TestClient(app)
    r = manager.post("/login", data={"username": "manager_b", "password": "secret-b"})
    assert "отключена" in r.text
    # назначить руководителю отключённую клинику нельзя
    r = admin.post("/admin/users", data={"username": "m2", "password": "secret-2", "role": "manager", "clinic_id": str(clinic_b)})
    assert "отключена" in r.text


def test_mango_settings_saved_when_key_missing(settings):
    app, client, _, _ = make(settings)
    r = client.post("/settings/mango", data={"enabled": "1", "api_key": "KEY", "api_salt": "", "interval_min": "60", "lines": "101"})
    assert "автозагрузка не включена" in r.text
    with app.state.session_factory() as s:
        c = s.get(db.Clinic, app.state.first_clinic_id)
        assert (c.mango_api_key, c.mango_interval_min, c.mango_lines, c.mango_enabled) == ("KEY", 60, "101", False)
