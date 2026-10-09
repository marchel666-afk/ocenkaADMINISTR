"""Сеть клиник: вход и роли, изоляция клиник, редактор критериев, перенос чек-листов, история оценок."""

import sqlite3

import yaml
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import db
from app.checklists import load_clinic_checklists
from app.web import create_app

from .conftest import ADMIN_LOGIN, FakeLLM, FakeTranscriber, login, sample_text, setup_admin
from .test_web import get_call, make, process_all


def upload_text(client, name="call.txt", **data):
    return client.post("/calls/upload", files=[("files", (name, sample_text().encode(), "text/plain"))], data=data)


def scenario_id(app, clinic_id, key):
    with app.state.session_factory() as s:
        return s.scalars(
            select(db.Scenario.id).where(db.Scenario.clinic_id == clinic_id, db.Scenario.key == key)
        ).first()


def criterion(app, clinic_id, code) -> db.CriterionDef:
    with app.state.session_factory() as s:
        return s.scalars(
            select(db.CriterionDef).where(db.CriterionDef.clinic_id == clinic_id, db.CriterionDef.code == code)
        ).first()


# ---------------------------------------------------------------- вход


def test_first_run_setup(settings):
    app = create_app(settings, transcriber=FakeTranscriber(), llm=FakeLLM())
    client = TestClient(app)
    r = client.get("/calls", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/setup"
    r = client.post("/setup", data={"username": "boss", "password": "123", "password2": "123"})
    assert "не короче" in r.text
    setup_admin(client)
    assert client.get("/admin/users").status_code == 200
    # второй раз создать администратора через /setup нельзя
    client.cookies.clear()
    r = client.post("/setup", data={"username": "hacker", "password": "123456", "password2": "123456"})
    assert "/login" in str(r.url)
    with app.state.session_factory() as s:
        assert [u.username for u in s.scalars(select(db.User))] == [ADMIN_LOGIN[0]]


def test_login_logout_and_tampered_cookie(settings):
    app, client, _, _ = make(settings)
    assert client.get("/calls").status_code == 200
    token = client.cookies.get("ocenka_session")
    client.cookies.set("ocenka_session", token[:-2] + "xx")
    assert "/login" in str(client.get("/calls").url)
    login(client, *ADMIN_LOGIN)
    client.post("/logout")
    assert "/login" in str(client.get("/calls").url)


def test_open_redirect_blocked(settings):
    app, client, _, _ = make(settings, login=False)
    setup_admin(client)
    client.cookies.clear()
    r = client.post(
        "/login",
        data={"username": ADMIN_LOGIN[0], "password": ADMIN_LOGIN[1], "next": "https://evil.example/"},
        follow_redirects=False,
    )
    assert r.headers["location"] == "/calls"


# ---------------------------------------------------------------- клиники и роли


def make_two_clinics(settings):
    app, client, transcriber, llm = make(settings)
    upload_text(client, "a.txt")  # звонок клиники A (первой)
    r = client.post("/admin/clinics", data={"name": "Клиника Б", "utc_offset": "7", "copy_from": "template"})
    assert "Клиника «Клиника Б» создана" in r.text
    with app.state.session_factory() as s:
        clinic_b = s.scalars(select(db.Clinic).where(db.Clinic.name == "Клиника Б")).one()
    client.post(
        "/admin/users",
        data={"username": "manager_b", "password": "secret-b", "role": "manager", "clinic_id": str(clinic_b.id)},
    )
    return app, client, clinic_b.id


def test_new_clinic_gets_template_checklist(settings):
    app, client, clinic_b = make_two_clinics(settings)
    with app.state.session_factory() as s:
        cl = load_clinic_checklists(s, clinic_b)
        assert len(cl.general.criteria) == 27 and "incoming" in cl.scenarios
        assert s.get(db.Clinic, clinic_b).utc_offset == 7


def test_manager_sees_only_own_clinic(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    manager = TestClient(app)
    login(manager, "manager_b", "secret-b")
    page = manager.get("/calls").text
    assert "Клиника Б" in page and "Звонки <span class=\"muted\">(0)</span>" in page
    assert manager.get("/calls/1").status_code == 404  # звонок клиники A
    assert manager.get("/admin/clinics").status_code == 403
    assert manager.get("/admin/users").status_code == 403
    r = manager.post("/switch-clinic", data={"clinic_id": str(app.state.first_clinic_id)})
    assert "Нет доступа" in r.text
    # загруженный руководителем звонок попадает в его клинику
    upload_text(manager, "b.txt")
    with app.state.session_factory() as s:
        assert s.scalars(select(db.Call.clinic_id).where(db.Call.original_filename == "b.txt")).one() == clinic_b


def test_admin_switches_clinic(settings):
    app, client, clinic_b = make_two_clinics(settings)  # после создания клиники админ переключён на неё
    assert "Звонки <span class=\"muted\">(0)</span>" in client.get("/calls").text
    client.post("/switch-clinic", data={"clinic_id": str(app.state.first_clinic_id)})
    assert "Звонки <span class=\"muted\">(1)</span>" in client.get("/calls").text


def test_disabled_user_and_clinic(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    with app.state.session_factory() as s:
        manager_id = s.scalars(select(db.User.id).where(db.User.username == "manager_b")).one()
    admin.post(f"/admin/users/{manager_id}", data={"role": "manager", "clinic_id": str(clinic_b), "active": ""})
    manager = TestClient(app)
    r = manager.post("/login", data={"username": "manager_b", "password": "secret-b"})
    assert "Неверный логин или пароль" in r.text
    # администратор не может отключить сам себя
    with app.state.session_factory() as s:
        admin_id = s.scalars(select(db.User.id).where(db.User.username == ADMIN_LOGIN[0])).one()
    r = admin.post(f"/admin/users/{admin_id}", data={"role": "admin", "active": ""})
    assert "Нельзя лишить прав" in r.text


# ---------------------------------------------------------------- редактор критериев


def test_checklist_editor_changes_evaluation(settings):
    app, client, _, llm = make(settings)
    clinic = app.state.first_clinic_id
    general = scenario_id(app, clinic, "general")
    assert client.get(f"/settings/checklist?section={general}").status_code == 200

    # добавить пункт
    r = client.post(
        "/settings/checklist/criteria",
        data={"scenario_id": general, "text": "Сотрудник предлагает скидку постоянным пациентам", "weight": "high",
              "evaluable": "1", "hint": "Если пациент первичный — «не применимо»."},
    )
    assert "Пункт G28 добавлен" in r.text
    # изменить и отключить другой пункт
    g01 = criterion(app, clinic, "G01")
    client.post(f"/settings/checklist/criteria/{g01.id}",
                data={"text": "Новая формулировка инициативы", "num": "1", "weight": "low", "evaluable": "1", "hint": ""})
    g02 = criterion(app, clinic, "G02")
    client.post(f"/settings/checklist/criteria/{g02.id}/delete")

    upload_text(client)
    process_all(app)
    system_parts, _ = llm.requests[-1]
    checklist = system_parts[1]
    assert "G28 [значимость: высокая] Сотрудник предлагает скидку постоянным пациентам" in checklist
    assert "Подсказка: Если пациент первичный" in checklist
    assert "G01 [значимость: низкая] Новая формулировка инициативы" in checklist
    assert "G02 [" not in checklist
    assert get_call(app).status == db.DONE


def test_history_keeps_criterion_text(settings):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    upload_text(client)
    process_all(app)
    in11 = criterion(app, clinic, "IN11")
    old_text = in11.text
    client.post(f"/settings/checklist/criteria/{in11.id}",
                data={"text": "Совсем другой текст пункта", "num": "11", "weight": "high", "evaluable": "1"})
    page = client.get("/calls/1").text
    assert old_text in page and "Совсем другой текст пункта" not in page
    # удаление сценария не ломает карточку старого звонка
    client.post(f"/settings/checklist/sections/{scenario_id(app, clinic, 'incoming')}/delete")
    page = client.get("/calls/1").text
    assert "Входящий звонок" in page and old_text in page
    assert client.get("/stats").status_code == 200 and client.get("/export.xlsx").status_code == 200


def test_custom_scenario_and_move(settings):
    app, client, _, llm = make(settings)
    clinic = app.state.first_clinic_id
    r = client.post("/settings/checklist/sections",
                    data={"title": "Повторная консультация", "direction": "out", "description": "Звонок после приёма", "prefix": "pk"})
    assert "Сценарий добавлен" in r.text
    sid = scenario_id(app, clinic, "custom_1")
    client.post("/settings/checklist/criteria", data={"scenario_id": sid, "text": "Пункт один", "weight": "medium", "evaluable": "1"})
    client.post("/settings/checklist/criteria", data={"scenario_id": sid, "text": "Пункт два", "weight": "medium", "evaluable": "1"})
    assert criterion(app, clinic, "PK01").text == "Пункт один"
    second = criterion(app, clinic, "PK02")
    client.post(f"/settings/checklist/criteria/{second.id}/move", data={"direction": "up"})
    with app.state.session_factory() as s:
        cl = load_clinic_checklists(s, clinic)
    assert [c.id for c in cl.scenarios["custom_1"].criteria] == ["PK02", "PK01"]
    assert cl.scenarios["custom_1"].direction == "out"

    # ИИ видит новый сценарий при выборе типа звонка, и по нему можно оценить звонок
    upload_text(client, direction="out")
    llm.call_type = "custom_1"
    process_all(app)
    classify_system = llm.requests[0][0][0]
    assert "custom_1: Повторная консультация. Звонок после приёма" in classify_system
    call = get_call(app)
    assert call.status == db.DONE and call.call_type == "custom_1"

    # нельзя удалить общие правила и задать занятый код раздела
    general = scenario_id(app, clinic, "general")
    assert "удалить нельзя" in client.post(f"/settings/checklist/sections/{general}/delete").text
    r = client.post("/settings/checklist/sections", data={"title": "x", "direction": "in", "prefix": "PK"})
    assert "не занятые" in r.text


def test_weights_change_scores(settings):
    app, client, _, llm = make(settings)
    upload_text(client)
    process_all(app)
    before = get_call(app).score_total
    client.post("/settings/checklist/weights", data={"high": "10", "medium": "1", "low": "1"})
    client.post("/calls/1/reevaluate")
    process_all(app)
    after = get_call(app).score_total
    assert before != after  # невыполненные пункты IN11/G04 высокой значимости стали весить больше
    assert after < before


def test_yaml_export_import_roundtrip(settings):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    exported = client.get("/settings/checklist/export.yaml")
    assert exported.status_code == 200
    data = yaml.safe_load(exported.content.decode("utf-8"))
    assert len(data["general"]["criteria"]) == 27
    data["general"]["criteria"][0]["text"] = "Изменено через файл"
    data["scenarios"].pop("out_noshow")
    r = client.post("/settings/checklist/import",
                    files=[("file", ("c.yaml", yaml.safe_dump(data, allow_unicode=True).encode(), "application/x-yaml"))])
    assert "Чек-лист загружен" in r.text
    with app.state.session_factory() as s:
        cl = load_clinic_checklists(s, clinic)
    assert cl.criterion("G01").text == "Изменено через файл" and "out_noshow" not in cl.scenarios
    # ошибочный файл не ломает чек-лист
    bad = {"general": {"title": "x", "criteria": [{"id": "A1", "text": "a"}, {"id": "A1", "text": "b"}]}, "scenarios": {}}
    r = client.post("/settings/checklist/import",
                    files=[("file", ("bad.yaml", yaml.safe_dump(bad).encode(), "application/x-yaml"))])
    assert "Повторяющиеся коды" in r.text
    r = client.post("/settings/checklist/import", files=[("file", ("bad.yaml", b"{: [", "application/x-yaml"))])
    assert "Не удалось прочитать файл" in r.text
    # сброс к шаблону возвращает сценарий
    client.post("/settings/checklist/reset")
    with app.state.session_factory() as s:
        cl = load_clinic_checklists(s, clinic)
    assert "out_noshow" in cl.scenarios and cl.criterion("G01").text.startswith("Сотрудник в общении")


def test_copy_checklist_between_clinics(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    g01 = criterion(app, a, "G01")
    admin.post(f"/settings/checklist/criteria/{g01.id}",
               data={"text": "Особое правило клиники А", "num": "1", "weight": "high", "evaluable": "1"})
    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    assert criterion(app, clinic_b, "G01").text == "Особое правило клиники А"


def test_clinic_settings_and_ai_context(settings):
    app, client, _, llm = make(settings)
    r = client.post("/settings/clinic", data={"name": "Варикоза нет — Владивосток", "ai_context": "Клиника флебологии на ул. Лазо.",
                                             "utc_offset": "10", "min_call_seconds": "45"})
    assert "сохранены" in r.text
    with app.state.session_factory() as s:
        clinic = s.get(db.Clinic, app.state.first_clinic_id)
        assert (clinic.name, clinic.utc_offset, clinic.min_call_seconds) == ("Варикоза нет — Владивосток", 10, 45)
    upload_text(client)
    process_all(app)
    assert "Клиника флебологии на ул. Лазо." in llm.requests[-1][0][0]
    # ключи Mango не стираются, если поле оставили пустым
    client.post("/settings/mango", data={"api_key": "KEY", "api_salt": "SALT", "interval_min": "30"})
    client.post("/settings/mango", data={"enabled": "1", "api_key": "", "api_salt": "", "interval_min": "5", "lines": "101; 102"})
    with app.state.session_factory() as s:
        clinic = s.get(db.Clinic, app.state.first_clinic_id)
        assert (clinic.mango_api_key, clinic.mango_api_salt, clinic.mango_enabled) == ("KEY", "SALT", True)
        assert clinic.mango_interval_min == 10 and clinic.mango_lines == "101, 102"


# ---------------------------------------------------------------- миграция старой базы


def test_migration_from_first_version(settings):
    """База первой версии: таблицы без новых колонок — сервис дополняет их значениями по умолчанию."""
    path = settings.data_dir / "ocenka.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE clinics (id INTEGER PRIMARY KEY, name VARCHAR(200));
        INSERT INTO clinics (id, name) VALUES (1, 'Старая клиника');
        CREATE TABLE criterion_results (id INTEGER PRIMARY KEY, call_id INTEGER, checklist VARCHAR(50),
            criterion_id VARCHAR(20), result VARCHAR(5), weight FLOAT, evidence TEXT, comment TEXT);
        """
    )
    con.commit()
    con.close()
    app, client, _, _ = make(settings)
    with app.state.session_factory() as s:
        clinic = s.get(db.Clinic, 1)
        assert clinic.name == "Старая клиника" and clinic.active is True and clinic.mango_enabled is False
        assert clinic.utc_offset == 3 and clinic.weights == {"high": 3.0, "medium": 2.0, "low": 1.0}
        assert len(load_clinic_checklists(s, 1).general.criteria) == 27  # чек-лист заполнен из шаблона
    upload_text(client)
    process_all(app)
    assert get_call(app).status == db.DONE
