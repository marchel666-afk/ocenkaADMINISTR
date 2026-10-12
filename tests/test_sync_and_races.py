"""Повторное копирование чек-листов, гонки обработки и запросов, ограничения входа, сообщения после действий."""

import sqlite3
import threading

import yaml
from fastapi.testclient import TestClient
from sqlalchemy import event, select
from sqlalchemy import text as text_sql

from app import db, routes_auth, routes_settings
from app.checklists import load_clinic_checklists
from app.pipeline import Worker

from .conftest import ADMIN_LOGIN, FakeLLM, FakeTranscriber, login
from .test_clinics import criterion, make_two_clinics, upload_text
from .test_web import get_call, make, process_all


def active_codes(app, clinic_id):
    with app.state.session_factory() as s:
        cl = load_clinic_checklists(s, clinic_id)
        rows = s.scalars(select(db.CriterionDef).where(db.CriterionDef.clinic_id == clinic_id)).all()
        scenarios = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == clinic_id)).all()
    codes = sorted([c.id for c in cl.general.criteria] + [c.id for sc in cl.scenarios.values() for c in sc.criteria])
    return codes, sorted(cl.scenarios), len(rows), len(scenarios)


# ---------------------------------------------------------------- копирование чек-листов


def test_repeated_copy_and_round_trip_keep_codes(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    g01 = criterion(app, a, "G01")
    admin.post(f"/settings/checklist/criteria/{g01.id}", data={"text": "Правило клиники А", "num": "1", "weight": "high", "evaluable": "1"})
    with app.state.session_factory() as s:
        sc = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == a, db.Scenario.key == "incoming")).one()
        sc_id = sc.id
    admin.post(f"/settings/checklist/sections/{sc_id}", data={"title": "Входящий (А)", "direction": "in"})

    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    first = active_codes(app, clinic_b)
    exported_a = None
    for _ in range(2):  # повторное копирование того же чек-листа ничего не меняет
        admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
        assert active_codes(app, clinic_b) == first
    # повторная загрузка файла, выгруженного в А, тоже
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    exported_a = admin.get("/settings/checklist/export.yaml").content
    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/import", files=[("file", ("a.yaml", exported_a, "application/x-yaml"))])
    assert active_codes(app, clinic_b) == first

    # обратно Б → А: у самой А ключи и коды остаются прежними
    before_a = active_codes(app, a)
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(clinic_b)})
    after_a = active_codes(app, a)
    assert after_a[:2] == before_a[:2] and "incoming" in after_a[1] and "G01" in after_a[0]


def test_copy_from_clinic_with_legacy_codes_is_friendly(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    with app.state.session_factory() as s:
        s.scalars(select(db.CriterionDef).where(db.CriterionDef.clinic_id == a, db.CriterionDef.code == "G01")).one().code = "1.1"
        s.commit()
    r = admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    assert r.status_code == 200 and "Не удалось скопировать" in r.text
    r = admin.post("/admin/clinics", data={"name": "Клиника В", "copy_from": str(a)})
    assert r.status_code == 200 and "Не удалось скопировать" in r.text


def test_double_add_criterion(settings, monkeypatch):
    app, client, _, _ = make(settings)
    clinic = app.state.first_clinic_id
    real = routes_settings.next_criterion_code
    calls = []

    def racy(s, clinic_id, prefix):  # первый раз отдаём код, который «только что» занял другой запрос
        calls.append(1)
        return "G01" if len(calls) == 1 else real(s, clinic_id, prefix)

    monkeypatch.setattr(routes_settings, "next_criterion_code", racy)
    with app.state.session_factory() as s:
        general = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == clinic, db.Scenario.key == "general")).one()
    r = client.post("/settings/checklist/criteria", data={"scenario_id": general.id, "text": "Новый пункт", "weight": "low", "evaluable": "1"})
    assert r.status_code == 200 and "добавлен" in r.text
    assert criterion(app, clinic, "G28").text == "Новый пункт"


# ---------------------------------------------------------------- обработка звонков


class HookTranscriber(FakeTranscriber):
    def __init__(self, hook=None):
        super().__init__()
        self.hook = hook

    def transcribe(self, path):
        if self.hook:
            hook, self.hook = self.hook, None
            hook()
        return super().transcribe(path)


def upload_audio(client, name="call.mp3"):
    return client.post("/calls/upload", files=[("files", (name, b"ID3fake", "audio/mpeg"))])


def test_transcript_kept_when_type_changed_during_transcription(settings):
    holder = {}
    transcriber = HookTranscriber(lambda: holder["c"].post("/calls/1/update", data={"call_type": "incoming"}))
    app, client, _, _ = make(settings, transcriber=transcriber)
    holder["c"] = client
    upload_audio(client)
    process_all(app)
    call = get_call(app)
    assert call.status == db.QUEUED and call.transcript_json is not None  # оценка отброшена, расшифровка — нет
    process_all(app)
    assert get_call(app).status == db.DONE and transcriber.calls == 1


def test_usage_of_superseded_run_is_counted(settings):
    holder = {}

    class HookLLM(FakeLLM):
        def complete(self, system_parts, user, max_tokens=None, model=None):
            if "определи его тип" not in "\n".join(system_parts) and holder.get("hook"):
                holder.pop("hook")()
            return super().complete(system_parts, user, max_tokens, model)

    app, client, _, llm = make(settings, llm=HookLLM())
    holder["hook"] = lambda: client.post("/calls/1/reevaluate")
    upload_text(client)
    process_all(app)
    process_all(app)
    call = get_call(app)
    assert call.status == db.DONE and len(llm.requests) == 4
    assert call.llm_tokens == 4 * 1500  # оплачены все четыре запроса, в том числе отброшенного прогона


def test_requeue_always_writes_status(settings):
    """Обработчик взял звонок, пока шёл запрос «Переоценить»: статус всё равно возвращается в очередь."""
    app, client, _, _ = make(settings)
    upload_text(client)
    engine = app.state.session_factory.kw["bind"]
    fired = []

    def worker_claims(conn, cursor, statement, parameters, context, executemany):
        if not fired and statement.lstrip().upper().startswith("SELECT") and "FROM calls" in statement:
            fired.append(1)
            con = sqlite3.connect(settings.data_dir / "ocenka.db", timeout=5)
            con.execute("UPDATE calls SET status = 'transcribing' WHERE id = 1")
            con.commit()
            con.close()

    event.listen(engine, "after_cursor_execute", worker_claims)
    try:
        client.post("/calls/1/reevaluate")
    finally:
        event.remove(engine, "after_cursor_execute", worker_claims)
    assert fired and get_call(app).status == db.QUEUED


def test_queue_round_robin_with_manual_first_inside_clinic(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    with app.state.session_factory() as s:
        s.get(db.Call, 1).status = db.DONE
        s.add(db.Call(clinic_id=clinic_b, source="mango", status=db.QUEUED, transcript_json="{}", original_filename="b-mango"))
        for i in range(5):
            s.add(db.Call(clinic_id=a, source="upload", status=db.QUEUED, transcript_json="{}", original_filename=f"a{i}"))
        s.add(db.Call(clinic_id=a, source="mango", status=db.QUEUED, transcript_json="{}", original_filename="a-mango"))
        s.add(db.Call(clinic_id=clinic_b, source="upload", status=db.QUEUED, transcript_json="{}", original_filename="b-manual"))
        s.commit()
    worker = Worker(app.state.processor)
    order = []
    while (call_id := worker._next_call_id()) is not None:
        with app.state.session_factory() as s:
            call = s.get(db.Call, call_id)
            order.append(call.original_filename)
            call.status = db.DONE
            s.commit()
    # архив клиники А не задерживает клинику Б, внутри клиники ручные — раньше Mango
    assert order[:4] == ["a0", "b-manual", "a1", "b-mango"]
    assert order.index("a-mango") == len(order) - 1


# ---------------------------------------------------------------- вход, пароли, сообщения


def test_login_throttle_holds_under_parallel_attempts(settings, monkeypatch):
    app, client, _, _ = make(settings)
    checked = []
    real = routes_auth.verify_password

    def slow_verify(password, hashed):
        checked.append(1)
        threading.Event().wait(0.05)
        return real(password, hashed)

    monkeypatch.setattr(routes_auth, "verify_password", slow_verify)
    start = threading.Barrier(12)

    def attempt():
        guest = TestClient(app)
        start.wait()
        guest.post("/login", data={"username": "admin", "password": "wrong"})

    threads = [threading.Thread(target=attempt) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(checked) <= routes_auth.MAX_FAILURES


def test_account_password_check_is_throttled(settings):
    app, client, _, _ = make(settings)
    for _ in range(routes_auth.MAX_FAILURES):
        r = client.post("/account/password", data={"old_password": "wrong", "new_password": "x" * 8, "new_password2": "x" * 8})
        assert "Текущий пароль указан неверно" in r.text
    r = client.post("/account/password", data={"old_password": ADMIN_LOGIN[1], "new_password": "x" * 8, "new_password2": "x" * 8})
    assert "Слишком много неудачных попыток" in r.text


def test_admin_own_password_change_closes_other_sessions(settings):
    app, admin, _ = make_two_clinics(settings)
    thief = TestClient(app)
    login(thief, *ADMIN_LOGIN)
    with app.state.session_factory() as s:
        me = s.scalars(select(db.User).where(db.User.username == ADMIN_LOGIN[0])).one()
    r = admin.post(f"/admin/users/{me.id}", data={"role": "admin", "password": "brand-new-pass", "active": "1"})
    assert "сохранён" in r.text
    assert admin.get("/admin/users").status_code == 200 and "/login" not in str(admin.get("/admin/users").url)
    assert "/login" in str(thief.get("/admin/users").url)


def test_flash_is_one_time_cookie(settings):
    app, client, _, _ = make(settings)
    guest = TestClient(app)  # страница входа без сессии с испорченной ссылкой
    r = guest.get("/login?msg=x&fs=%D1%84")
    assert r.status_code == 200 and ">x<" not in r.text
    for bogus in ("abc.zz", "W10=." + "0" * 24, "!!!." + "f" * 24):  # испорченная или подделанная cookie
        client.cookies.set("ocenka_flash", bogus)
        assert client.get("/calls").status_code == 200
    client.cookies.delete("ocenka_flash")  # в браузере сервер просто заменил бы её своей
    r = client.post("/settings/clinic", data={"name": "Клиника"})
    assert "сохранены" in r.text
    assert "сохранены" not in client.get("/settings/clinic").text  # показывается один раз


def test_admin_clinic_is_pinned_after_login(settings):
    app, admin, clinic_b = make_two_clinics(settings)  # «Варикоза нет» и «Клиника Б»
    a = app.state.first_clinic_id
    fresh = TestClient(app)
    login(fresh, *ADMIN_LOGIN)
    assert f'data-clinic="{a}"' in fresh.get("/calls").text
    fresh.post("/settings/clinic", data={"name": "Ясная клиника"})  # теперь по алфавиту она после «Клиника Б»
    assert f'data-clinic="{a}"' in fresh.get("/calls").text


# ---------------------------------------------------------------- третий круг проверки


def placement(app, clinic_id, codes):
    with app.state.session_factory() as s:
        rows = s.scalars(select(db.CriterionDef).where(db.CriterionDef.clinic_id == clinic_id, db.CriterionDef.code.in_(codes)))
        keys = {sc.id: sc.key for sc in s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == clinic_id))}
        return {c.code: (keys[c.scenario_id], c.active) for c in rows}


def test_copy_does_not_move_items_between_sections(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    codes = ["IN23", "RQ26", "RS12", "NS12"]
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    with app.state.session_factory() as s:
        general = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == a, db.Scenario.key == "general")).one()
        text = criterion(app, a, "IN23").text
    admin.post("/settings/checklist/criteria", data={"scenario_id": general.id, "text": text, "weight": "low", "evaluable": "1"})
    before_a, before_b = placement(app, a, codes), placement(app, clinic_b, codes)
    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    assert placement(app, clinic_b, codes) == before_b  # пункты сценариев остались на своих местах
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(clinic_b)})
    assert placement(app, a, codes) == before_a


def test_deleted_item_of_other_section_is_not_revived(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    for clinic in (a, clinic_b):  # в обеих клиниках удалили сценарий «неявка»
        admin.post("/switch-clinic", data={"clinic_id": str(clinic)})
        with app.state.session_factory() as s:
            sc = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == clinic, db.Scenario.key == "out_noshow")).one()
        admin.post(f"/settings/checklist/sections/{sc.id}/delete")
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    with app.state.session_factory() as s:
        rs = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == a, db.Scenario.key == "out_reschedule")).one()
        ns01_text = criterion(app, a, "NS01").text
    admin.post("/settings/checklist/criteria", data={"scenario_id": rs.id, "text": ns01_text, "weight": "low", "evaluable": "1"})
    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    assert placement(app, clinic_b, ["NS01"]) == {"NS01": ("out_noshow", False)}  # история неявок не переехала


class ReuseIdTranscriber(FakeTranscriber):
    """Во время расшифровки звонок удаляют, а новый получает тот же номер (так бывает в старых базах)."""

    def __init__(self, app_holder):
        super().__init__()
        self.app_holder = app_holder

    def transcribe(self, path):
        if self.calls == 0:
            app = self.app_holder["app"]
            with app.state.session_factory() as s:
                old = s.get(db.Call, 1)
                clinic_id = old.clinic_id
                s.delete(old)
                s.commit()
                s.add(db.Call(id=1, clinic_id=clinic_id, source="upload", audio_path=old.audio_path,
                              original_filename="right.mp3", status=db.QUEUED))
                s.commit()
        return super().transcribe(path)


def test_transcript_of_deleted_call_not_written_to_new_call_with_same_id(settings):
    holder = {}
    transcriber = ReuseIdTranscriber(holder)
    app, client, _, _ = make(settings, transcriber=transcriber)
    holder["app"] = app
    upload_audio(client, "wrong.mp3")
    process_all(app)
    call = get_call(app)
    assert call.original_filename == "right.mp3" and call.transcript_json is None and call.status == db.QUEUED


def test_usage_of_deleted_call_not_added_to_new_call(settings):
    holder = {}

    class ReuseLLM(FakeLLM):
        def complete(self, system_parts, user, max_tokens=None, model=None):
            if "определи его тип" not in "\n".join(system_parts) and not holder.get("done"):
                holder["done"] = True
                with holder["app"].state.session_factory() as s:
                    old = s.get(db.Call, 1)
                    clinic_id = old.clinic_id
                    s.delete(old)
                    s.commit()
                    s.add(db.Call(id=1, clinic_id=clinic_id, source="text", transcript_json=None,
                                  original_filename="right.txt", status=db.QUEUED))
                    s.commit()
            return super().complete(system_parts, user, max_tokens, model)

    app, client, _, _ = make(settings, llm=ReuseLLM())
    holder["app"] = app
    upload_text(client)
    process_all(app)
    call = get_call(app)
    assert call.original_filename == "right.txt" and not call.llm_tokens


def test_switch_clinic_does_not_revive_revoked_session(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    other = TestClient(app)
    login(other, *ADMIN_LOGIN)
    svc = app.state.svc
    real_set = svc.set_session

    def logout_everywhere_meanwhile(response, user_id, clinic_id=None, sv=None):
        other.post("/account/logout-all")  # пока запрос переключения шёл, на другом устройстве нажали «выйти везде»
        return real_set(response, user_id, clinic_id, sv=sv)

    svc.set_session = logout_everywhere_meanwhile
    try:
        admin.post("/switch-clinic", data={"clinic_id": str(app.state.first_clinic_id)})
    finally:
        svc.set_session = real_set
    assert "/login" in str(admin.get("/calls").url)


def test_new_database_never_reuses_call_ids(settings):
    app, client, _, _ = make(settings)
    upload_text(client, "a.txt")
    upload_text(client, "b.txt")
    client.post("/calls/2/delete")
    upload_text(client, "c.txt")
    with app.state.session_factory() as s:
        assert s.scalars(select(db.Call.id).where(db.Call.original_filename == "c.txt")).one() == 3


# ---------------------------------------------------------------- четвёртый круг проверки


def test_old_database_gets_autoincrement(settings):
    """База, созданная до AUTOINCREMENT: таблица звонков перестраивается без потери данных и связей."""
    opts = db.Call.__table__.dialect_options["sqlite"]
    opts["autoincrement"] = False
    try:
        old = db.make_session_factory(settings.db_url)
    finally:
        opts["autoincrement"] = True
    with old() as s:
        clinic = db.Clinic(name="К")
        s.add(clinic)
        s.flush()
        for name in ("a.txt", "b.txt"):
            call = db.Call(clinic_id=clinic.id, source="text", original_filename=name, status=db.DONE)
            call.results.append(db.CriterionResult(checklist="general", criterion_id="G01", result="yes", weight=1.0))
            s.add(call)
        s.commit()
    con = sqlite3.connect(settings.data_dir / "ocenka.db")
    assert "AUTOINCREMENT" not in con.execute("SELECT sql FROM sqlite_master WHERE name='calls'").fetchone()[0]
    con.close()

    sf = db.make_session_factory(settings.db_url)  # запуск новой версии
    con = sqlite3.connect(settings.data_dir / "ocenka.db")
    assert "AUTOINCREMENT" in con.execute("SELECT sql FROM sqlite_master WHERE name='calls'").fetchone()[0]
    indexes = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='calls'")}
    assert {"ix_calls_status", "ix_calls_clinic", "ix_calls_external"} <= indexes
    con.close()
    with sf() as s:
        assert [c.original_filename for c in s.scalars(select(db.Call).order_by(db.Call.id))] == ["a.txt", "b.txt"]
        s.delete(s.get(db.Call, 2))  # результаты удаляются вместе со звонком (связь сохранилась)
        s.commit()
        assert s.scalar(select(db.CriterionResult.id).where(db.CriterionResult.call_id == 2)) is None
        new = db.Call(clinic_id=s.get(db.Call, 1).clinic_id, source="text", original_filename="c.txt", status=db.QUEUED)
        s.add(new)
        s.commit()
        assert new.id == 3  # номер удалённого звонка не достался новому
        assert s.scalar(select(db.CriterionResult.call_id).where(db.CriterionResult.call_id == 1)) == 1


def test_clinic_stays_put_for_old_cookie_and_after_disable(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    with app.state.session_factory() as s:
        me = s.scalars(select(db.User).where(db.User.username == ADMIN_LOGIN[0])).one()
    old_style = TestClient(app)  # вход старой версии: клиника в cookie не записана
    old_style.cookies.set("ocenka_session", app.state.svc.signer.dumps({"uid": me.id, "clinic": None, "sv": 0}))
    assert f'data-clinic="{a}"' in old_style.get("/calls").text
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    admin.post("/settings/clinic", data={"name": "Ясная клиника"})  # по алфавиту теперь после «Клиника Б»
    assert f'data-clinic="{a}"' in old_style.get("/calls").text

    # работавшему в Б другому администратору Б отключили: его переводит в другую клинику один раз и закрепляет
    worker = TestClient(app)
    login(worker, *ADMIN_LOGIN)
    worker.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post(f"/admin/clinics/{clinic_b}/toggle")
    assert f'data-clinic="{a}"' in worker.get("/calls").text
    admin.post(f"/admin/clinics/{clinic_b}/toggle")  # Б включили снова — молча обратно не перебрасывает
    assert f'data-clinic="{a}"' in worker.get("/calls").text


def test_moved_item_keeps_code_on_recopy(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    in07 = criterion(app, a, "IN07")
    admin.post(f"/settings/checklist/criteria/{in07.id}", data={"text": "Своя формулировка А", "num": "7", "weight": "medium", "evaluable": "1"})
    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    with app.state.session_factory() as s:
        code = s.scalars(select(db.CriterionDef.code).where(db.CriterionDef.clinic_id == clinic_b,
                                                            db.CriterionDef.text == "Своя формулировка А")).one()
        incoming = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == clinic_b, db.Scenario.key == "incoming")).one()
    admin.post(f"/settings/checklist/sections/{incoming.id}", data={"title": "Входящий (Б)", "direction": "in"})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    assert criterion(app, clinic_b, code).active  # пункт с историей не заменён новым кодом


# ---------------------------------------------------------------- пятый круг проверки


def test_item_moved_in_source_survives_copy_round_trip(settings):
    app, admin, clinic_b = make_two_clinics(settings)
    a = app.state.first_clinic_id
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    in07 = criterion(app, a, "IN07")
    admin.post(f"/settings/checklist/criteria/{in07.id}/delete")  # «перенос»: удалили и добавили в другой раздел
    with app.state.session_factory() as s:
        rm = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == a, db.Scenario.key == "out_reminder")).one()
    admin.post("/settings/checklist/criteria", data={"scenario_id": rm.id, "text": in07.text, "weight": "medium", "evaluable": "1"})
    with app.state.session_factory() as s:
        moved = s.scalars(select(db.CriterionDef.code).where(db.CriterionDef.clinic_id == a,
                                                             db.CriterionDef.scenario_id == rm.id,
                                                             db.CriterionDef.text == in07.text)).one()
    before = placement(app, a, ["IN07", moved])
    admin.post("/switch-clinic", data={"clinic_id": str(clinic_b)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(a)})
    admin.post("/switch-clinic", data={"clinic_id": str(a)})
    admin.post("/settings/checklist/copy", data={"source_clinic_id": str(clinic_b)})
    assert placement(app, a, ["IN07", moved]) == before == {"IN07": ("incoming", False), moved: ("out_reminder", True)}


def test_migration_waits_briefly_and_never_blocks_startup(settings, monkeypatch):
    opts = db.Call.__table__.dialect_options["sqlite"]
    opts["autoincrement"] = False
    try:
        db.make_session_factory(settings.db_url)
    finally:
        opts["autoincrement"] = True
    path = settings.data_dir / "ocenka.db"
    holder = sqlite3.connect(path, timeout=1, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")  # другая программа держит запись
    monkeypatch.setattr(db, "MIGRATION_BUSY_MS", 200)
    try:
        sf = db.make_session_factory(settings.db_url)  # запуск не падает
        with sf() as s:
            assert s.execute(text_sql("PRAGMA foreign_keys")).scalar() == 1
    finally:
        holder.execute("ROLLBACK")
        holder.close()
    con = sqlite3.connect(path)
    assert "AUTOINCREMENT" not in con.execute("SELECT sql FROM sqlite_master WHERE name='calls'").fetchone()[0]
    con.close()
    db.make_session_factory(settings.db_url)  # следующий запуск перестраивает таблицу
    con = sqlite3.connect(path)
    assert "AUTOINCREMENT" in con.execute("SELECT sql FROM sqlite_master WHERE name='calls'").fetchone()[0]
    con.close()
