"""Звонки не зависают «в обработке»: второй запуск сервиса, брошенные звонки, повтор звонков с ошибкой,
лимит длины ответа при определении типа звонка."""

import threading

import pytest

from app import __main__ as main_module
from app import db
from app.evaluation import CLASSIFY_MAX_TOKENS
from app.llm import LLMError, LLMResponse
from app.pipeline import ORPHAN_MESSAGE, Worker

from .conftest import FakeLLM
from .test_clinics import make_two_clinics, upload_text
from .test_web import get_call, make, process_all


def set_status(app, call_id, status, message=""):
    with app.state.session_factory() as s:
        call = s.get(db.Call, call_id)
        call.status, call.status_message = status, message
        s.commit()


def test_classification_leaves_room_for_thinking(settings):
    """Тип звонка определялся с лимитом 2000 токенов: модель тратила их на «размышления» над длинным
    разговором, и ответ обрывался («Ответ модели обрезан»)."""
    app, client, _, llm = make(settings, llm=FakeLLM())
    upload_text(client)
    process_all(app)
    assert get_call(app).status == db.DONE
    assert llm.max_tokens[0] == CLASSIFY_MAX_TOKENS >= 8000
    assert llm.max_tokens[1] is None  # оценка — с общим лимитом LLM_MAX_TOKENS


def test_cost_of_cut_off_answers_is_recorded(settings):
    """Ответ оборвался даже на наибольшем лимите: звонок с ошибкой, но потраченные деньги в его стоимости."""

    class CutOff(FakeLLM):
        def complete(self, system_parts, user, max_tokens=None, model=None):
            if "определи его тип" in "\n".join(system_parts):
                return super().complete(system_parts, user, max_tokens, model)
            spent = LLMResponse(text="", prompt_tokens=2000, completion_tokens=96000, cost_usd=0.5)
            raise LLMError("Ответ модели обрезан", spent=spent)

    app, client, _, _ = make(settings, llm=CutOff())
    upload_text(client)
    process_all(app)
    call = get_call(app)
    assert call.status == db.ERROR and "обрезан" in call.status_message
    assert call.llm_cost_usd == pytest.approx(0.501)  # классификация 0.001 + оборвавшиеся попытки оценки
    assert call.llm_tokens == 1500 + 98000


def test_second_instance_lock(tmp_path):
    path = tmp_path / "service.lock"
    first = main_module.acquire_instance_lock(path)
    assert first is not None
    assert main_module.acquire_instance_lock(path) is None
    first.close()
    again = main_module.acquire_instance_lock(path)
    assert again is not None
    again.close()


def test_second_start_does_not_touch_queue(settings, monkeypatch, capsys):
    """Повторный запуск start.bat, пока сервис работает: вторая копия возвращала в очередь и брала в работу
    звонки первой, а закрывшись (порт занят), оставляла звонок «в расшифровке» навсегда."""
    held = main_module.acquire_instance_lock(settings.data_dir / "service.lock")
    started = []
    monkeypatch.setattr(main_module.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(main_module, "create_app", lambda *a, **k: started.append("app"))
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **k: started.append("server"))
    monkeypatch.delenv("OPEN_BROWSER", raising=False)
    try:
        main_module.main()
    finally:
        held.close()
    assert started == []
    assert "уже запущен" in capsys.readouterr().out

    main_module.main()  # блокировку освободили — сервис запускается
    assert started == ["app", "server"]


def test_orphaned_call_marked_error_when_queue_is_empty(settings):
    """Звонок «в обработке», которым никто не занят (не записался итог обработки), не висит вечно."""
    app, client, _, _ = make(settings)
    upload_text(client, "a.txt")
    upload_text(client, "b.txt")
    set_status(app, 1, db.TRANSCRIBING)
    worker = Worker(app.state.processor, poll_sec=0.01)
    worker.start()
    try:
        for _ in range(500):
            if get_call(app, 1).status == db.ERROR:
                break
            threading.Event().wait(0.01)
    finally:
        worker.stop()
        worker.join(5)
    assert get_call(app, 2).status == db.DONE  # очередь обработана
    call = get_call(app, 1)
    assert call.status == db.ERROR and call.status_message == ORPHAN_MESSAGE


def test_orphan_does_not_block_queue(settings):
    app, client, _, _ = make(settings)
    upload_text(client, "a.txt")
    upload_text(client, "b.txt")
    set_status(app, 1, db.EVALUATING)
    worker = Worker(app.state.processor)
    assert worker._next_call_id() == 2
    set_status(app, 2, db.DONE)
    assert worker._next_call_id() is None
    worker._fail_orphans()
    assert get_call(app, 1).status == db.ERROR


def test_retry_errors_only_in_current_clinic(settings):
    app, admin, clinic_b = make_two_clinics(settings)  # звонок 1 — клиника А, админ переключён на Б
    upload_text(admin, "b1.txt")
    upload_text(admin, "b2.txt")
    upload_text(admin, "b3.txt")
    set_status(app, 1, db.ERROR, "LLMError: сбой")
    set_status(app, 2, db.ERROR, "LLMError: сбой")
    set_status(app, 3, db.ERROR, "LLMError: сбой")
    set_status(app, 4, db.DONE)

    page = admin.get("/calls").text
    assert "Звонков с ошибкой: 2" in page and 'action="/calls/retry-errors"' in page
    assert 'title="LLMError: сбой"' in page  # причину видно, наведя на статус

    r = admin.post("/calls/retry-errors", follow_redirects=False)
    assert r.status_code == 303
    page = admin.get("/calls").text
    assert "Звонков возвращено в очередь: 2" in page and "Звонков с ошибкой" not in page
    assert get_call(app, 1).status == db.ERROR  # чужая клиника не тронута
    for call_id in (2, 3):
        call = get_call(app, call_id)
        assert call.status == db.QUEUED and call.status_message == ""
    assert get_call(app, 4).status == db.DONE
    process_all(app)
    assert get_call(app, 2).status == db.DONE and get_call(app, 3).status == db.DONE


@pytest.mark.parametrize("status", [db.QUEUED, db.DONE])
def test_no_retry_button_without_errors(settings, status):
    app, client, _, _ = make(settings)
    upload_text(client)
    set_status(app, 1, status)
    assert "retry-errors" not in client.get("/calls").text
