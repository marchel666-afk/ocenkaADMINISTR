"""Автоматическая загрузка звонков из Mango Office (API виртуальной АТС) для каждой клиники.

Схема: раз в N минут запрашиваем статистику вызовов за прошедший период (stats/request → stats/result),
берём звонки с записью разговора и скачиваем записи (queries/recording/post). Последние часы
перепроверяются при каждом запуске (запись может появиться с задержкой), дубли отсекаются по id записи.

Формат API — по официальной документации «MangoOffice VPBX API» (v1.8) и рабочим интеграциям:
подпись sha256(vpbx_api_key + json + vpbx_api_salt), тело формы vpbx_api_key/sign/json, время — Unix,
статистика — CSV через «;» без заголовка, колонки в порядке fields, запись — редирект 302 на файл.
Лимиты: stats/request не чаще раза в 2 с на АТС, stats/result — раз в 5 с, при перегрузке — HTTP 429.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from . import db
from .config import Settings

log = logging.getLogger(__name__)

API_URL = "https://app.mango-office.ru/vpbx/"
STATS_FIELDS = (
    "records",
    "start",
    "finish",
    "answer",
    "from_extension",
    "from_number",
    "to_extension",
    "to_number",
    "disconnect_reason",
    "line_number",
    "location",
    "entry_id",
)
OVERLAP_SECONDS = 2 * 3600  # перепроверяем последние 2 часа: запись разговора появляется с задержкой
SETTLE_SECONDS = 5 * 60  # не берём звонки, закончившиеся меньше 5 минут назад
FIRST_RUN_SECONDS = 24 * 3600  # при первом подключении — звонки за последние сутки
MAX_WINDOW_SECONDS = 24 * 3600  # статистику запрашиваем кусками не длиннее суток (Mango даёт не больше месяца)

# коды результата из документации Mango
RESULT_MESSAGES = {
    "2000": "ограничение биллинга — проверьте баланс и тариф в Mango",
    "3100": "переданы неверные параметры",
    "3101": "запрос выполнен не методом POST",
    "3102": "подпись не совпала — проверьте ключ для создания подписи",
    "3103": "не передан обязательный параметр",
    "3104": "неверный формат параметра",
    "3105": "неверный уникальный код АТС",
    "3300": "объект не найден",
    "3320": "запись разговора не найдена",
    "3340": "файл записи не найден",
    "4000": "действие не может быть выполнено",
    "5000": "ошибка сервера Mango",
    "5001": "сервер Mango перегружен",
    "5002": "сервер Mango перезапускается",
    "5003": "технические проблемы на стороне Mango",
    "5004": "ошибка базы данных Mango",
}
KEY_ERRORS = {"3102", "3105"}


class MangoError(RuntimeError):
    """Ошибка обмена с Mango: загрузка прерывается и повторяется при следующей проверке."""


class MangoRecordingError(MangoError):
    """Не удалось скачать одну запись (удалена, ещё не сохранена) — остальные звонки загружаются дальше."""


def _result_message(code) -> str:
    code = str(code or "")
    text = RESULT_MESSAGES.get(code) or RESULT_MESSAGES.get(code[:1] + "000" if code[:1] == "2" else "", "")
    return f"код {code}" + (f" — {text}" if text else "") if code else ""


def _error_code(resp: httpx.Response) -> str:
    try:
        data = resp.json()
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    return str(data.get("result") or data.get("code") or "")


# ---------------------------------------------------------------- строки статистики


@dataclass
class MangoCall:
    recording_ids: list[str]
    start: int
    finish: int
    answer: int | None
    from_extension: str = ""
    from_number: str = ""
    to_extension: str = ""
    to_number: str = ""
    line_number: str = ""
    entry_id: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def direction(self) -> str | None:
        """Исходящий — если звонил сотрудник (есть добавочный звонящего) клиенту; входящий — наоборот."""
        if self.from_extension and not self.to_extension:
            return "out"
        if self.to_extension and not self.from_extension:
            return "in"
        if self.from_extension and self.to_extension:
            return None  # внутренний звонок между сотрудниками
        # добавочных нет: сотрудника видно только по SIP-адресу вида sip:login@domain.mangosip.ru
        from_sip, to_sip = _is_sip(self.from_number), _is_sip(self.to_number)
        if from_sip and to_sip:
            return None
        if from_sip:
            return "out"
        return "in"  # обычно клиент, которого ещё не соединили с сотрудником (IVR, очередь)

    @property
    def phone(self) -> str:
        """Номер клиента."""
        return _clean_number(self.to_number if self.direction == "out" else self.from_number)

    @property
    def extension(self) -> str:
        """Добавочный сотрудника, а если его нет — логин SIP."""
        if self.direction == "out":
            return self.from_extension or _sip_login(self.from_number)
        return self.to_extension or _sip_login(self.to_number)

    @property
    def answered(self) -> bool:
        return bool(self.answer)

    @property
    def talk_seconds(self) -> int:
        """Длительность разговора: от ответа до окончания (0 — трубку не сняли)."""
        return max(0, self.finish - self.answer) if self.answer and self.finish else 0


def _is_sip(value: str) -> bool:
    return (value or "").strip().lower().startswith("sip:")


def _sip_login(value: str) -> str:
    return value.strip()[4:].split("@", 1)[0] if _is_sip(value) else ""


def _clean_number(value: str) -> str:
    value = (value or "").strip()
    return _sip_login(value) or value


def _parse_records(value: str) -> list[str]:
    """Поле records: «[id1,id2]» или «id1» → список id записей."""
    value = (value or "").strip().strip("[]").strip()
    return [x.strip().strip('"') for x in value.split(",") if x.strip().strip('"')]


def _to_int(value: str) -> int | None:
    try:
        return int(float(value)) if str(value).strip() else None
    except ValueError:
        return None


def parse_stats_csv(text: str, fields: tuple[str, ...] = STATS_FIELDS) -> list[MangoCall]:
    """Разбор ответа stats/result: строки через перевод строки, поля через «;» в порядке fields."""
    calls = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split(";")
        row = {name: (values[i].strip() if i < len(values) else "") for i, name in enumerate(fields)}
        start = _to_int(row.get("start", ""))
        if start is None:  # строка заголовка или мусор
            continue
        calls.append(
            MangoCall(
                recording_ids=_parse_records(row.get("records", "")),
                start=start,
                finish=_to_int(row.get("finish", "")) or start,
                answer=_to_int(row.get("answer", "")) or None,  # 0 — трубку не сняли
                from_extension=row.get("from_extension", ""),
                from_number=row.get("from_number", ""),
                to_extension=row.get("to_extension", ""),
                to_number=row.get("to_number", ""),
                line_number=row.get("line_number", ""),
                entry_id=row.get("entry_id", ""),
                extra={k: v for k, v in row.items() if k in ("disconnect_reason", "location")},
            )
        )
    return calls


# ---------------------------------------------------------------- клиент API


class MangoClient:
    def __init__(
        self,
        api_key: str,
        api_salt: str,
        http: httpx.Client | None = None,
        base_url: str = API_URL,
        poll_interval: float = 5.0,
        poll_timeout: float = 300.0,
        request_gap: float = 2.1,
        retry_delays: tuple[float, ...] = (5, 15, 30),
    ):
        if not api_key or not api_salt:
            raise MangoError("Не указаны уникальный код АТС и ключ подписи")
        self.api_key = api_key
        self.api_salt = api_salt
        self.http = http or httpx.Client(timeout=httpx.Timeout(30, read=120), follow_redirects=True)
        self.base_url = base_url.rstrip("/") + "/"
        self.poll_interval = poll_interval  # Mango просит опрашивать stats/result не чаще раза в 5 с
        self.poll_timeout = poll_timeout
        self.request_gap = request_gap  # stats/request — не чаще раза в 2 с на АТС
        self.retry_delays = retry_delays  # паузы при перегрузке (HTTP 429/503, коды 5001–5003)
        self._last_stats_request = 0.0

    def sign(self, payload_json: str) -> str:
        return hashlib.sha256((self.api_key + payload_json + self.api_salt).encode("utf-8")).hexdigest()

    def _form(self, payload: dict) -> dict:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return {"vpbx_api_key": self.api_key, "sign": self.sign(body), "json": body}

    @staticmethod
    def _keys_error() -> MangoError:
        return MangoError("Mango отклонил ключи доступа — проверьте уникальный код АТС и ключ для создания подписи")

    @staticmethod
    def _overloaded(status: int, code: str) -> bool:
        return status in (429, 503) or code in ("5001", "5002", "5003")

    def _post(self, method: str, payload: dict, allow: tuple[int, ...] = ()) -> httpx.Response:
        data = self._form(payload)
        delays = list(self.retry_delays)
        while True:
            try:
                resp = self.http.post(self.base_url + method, data=data)
            except httpx.TransportError as e:
                raise MangoError(f"Mango недоступен: {e}") from e
            if resp.status_code < 400 or resp.status_code in allow:
                return resp
            code = _error_code(resp)
            if resp.status_code in (401, 403) or code in KEY_ERRORS:
                raise self._keys_error()
            if self._overloaded(resp.status_code, code) and delays:
                time.sleep(delays.pop(0))
                continue
            if self._overloaded(resp.status_code, code):
                raise MangoError("Mango перегружен или ограничил частоту запросов — повторим при следующей проверке")
            detail = _result_message(code) or resp.text[:200]
            raise MangoError(f"Mango вернул ошибку HTTP {resp.status_code}: {detail}")

    @staticmethod
    def _json(resp: httpx.Response) -> dict:
        try:
            data = resp.json()
        except ValueError as e:
            raise MangoError(f"Неожиданный ответ Mango: {resp.text[:200]}") from e
        return data if isinstance(data, dict) else {}

    def stats(self, date_from: int, date_to: int, fields: tuple[str, ...] = STATS_FIELDS) -> list[MangoCall]:
        """Звонки АТС за период (unix-время). Ждёт, пока Mango подготовит выгрузку."""
        wait = self._last_stats_request + self.request_gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_stats_request = time.monotonic()
        resp = self._post(
            "stats/request",
            {
                "date_from": str(int(date_from)),
                "date_to": str(int(date_to)),
                "fields": ",".join(fields),
                "request_id": uuid.uuid4().hex,
            },
        )
        data = self._json(resp)
        key = data.get("key")
        if not key:
            code = str(data.get("result") or data.get("code") or "")
            if code in KEY_ERRORS:
                raise self._keys_error()
            detail = _result_message(code) or data.get("message", "") or resp.text[:200]
            raise MangoError(f"Mango не принял запрос статистики: {detail}")
        deadline = time.monotonic() + self.poll_timeout
        while True:
            time.sleep(self.poll_interval)  # выгрузка готовится несколько секунд
            resp = self._post("stats/result", {"key": key}, allow=(404,))
            if resp.status_code == 404:
                raise MangoError("Mango не нашёл подготовленную выгрузку статистики — повторим при следующей проверке")
            if resp.status_code == 200 and resp.content:
                text = resp.content.decode("utf-8", errors="replace")
                if text.lstrip().startswith("{"):  # JSON — значит, ошибка, а не выгрузка
                    code = _error_code(resp)
                    if code in KEY_ERRORS:
                        raise self._keys_error()
                    raise MangoError(f"Mango вернул ошибку статистики: {_result_message(code) or text[:200]}")
                return parse_stats_csv(text, fields)
            if resp.status_code == 200:  # пустая выгрузка — звонков нет
                return []
            if time.monotonic() > deadline:  # 204 — ещё готовится
                raise MangoError("Mango слишком долго готовит выгрузку статистики")

    def download_recording(self, recording_id: str, target: Path) -> None:
        """Скачивает запись: Mango отвечает редиректом 302 на одноразовую ссылку с файлом mp3."""
        data = self._form({"recording_id": recording_id, "action": "download"})
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        delays = list(self.retry_delays)
        try:
            while True:
                with self.http.stream("POST", self.base_url + "queries/recording/post", data=data) as resp:
                    ctype = resp.headers.get("content-type", "")
                    if resp.status_code >= 400 or "json" in ctype or "text/html" in ctype:
                        resp.read()
                        code = _error_code(resp)
                        if resp.status_code in (401, 403) or code in KEY_ERRORS:
                            raise self._keys_error()
                        if self._overloaded(resp.status_code, code):
                            if delays:
                                time.sleep(delays.pop(0))
                                continue
                            raise MangoError("Mango перегружен или ограничил частоту запросов — повторим при следующей проверке")
                        detail = _result_message(code) or (f"HTTP {resp.status_code}" if resp.status_code >= 400 else resp.text[:200])
                        raise MangoRecordingError(f"Mango не отдал запись {recording_id}: {detail}")
                    with open(tmp, "wb") as f:
                        for chunk in resp.iter_bytes(1 << 16):
                            f.write(chunk)
                break
        except httpx.TransportError as e:
            tmp.unlink(missing_ok=True)
            raise MangoError(f"Обрыв при скачивании записи {recording_id}: {e}") from e
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise MangoRecordingError(f"Mango вернул пустую запись {recording_id}")
        tmp.replace(target)


# ---------------------------------------------------------------- загрузка в сервис


def _local_time(ts: int, utc_offset: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None) + timedelta(hours=utc_offset)


def _lines_filter(value: str) -> set[str]:
    return {x.strip() for x in (value or "").replace(";", ",").split(",") if x.strip()}


def _by_recording(calls: list[MangoCall]) -> dict[str, MangoCall]:
    """id записи → строка статистики. Одна запись может стоять в нескольких строках (плечах) звонка:
    берём ту, где ответили и разговор дольше, — по ней определяется сотрудник."""
    best: dict[str, MangoCall] = {}
    for call in calls:
        if call.direction is None:
            continue
        for recording_id in call.recording_ids:
            cur = best.get(recording_id)
            if cur is None or (call.answered, call.talk_seconds) > (cur.answered, cur.talk_seconds):
                best[recording_id] = call
    return best


class MangoScheduler:
    """Фоновый поток: проверяет клиники с включённой автозагрузкой и забирает их звонки по расписанию."""

    def __init__(self, settings: Settings, session_factory: sessionmaker, worker=None, http: httpx.Client | None = None):
        self.settings = settings
        self.session_factory = session_factory
        self.worker = worker
        self.http = http or httpx.Client(timeout=httpx.Timeout(30, read=120), follow_redirects=True)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._locks: dict[int, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # управление потоком
    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="mango-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def _lock(self, clinic_id: int) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(clinic_id, threading.Lock())

    def is_running(self, clinic_id: int) -> bool:
        return self._lock(clinic_id).locked()

    def client_for(self, clinic: db.Clinic) -> MangoClient:
        return MangoClient(clinic.mango_api_key, clinic.mango_api_salt, http=self.http)

    def due_clinics(self, now: datetime) -> list[int]:
        with self.session_factory() as s:
            clinics = s.scalars(
                select(db.Clinic).where(db.Clinic.active.is_(True), db.Clinic.mango_enabled.is_(True))
            ).all()
            return [
                c.id
                for c in clinics
                if c.mango_last_run is None or now - c.mango_last_run >= timedelta(minutes=c.mango_interval_min)
            ]

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                for clinic_id in self.due_clinics(datetime.now()):
                    if self._stop.is_set():
                        break
                    self.run_clinic(clinic_id)
            except Exception:
                log.exception("Ошибка планировщика Mango")
            self._wake.wait(60)
            self._wake.clear()

    # проверка подключения
    def test_connection(self, clinic_id: int) -> tuple[bool, str]:
        with self.session_factory() as s:
            clinic = s.get(db.Clinic, clinic_id)
            try:
                client = self.client_for(clinic)
                now = int(time.time())
                calls = client.stats(now - 3600, now)
            except MangoError as e:
                return False, str(e)
        with_records = sum(1 for c in calls if c.recording_ids)
        return True, f"Подключение работает: за последний час звонков — {len(calls)}, из них с записью — {with_records}."

    # загрузка
    def run_clinic(self, clinic_id: int, now: int | None = None) -> int:
        """Забирает новые звонки клиники. Возвращает число добавленных звонков."""
        lock = self._lock(clinic_id)
        if not lock.acquire(blocking=False):
            return 0
        try:
            return self._run_clinic(clinic_id, now or int(time.time()))
        finally:
            lock.release()

    def _run_clinic(self, clinic_id: int, now: int) -> int:
        with self.session_factory() as s:
            clinic = s.get(db.Clinic, clinic_id)
            if clinic is None:
                return 0
            date_to = now - SETTLE_SECONDS
            start_from = clinic.mango_synced_until or (now - FIRST_RUN_SECONDS)
            date_from = max(0, start_from - OVERLAP_SECONDS)
            added = 0
            skipped: list[str] = []
            try:
                client = self.client_for(clinic)
                cursor = date_from
                while cursor < date_to:
                    window_end = min(cursor + MAX_WINDOW_SECONDS, date_to)
                    added += self._import_window(s, clinic, client, client.stats(cursor, window_end), skipped)
                    clinic.mango_synced_until = window_end
                    s.commit()
                    cursor = window_end
                clinic.mango_last_error = (
                    f"Не удалось скачать записей: {len(skipped)} (попробуем ещё раз при следующих проверках). {skipped[-1]}"
                    if skipped
                    else ""
                )
            except MangoError as e:
                s.rollback()
                clinic = s.get(db.Clinic, clinic_id)
                clinic.mango_last_error = str(e)
                log.warning("Mango, клиника %s: %s", clinic_id, e)
            except Exception as e:  # неожиданная ошибка не должна останавливать планировщик
                s.rollback()
                clinic = s.get(db.Clinic, clinic_id)
                clinic.mango_last_error = f"{type(e).__name__}: {e}"
                log.exception("Mango, клиника %s", clinic_id)
            clinic.mango_last_run = datetime.now()
            clinic.mango_last_count = added
            s.commit()
        if added and self.worker:
            self.worker.wake()
        return added

    def _import_window(
        self, s, clinic: db.Clinic, client: MangoClient, calls: list[MangoCall], skipped: list[str] | None = None
    ) -> int:
        lines = _lines_filter(clinic.mango_lines)
        min_seconds = clinic.min_call_seconds if clinic.min_call_seconds is not None else self.settings.min_call_seconds
        employees = {
            e.mango_id: e.id
            for e in s.scalars(
                select(db.Employee).where(db.Employee.clinic_id == clinic.id, db.Employee.active.is_(True))
            )
            if e.mango_id
        }
        added = 0
        for recording_id, call in _by_recording(calls).items():
            if lines and not ({call.from_extension, call.to_extension, call.line_number, call.extension} & lines):
                continue
            if call.answered and call.talk_seconds < min_seconds:
                continue
            exists = s.scalars(
                select(db.Call.id).where(
                    db.Call.clinic_id == clinic.id, db.Call.source == "mango", db.Call.external_id == recording_id
                )
            ).first()
            if exists:
                continue
            started = _local_time(call.start, clinic.utc_offset)
            stored = f"{started:%Y%m}/{uuid.uuid4().hex}.mp3"
            try:
                client.download_recording(recording_id, self.settings.audio_dir / stored)
            except MangoRecordingError as e:  # запись удалена или ещё не сохранена — остальные звонки не ждут
                log.warning("Mango, клиника %s: %s", clinic.id, e)
                if skipped is not None:
                    skipped.append(str(e))
                continue
            s.add(
                db.Call(
                    clinic_id=clinic.id,
                    employee_id=employees.get(call.extension),
                    source="mango",
                    external_id=recording_id,
                    original_filename=f"mango_{started:%Y-%m-%d_%H-%M-%S}_{call.phone or 'unknown'}.mp3",
                    audio_path=stored,
                    direction=call.direction,
                    phone=call.phone or None,
                    mango_line=call.extension or call.line_number or None,
                    started_at=started,
                    duration_sec=float(call.talk_seconds) if call.talk_seconds else None,
                    status=db.QUEUED,
                )
            )
            s.commit()  # каждый звонок сохраняем сразу — сбой на следующем не потеряет уже скачанные
            added += 1
        return added
