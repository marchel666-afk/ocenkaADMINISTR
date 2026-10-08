"""Веб-интерфейс сервиса оценки звонков."""

import logging
import re
import secrets
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from . import db
from .checklists import GENERAL_KEY, NOT_TARGET, load_checklists
from .config import Settings
from .evaluation import Evaluation, Evaluator
from .llm import LLMClient, OpenRouterClient
from .mango import parse_mango_filename
from .pipeline import Processor, Worker, requeue_interrupted
from .reports import build_excel, criterion_stats, employee_stats, period_start
from .scoring import RESULT_LABELS
from .transcription import SPEAKER_LABELS, SherpaTranscriber, Transcriber, Transcript, transcript_from_text

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
TEXT_EXTENSIONS = {".txt"}
DIRECTIONS = {"in": "Входящий", "out": "Исходящий"}
PERIODS = {7: "7 дней", 30: "30 дней", 90: "90 дней", 0: "всё время"}

_DT_PATTERNS = (
    (re.compile(r"(\d{4})[-_.]?(\d{2})[-_.]?(\d{2})[ _T-]+(\d{2})[-_.:]?(\d{2})(?:[-_.:]?(\d{2}))?"), "ymd"),
    (re.compile(r"(\d{2})\.(\d{2})\.(\d{4})[ _T-]+(\d{2})[-_.:](\d{2})(?:[-_.:](\d{2}))?"), "dmy"),
)


def guess_datetime(filename: str) -> datetime | None:
    """Пытается достать дату и время звонка из имени файла записи."""
    for pattern, order in _DT_PATTERNS:
        m = pattern.search(filename)
        if not m:
            continue
        g = m.groups()
        y, mo, d = (g[0], g[1], g[2]) if order == "ymd" else (g[2], g[1], g[0])
        try:
            return datetime(int(y), int(mo), int(d), int(g[3]), int(g[4]), int(g[5] or 0))
        except ValueError:
            continue
    return None


def _decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_dt(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ---------------------------------------------------------------- шаблоны


def _fmt_dt(value: datetime | None) -> str:
    return value.strftime("%d.%m.%Y %H:%M") if value else "—"


def _fmt_duration(value: float | None) -> str:
    if not value:
        return "—"
    sec = int(round(value))
    return f"{sec // 60}:{sec % 60:02d}"


def _score_class(value: float | None) -> str:
    if value is None:
        return "score-none"
    if value >= 85:
        return "score-good"
    if value >= 70:
        return "score-mid"
    return "score-bad"


def _fmt_score(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.0f}%" if float(value).is_integer() else f"{value:.1f}%"


templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
templates.env.filters.update(dt=_fmt_dt, duration=_fmt_duration, score_class=_score_class, score=_fmt_score)
templates.env.globals.update(
    STATUS_LABELS=db.STATUS_LABELS,
    RESULT_LABELS=RESULT_LABELS,
    DIRECTIONS=DIRECTIONS,
    PERIODS=PERIODS,
    SPEAKER_LABELS=SPEAKER_LABELS,
)


# ---------------------------------------------------------------- приложение


def create_app(
    settings: Settings | None = None,
    transcriber: Transcriber | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    checklists = load_checklists(settings.checklists_path)
    session_factory = db.make_session_factory(settings.db_url)
    clinic_id = db.ensure_default_clinic(session_factory, settings.clinic_name)
    evaluator = Evaluator(checklists, llm or OpenRouterClient(settings))
    processor = Processor(
        settings, session_factory, checklists, transcriber or SherpaTranscriber(settings), evaluator
    )
    worker = Worker(processor, settings.worker_poll_sec) if settings.worker_enabled else None

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        requeue_interrupted(session_factory)
        if worker:
            worker.start()
        yield
        if worker:
            worker.stop()

    security = HTTPBasic(auto_error=False)

    def check_auth(credentials: Annotated[HTTPBasicCredentials | None, Depends(security)]) -> None:
        if not settings.app_password:
            return
        ok = credentials is not None and secrets.compare_digest(
            credentials.password.encode(), settings.app_password.encode()
        )
        if not ok:
            raise HTTPException(status_code=401, detail="Нужен пароль", headers={"WWW-Authenticate": "Basic"})

    app = FastAPI(title="Оценка звонков", lifespan=lifespan, dependencies=[Depends(check_auth)])
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
    app.state.settings = settings
    app.state.checklists = checklists
    app.state.session_factory = session_factory
    app.state.processor = processor
    app.state.worker = worker
    app.state.clinic_id = clinic_id

    def wake_worker() -> None:
        if worker:
            worker.wake()

    def render(request: Request, name: str, **ctx) -> Response:
        ctx.setdefault("clinic_name", settings.clinic_name)
        ctx.setdefault("api_key_missing", not settings.openrouter_api_key)
        return templates.TemplateResponse(request, name, ctx)

    def employees(s, active_only: bool = False):
        q = select(db.Employee).where(db.Employee.clinic_id == clinic_id).order_by(db.Employee.name)
        if active_only:
            q = q.where(db.Employee.active.is_(True))
        return s.scalars(q).all()

    def get_or_create_employee(s, employee_id: str, new_name: str) -> int | None:
        new_name = " ".join(new_name.split())
        if new_name:
            existing = s.scalars(
                select(db.Employee).where(db.Employee.clinic_id == clinic_id, db.Employee.name == new_name)
            ).first()
            if existing:
                return existing.id
            emp = db.Employee(clinic_id=clinic_id, name=new_name)
            s.add(emp)
            s.flush()
            return emp.id
        return int(employee_id) if employee_id.isdigit() else None

    # ------------------------------------------------------------ звонки

    @app.get("/")
    def index():
        return RedirectResponse("/calls", status_code=303)

    @app.get("/calls")
    def calls_list(request: Request, employee: str = "", status: str = "", days: int = 30, page: int = 1):
        per_page = 50
        with session_factory() as s:
            q = select(db.Call).where(db.Call.clinic_id == clinic_id).options(selectinload(db.Call.employee))
            since = period_start(days)
            if since:
                q = q.where(func.coalesce(db.Call.started_at, db.Call.created_at) >= since)
            if employee == "none":
                q = q.where(db.Call.employee_id.is_(None))
            elif employee.isdigit():
                q = q.where(db.Call.employee_id == int(employee))
            if status in db.STATUS_LABELS:
                q = q.where(db.Call.status == status)
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            calls = s.scalars(
                q.order_by(func.coalesce(db.Call.started_at, db.Call.created_at).desc(), db.Call.id.desc())
                .offset((max(page, 1) - 1) * per_page)
                .limit(per_page)
            ).all()
            pending = s.scalar(
                select(func.count()).where(
                    db.Call.clinic_id == clinic_id, db.Call.status.in_((db.QUEUED, *db.IN_PROGRESS))
                )
            )
            return render(
                request,
                "calls.html",
                calls=calls,
                employees=employees(s),
                active_employees=employees(s, active_only=True),
                scenarios=checklists.scenarios,
                scenario_title=checklists.scenario_title,
                f_employee=employee,
                f_status=status,
                f_days=days,
                page=max(page, 1),
                pages=max(1, -(-total // per_page)),
                total=total,
                pending=pending,
            )

    @app.post("/calls/upload")
    def upload(
        files: Annotated[list[UploadFile], File()],
        employee_id: Annotated[str, Form()] = "",
        new_employee: Annotated[str, Form()] = "",
        direction: Annotated[str, Form()] = "",
        call_type: Annotated[str, Form()] = "",
        started_at: Annotated[str, Form()] = "",
    ):
        manual_dt = _parse_dt(started_at)
        if call_type and call_type not in checklists.scenarios:
            call_type = ""
        if direction not in DIRECTIONS:
            direction = ""
        if call_type:
            direction = checklists.scenarios[call_type].direction or direction

        with session_factory() as s:
            emp_id = get_or_create_employee(s, employee_id, new_employee)
            for upload_file in files:
                if not upload_file.filename:
                    continue
                name = Path(upload_file.filename).name
                ext = Path(name).suffix.lower()
                mango = parse_mango_filename(name, settings.mango_time_shift_hours)
                call = db.Call(
                    clinic_id=clinic_id,
                    employee_id=emp_id,
                    original_filename=name,
                    direction=direction or (mango.direction if mango else None),
                    call_type=call_type or None,
                    call_type_manual=bool(call_type),
                    started_at=manual_dt or (mango.started_at if mango else guess_datetime(name)),
                    phone=mango.phone if mango else None,
                    mango_line=mango.line if mango else None,
                    status=db.QUEUED,
                )
                if ext in TEXT_EXTENSIONS:
                    transcript = transcript_from_text(_decode_text(upload_file.file.read()))
                    call.source = "text"
                    call.transcript_json = transcript.to_json()
                else:
                    stored = f"{datetime.now():%Y%m}/{uuid.uuid4().hex}{ext or '.audio'}"
                    target = settings.audio_dir / stored
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with open(target, "wb") as out:
                        shutil.copyfileobj(upload_file.file, out)
                    call.source = "upload"
                    call.audio_path = stored
                s.add(call)
            s.commit()
        wake_worker()
        return RedirectResponse("/calls", status_code=303)

    def load_call(s, call_id: int) -> db.Call:
        call = s.get(db.Call, call_id, options=[selectinload(db.Call.results), selectinload(db.Call.employee)])
        if call is None or call.clinic_id != clinic_id:
            raise HTTPException(status_code=404, detail="Звонок не найден")
        return call

    @app.get("/calls/{call_id}")
    def call_card(request: Request, call_id: int):
        with session_factory() as s:
            call = load_call(s, call_id)
            evaluation = Evaluation.from_json(call.evaluation_json) if call.evaluation_json else None
            transcript = Transcript.from_json(call.transcript_json) if call.transcript_json else None
            sections = []
            if evaluation and call.call_type in checklists.scenarios:
                scenario = checklists.scenario(call.call_type)
                for key, checklist, score in (
                    (GENERAL_KEY, checklists.general, call.score_general),
                    (scenario.key, scenario, call.score_scenario),
                ):
                    sections.append(
                        {"key": key, "title": checklist.title, "score": score, "criteria": checklist.criteria}
                    )
            return render(
                request,
                "call.html",
                call=call,
                evaluation=evaluation,
                transcript=transcript,
                sections=sections,
                weight_labels=checklists.weight_labels,
                scenario_title=checklists.scenario_title,
                scenarios=checklists.scenarios,
                employees=employees(s, active_only=True),
                NOT_TARGET=NOT_TARGET,
            )

    @app.get("/calls/{call_id}/audio")
    def call_audio(call_id: int):
        with session_factory() as s:
            call = load_call(s, call_id)
            if not call.audio_path:
                raise HTTPException(status_code=404, detail="Записи нет")
            path = (settings.audio_dir / call.audio_path).resolve()
            if not path.is_file() or settings.audio_dir.resolve() not in path.parents:
                raise HTTPException(status_code=404, detail="Файл записи не найден")
            return FileResponse(path, filename=call.original_filename or path.name)

    def requeue(s, call: db.Call, retranscribe: bool = False) -> None:
        if retranscribe and call.audio_path:
            call.transcript_json = None
        if not call.call_type_manual:
            call.call_type = None
        call.status = db.QUEUED
        call.status_message = ""
        s.commit()

    @app.post("/calls/{call_id}/update")
    def call_update(
        call_id: int,
        employee_id: Annotated[str, Form()] = "",
        new_employee: Annotated[str, Form()] = "",
        call_type: Annotated[str, Form()] = "",
        started_at: Annotated[str, Form()] = "",
        reevaluate: Annotated[str, Form()] = "",
    ):
        with session_factory() as s:
            call = load_call(s, call_id)
            call.employee_id = get_or_create_employee(s, employee_id, new_employee)
            type_changed = False
            if call_type in checklists.scenarios:
                type_changed = call.call_type != call_type or not call.call_type_manual
                call.call_type, call.call_type_manual = call_type, True
                call.direction = checklists.scenarios[call_type].direction or call.direction
            elif call_type == "auto" and call.call_type_manual:
                call.call_type_manual, type_changed = False, True
            dt = _parse_dt(started_at)
            if dt:
                call.started_at = dt
            if reevaluate or type_changed:
                requeue(s, call)
                wake_worker()
            else:
                s.commit()
        return RedirectResponse(f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/reevaluate")
    def call_reevaluate(call_id: int):
        with session_factory() as s:
            requeue(s, load_call(s, call_id))
        wake_worker()
        return RedirectResponse(f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/retranscribe")
    def call_retranscribe(call_id: int):
        with session_factory() as s:
            requeue(s, load_call(s, call_id), retranscribe=True)
        wake_worker()
        return RedirectResponse(f"/calls/{call_id}", status_code=303)

    @app.post("/calls/{call_id}/delete")
    def call_delete(call_id: int):
        with session_factory() as s:
            call = load_call(s, call_id)
            if call.audio_path:
                (settings.audio_dir / call.audio_path).unlink(missing_ok=True)
            s.delete(call)
            s.commit()
        return RedirectResponse("/calls", status_code=303)

    # ------------------------------------------------------------ сводка и выгрузка

    @app.get("/stats")
    def stats(request: Request, days: int = 30, employee: str = ""):
        emp_id = int(employee) if employee.isdigit() else None
        with session_factory() as s:
            since = period_start(days)
            rows = employee_stats(s, clinic_id, since)
            crit = [c for c in criterion_stats(s, checklists, clinic_id, since, emp_id) if c.failed][:15]
            return render(
                request,
                "stats.html",
                rows=rows,
                criteria=crit,
                employees=employees(s),
                f_days=days,
                f_employee=employee,
            )

    @app.get("/export.xlsx")
    def export(days: int = 30):
        with session_factory() as s:
            content = build_excel(s, checklists, clinic_id, period_start(days))
        filename = f"ocenka_zvonkov_{datetime.now():%Y-%m-%d}.xlsx"
        return Response(
            content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ------------------------------------------------------------ сотрудники

    @app.get("/employees")
    def employees_page(request: Request):
        with session_factory() as s:
            counts = dict(
                s.execute(
                    select(db.Call.employee_id, func.count())
                    .where(db.Call.clinic_id == clinic_id)
                    .group_by(db.Call.employee_id)
                ).all()
            )
            return render(request, "employees.html", employees=employees(s), counts=counts)

    @app.post("/employees")
    def employees_add(name: Annotated[str, Form()]):
        with session_factory() as s:
            get_or_create_employee(s, "", name)
            s.commit()
        return RedirectResponse("/employees", status_code=303)

    @app.post("/employees/{employee_id}/toggle")
    def employees_toggle(employee_id: int):
        with session_factory() as s:
            emp = s.get(db.Employee, employee_id)
            if emp is None or emp.clinic_id != clinic_id:
                raise HTTPException(status_code=404)
            emp.active = not emp.active
            s.commit()
        return RedirectResponse("/employees", status_code=303)

    @app.post("/employees/{employee_id}/rename")
    def employees_rename(employee_id: int, name: Annotated[str, Form()]):
        name = " ".join(name.split())
        with session_factory() as s:
            emp = s.get(db.Employee, employee_id)
            if emp is None or emp.clinic_id != clinic_id:
                raise HTTPException(status_code=404)
            if name:
                emp.name = name
                s.commit()
        return RedirectResponse("/employees", status_code=303)

    return app
