"""Веб-интерфейс: сборка приложения, вход, выбор клиники, общие функции шаблонов."""

import logging
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker
from starlette.concurrency import run_in_threadpool

from . import db
from .auth import COOKIE_NAME, SESSION_MAX_AGE, SessionSigner, hash_password, load_or_create_secret
from .checklists import DIRECTION_LABELS, WEIGHT_LABELS, seed_clinic_checklist
from .config import Settings
from .llm import LLMClient, OpenRouterClient
from .mango_import import MangoScheduler
from .pipeline import Processor, Worker, requeue_interrupted
from .scoring import RESULT_LABELS
from .transcription import SPEAKER_LABELS, SherpaTranscriber, Transcriber

log = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
DIRECTIONS = {"in": "Входящий", "out": "Исходящий"}
PERIODS = {7: "7 дней", 30: "30 дней", 90: "90 дней", 0: "всё время"}
PUBLIC_PATHS = ("/login", "/setup")

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


def parse_dt(value: str) -> datetime | None:
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


def _fmt_weight(value: float) -> str:
    return f"{value:g}"


templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
templates.env.filters.update(
    dt=_fmt_dt, duration=_fmt_duration, score_class=_score_class, score=_fmt_score, weight=_fmt_weight
)
templates.env.globals.update(
    STATUS_LABELS=db.STATUS_LABELS,
    RESULT_LABELS=RESULT_LABELS,
    DIRECTIONS=DIRECTIONS,
    DIRECTION_LABELS=DIRECTION_LABELS,
    PERIODS=PERIODS,
    SPEAKER_LABELS=SPEAKER_LABELS,
    WEIGHT_LABELS=WEIGHT_LABELS,
    ROLE_LABELS=db.ROLE_LABELS,
)


# ---------------------------------------------------------------- контекст запроса


@dataclass
class Ctx:
    """Кто работает с сервисом и с какой клиникой."""

    user_id: int
    username: str
    role: str
    clinic_id: int
    clinic_name: str
    clinics: list[tuple[int, str]] = field(default_factory=list)  # доступные для переключения

    @property
    def is_admin(self) -> bool:
        return self.role == db.ADMIN


@dataclass
class Services:
    settings: Settings
    session_factory: sessionmaker
    processor: Processor
    worker: Worker | None
    scheduler: MangoScheduler | None
    signer: SessionSigner

    def wake_worker(self) -> None:
        if self.worker:
            self.worker.wake()

    def render(self, request: Request, name: str, **context) -> Response:
        context.setdefault("ctx", getattr(request.state, "ctx", None))
        context.setdefault("api_key_missing", not self.settings.openrouter_api_key)
        context.setdefault("flash", request.query_params.get("msg", ""))
        context.setdefault("flash_error", request.query_params.get("err", ""))
        return templates.TemplateResponse(request, name, context)

    def set_session(self, response: Response, user_id: int, clinic_id: int | None = None) -> None:
        response.set_cookie(
            COOKIE_NAME,
            self.signer.dumps({"uid": user_id, "clinic": clinic_id}),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
        )


def redirect(url: str, msg: str = "", err: str = "") -> RedirectResponse:
    if msg or err:
        sep = "&" if "?" in url else "?"
        url += sep + ("msg=" + quote(msg) if msg else "err=" + quote(err))
    return RedirectResponse(url, status_code=303)


def ctx_of(request: Request) -> Ctx:
    return request.state.ctx


def resolve_ctx(svc: Services, token: str | None) -> Ctx | None:
    data = svc.signer.loads(token)
    if not data:
        return None
    with svc.session_factory() as s:
        user = s.get(db.User, int(data.get("uid") or 0))
        if user is None or not user.active:
            return None
        if user.role == db.ADMIN:
            clinics = s.scalars(select(db.Clinic).where(db.Clinic.active.is_(True)).order_by(db.Clinic.name)).all()
        else:
            clinic = s.get(db.Clinic, user.clinic_id) if user.clinic_id else None
            clinics = [clinic] if clinic is not None and clinic.active else []
        if not clinics:
            return None
        wanted = data.get("clinic")
        current = next((c for c in clinics if c.id == wanted), clinics[0])
        return Ctx(
            user_id=user.id,
            username=user.username,
            role=user.role,
            clinic_id=current.id,
            clinic_name=current.name,
            clinics=[(c.id, c.name) for c in clinics],
        )


def users_exist(svc: Services) -> bool:
    with svc.session_factory() as s:
        return bool(s.scalar(select(func.count()).select_from(db.User)))


# ---------------------------------------------------------------- приложение


def create_app(
    settings: Settings | None = None,
    transcriber: Transcriber | None = None,
    llm: LLMClient | None = None,
    mango_http=None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    session_factory = db.make_session_factory(settings.db_url)

    # Первый запуск: первая клиника с чек-листом из шаблона; вход по APP_PASSWORD (старые установки)
    first_clinic_id = db.ensure_default_clinic(
        session_factory, settings.clinic_name, utc_offset=3 + settings.mango_time_shift_hours
    )
    with session_factory() as s:
        for clinic in s.scalars(select(db.Clinic)):
            seed_clinic_checklist(s, clinic, settings.checklists_path)
        if settings.app_password and not s.scalar(select(func.count()).select_from(db.User)):
            s.add(db.User(username="admin", password_hash=hash_password(settings.app_password), role=db.ADMIN))
        s.commit()

    llm = llm or OpenRouterClient(settings)
    processor = Processor(settings, session_factory, transcriber or SherpaTranscriber(settings), llm)
    worker = Worker(processor, settings.worker_poll_sec) if settings.worker_enabled else None
    scheduler = MangoScheduler(settings, session_factory, worker, http=mango_http) if settings.worker_enabled else None
    svc = Services(
        settings=settings,
        session_factory=session_factory,
        processor=processor,
        worker=worker,
        scheduler=scheduler,
        signer=SessionSigner(load_or_create_secret(settings.data_dir / "secret.key")),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        requeue_interrupted(session_factory)
        if worker:
            worker.start()
        if scheduler:
            scheduler.start()
        yield
        if scheduler:
            scheduler.stop()
        if worker:
            worker.stop()

    app = FastAPI(title="Оценка звонков", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
    app.state.svc = svc
    app.state.settings = settings
    app.state.session_factory = session_factory
    app.state.processor = processor
    app.state.first_clinic_id = first_clinic_id

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith("/static/"):
            return await call_next(request)
        ctx = await run_in_threadpool(resolve_ctx, svc, request.cookies.get(COOKIE_NAME))
        request.state.ctx = ctx
        if path in PUBLIC_PATHS:
            return await call_next(request)
        if ctx is None:
            if not await run_in_threadpool(users_exist, svc):
                return RedirectResponse("/setup", status_code=303)
            return RedirectResponse(f"/login?next={quote(path)}", status_code=303)
        return await call_next(request)

    from . import routes_admin, routes_auth, routes_calls, routes_settings

    routes_auth.register(app, svc)
    routes_calls.register(app, svc)
    routes_settings.register(app, svc)
    routes_admin.register(app, svc)
    return app
