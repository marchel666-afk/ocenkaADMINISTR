"""Веб-интерфейс: сборка приложения, вход, выбор клиники, общие функции шаблонов."""

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote, urlparse

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

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
    pinned: bool = True  # клиника записана в cookie (иначе выбрана по умолчанию и её надо закрепить)

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
        kind, text = read_flash(request)
        context.setdefault("flash", text if kind == "msg" else "")
        context.setdefault("flash_error", text if kind == "err" else "")
        response = templates.TemplateResponse(request, name, context)
        if FLASH_COOKIE in request.cookies:  # сообщение показывается один раз
            response.delete_cookie(FLASH_COOKIE)
        return response

    def set_session(self, response: Response, user_id: int, clinic_id: int | None = None) -> None:
        with self.session_factory() as s:
            user = s.get(db.User, user_id)
            version = (user.session_version or 0) if user else 0
        response.set_cookie(
            COOKIE_NAME,
            self.signer.dumps({"uid": user_id, "clinic": clinic_id, "sv": version}),
            max_age=SESSION_MAX_AGE,
            httponly=True,
            samesite="lax",
        )


def to_int(value) -> int | None:
    """Целое из формы или адреса: только цифры ASCII (str.isdigit пропускает «²»), без огромных чисел."""
    value = str(value or "").strip()
    return int(value) if re.fullmatch(r"\d{1,9}", value, re.ASCII) else None


# Сообщение после действия («Сохранено», ошибка) передаётся в одноразовой подписанной cookie, а не в адресе:
# ссылкой его нельзя переслать другому человеку и показать чужой текст в оформлении сервиса.
# Ключ живёт до перезапуска — сообщениям этого хватает.
FLASH_COOKIE = "ocenka_flash"
_FLASH_KEY = secrets.token_bytes(32)


def _flash_sig(payload: str) -> str:
    return hmac.new(_FLASH_KEY, payload.encode(), hashlib.sha256).hexdigest()[:24]


def read_flash(request: Request) -> tuple[str, str]:
    """(вид, текст) сообщения из cookie или ("", ""), если его нет или подпись не сходится."""
    payload, _, sig = request.cookies.get(FLASH_COOKIE, "").rpartition(".")
    if not payload or not re.fullmatch(r"[0-9a-f]{24}", sig):
        return "", ""
    if not hmac.compare_digest(sig.encode(), _flash_sig(payload).encode()):
        return "", ""
    try:
        kind, text = json.loads(base64.urlsafe_b64decode(payload.encode()).decode("utf-8"))
    except (ValueError, TypeError):
        return "", ""
    return (kind, str(text)) if kind in ("msg", "err") else ("", "")


def redirect(url: str, msg: str = "", err: str = "") -> RedirectResponse:
    response = RedirectResponse(url, status_code=303)
    if msg or err:
        kind, text = ("msg", msg) if msg else ("err", err)
        payload = base64.urlsafe_b64encode(json.dumps([kind, text[:800]], ensure_ascii=False).encode()).decode()
        response.set_cookie(
            FLASH_COOKIE, f"{payload}.{_flash_sig(payload)}", max_age=120, httponly=True, samesite="lax"
        )
    return response


def ctx_of(request: Request) -> Ctx:
    return request.state.ctx


def resolve_ctx(svc: Services, token: str | None) -> Ctx | None:
    data = svc.signer.loads(token)
    if not data:
        return None
    with svc.session_factory() as s:
        user = s.get(db.User, to_int(data.get("uid")) or 0)
        if user is None or not user.active or (to_int(data.get("sv")) or 0) != (user.session_version or 0):
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
            pinned=current.id == wanted,
        )


def default_clinic_id(s, user: db.User) -> int | None:
    """Клиника, которая открывается при входе: у руководителя — своя, у администратора сети — первая работающая."""
    if user.role == db.ADMIN:
        return s.scalar(select(db.Clinic.id).where(db.Clinic.active.is_(True)).order_by(db.Clinic.name).limit(1))
    return user.clinic_id


UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
# действия, не относящиеся к выбранной клинике: их не проверяем на «вкладку от другой клиники»
CLINIC_FREE_PREFIXES = ("/switch-clinic", "/logout", "/login", "/setup", "/account", "/admin/")


SESSION_ENDING_PATHS = ("/logout", "/account/logout-all")


def _sets_session(response: Response) -> bool:
    return any(h.startswith(COOKIE_NAME + "=") for h in response.headers.getlist("set-cookie"))


def same_origin(request: Request) -> bool:
    """Защита от подделки запросов с чужих сайтов: браузер всегда сообщает, с какой страницы отправлена форма."""
    source = request.headers.get("origin") or request.headers.get("referer")
    if not source:
        return True  # не браузер: скрипты и тесты
    hosts = {request.headers.get("host", ""), request.headers.get("x-forwarded-host", "")} - {""}
    return source != "null" and urlparse(source).netloc in hosts


def back_url(request: Request) -> str:
    """Страница, с которой пришла форма (только путь этого сервиса)."""
    path = urlparse(request.headers.get("referer", "")).path
    return path if path.startswith("/") and not path.startswith("//") else "/calls"


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
        if request.method in UNSAFE_METHODS and not same_origin(request):
            return Response("Запрос отправлен с другого сайта и отклонён", status_code=403)
        ctx = await run_in_threadpool(resolve_ctx, svc, request.cookies.get(COOKIE_NAME))
        request.state.ctx = ctx
        if path in PUBLIC_PATHS:
            return await call_next(request)
        if ctx is None:
            if not await run_in_threadpool(users_exist, svc):
                return RedirectResponse("/setup", status_code=303)
            target = path + (f"?{request.url.query}" if request.url.query and request.method == "GET" else "")
            return RedirectResponse(f"/login?next={quote(target)}", status_code=303)
        # форма из вкладки, открытой до переключения клиники в другой вкладке, не должна писать в чужую клинику
        stamp = request.query_params.get("_clinic")
        if (
            request.method in UNSAFE_METHODS
            and stamp is not None
            and not path.startswith(CLINIC_FREE_PREFIXES)
            and to_int(stamp) != ctx.clinic_id
        ):
            return redirect(
                back_url(request),
                err=f"Страница была открыта для другой клиники, а сейчас выбрана «{ctx.clinic_name}». "
                "Проверьте, что всё верно, и повторите действие.",
            )
        response = await call_next(request)
        # клиника, выбранная по умолчанию, закрепляется в cookie: иначе после переименования или добавления
        # клиник «первая» клиника сменится сама, и работа незаметно продолжится в другой клинике
        if not ctx.pinned and path not in SESSION_ENDING_PATHS and not _sets_session(response):
            await run_in_threadpool(svc.set_session, response, ctx.user_id, ctx.clinic_id)
        return response

    ERROR_TITLES = {403: "Нет доступа", 404: "Не найдено", 405: "Действие недоступно", 409: "Конфликт", 422: "Неверные данные"}
    DEFAULT_DETAILS = {"Not Found", "Forbidden", "Method Not Allowed"}

    def error_response(request: Request, status: int, detail: str = ""):
        if not detail or detail in DEFAULT_DETAILS:
            detail = {
                403: "Этот раздел вам недоступен.",
                404: "Такой страницы или записи нет — возможно, её удалили.",
                409: "Эти данные одновременно изменили в другом окне. Обновите страницу и повторите действие.",
                422: "В запросе неверные данные. Обновите страницу и попробуйте ещё раз.",
            }.get(status, "Не удалось выполнить действие.")
        if request.method in UNSAFE_METHODS and status in (404, 409, 422) and getattr(request.state, "ctx", None):
            return redirect(back_url(request), err=detail)  # например, повторное «Удалить» по уже удалённой записи
        if "text/html" in request.headers.get("accept", ""):
            page = svc.render(request, "error.html", title=ERROR_TITLES.get(status, "Ошибка"), message=detail)
            page.status_code = status
            return page
        return JSONResponse({"detail": detail}, status_code=status)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        return error_response(request, exc.status_code, str(exc.detail or ""))

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        return error_response(request, 422)

    @app.exception_handler(IntegrityError)  # например, двойное нажатие «Добавить» в двух окнах сразу
    async def integrity_error(request: Request, exc: IntegrityError):
        log.warning("Конфликт записи: %s", exc)
        return error_response(request, 409)

    @app.exception_handler(OverflowError)  # огромный номер в адресе (/calls/99999999999999999999)
    async def overflow_error(request: Request, exc: OverflowError):
        return error_response(request, 404)

    from . import routes_admin, routes_auth, routes_calls, routes_settings

    routes_auth.register(app, svc)
    routes_calls.register(app, svc)
    routes_settings.register(app, svc)
    routes_admin.register(app, svc)
    return app
