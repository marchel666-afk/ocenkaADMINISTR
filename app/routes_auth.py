"""Вход, выход, первый запуск, переключение клиники, смена пароля."""

import threading
import time
from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, Form, Request
from sqlalchemy import func, select

from . import db
from .auth import COOKIE_NAME, hash_password, verify_password
from .web import Services, ctx_of, default_clinic_id, redirect

MIN_PASSWORD = 6
MAX_FAILURES = 5  # неудачных входов подряд на один логин с одного адреса
MAX_FAILURES_PER_IP = 30
LOCK_SECONDS = 10 * 60
_DUMMY_HASH = hash_password("timing-equalizer")  # чтобы ответ для несуществующего логина шёл столько же времени


class LoginThrottle:
    """Ограничение подбора пароля: после серии неудачных попыток проверка пароля временно закрыта.

    Попытка учитывается как неудачная сразу, до проверки пароля, и снимается при успехе — так
    одновременные запросы не проскочат лимит, пока идёт медленная проверка пароля.
    """

    def __init__(self) -> None:
        self._failures: dict[tuple, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: tuple, now: float) -> list[float]:
        items = [t for t in self._failures.get(key, []) if now - t < LOCK_SECONDS]
        if items:
            self._failures[key] = items
        else:
            self._failures.pop(key, None)
        return items

    def try_begin(self, limits: list[tuple[tuple, int]]) -> int:
        """Резервирует попытку по всем ключам. Возвращает, сколько секунд ещё ждать (0 — можно пробовать)."""
        now = time.monotonic()
        with self._lock:
            for key, limit in limits:
                items = self._recent(key, now)
                if len(items) >= limit:
                    return int(LOCK_SECONDS - (now - items[-limit])) + 1
            for key, _ in limits:
                self._failures.setdefault(key, []).append(now)
        return 0

    def succeeded(self, reset: tuple, release: tuple | None = None) -> None:
        """Успех: счётчик reset обнуляется, у общего счётчика release снимается резерв этой попытки."""
        with self._lock:
            self._failures.pop(reset, None)
            items = self._failures.get(release) if release else None
            if items:
                items.pop()


def login_limits(ip: str, username: str) -> list[tuple[tuple, int]]:
    return [((ip, username), MAX_FAILURES), ((ip,), MAX_FAILURES_PER_IP)]


def _wait_text(wait: int) -> str:
    return f"Слишком много неудачных попыток. Попробуйте через {wait // 60 + 1} мин."


def _safe_next(url: str) -> str:
    """Переход после входа — только на страницы этого сервиса."""
    parsed = urlparse(url or "")
    if parsed.scheme or parsed.netloc or not (url or "").startswith("/") or url.startswith("//"):
        return "/calls"
    return url


def register(app: FastAPI, svc: Services) -> None:
    throttle = LoginThrottle()
    app.state.login_throttle = throttle

    @app.get("/login")
    def login_page(request: Request, next: str = "/calls"):
        if request.state.ctx is not None:
            return redirect(_safe_next(next))
        return svc.render(request, "login.html", next=_safe_next(next))

    @app.post("/login")
    def login(
        request: Request,
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        next: Annotated[str, Form()] = "/calls",
    ):
        ip = request.client.host if request.client else ""
        login_name = username.strip().casefold()
        wait = throttle.try_begin(login_limits(ip, login_name))
        if wait:
            return svc.render(request, "login.html", next=_safe_next(next), error=_wait_text(wait))
        with svc.session_factory() as s:
            user = s.scalars(select(db.User).where(db.User.username == username.strip())).first()
            ok = verify_password(password, user.password_hash if user else _DUMMY_HASH)
            if not (ok and user is not None and user.active):
                return svc.render(request, "login.html", next=_safe_next(next), error="Неверный логин или пароль")
            throttle.succeeded((ip, login_name), release=(ip,))
            if user.role != db.ADMIN and (user.clinic is None or not user.clinic.active):
                return svc.render(
                    request,
                    "login.html",
                    next=_safe_next(next),
                    error="Клиника, к которой привязана ваша учётная запись, отключена. Обратитесь к администратору сети.",
                )
            response = redirect(_safe_next(next))
            svc.set_session(response, user.id, default_clinic_id(s, user))
            return response

    @app.post("/logout")
    def logout():
        response = redirect("/login")
        response.delete_cookie(COOKIE_NAME)
        return response

    @app.post("/account/logout-all")
    def logout_all(request: Request):
        ctx = ctx_of(request)
        with svc.session_factory() as s:
            user = s.get(db.User, ctx.user_id)
            user.session_version = (user.session_version or 0) + 1
            s.commit()
        response = redirect("/login")
        response.delete_cookie(COOKIE_NAME)
        return response

    @app.get("/setup")
    def setup_page(request: Request):
        with svc.session_factory() as s:
            if s.scalar(select(func.count()).select_from(db.User)):
                return redirect("/login")
        return svc.render(request, "setup.html")

    @app.post("/setup")
    def setup(
        request: Request,
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        password2: Annotated[str, Form()],
    ):
        username = username.strip()
        with svc.session_factory() as s:
            if s.scalar(select(func.count()).select_from(db.User)):
                return redirect("/login")
            error = ""
            if not username:
                error = "Укажите логин"
            elif len(password) < MIN_PASSWORD:
                error = f"Пароль — не короче {MIN_PASSWORD} символов"
            elif password != password2:
                error = "Пароли не совпадают"
            if error:
                return svc.render(request, "setup.html", error=error, username=username)
            user = db.User(username=username, password_hash=hash_password(password), role=db.ADMIN)
            s.add(user)
            s.commit()
            response = redirect("/calls", msg="Администратор создан. Добро пожаловать!")
            svc.set_session(response, user.id, default_clinic_id(s, user))
            return response

    @app.post("/switch-clinic")
    def switch_clinic(request: Request, clinic_id: Annotated[int, Form()], back: Annotated[str, Form()] = "/calls"):
        ctx = ctx_of(request)
        if clinic_id not in {cid for cid, _ in ctx.clinics}:
            return redirect("/calls", err="Нет доступа к этой клинике")
        target = _safe_next(back)
        if re_call_card(target):  # карточка звонка другой клиники не откроется — возвращаемся к списку
            target = "/calls"
        response = redirect(target)
        svc.set_session(response, ctx.user_id, clinic_id)
        return response

    @app.get("/account")
    def account_page(request: Request):
        return svc.render(request, "account.html")

    @app.post("/account/password")
    def account_password(
        request: Request,
        old_password: Annotated[str, Form()],
        new_password: Annotated[str, Form()],
        new_password2: Annotated[str, Form()],
    ):
        ctx = ctx_of(request)
        # проверку текущего пароля ограничиваем так же, как вход: иначе по чужой открытой сессии
        # пароль можно подбирать без ограничений
        limits = [(("account", ctx.user_id), MAX_FAILURES)]
        wait = throttle.try_begin(limits)
        if wait:
            return redirect("/account", err=_wait_text(wait))
        with svc.session_factory() as s:
            user = s.get(db.User, ctx.user_id)
            if not verify_password(old_password, user.password_hash):
                return redirect("/account", err="Текущий пароль указан неверно")
            throttle.succeeded(("account", ctx.user_id))
            if len(new_password) < MIN_PASSWORD:
                return redirect("/account", err=f"Новый пароль — не короче {MIN_PASSWORD} символов")
            if new_password != new_password2:
                return redirect("/account", err="Пароли не совпадают")
            user.password_hash = hash_password(new_password)
            user.session_version = (user.session_version or 0) + 1  # входы на других устройствах закрываются
            s.commit()
        response = redirect("/account", msg="Пароль изменён. На других устройствах нужно войти заново.")
        svc.set_session(response, ctx.user_id, ctx.clinic_id)
        return response


def re_call_card(path: str) -> bool:
    parts = path.strip("/").split("/")
    return len(parts) >= 2 and parts[0] == "calls" and parts[1].isascii() and parts[1].isdigit()
