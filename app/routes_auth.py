"""Вход, выход, первый запуск, переключение клиники, смена пароля."""

from typing import Annotated
from urllib.parse import urlparse

from fastapi import FastAPI, Form, Request
from sqlalchemy import func, select

from . import db
from .auth import COOKIE_NAME, hash_password, verify_password
from .web import Services, ctx_of, redirect

MIN_PASSWORD = 6


def _safe_next(url: str) -> str:
    """Переход после входа — только на страницы этого сервиса."""
    parsed = urlparse(url or "")
    if parsed.scheme or parsed.netloc or not (url or "").startswith("/") or url.startswith("//"):
        return "/calls"
    return url


def register(app: FastAPI, svc: Services) -> None:
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
        with svc.session_factory() as s:
            user = s.scalars(select(db.User).where(db.User.username == username.strip())).first()
            ok = user is not None and user.active and verify_password(password, user.password_hash)
            if not ok:
                return svc.render(request, "login.html", next=_safe_next(next), error="Неверный логин или пароль")
            response = redirect(_safe_next(next))
            svc.set_session(response, user.id)
            return response

    @app.post("/logout")
    def logout():
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
            svc.set_session(response, user.id)
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
        with svc.session_factory() as s:
            user = s.get(db.User, ctx.user_id)
            if not verify_password(old_password, user.password_hash):
                return redirect("/account", err="Текущий пароль указан неверно")
            if len(new_password) < MIN_PASSWORD:
                return redirect("/account", err=f"Новый пароль — не короче {MIN_PASSWORD} символов")
            if new_password != new_password2:
                return redirect("/account", err="Пароли не совпадают")
            user.password_hash = hash_password(new_password)
            s.commit()
        return redirect("/account", msg="Пароль изменён")


def re_call_card(path: str) -> bool:
    parts = path.strip("/").split("/")
    return len(parts) >= 2 and parts[0] == "calls" and parts[1].isdigit()
