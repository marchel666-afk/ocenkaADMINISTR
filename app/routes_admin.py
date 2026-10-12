"""Администрирование сети: клиники и пользователи (только для администратора сети)."""

from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Request
from sqlalchemy import func, select

from . import db
from .auth import hash_password
from .checklists import copy_checklist, import_checklist_data, read_template
from .routes_auth import MIN_PASSWORD
from .web import Services, ctx_of, default_clinic_id, redirect, to_int


def register(app: FastAPI, svc: Services) -> None:
    sf = svc.session_factory

    def require_admin(request: Request):
        ctx = ctx_of(request)
        if not ctx.is_admin:
            raise HTTPException(status_code=403, detail="Раздел доступен только администратору сети")
        return ctx

    # ------------------------------------------------------------ клиники

    @app.get("/admin/clinics")
    def clinics_page(request: Request):
        require_admin(request)
        with sf() as s:
            clinics = s.scalars(select(db.Clinic).order_by(db.Clinic.active.desc(), db.Clinic.name)).all()
            calls = dict(s.execute(select(db.Call.clinic_id, func.count()).group_by(db.Call.clinic_id)).all())
            users = dict(
                s.execute(
                    select(db.User.clinic_id, func.count()).where(db.User.active.is_(True)).group_by(db.User.clinic_id)
                ).all()
            )
            return svc.render(request, "admin_clinics.html", clinics=clinics, calls=calls, users=users)

    @app.post("/admin/clinics")
    def clinic_add(
        request: Request,
        name: Annotated[str, Form()],
        utc_offset: Annotated[int, Form()] = 3,
        copy_from: Annotated[str, Form()] = "template",
    ):
        ctx = require_admin(request)
        name = " ".join(name.split())
        if not name:
            return redirect("/admin/clinics", err="Укажите название клиники")
        with sf() as s:
            clinic = db.Clinic(name=name, utc_offset=utc_offset)
            s.add(clinic)
            s.flush()
            source = s.get(db.Clinic, to_int(copy_from)) if to_int(copy_from) is not None else None
            try:
                if source is not None:
                    copy_checklist(s, source, clinic)  # сведения о клинике для ИИ не копируются — у каждой свои
                else:
                    import_checklist_data(s, clinic, read_template(svc.settings.checklists_path))
            except ValueError as e:
                s.rollback()
                return redirect("/admin/clinics", err=f"Не удалось скопировать критерии: {e}")
            s.commit()
            response = redirect("/settings/clinic", msg=f"Клиника «{name}» создана. Заполните сведения о ней для ИИ.")
            svc.set_session(response, ctx.user_id, clinic.id, sv=ctx.session_version)
            return response

    @app.post("/admin/clinics/{clinic_id}/toggle")
    def clinic_toggle(request: Request, clinic_id: int):
        ctx = require_admin(request)
        with sf() as s:
            clinic = s.get(db.Clinic, clinic_id)
            if clinic is None:
                raise HTTPException(status_code=404)
            if clinic.active and s.scalar(select(func.count()).where(db.Clinic.active.is_(True))) <= 1:
                return redirect("/admin/clinics", err="Нельзя отключить единственную клинику")
            clinic.active = not clinic.active
            if not clinic.active:
                clinic.mango_enabled = False
            s.commit()
            state = "включена" if clinic.active else "отключена"
        response = redirect("/admin/clinics", msg=f"Клиника {state}")
        if clinic_id == ctx.clinic_id and state == "отключена":
            with sf() as s:
                svc.set_session(
                    response, ctx.user_id, default_clinic_id(s, s.get(db.User, ctx.user_id)), sv=ctx.session_version
                )
        return response

    # ------------------------------------------------------------ пользователи

    @app.get("/admin/users")
    def users_page(request: Request):
        require_admin(request)
        with sf() as s:
            users = s.scalars(select(db.User).order_by(db.User.role, db.User.username)).all()
            clinics = s.scalars(select(db.Clinic).order_by(db.Clinic.name)).all()
            return svc.render(request, "admin_users.html", users=users, clinics=clinics, min_password=MIN_PASSWORD)

    def _role_and_clinic(s, role: str, clinic_id: str, current: int | None = None) -> tuple[str, int | None, str]:
        if role not in db.ROLE_LABELS:
            return "", None, "Неизвестная роль"
        if role == db.MANAGER:
            clinic = s.get(db.Clinic, to_int(clinic_id)) if to_int(clinic_id) is not None else None
            if clinic is None:
                return "", None, "Руководителю клиники нужно выбрать клинику"
            if not clinic.active and clinic.id != current:
                return "", None, f"Клиника «{clinic.name}» отключена — выберите работающую"
            return role, clinic.id, ""
        return role, None, ""

    @app.post("/admin/users")
    def user_add(
        request: Request,
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        role: Annotated[str, Form()] = db.MANAGER,
        clinic_id: Annotated[str, Form()] = "",
    ):
        require_admin(request)
        username = username.strip()
        if not username:
            return redirect("/admin/users", err="Укажите логин")
        if len(password) < MIN_PASSWORD:
            return redirect("/admin/users", err=f"Пароль — не короче {MIN_PASSWORD} символов")
        with sf() as s:
            if s.scalars(select(db.User).where(db.User.username == username)).first():
                return redirect("/admin/users", err="Такой логин уже есть")
            role, cid, error = _role_and_clinic(s, role, clinic_id)
            if error:
                return redirect("/admin/users", err=error)
            s.add(db.User(username=username, password_hash=hash_password(password), role=role, clinic_id=cid))
            s.commit()
        return redirect("/admin/users", msg=f"Пользователь {username} создан")

    @app.post("/admin/users/{user_id}")
    def user_update(
        request: Request,
        user_id: int,
        role: Annotated[str, Form()],
        clinic_id: Annotated[str, Form()] = "",
        password: Annotated[str, Form()] = "",
        active: Annotated[str, Form()] = "",
    ):
        ctx = require_admin(request)
        with sf() as s:
            user = s.get(db.User, user_id)
            if user is None:
                raise HTTPException(status_code=404)
            role, cid, error = _role_and_clinic(s, role, clinic_id, current=user.clinic_id)
            if error:
                return redirect("/admin/users", err=error)
            if user.id == ctx.user_id and (role != db.ADMIN or not active):
                return redirect("/admin/users", err="Нельзя лишить прав или отключить самого себя")
            changed = (user.role, user.clinic_id, user.active) != (role, cid, bool(active))
            if password:
                if len(password) < MIN_PASSWORD:
                    return redirect("/admin/users", err=f"Пароль — не короче {MIN_PASSWORD} символов")
                user.password_hash = hash_password(password)
                changed = True
            user.role, user.clinic_id, user.active = role, cid, bool(active)
            is_self = user.id == ctx.user_id
            # пользователю придётся войти заново; себе — только при смене пароля (другие входы закрываются)
            if changed and (not is_self or password):
                user.session_version = (user.session_version or 0) + 1
            new_sv = user.session_version or 0
            s.commit()
            response = redirect("/admin/users", msg=f"Пользователь {user.username} сохранён")
        if is_self and password:  # текущий вход продолжает работать
            svc.set_session(response, ctx.user_id, ctx.clinic_id, sv=new_sv)
        return response
