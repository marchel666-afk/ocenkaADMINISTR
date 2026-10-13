"""Звонки, сводка, выгрузка, сотрудники — всё в рамках текущей клиники."""

import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import flag_modified

from . import db
from .checklists import GENERAL_KEY, NOT_TARGET, ChecklistSet, load_clinic_checklists
from .evaluation import Evaluation
from .mango import parse_mango_filename
from .reports import build_excel, call_type_title, criterion_stats, employee_stats, period_start
from .transcription import Transcript, transcript_from_text
from .web import DIRECTIONS, Services, ctx_of, guess_datetime, parse_dt, redirect, to_int

TEXT_EXTENSIONS = {".txt"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".flac", ".amr", ".webm", ".wma", ".mp4", ".3gp"}


def _decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def card_sections(call: db.Call, checklists: ChecklistSet) -> list[dict]:
    """Разделы карточки звонка: пункты в том виде, в каком их оценили (с текущими «не оценивается по записи»).

    Сценарий берётся из самих оценок, а не из типа звонка: если переоценка не удалась, карточка
    по-прежнему показывает прошлую оценку целиком.
    """
    keys = {r.checklist for r in call.results} - {GENERAL_KEY}
    order = [GENERAL_KEY] + sorted(keys, key=lambda k: (k != call.call_type, k))
    sections = []
    for key in order:
        results = [r for r in call.results if r.checklist == key]
        if not results:
            continue
        current = checklists.general if key == GENERAL_KEY else checklists.scenarios.get(key)
        rows = []
        for r in results:
            c = checklists.criterion(r.criterion_id)
            rows.append(
                {
                    "num": r.criterion_num or (c.num if c else ""),
                    "text": r.criterion_text or (c.text if c else r.criterion_id),
                    "weight_key": r.weight_key or (c.weight_key if c else ""),
                    "evaluable": True,
                    "result": r.result,
                    "evidence": r.evidence,
                    "comment": r.comment,
                }
            )
        if current is not None:
            rated = {r.criterion_id.upper() for r in results}
            rows += [
                {"num": c.num, "text": c.text, "weight_key": c.weight_key, "evaluable": False}
                for c in current.criteria
                if not c.evaluable and c.id.upper() not in rated
            ]
        title = (current.title if current else "") or results[0].checklist_title or key
        score = call.score_general if key == GENERAL_KEY else call.score_scenario
        sections.append({"key": key, "title": title, "score": score, "rows": rows})
    return sections


def register(app: FastAPI, svc: Services) -> None:
    sf = svc.session_factory
    settings = svc.settings

    def employees(s, clinic_id: int, active_only: bool = False):
        q = select(db.Employee).where(db.Employee.clinic_id == clinic_id).order_by(db.Employee.name)
        if active_only:
            q = q.where(db.Employee.active.is_(True))
        return s.scalars(q).all()

    def get_or_create_employee(s, clinic_id: int, employee_id: str, new_name: str) -> int | None:
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
        if to_int(employee_id) is not None:
            emp = s.get(db.Employee, to_int(employee_id))
            return emp.id if emp is not None and emp.clinic_id == clinic_id else None
        return None

    def load_call(s, clinic_id: int, call_id: int) -> db.Call:
        call = s.get(db.Call, call_id, options=[selectinload(db.Call.results), selectinload(db.Call.employee)])
        if call is None or call.clinic_id != clinic_id:
            raise HTTPException(status_code=404, detail="Звонок не найден")
        return call

    # ------------------------------------------------------------ звонки

    @app.get("/")
    def index():
        return redirect("/calls")

    @app.get("/calls")
    def calls_list(request: Request, employee: str = "", status: str = "", days: int = 30, page: int = 1):
        ctx = ctx_of(request)
        per_page = 50
        with sf() as s:
            checklists = load_clinic_checklists(s, ctx.clinic_id)
            q = (
                select(db.Call)
                .where(db.Call.clinic_id == ctx.clinic_id)
                .options(selectinload(db.Call.employee), selectinload(db.Call.results))
            )
            since = period_start(days)
            if since:
                q = q.where(func.coalesce(db.Call.started_at, db.Call.created_at) >= since)
            if employee == "none":
                q = q.where(db.Call.employee_id.is_(None))
            elif to_int(employee) is not None:
                q = q.where(db.Call.employee_id == to_int(employee))
            if status in db.STATUS_LABELS:
                q = q.where(db.Call.status == status)
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            calls = s.scalars(
                q.order_by(func.coalesce(db.Call.started_at, db.Call.created_at).desc(), db.Call.id.desc())
                .offset((min(max(page, 1), 100_000) - 1) * per_page)
                .limit(per_page)
            ).all()
            pending = s.scalar(
                select(func.count()).where(
                    db.Call.clinic_id == ctx.clinic_id, db.Call.status.in_((db.QUEUED, *db.IN_PROGRESS))
                )
            )
            errors = s.scalar(
                select(func.count()).where(db.Call.clinic_id == ctx.clinic_id, db.Call.status == db.ERROR)
            )
            return svc.render(
                request,
                "calls.html",
                calls=calls,
                employees=employees(s, ctx.clinic_id),
                active_employees=employees(s, ctx.clinic_id, active_only=True),
                scenarios=checklists.scenarios,
                type_title=lambda c: call_type_title(c, checklists),
                f_employee=employee,
                f_status=status,
                f_days=days,
                page=min(max(page, 1), 100_000),
                pages=max(1, -(-total // per_page)),
                total=total,
                pending=pending,
                errors=errors,
            )

    @app.post("/calls/retry-errors")
    def calls_retry_errors(request: Request):
        """Все звонки клиники с ошибкой — снова в очередь (например, после сбоя связи с ИИ)."""
        ctx = ctx_of(request)
        with sf() as s:
            count = s.execute(
                update(db.Call)
                .where(db.Call.clinic_id == ctx.clinic_id, db.Call.status == db.ERROR)
                .values(status=db.QUEUED, status_message=""),
                execution_options={"synchronize_session": False},
            ).rowcount
            s.commit()
        svc.wake_worker()
        return redirect("/calls", msg=f"Звонков возвращено в очередь: {count}")

    @app.post("/calls/upload")
    def upload(
        request: Request,
        files: Annotated[list[UploadFile], File()],
        employee_id: Annotated[str, Form()] = "",
        new_employee: Annotated[str, Form()] = "",
        direction: Annotated[str, Form()] = "",
        call_type: Annotated[str, Form()] = "",
        started_at: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        manual_dt = parse_dt(started_at)
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            checklists = load_clinic_checklists(s, ctx.clinic_id)
            if call_type and call_type not in checklists.scenarios:
                call_type = ""
            if direction not in DIRECTIONS:
                direction = ""
            if call_type:
                direction = checklists.scenarios[call_type].direction or direction
            emp_id = get_or_create_employee(s, ctx.clinic_id, employee_id, new_employee)
            shift = clinic.utc_offset - clinic.mango_utc_offset
            count = 0
            rejected: list[str] = []
            written: list[Path] = []
            for upload_file in files:
                if not upload_file.filename:
                    continue
                name = Path(upload_file.filename).name[-200:]
                ext = Path(name).suffix.lower()
                if ext not in AUDIO_EXTENSIONS | TEXT_EXTENSIONS:
                    rejected.append(name)
                    continue
                mango = parse_mango_filename(name, shift)
                call = db.Call(
                    clinic_id=ctx.clinic_id,
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
                if emp_id is None and mango and mango.line:
                    by_line = s.scalars(
                        select(db.Employee).where(
                            db.Employee.clinic_id == ctx.clinic_id, db.Employee.mango_id == mango.line
                        )
                    ).first()
                    call.employee_id = by_line.id if by_line else None
                if ext in TEXT_EXTENSIONS:
                    transcript = transcript_from_text(_decode_text(upload_file.file.read()))
                    call.source = "text"
                    call.transcript_json = transcript.to_json()
                else:
                    stored = f"{datetime.now():%Y%m}/{uuid.uuid4().hex}{ext}"
                    target = settings.audio_dir / stored
                    target.parent.mkdir(parents=True, exist_ok=True)
                    written.append(target)
                    try:
                        with open(target, "wb") as out:
                            shutil.copyfileobj(upload_file.file, out)
                    except BaseException:
                        for path in written:  # пачка не сохранится — не оставляем файлы без звонков
                            path.unlink(missing_ok=True)
                        raise
                    call.source = "upload"
                    call.audio_path = stored
                s.add(call)
                count += 1
            s.commit()
        svc.wake_worker()
        names = ", ".join(n if len(n) <= 40 else n[:37] + "…" for n in rejected[:5])
        skipped = f" Пропущены файлы неподходящего формата: {names}{' и др.' if len(rejected) > 5 else ''}." if rejected else ""
        if not count:
            return redirect("/calls", err=f"Нет файлов для загрузки.{skipped}")
        return redirect("/calls", msg=f"Загружено файлов: {count}. Они обрабатываются по очереди.{skipped}")

    @app.get("/calls/{call_id}")
    def call_card(request: Request, call_id: int):
        ctx = ctx_of(request)
        with sf() as s:
            call = load_call(s, ctx.clinic_id, call_id)
            checklists = load_clinic_checklists(s, ctx.clinic_id)
            evaluation = Evaluation.from_json(call.evaluation_json) if call.evaluation_json else None
            transcript = Transcript.from_json(call.transcript_json) if call.transcript_json else None
            precise = settings.llm_precise_model if settings.llm_precise_model != settings.llm_model else ""
            return svc.render(
                request,
                "call.html",
                call=call,
                evaluation=evaluation,
                transcript=transcript,
                sections=card_sections(call, checklists) if evaluation else [],
                type_title=call_type_title(call, checklists),
                scenarios=checklists.scenarios,
                # отключённый сотрудник звонка остаётся в списке, иначе сохранение формы его сотрёт
                employees=[e for e in employees(s, ctx.clinic_id) if e.active or e.id == call.employee_id],
                NOT_TARGET=NOT_TARGET,
                precise_model=precise,
            )

    @app.get("/calls/{call_id}/audio")
    def call_audio(request: Request, call_id: int):
        ctx = ctx_of(request)
        with sf() as s:
            call = load_call(s, ctx.clinic_id, call_id)
            if not call.audio_path:
                raise HTTPException(status_code=404, detail="Записи нет")
            path = (settings.audio_dir / call.audio_path).resolve()
            if not path.is_file() or settings.audio_dir.resolve() not in path.parents:
                raise HTTPException(status_code=404, detail="Файл записи не найден")
            return FileResponse(path, filename=call.original_filename or path.name)

    def requeue(s, call: db.Call, retranscribe: bool = False, model: str | None = None) -> None:
        call.llm_model_override = model
        if retranscribe and call.audio_path:
            call.transcript_json = None
        # тип звонка не сбрасываем: автоматический тип всё равно определяется заново,
        # а при сбое переоценки звонок остаётся со старым типом и старой оценкой
        call.status = db.QUEUED
        call.status_message = ""
        # статус пишем всегда, даже если он уже «в очереди»: если обработчик успел взять звонок, пока шёл
        # этот запрос, его результат будет отброшен, и звонок обработается заново с правками пользователя
        flag_modified(call, "status")
        s.commit()

    @app.post("/calls/{call_id}/update")
    def call_update(
        request: Request,
        call_id: int,
        employee_id: Annotated[str, Form()] = "",
        new_employee: Annotated[str, Form()] = "",
        call_type: Annotated[str, Form()] = "",
        direction: Annotated[str, Form()] = "keep",
        started_at: Annotated[str, Form()] = "",
        reevaluate: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        with sf() as s:
            call = load_call(s, ctx.clinic_id, call_id)
            checklists = load_clinic_checklists(s, ctx.clinic_id)
            call.employee_id = get_or_create_employee(s, ctx.clinic_id, employee_id, new_employee)
            type_changed = False
            if direction in DIRECTIONS or direction == "":
                new_direction = direction or None
                type_changed = new_direction != call.direction and not call.call_type_manual
                call.direction = new_direction
            if call.call_type_manual and call_type == call.call_type:
                pass  # тип не меняли (в том числе удалённый из критериев сценарий)
            elif call_type in checklists.scenarios:
                type_changed = True
                call.call_type, call.call_type_manual = call_type, True
                call.direction = checklists.scenarios[call_type].direction or call.direction
            elif call_type == "auto" and call.call_type_manual:
                call.call_type_manual, type_changed = False, True
            dt = parse_dt(started_at)
            if dt and dt != call.started_at:
                call.started_at = dt
            if reevaluate or type_changed:
                requeue(s, call)
                svc.wake_worker()
            else:
                s.commit()
        return redirect(f"/calls/{call_id}")

    @app.post("/calls/{call_id}/reevaluate")
    def call_reevaluate(request: Request, call_id: int):
        with sf() as s:
            requeue(s, load_call(s, ctx_of(request).clinic_id, call_id))
        svc.wake_worker()
        return redirect(f"/calls/{call_id}")

    @app.post("/calls/{call_id}/reevaluate-precise")
    def call_reevaluate_precise(request: Request, call_id: int):
        with sf() as s:
            requeue(s, load_call(s, ctx_of(request).clinic_id, call_id), model=settings.llm_precise_model or None)
        svc.wake_worker()
        return redirect(f"/calls/{call_id}")

    @app.post("/calls/{call_id}/retranscribe")
    def call_retranscribe(request: Request, call_id: int):
        with sf() as s:
            requeue(s, load_call(s, ctx_of(request).clinic_id, call_id), retranscribe=True)
        svc.wake_worker()
        return redirect(f"/calls/{call_id}")

    @app.post("/calls/{call_id}/delete")
    def call_delete(request: Request, call_id: int):
        with sf() as s:
            call = load_call(s, ctx_of(request).clinic_id, call_id)
            if call.audio_path:
                (settings.audio_dir / call.audio_path).unlink(missing_ok=True)
            s.delete(call)
            s.commit()
        return redirect("/calls", msg="Звонок удалён")

    # ------------------------------------------------------------ сводка и выгрузка

    @app.get("/stats")
    def stats(request: Request, days: int = 30, employee: str = ""):
        ctx = ctx_of(request)
        emp_id = to_int(employee)
        with sf() as s:
            checklists = load_clinic_checklists(s, ctx.clinic_id)
            since = period_start(days)
            rows = employee_stats(s, ctx.clinic_id, since)
            crit = [c for c in criterion_stats(s, checklists, ctx.clinic_id, since, emp_id) if c.failed][:15]
            return svc.render(
                request,
                "stats.html",
                rows=rows,
                criteria=crit,
                employees=employees(s, ctx.clinic_id),
                f_days=days,
                f_employee=employee,
            )

    @app.get("/export.xlsx")
    def export(request: Request, days: int = 30):
        ctx = ctx_of(request)
        with sf() as s:
            checklists = load_clinic_checklists(s, ctx.clinic_id)
            content = build_excel(s, checklists, ctx.clinic_id, period_start(days))
        filename = f"ocenka_zvonkov_{datetime.now():%Y-%m-%d}.xlsx"
        return Response(
            content,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ------------------------------------------------------------ сотрудники

    def own_employee(s, clinic_id: int, employee_id: int) -> db.Employee:
        emp = s.get(db.Employee, employee_id)
        if emp is None or emp.clinic_id != clinic_id:
            raise HTTPException(status_code=404)
        return emp

    @app.get("/employees")
    def employees_page(request: Request):
        ctx = ctx_of(request)
        with sf() as s:
            counts = dict(
                s.execute(
                    select(db.Call.employee_id, func.count())
                    .where(db.Call.clinic_id == ctx.clinic_id)
                    .group_by(db.Call.employee_id)
                ).all()
            )
            return svc.render(request, "employees.html", employees=employees(s, ctx.clinic_id), counts=counts)

    @app.post("/employees")
    def employees_add(request: Request, name: Annotated[str, Form()], mango_id: Annotated[str, Form()] = ""):
        ctx = ctx_of(request)
        with sf() as s:
            emp_id = get_or_create_employee(s, ctx.clinic_id, "", name)
            if emp_id and mango_id.strip():
                s.get(db.Employee, emp_id).mango_id = mango_id.strip()
            s.commit()
        return redirect("/employees")

    @app.post("/employees/{employee_id}/toggle")
    def employees_toggle(request: Request, employee_id: int):
        with sf() as s:
            emp = own_employee(s, ctx_of(request).clinic_id, employee_id)
            emp.active = not emp.active
            s.commit()
        return redirect("/employees")

    @app.post("/employees/{employee_id}/rename")
    def employees_rename(
        request: Request,
        employee_id: int,
        name: Annotated[str, Form()],
        mango_id: Annotated[str, Form()] = "",
    ):
        name = " ".join(name.split())
        with sf() as s:
            emp = own_employee(s, ctx_of(request).clinic_id, employee_id)
            if name:
                emp.name = name
            emp.mango_id = mango_id.strip() or None
            s.commit()
        return redirect("/employees")
