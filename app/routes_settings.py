"""Настройки текущей клиники: редактор критериев оценки, сведения для ИИ, подключение к Mango."""

import json
import re
import threading
from datetime import datetime
from typing import Annotated

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from sqlalchemy import func, select

from . import db
from .checklists import (
    GENERAL_KEY,
    WEIGHT_KEYS,
    checklist_to_yaml,
    copy_checklist,
    export_checklist_data,
    import_checklist_data,
    next_criterion_code,
    read_template,
)
from .web import Services, ctx_of, redirect

CHECKLIST_URL = "/settings/checklist"


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def register(app: FastAPI, svc: Services) -> None:
    sf = svc.session_factory

    # ------------------------------------------------------------ редактор критериев

    def own_scenario(s, clinic_id: int, scenario_id: int) -> db.Scenario:
        sc = s.get(db.Scenario, scenario_id)
        if sc is None or sc.clinic_id != clinic_id or not sc.active:
            raise HTTPException(status_code=404, detail="Раздел не найден")
        return sc

    def own_criterion(s, clinic_id: int, criterion_id: int) -> db.CriterionDef:
        c = s.get(db.CriterionDef, criterion_id)
        if c is None or c.clinic_id != clinic_id or not c.active:
            raise HTTPException(status_code=404, detail="Пункт не найден")
        return c

    def active_scenarios(s, clinic_id: int) -> list[db.Scenario]:
        return s.scalars(
            select(db.Scenario)
            .where(db.Scenario.clinic_id == clinic_id, db.Scenario.active.is_(True))
            .order_by(db.Scenario.position, db.Scenario.id)
        ).all()

    def active_criteria(s, scenario_id: int) -> list[db.CriterionDef]:
        return s.scalars(
            select(db.CriterionDef)
            .where(db.CriterionDef.scenario_id == scenario_id, db.CriterionDef.active.is_(True))
            .order_by(db.CriterionDef.position, db.CriterionDef.id)
        ).all()

    def section_url(scenario_id: int) -> str:
        return f"{CHECKLIST_URL}?section={scenario_id}"

    @app.get(CHECKLIST_URL)
    def checklist_page(request: Request, section: int = 0):
        ctx = ctx_of(request)
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            scenarios = active_scenarios(s, ctx.clinic_id)
            counts = dict(
                s.execute(
                    select(db.CriterionDef.scenario_id, func.count())
                    .where(db.CriterionDef.clinic_id == ctx.clinic_id, db.CriterionDef.active.is_(True))
                    .group_by(db.CriterionDef.scenario_id)
                ).all()
            )
            current = next((sc for sc in scenarios if sc.id == section), scenarios[0] if scenarios else None)
            other_clinics = [(cid, name) for cid, name in ctx.clinics if cid != ctx.clinic_id]
            return svc.render(
                request,
                "settings_checklist.html",
                clinic=clinic,
                scenarios=scenarios,
                counts=counts,
                current=current,
                criteria=active_criteria(s, current.id) if current else [],
                weights=clinic.weights,
                other_clinics=other_clinics if ctx.is_admin else [],
                GENERAL_KEY=GENERAL_KEY,
            )

    @app.post(f"{CHECKLIST_URL}/weights")
    def checklist_weights(
        request: Request,
        high: Annotated[float, Form()],
        medium: Annotated[float, Form()],
        low: Annotated[float, Form()],
    ):
        ctx = ctx_of(request)
        if min(high, medium, low) < 0:
            return redirect(CHECKLIST_URL, err="Баллы не могут быть отрицательными")
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            clinic.weights_json = json.dumps({"high": high, "medium": medium, "low": low})
            s.commit()
        return redirect(CHECKLIST_URL, msg="Баллы за значимость сохранены. Они применяются к новым оценкам.")

    @app.post(f"{CHECKLIST_URL}/sections")
    def section_add(
        request: Request,
        title: Annotated[str, Form()],
        direction: Annotated[str, Form()] = "",
        description: Annotated[str, Form()] = "",
        prefix: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        title, prefix = _clean(title), _clean(prefix).upper()
        if not title:
            return redirect(CHECKLIST_URL, err="Укажите название сценария")
        if direction not in ("in", "out"):
            return redirect(CHECKLIST_URL, err="Укажите направление звонка: входящий или исходящий")
        with sf() as s:
            existing = s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == ctx.clinic_id)).all()
            used_prefixes = {sc.prefix for sc in existing}
            if prefix and (not re.fullmatch(r"[A-Z]{1,4}", prefix) or prefix in used_prefixes):
                return redirect(CHECKLIST_URL, err="Код раздела — 1–4 латинские буквы, не занятые другим разделом")
            if not prefix:
                prefix = next(
                    f"S{chr(c)}" for c in range(ord("A"), ord("Z") + 1) if f"S{chr(c)}" not in used_prefixes
                )
            n = 1
            used_keys = {sc.key for sc in existing}
            while f"custom_{n}" in used_keys:
                n += 1
            sc = db.Scenario(
                clinic_id=ctx.clinic_id,
                key=f"custom_{n}",
                title=title,
                direction=direction,
                description=_clean(description),
                prefix=prefix,
                position=max((x.position for x in existing), default=0) + 1,
            )
            s.add(sc)
            s.commit()
            return redirect(section_url(sc.id), msg="Сценарий добавлен. Теперь добавьте в него пункты.")

    @app.post(CHECKLIST_URL + "/sections/{scenario_id}")
    def section_update(
        request: Request,
        scenario_id: int,
        title: Annotated[str, Form()],
        direction: Annotated[str, Form()] = "",
        description: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        with sf() as s:
            sc = own_scenario(s, ctx.clinic_id, scenario_id)
            if _clean(title):
                sc.title = _clean(title)
            if sc.key != GENERAL_KEY:
                if direction in ("in", "out"):
                    sc.direction = direction
                sc.description = _clean(description)
            s.commit()
        return redirect(section_url(scenario_id), msg="Раздел сохранён")

    @app.post(CHECKLIST_URL + "/sections/{scenario_id}/delete")
    def section_delete(request: Request, scenario_id: int):
        ctx = ctx_of(request)
        with sf() as s:
            sc = own_scenario(s, ctx.clinic_id, scenario_id)
            if sc.key == GENERAL_KEY:
                return redirect(section_url(scenario_id), err="Общие правила удалить нельзя")
            sc.active = False
            for c in active_criteria(s, sc.id):
                c.active = False
            s.commit()
        return redirect(CHECKLIST_URL, msg="Сценарий удалён. Уже выставленные оценки сохранились.")

    @app.post(CHECKLIST_URL + "/sections/{scenario_id}/move")
    def section_move(request: Request, scenario_id: int, direction: Annotated[str, Form()]):
        ctx = ctx_of(request)
        with sf() as s:
            sc = own_scenario(s, ctx.clinic_id, scenario_id)
            ordered = [x for x in active_scenarios(s, ctx.clinic_id) if x.key != GENERAL_KEY]
            _move(ordered, sc, direction, offset=1)
            s.commit()
        return redirect(section_url(scenario_id))

    def _criterion_fields(
        num: str, text: str, weight: str, evaluable: str, hint: str
    ) -> tuple[dict | None, str]:
        text = _clean(text)
        if not text:
            return None, "Текст пункта не может быть пустым"
        if weight not in WEIGHT_KEYS:
            return None, "Неизвестная значимость"
        return {"num": _clean(num), "text": text, "weight_key": weight, "evaluable": bool(evaluable), "hint": _clean(hint)}, ""

    @app.post(CHECKLIST_URL + "/criteria")
    def criterion_add(
        request: Request,
        scenario_id: Annotated[int, Form()],
        text: Annotated[str, Form()],
        num: Annotated[str, Form()] = "",
        weight: Annotated[str, Form()] = "medium",
        evaluable: Annotated[str, Form()] = "",
        hint: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        fields, error = _criterion_fields(num, text, weight, evaluable, hint)
        if error:
            return redirect(section_url(scenario_id), err=error)
        with sf() as s:
            sc = own_scenario(s, ctx.clinic_id, scenario_id)
            existing = active_criteria(s, sc.id)
            if not fields["num"]:
                fields["num"] = str(len(existing) + 1)
            c = db.CriterionDef(
                clinic_id=ctx.clinic_id,
                scenario_id=sc.id,
                code=next_criterion_code(s, ctx.clinic_id, sc.prefix or "C"),
                position=max((x.position for x in existing), default=-1) + 1,
                **fields,
            )
            s.add(c)
            s.commit()
            return redirect(section_url(scenario_id), msg=f"Пункт {c.code} добавлен")

    @app.post(CHECKLIST_URL + "/criteria/{criterion_id}")
    def criterion_update(
        request: Request,
        criterion_id: int,
        text: Annotated[str, Form()],
        num: Annotated[str, Form()] = "",
        weight: Annotated[str, Form()] = "medium",
        evaluable: Annotated[str, Form()] = "",
        hint: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        with sf() as s:
            c = own_criterion(s, ctx.clinic_id, criterion_id)
            fields, error = _criterion_fields(num, text, weight, evaluable, hint)
            if error:
                return redirect(section_url(c.scenario_id), err=error)
            for k, v in fields.items():
                setattr(c, k, v)
            s.commit()
            return redirect(section_url(c.scenario_id) + f"#c{c.id}", msg=f"Пункт {c.code} сохранён")

    @app.post(CHECKLIST_URL + "/criteria/{criterion_id}/delete")
    def criterion_delete(request: Request, criterion_id: int):
        ctx = ctx_of(request)
        with sf() as s:
            c = own_criterion(s, ctx.clinic_id, criterion_id)
            c.active = False
            s.commit()
            return redirect(section_url(c.scenario_id), msg=f"Пункт {c.code} удалён")

    @app.post(CHECKLIST_URL + "/criteria/{criterion_id}/move")
    def criterion_move(request: Request, criterion_id: int, direction: Annotated[str, Form()]):
        ctx = ctx_of(request)
        with sf() as s:
            c = own_criterion(s, ctx.clinic_id, criterion_id)
            _move(active_criteria(s, c.scenario_id), c, direction)
            s.commit()
            return redirect(section_url(c.scenario_id) + f"#c{c.id}")

    @app.get(CHECKLIST_URL + "/export.yaml")
    def checklist_export(request: Request):
        ctx = ctx_of(request)
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            content = checklist_to_yaml(export_checklist_data(s, clinic))
        filename = f"checklist_{ctx.clinic_id}_{datetime.now():%Y-%m-%d}.yaml"
        return Response(
            content.encode("utf-8"),
            media_type="application/x-yaml; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @app.post(CHECKLIST_URL + "/import")
    def checklist_import(request: Request, file: Annotated[UploadFile, File()]):
        ctx = ctx_of(request)
        try:
            data = yaml.safe_load(file.file.read().decode("utf-8-sig"))
        except (UnicodeDecodeError, yaml.YAMLError) as e:
            return redirect(CHECKLIST_URL, err=f"Не удалось прочитать файл: {e}")
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            try:
                import_checklist_data(s, clinic, data)
            except ValueError as e:
                s.rollback()
                return redirect(CHECKLIST_URL, err=str(e))
            s.commit()
        return redirect(CHECKLIST_URL, msg="Чек-лист загружен из файла")

    @app.post(CHECKLIST_URL + "/reset")
    def checklist_reset(request: Request):
        ctx = ctx_of(request)
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            import_checklist_data(s, clinic, read_template(svc.settings.checklists_path), replace_context=False)
            s.commit()
        return redirect(CHECKLIST_URL, msg="Чек-лист заменён шаблоном")

    @app.post(CHECKLIST_URL + "/copy")
    def checklist_copy(request: Request, source_clinic_id: Annotated[int, Form()]):
        ctx = ctx_of(request)
        if not ctx.is_admin or source_clinic_id not in {cid for cid, _ in ctx.clinics}:
            return redirect(CHECKLIST_URL, err="Нет доступа к этой клинике")
        with sf() as s:
            copy_checklist(s, s.get(db.Clinic, source_clinic_id), s.get(db.Clinic, ctx.clinic_id))
            s.commit()
        return redirect(CHECKLIST_URL, msg="Чек-лист скопирован из другой клиники")

    # ------------------------------------------------------------ клиника и Mango

    @app.get("/settings/clinic")
    def clinic_page(request: Request):
        ctx = ctx_of(request)
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            return svc.render(
                request,
                "settings_clinic.html",
                clinic=clinic,
                default_min_seconds=svc.settings.min_call_seconds,
                scheduler_enabled=svc.scheduler is not None,
                running=svc.scheduler.is_running(ctx.clinic_id) if svc.scheduler else False,
            )

    @app.post("/settings/clinic")
    def clinic_save(
        request: Request,
        name: Annotated[str, Form()],
        ai_context: Annotated[str, Form()] = "",
        utc_offset: Annotated[int, Form()] = 3,
        min_call_seconds: Annotated[str, Form()] = "",
    ):
        ctx = ctx_of(request)
        if not -12 <= utc_offset <= 14:
            return redirect("/settings/clinic", err="Часовой пояс — от UTC−12 до UTC+14")
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            if _clean(name):
                clinic.name = _clean(name)
            clinic.ai_context = (ai_context or "").strip()
            clinic.utc_offset = utc_offset
            clinic.min_call_seconds = int(min_call_seconds) if min_call_seconds.strip().isdigit() else None
            s.commit()
        return redirect("/settings/clinic", msg="Настройки клиники сохранены")

    @app.post("/settings/mango")
    def mango_save(
        request: Request,
        enabled: Annotated[str, Form()] = "",
        api_key: Annotated[str, Form()] = "",
        api_salt: Annotated[str, Form()] = "",
        interval_min: Annotated[int, Form()] = 30,
        lines: Annotated[str, Form()] = "",
        mango_utc_offset: Annotated[int, Form()] = 3,
    ):
        ctx = ctx_of(request)
        with sf() as s:
            clinic = s.get(db.Clinic, ctx.clinic_id)
            if api_key.strip():
                clinic.mango_api_key = api_key.strip()
            if api_salt.strip():
                clinic.mango_api_salt = api_salt.strip()
            if enabled and not (clinic.mango_api_key and clinic.mango_api_salt):
                return redirect("/settings/clinic", err="Для автозагрузки укажите уникальный код АТС и ключ подписи")
            clinic.mango_enabled = bool(enabled)
            clinic.mango_interval_min = max(10, min(int(interval_min), 24 * 60))
            clinic.mango_lines = ", ".join(x for x in re.split(r"[\s,;]+", lines) if x)
            clinic.mango_utc_offset = mango_utc_offset
            s.commit()
        if enabled and svc.scheduler:
            svc.scheduler.wake()
        return redirect("/settings/clinic", msg="Настройки Mango сохранены")

    @app.post("/settings/mango/test")
    def mango_test(request: Request):
        ctx = ctx_of(request)
        if svc.scheduler is None:
            return redirect("/settings/clinic", err="Фоновая обработка выключена (WORKER_ENABLED=false)")
        ok, message = svc.scheduler.test_connection(ctx.clinic_id)
        return redirect("/settings/clinic", **({"msg": message} if ok else {"err": message}))

    @app.post("/settings/mango/import")
    def mango_import_now(request: Request, days: Annotated[int, Form()] = 0):
        ctx = ctx_of(request)
        if svc.scheduler is None:
            return redirect("/settings/clinic", err="Фоновая обработка выключена (WORKER_ENABLED=false)")
        if days > 0:
            with sf() as s:
                clinic = s.get(db.Clinic, ctx.clinic_id)
                clinic.mango_synced_until = int(datetime.now().timestamp()) - min(days, 31) * 86400
                s.commit()
        threading.Thread(target=svc.scheduler.run_clinic, args=(ctx.clinic_id,), daemon=True).start()
        return redirect("/settings/clinic", msg="Загрузка звонков из Mango запущена — обновите страницу через минуту")


def _move(items: list, item, direction: str, offset: int = 0) -> None:
    """Меняет местами элемент с соседним и переписывает позиции подряд."""
    idx = next(i for i, x in enumerate(items) if x.id == item.id)
    swap = idx - 1 if direction == "up" else idx + 1
    if 0 <= swap < len(items):
        items[idx], items[swap] = items[swap], items[idx]
    for pos, x in enumerate(items):
        x.position = pos + offset
