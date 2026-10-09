"""Чек-листы оценки: разбор YAML-шаблона и хранение чек-листов клиник в базе данных."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GENERAL_KEY = "general"
NOT_TARGET = "not_target"


@dataclass(frozen=True)
class Criterion:
    id: str
    num: str
    text: str
    weight_key: str
    weight: float
    evaluable: bool = True
    hint: str = ""


@dataclass(frozen=True)
class Checklist:
    key: str
    title: str
    criteria: tuple[Criterion, ...]
    direction: str = ""  # in / out / "" для общих правил
    description: str = ""

    @property
    def evaluable(self) -> tuple[Criterion, ...]:
        return tuple(c for c in self.criteria if c.evaluable)


@dataclass
class ChecklistSet:
    general: Checklist
    scenarios: dict[str, Checklist]
    weight_labels: dict[str, str] = field(default_factory=dict)
    clinic_context: str = ""

    def scenario(self, key: str) -> Checklist:
        return self.scenarios[key]

    def criterion(self, criterion_id: str) -> Criterion | None:
        for checklist in (self.general, *self.scenarios.values()):
            for c in checklist.criteria:
                if c.id == criterion_id:
                    return c
        return None

    def scenario_title(self, key: str | None) -> str:
        if key == NOT_TARGET:
            return "Нецелевой звонок"
        if key and key in self.scenarios:
            return self.scenarios[key].title
        return "Не определён"


def _parse_checklist(key: str, raw: dict, weights: dict[str, float]) -> Checklist:
    criteria = []
    for item in raw.get("criteria", []):
        weight_key = str(item.get("weight", "medium"))
        if weight_key not in weights:
            raise ValueError(f"Пункт {item.get('id')}: неизвестная значимость «{weight_key}»")
        criteria.append(
            Criterion(
                id=str(item["id"]),
                num=str(item.get("num", "")),
                text=" ".join(str(item["text"]).split()),
                weight_key=weight_key,
                weight=float(weights[weight_key]),
                evaluable=bool(item.get("evaluable", True)),
                hint=" ".join(str(item.get("hint", "")).split()),
            )
        )
    return Checklist(
        key=key,
        title=str(raw.get("title", key)),
        criteria=tuple(criteria),
        direction=str(raw.get("direction", "")),
        description=str(raw.get("description", "")),
    )


def load_checklists(path: Path) -> ChecklistSet:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    weights = {str(k): float(v) for k, v in data["weights"].items()}
    general = _parse_checklist(GENERAL_KEY, data["general"], weights)
    scenarios = {key: _parse_checklist(key, raw, weights) for key, raw in data["scenarios"].items()}

    ids = [c.id for c in general.criteria] + [c.id for s in scenarios.values() for c in s.criteria]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Повторяющиеся коды пунктов: {', '.join(sorted(duplicates))}")

    return ChecklistSet(
        general=general,
        scenarios=scenarios,
        weight_labels={str(k): str(v) for k, v in data.get("weight_labels", {}).items()},
        clinic_context=str(data.get("clinic_context", "")).strip(),
    )


# ---------------------------------------------------------------- чек-листы клиник в базе данных

WEIGHT_KEYS = ("high", "medium", "low")
WEIGHT_LABELS = {"high": "высокая", "medium": "средняя", "low": "низкая"}
DIRECTION_LABELS = {"in": "входящий", "out": "исходящий", "": "любой"}


def read_template(path: Path) -> dict:
    """Чек-лист в формате config/checklists.yaml как словарь."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _code_prefix(code: str) -> str:
    import re

    m = re.match(r"^([A-Za-zА-Яа-я]+)", code or "")
    return m.group(1) if m else ""


def validate_checklist_data(data: dict) -> None:
    """Проверяет словарь чек-листа перед загрузкой. Бросает ValueError с понятным текстом."""
    if not isinstance(data, dict) or "general" not in data or "scenarios" not in data:
        raise ValueError("В файле нет разделов «general» и «scenarios»")
    codes: list[str] = []
    sections = [("general", data["general"])] + list((data.get("scenarios") or {}).items())
    for key, section in sections:
        if not isinstance(section, dict) or not section.get("title"):
            raise ValueError(f"У раздела «{key}» нет названия (title)")
        for item in section.get("criteria") or []:
            if not isinstance(item, dict) or not item.get("id") or not str(item.get("text", "")).strip():
                raise ValueError(f"В разделе «{section.get('title')}» есть пункт без кода (id) или текста (text)")
            if str(item.get("weight", "medium")) not in WEIGHT_KEYS:
                raise ValueError(f"Пункт {item['id']}: значимость должна быть high, medium или low")
            codes.append(str(item["id"]))
    duplicates = sorted({c for c in codes if codes.count(c) > 1})
    if duplicates:
        raise ValueError(f"Повторяющиеся коды пунктов: {', '.join(duplicates)}")


def import_checklist_data(s, clinic, data: dict, replace_context: bool = True) -> None:
    """Загружает чек-лист в клинику: разделы — по ключу, пункты — по коду.

    Существующие пункты с тем же кодом обновляются (история оценок по ним сохраняется),
    пункты и разделы, которых нет в данных, отключаются.
    """
    from sqlalchemy import select

    from . import db

    validate_checklist_data(data)
    weights = data.get("weights") or {}
    if weights:
        clinic.weights_json = json.dumps(
            {k: float(weights.get(k, clinic.weights[k])) for k in WEIGHT_KEYS}, ensure_ascii=False
        )
    if replace_context and data.get("clinic_context") is not None:
        clinic.ai_context = str(data.get("clinic_context") or "").strip()

    scenarios = {sc.key: sc for sc in s.scalars(select(db.Scenario).where(db.Scenario.clinic_id == clinic.id))}
    criteria = {c.code: c for c in s.scalars(select(db.CriterionDef).where(db.CriterionDef.clinic_id == clinic.id))}
    seen_scenarios, seen_codes = set(), set()

    sections = [(GENERAL_KEY, data["general"])] + list((data.get("scenarios") or {}).items())
    for pos, (key, section) in enumerate(sections):
        items = section.get("criteria") or []
        sc = scenarios.get(key)
        if sc is None:
            sc = db.Scenario(clinic_id=clinic.id, key=key)
            s.add(sc)
            scenarios[key] = sc
        sc.title = str(section.get("title", key))
        sc.description = " ".join(str(section.get("description", "")).split())
        sc.direction = "" if key == GENERAL_KEY else str(section.get("direction", "") or "")
        sc.prefix = sc.prefix or (_code_prefix(str(items[0]["id"])) if items else "") or ("G" if key == GENERAL_KEY else "S")
        sc.position = pos
        sc.active = True
        s.flush()
        seen_scenarios.add(key)
        for cpos, item in enumerate(items):
            code = str(item["id"])
            c = criteria.get(code)
            if c is None:
                c = db.CriterionDef(clinic_id=clinic.id, code=code, text="")
                s.add(c)
                criteria[code] = c
            c.scenario_id = sc.id
            c.num = str(item.get("num", ""))
            c.text = " ".join(str(item["text"]).split())
            c.weight_key = str(item.get("weight", "medium"))
            c.evaluable = bool(item.get("evaluable", True))
            c.hint = " ".join(str(item.get("hint", "")).split())
            c.position = cpos
            c.active = True
            seen_codes.add(code)

    for key, sc in scenarios.items():
        if key not in seen_scenarios:
            sc.active = False
    for code, c in criteria.items():
        if code not in seen_codes:
            c.active = False
    s.flush()


def export_checklist_data(s, clinic) -> dict:
    """Чек-лист клиники в формате config/checklists.yaml (только действующие разделы и пункты)."""
    from sqlalchemy import select

    from . import db

    def criteria_of(sc) -> list[dict]:
        rows = s.scalars(
            select(db.CriterionDef)
            .where(db.CriterionDef.scenario_id == sc.id, db.CriterionDef.active.is_(True))
            .order_by(db.CriterionDef.position, db.CriterionDef.id)
        )
        out = []
        for c in rows:
            item = {"id": c.code, "num": c.num, "text": c.text, "weight": c.weight_key}
            if not c.evaluable:
                item["evaluable"] = False
            if c.hint:
                item["hint"] = c.hint
            out.append(item)
        return out

    scenarios = s.scalars(
        select(db.Scenario)
        .where(db.Scenario.clinic_id == clinic.id, db.Scenario.active.is_(True))
        .order_by(db.Scenario.position, db.Scenario.id)
    ).all()
    general = next((sc for sc in scenarios if sc.key == GENERAL_KEY), None)
    data = {
        "weights": clinic.weights,
        "weight_labels": dict(WEIGHT_LABELS),
        "clinic_context": clinic.ai_context or "",
        "general": {"title": general.title if general else "Общие правила", "criteria": criteria_of(general) if general else []},
        "scenarios": {},
    }
    for sc in scenarios:
        if sc.key == GENERAL_KEY:
            continue
        data["scenarios"][sc.key] = {
            "title": sc.title,
            "direction": sc.direction,
            "description": sc.description,
            "criteria": criteria_of(sc),
        }
    return data


def checklist_to_yaml(data: dict) -> str:
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110)


def seed_clinic_checklist(s, clinic, template_path: Path) -> None:
    """Заполняет чек-лист новой клиники из шаблона, если у неё ещё нет разделов."""
    from sqlalchemy import func, select

    from . import db

    exists = s.scalar(select(func.count()).select_from(db.Scenario).where(db.Scenario.clinic_id == clinic.id))
    if not exists:
        import_checklist_data(s, clinic, read_template(template_path), replace_context=not clinic.ai_context)


def copy_checklist(s, source_clinic, target_clinic) -> None:
    import_checklist_data(s, target_clinic, export_checklist_data(s, source_clinic))


def next_criterion_code(s, clinic_id: int, prefix: str) -> str:
    """Следующий свободный код пункта с заданным началом (G28, IN25…)."""
    import re

    from sqlalchemy import select

    from . import db

    used = set(s.scalars(select(db.CriterionDef.code).where(db.CriterionDef.clinic_id == clinic_id)))
    numbers = [int(m.group(1)) for code in used if (m := re.match(rf"^{re.escape(prefix)}(\d+)$", code))]
    n = max(numbers, default=0) + 1
    while f"{prefix}{n:02d}" in used:
        n += 1
    return f"{prefix}{n:02d}"


def load_clinic_checklists(s, clinic_id: int) -> ChecklistSet:
    """Действующий чек-лист клиники из базы — в том же виде, что и из YAML."""
    from sqlalchemy import select

    from . import db

    clinic = s.get(db.Clinic, clinic_id)
    weights = clinic.weights
    scenarios = s.scalars(
        select(db.Scenario)
        .where(db.Scenario.clinic_id == clinic_id, db.Scenario.active.is_(True))
        .order_by(db.Scenario.position, db.Scenario.id)
    ).all()
    rows = s.scalars(
        select(db.CriterionDef)
        .where(db.CriterionDef.clinic_id == clinic_id, db.CriterionDef.active.is_(True))
        .order_by(db.CriterionDef.position, db.CriterionDef.id)
    ).all()
    by_scenario: dict[int, list] = {}
    for c in rows:
        by_scenario.setdefault(c.scenario_id, []).append(c)

    def build(sc) -> Checklist:
        return Checklist(
            key=sc.key,
            title=sc.title,
            direction=sc.direction,
            description=sc.description,
            criteria=tuple(
                Criterion(
                    id=c.code,
                    num=c.num,
                    text=c.text,
                    weight_key=c.weight_key,
                    weight=float(weights.get(c.weight_key, 1.0)),
                    evaluable=c.evaluable,
                    hint=c.hint,
                )
                for c in by_scenario.get(sc.id, [])
            ),
        )

    general = next((build(sc) for sc in scenarios if sc.key == GENERAL_KEY), None)
    if general is None:
        general = Checklist(key=GENERAL_KEY, title="Общие правила", criteria=())
    return ChecklistSet(
        general=general,
        scenarios={sc.key: build(sc) for sc in scenarios if sc.key != GENERAL_KEY},
        weight_labels=dict(WEIGHT_LABELS),
        clinic_context=clinic.ai_context or "",
    )
