"""Загрузка чек-листов из config/checklists.yaml."""

from __future__ import annotations

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
