"""Определение типа звонка и оценка разговора по чек-листу с помощью языковой модели."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .checklists import NOT_TARGET, Checklist, ChecklistSet
from .llm import LLMClient, LLMError, LLMResponse
from .scoring import NA, NO, YES
from .transcription import Transcript

log = logging.getLogger(__name__)

DIRECTION_LABELS = {"in": "входящий", "out": "исходящий"}


# ---------------------------------------------------------------- данные результата


@dataclass
class Usage:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: LLMResponse) -> None:
        self.model = r.model or self.model
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd


@dataclass
class Classification:
    call_type: str
    admin_name: str | None
    reason: str


@dataclass
class CriterionVerdict:
    id: str
    result: str
    evidence: str = ""
    comment: str = ""


@dataclass
class Improvement:
    criterion_id: str = ""
    issue: str = ""
    recommendation: str = ""
    example: str = ""


@dataclass
class Evaluation:
    call_type: str
    verdicts: dict[str, CriterionVerdict]
    summary: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[Improvement] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "call_type": self.call_type,
                "verdicts": [asdict(v) for v in self.verdicts.values()],
                "summary": self.summary,
                "strengths": self.strengths,
                "improvements": [asdict(i) for i in self.improvements],
                "warnings": self.warnings,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> "Evaluation":
        data = json.loads(raw)
        return cls(
            call_type=data.get("call_type", ""),
            verdicts={v["id"]: CriterionVerdict(**v) for v in data.get("verdicts", [])},
            summary=data.get("summary", ""),
            strengths=list(data.get("strengths", [])),
            improvements=[Improvement(**i) for i in data.get("improvements", [])],
            warnings=list(data.get("warnings", [])),
        )


# ---------------------------------------------------------------- разбор ответа модели


def extract_json(text: str) -> dict:
    """Достаёт JSON-объект из ответа модели (с учётом возможных ```json-обёрток и текста вокруг)."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе модели нет JSON-объекта")
    data = json.loads(cleaned[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("ответ модели — не JSON-объект")
    return data


_RESULT_ALIASES = {
    YES: YES, "да": YES, "y": YES, "true": YES, "1": YES, "выполнено": YES,
    NO: NO, "нет": NO, "n": NO, "false": NO, "0": NO, "не выполнено": NO, "partial": NO, "частично": NO,
    NA: NA, "n/a": NA, "не применимо": NA, "неприменимо": NA, "none": NA, "null": NA, "": NA,
}


def normalize_result(value) -> str | None:
    if value is None:
        return NA
    if isinstance(value, bool):
        return YES if value else NO
    return _RESULT_ALIASES.get(str(value).strip().lower())


def _clip(value, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---------------------------------------------------------------- тексты запросов

_CLASSIFY_RULES = """Ты — помощник отдела контроля качества медицинской клиники.
По автоматической расшифровке телефонного разговора определи его тип.

{clinic_context}

Типы разговоров:
{types}
- {not_target}: не оцениваем — автоответчик или голосовая почта, не дозвонились, ошибка номера, разговор не с пациентом (поставщик, коллега, реклама, внутренний звонок), разговор оборвался в первые секунды без сути.

Ответь ТОЛЬКО JSON-объектом без пояснений и markdown:
{{"call_type": "<код типа>", "admin_name": "<имя, которым представился администратор клиники, или null>", "reason": "<одно предложение: почему такой тип>"}}"""

_EVAL_RULES = """Ты — опытный специалист по контролю качества телефонных разговоров в медицинской клинике.
Тебе дают автоматическую расшифровку разговора администратора клиники с пациентом и чек-лист.
Оцени работу АДМИНИСТРАТОРА (не пациента) по каждому пункту чек-листа.

Правила оценки:
1. Для каждого пункта поставь ровно одно значение result:
   "yes" — требование выполнено полностью;
   "no" — требование не выполнено ИЛИ выполнено не в полной мере (по правилам клиники частичное выполнение считается невыполнением);
   "na" — пункт не применим к этому разговору: условие пункта не наступило (см. подсказку пункта) или разговор по независящей от администратора причине не дошёл до этого этапа (пациенту неудобно говорить и договорились о перезвоне, обрыв связи, пациент сам прервал разговор).
2. Если администратор сам пропустил этап, который должен был пройти, — это "no", а не "na".
3. Пункты-запреты («не перебивает», «не использует…») — "yes", если нарушений нет.
4. evidence — точная короткая цитата из расшифровки, подтверждающая оценку (до 200 символов). Если нужной фразы в разговоре нет — пустая строка.
5. comment — одно короткое предложение: почему такая оценка; для "no" — чего именно не хватило.
6. Расшифровка сделана автоматически и может содержать ошибки распознавания: искажённые слова и имена, слитые реплики. Оценивай по смыслу и не снижай оценку за очевидные ошибки распознавания.
7. Если в расшифровке нет разметки ролей, определи по смыслу, какие фразы произносит администратор (представляется от имени клиники, предлагает запись, называет цены), а какие — пациент.
8. Не придумывай того, чего нет в расшифровке.

Обратная связь администратору — на «вы», доброжелательно и конкретно:
- summary — 2–3 предложения: общее впечатление от разговора и главный вывод;
- strengths — 2–4 сильные стороны с опорой на конкретные моменты разговора;
- improvements — до 3 самых важных зон роста по убыванию значимости. Для каждой: criterion_id — код пункта, issue — что было не так, recommendation — что делать, example — готовая фраза, которую стоило сказать в этой ситуации.

Ответь ТОЛЬКО JSON-объектом без markdown и пояснений, строго такого вида:
{{"criteria": [{{"id": "G01", "result": "yes", "evidence": "цитата", "comment": "пояснение"}}],
 "summary": "...", "strengths": ["..."],
 "improvements": [{{"criterion_id": "IN11", "issue": "...", "recommendation": "...", "example": "..."}}]}}
В массиве criteria должны быть ВСЕ пункты чек-листа в том же порядке и с теми же кодами.

Сведения о клинике:
{clinic_context}"""


def _format_checklist(checklist: Checklist, weight_labels: dict[str, str]) -> str:
    lines = []
    for c in checklist.evaluable:
        label = weight_labels.get(c.weight_key, c.weight_key)
        lines.append(f"{c.id} [значимость: {label}] {c.text}")
        if c.hint:
            lines.append(f"    Подсказка: {c.hint}")
    return "\n".join(lines)


def _call_facts(transcript: Transcript, direction: str | None, started_at: datetime | None) -> str:
    duration = int(transcript.duration or 0)
    facts = [
        f"- Направление звонка: {DIRECTION_LABELS.get(direction or '', 'неизвестно')}",
        f"- Дата и время начала: {started_at.strftime('%d.%m.%Y %H:%M') if started_at else 'неизвестно'}",
        f"- Длительность: {duration // 60} мин {duration % 60} с" if duration else "- Длительность: неизвестна",
    ]
    if transcript.roles_known:
        facts.append("- Роли в расшифровке размечены (запись в стерео).")
    elif transcript.has_speakers:
        facts.append("- Реплики разделены на двух собеседников, но кто из них администратор — определи по смыслу.")
    else:
        facts.append("- Разметки ролей нет — определи по смыслу, кто говорит.")
    return "\n".join(facts)


# ---------------------------------------------------------------- оценщик


class Evaluator:
    def __init__(self, checklists: ChecklistSet, llm: LLMClient):
        self.checklists = checklists
        self.llm = llm

    def _ask_json(self, system_parts: list[str], user: str, usage: Usage, max_tokens: int | None = None) -> dict:
        """Запрос к модели с одной повторной попыткой, если ответ не разобрался как JSON."""
        last_error = ""
        for _ in range(2):
            response = self.llm.complete(system_parts, user, max_tokens=max_tokens)
            usage.add(response)
            try:
                return extract_json(response.text)
            except ValueError as e:  # json.JSONDecodeError — подкласс ValueError
                last_error = str(e)
                log.warning("Не удалось разобрать ответ модели: %s", e)
        raise LLMError(f"Модель вернула ответ не в формате JSON ({last_error})")

    def classify(
        self, transcript: Transcript, direction: str | None, started_at: datetime | None, usage: Usage
    ) -> Classification:
        allowed = [
            key for key, s in self.checklists.scenarios.items() if not direction or s.direction in ("", direction)
        ]
        types = "\n".join(
            f"- {key}: {self.checklists.scenarios[key].title}. {self.checklists.scenarios[key].description}"
            for key in allowed
        )
        system = _CLASSIFY_RULES.format(
            clinic_context=self.checklists.clinic_context, types=types, not_target=NOT_TARGET
        )
        user = f"Данные звонка:\n{_call_facts(transcript, direction, started_at)}\n\nРасшифровка:\n{transcript.to_text()}"
        data = self._ask_json([system], user, usage, max_tokens=2000)

        call_type = str(data.get("call_type", "")).strip()
        if call_type not in allowed and call_type != NOT_TARGET:
            raise LLMError(f"Модель вернула неизвестный тип звонка «{call_type}»")
        admin_name = data.get("admin_name")
        admin_name = str(admin_name).strip() if admin_name and str(admin_name).lower() != "null" else None
        return Classification(call_type=call_type, admin_name=admin_name, reason=_clip(data.get("reason"), 300))

    def system_parts(self, call_type: str) -> list[str]:
        scenario = self.checklists.scenario(call_type)
        rules = _EVAL_RULES.format(clinic_context=self.checklists.clinic_context)
        checklist = (
            "Чек-лист для оценки.\n\n"
            f"## {self.checklists.general.title}\n"
            f"{_format_checklist(self.checklists.general, self.checklists.weight_labels)}\n\n"
            f"## Сценарий: {scenario.title}\n"
            f"{_format_checklist(scenario, self.checklists.weight_labels)}"
        )
        return [rules, checklist]

    def evaluate(
        self,
        transcript: Transcript,
        call_type: str,
        direction: str | None,
        started_at: datetime | None,
        usage: Usage,
    ) -> Evaluation:
        scenario = self.checklists.scenario(call_type)
        expected = [c.id for c in self.checklists.general.evaluable] + [c.id for c in scenario.evaluable]

        user = (
            f"Сценарий звонка: {scenario.title}\n\n"
            f"Данные звонка:\n{_call_facts(transcript, direction, started_at)}\n\n"
            f"Расшифровка:\n{transcript.to_text()}"
        )
        data = self._ask_json(self.system_parts(call_type), user, usage)
        return self._build(call_type, expected, data)

    @staticmethod
    def _build(call_type: str, expected: list[str], data: dict) -> Evaluation:
        warnings: list[str] = []
        raw_items = data.get("criteria") or []
        by_id: dict[str, dict] = {}
        for item in raw_items:
            if isinstance(item, dict) and item.get("id"):
                by_id[str(item["id"]).strip().upper()] = item

        verdicts: dict[str, CriterionVerdict] = {}
        missing, invalid = [], []
        for cid in expected:
            item = by_id.get(cid.upper())
            if item is None:
                missing.append(cid)
                verdicts[cid] = CriterionVerdict(cid, NA, comment="ИИ не дал оценку по этому пункту")
                continue
            result = normalize_result(item.get("result"))
            if result is None:
                invalid.append(cid)
                result = NA
            verdicts[cid] = CriterionVerdict(
                cid, result, evidence=_clip(item.get("evidence"), 400), comment=_clip(item.get("comment"), 400)
            )
        if missing:
            warnings.append(f"ИИ не оценил пункты: {', '.join(missing)} (учтены как «не применимо»)")
        if invalid:
            warnings.append(f"Непонятная оценка по пунктам: {', '.join(invalid)} (учтены как «не применимо»)")

        improvements = []
        for item in data.get("improvements") or []:
            if isinstance(item, dict):
                improvements.append(
                    Improvement(
                        criterion_id=_clip(item.get("criterion_id"), 20),
                        issue=_clip(item.get("issue"), 500),
                        recommendation=_clip(item.get("recommendation"), 500),
                        example=_clip(item.get("example"), 500),
                    )
                )
        strengths = [_clip(s, 400) for s in (data.get("strengths") or []) if str(s).strip()]
        return Evaluation(
            call_type=call_type,
            verdicts=verdicts,
            summary=_clip(data.get("summary"), 1200),
            strengths=strengths[:5],
            improvements=improvements[:3],
            warnings=warnings,
        )
