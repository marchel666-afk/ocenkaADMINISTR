"""Обработка звонков: расшифровка → определение типа → оценка → сохранение. Фоновая очередь."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from . import db
from .checklists import GENERAL_KEY, NOT_TARGET, ChecklistSet
from .config import Settings
from .evaluation import Evaluation, Evaluator, Usage
from .scoring import weighted_score
from .transcription import Transcriber, Transcript

log = logging.getLogger(__name__)


def match_employee(session: Session, clinic_id: int, name: str | None) -> db.Employee | None:
    """Находит сотрудника клиники по имени, которым администратор представился в разговоре."""
    def words(text: str) -> set[str]:
        return {w for w in text.lower().replace("ё", "е").split() if len(w) > 1}

    wanted = words(name or "")
    if not wanted:
        return None
    employees = session.scalars(
        select(db.Employee).where(db.Employee.clinic_id == clinic_id, db.Employee.active.is_(True))
    ).all()
    matches = [e for e in employees if wanted & words(e.name)]
    return matches[0] if len(matches) == 1 else None


class Processor:
    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        checklists: ChecklistSet,
        transcriber: Transcriber,
        evaluator: Evaluator,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.checklists = checklists
        self.transcriber = transcriber
        self.evaluator = evaluator

    def _set_status(self, call_id: int, status: str, message: str = "") -> None:
        with self.session_factory() as s:
            s.execute(update(db.Call).where(db.Call.id == call_id).values(status=status, status_message=message))
            s.commit()

    def process(self, call_id: int) -> None:
        try:
            self._process(call_id)
        except Exception as e:  # ошибку показываем в карточке звонка, очередь продолжает работу
            log.exception("Ошибка обработки звонка %s", call_id)
            self._set_status(call_id, db.ERROR, f"{type(e).__name__}: {e}")

    def _process(self, call_id: int) -> None:
        with self.session_factory() as s:
            call = s.get(db.Call, call_id)
            if call is None:
                return

            if call.transcript_json is None:
                if not call.audio_path:
                    raise RuntimeError("У звонка нет ни записи, ни расшифровки")
                call.status = db.TRANSCRIBING
                s.commit()
                transcript = self.transcriber.transcribe(self.settings.audio_dir / call.audio_path)
                call.transcript_json = transcript.to_json()
                if transcript.duration:
                    call.duration_sec = transcript.duration
                s.commit()
            transcript = Transcript.from_json(call.transcript_json)

            if call.source != "text" and transcript.duration and transcript.duration < self.settings.min_call_seconds:
                return self._skip(s, call, f"Слишком короткий звонок (меньше {self.settings.min_call_seconds} с)")
            if transcript.is_empty:
                return self._skip(s, call, "В записи не распознана речь")

            call.status = db.EVALUATING
            s.commit()
            usage = Usage()

            if not call.call_type_manual or not call.call_type:
                cls = self.evaluator.classify(transcript, call.direction, call.started_at, usage)
                call.call_type = cls.call_type
                call.classification_reason = cls.reason
                if call.employee_id is None:
                    employee = match_employee(s, call.clinic_id, cls.admin_name)
                    if employee is not None:
                        call.employee_id = employee.id
                if cls.call_type == NOT_TARGET:
                    self._save_usage(call, usage)
                    return self._skip(s, call, f"Нецелевой звонок: {cls.reason}")

            evaluation = self.evaluator.evaluate(transcript, call.call_type, call.direction, call.started_at, usage)
            self._save_evaluation(call, evaluation)
            self._save_usage(call, usage)
            call.status = db.DONE
            call.status_message = "; ".join(evaluation.warnings)
            call.processed_at = datetime.now()
            s.commit()

    def _skip(self, s: Session, call: db.Call, reason: str) -> None:
        call.status = db.SKIPPED
        call.status_message = reason
        call.evaluation_json = None
        call.score_total = call.score_general = call.score_scenario = None
        call.results.clear()
        call.processed_at = datetime.now()
        s.commit()

    @staticmethod
    def _save_usage(call: db.Call, usage: Usage) -> None:
        call.llm_model = usage.model
        call.llm_tokens = (call.llm_tokens or 0) + usage.prompt_tokens + usage.completion_tokens
        call.llm_cost_usd = (call.llm_cost_usd or 0.0) + usage.cost_usd

    def _save_evaluation(self, call: db.Call, evaluation: Evaluation) -> None:
        general = self.checklists.general
        scenario = self.checklists.scenario(evaluation.call_type)
        results = {cid: v.result for cid, v in evaluation.verdicts.items()}

        call.evaluation_json = evaluation.to_json()
        call.score_general = weighted_score(general.criteria, results)
        call.score_scenario = weighted_score(scenario.criteria, results)
        call.score_total = weighted_score(general.criteria + scenario.criteria, results)

        call.results.clear()
        for key, checklist in ((GENERAL_KEY, general), (scenario.key, scenario)):
            for c in checklist.evaluable:
                v = evaluation.verdicts[c.id]
                call.results.append(
                    db.CriterionResult(
                        checklist=key,
                        criterion_id=c.id,
                        result=v.result,
                        weight=c.weight,
                        evidence=v.evidence,
                        comment=v.comment,
                    )
                )


class Worker(threading.Thread):
    """Фоновый поток: по одному берёт звонки из очереди и обрабатывает."""

    def __init__(self, processor: Processor, poll_sec: float = 3.0):
        super().__init__(name="call-worker", daemon=True)
        self.processor = processor
        self.poll_sec = poll_sec
        self._stop_event = threading.Event()
        self._wake = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def _next_call_id(self) -> int | None:
        with self.processor.session_factory() as s:
            return s.scalars(
                select(db.Call.id).where(db.Call.status == db.QUEUED).order_by(db.Call.id).limit(1)
            ).first()

    def run(self) -> None:
        while not self._stop_event.is_set():
            call_id = None
            try:
                call_id = self._next_call_id()
            except Exception:
                log.exception("Ошибка чтения очереди")
            if call_id is None:
                self._wake.wait(self.poll_sec)
                self._wake.clear()
                continue
            self.processor.process(call_id)


def requeue_interrupted(session_factory: sessionmaker) -> None:
    """После перезапуска возвращает в очередь звонки, обработка которых была прервана."""
    with session_factory() as s:
        s.execute(update(db.Call).where(db.Call.status.in_(db.IN_PROGRESS)).values(status=db.QUEUED))
        s.commit()
