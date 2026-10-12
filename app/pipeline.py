"""Обработка звонков: расшифровка → определение типа → оценка → сохранение. Фоновая очередь."""

from __future__ import annotations

import logging
import threading
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from . import db
from .checklists import GENERAL_KEY, NOT_TARGET, ChecklistSet, load_clinic_checklists
from .config import Settings
from .evaluation import Evaluation, Evaluator, Usage
from .llm import LLMClient, LLMError
from .scoring import weighted_score
from .transcription import Transcriber, Transcript, apply_admin_speaker

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


class Superseded(Exception):
    """Звонок изменили, вернули в очередь или удалили, пока он обрабатывался: результат не нужен."""


def _claim(s: Session, call_id: int, expected: str | tuple[str, ...], **values) -> None:
    """Меняет звонок, только если его статус всё ещё expected, иначе — Superseded.

    В SQLite этот UPDATE сразу берёт блокировку записи, поэтому до commit звонок никто не изменит.
    Так фоновая обработка не затирает действия пользователя (переоценку, смену типа, удаление).
    """
    expected = (expected,) if isinstance(expected, str) else tuple(expected)
    res = s.execute(
        update(db.Call).where(db.Call.id == call_id, db.Call.status.in_(expected)).values(**values),
        execution_options={"synchronize_session": False},
    )
    if res.rowcount != 1:
        s.rollback()
        raise Superseded(call_id)


class Processor:
    """Обрабатывает звонок по чек-листу и настройкам его клиники.

    Долгие шаги (расшифровка, запросы к ИИ) идут без открытой транзакции: база в это время свободна
    для веб-интерфейса и загрузки из Mango. Результат записывается короткой транзакцией в конце.
    """

    def __init__(
        self,
        settings: Settings,
        session_factory: sessionmaker,
        transcriber: Transcriber,
        llm: LLMClient,
    ):
        self.settings = settings
        self.session_factory = session_factory
        self.transcriber = transcriber
        self.llm = llm

    def _set_status(self, call_id: int, status: str, message: str = "") -> None:
        """Статус по итогам обработки — если звонок за это время не вернули в очередь."""
        with self.session_factory() as s:
            s.execute(
                update(db.Call)
                .where(db.Call.id == call_id, db.Call.status.in_(db.IN_PROGRESS))
                .values(status=status, status_message=message)
            )
            s.commit()

    def process(self, call_id: int) -> None:
        try:
            self._process(call_id)
        except Superseded:
            log.info("Звонок %s изменили во время обработки — результат отброшен", call_id)
        except Exception as e:  # ошибку показываем в карточке звонка, очередь продолжает работу
            log.exception("Ошибка обработки звонка %s", call_id)
            try:
                self._set_status(call_id, db.ERROR, f"{type(e).__name__}: {e}")
            except Exception:
                log.exception("Не удалось записать ошибку звонка %s", call_id)

    def _process(self, call_id: int) -> None:
        # 1. берём звонок из очереди
        with self.session_factory() as s:
            call = s.get(db.Call, call_id)
            if call is None:
                return
            need_transcript = call.transcript_json is None
            if need_transcript and not call.audio_path:
                raise RuntimeError("У звонка нет ни записи, ни расшифровки")
            _claim(s, call_id, db.QUEUED, status=db.TRANSCRIBING if need_transcript else db.EVALUATING, status_message="")
            s.commit()
            audio_path = self.settings.audio_dir / call.audio_path if call.audio_path else None
            transcript_json = call.transcript_json

        # 2. расшифровка (минуты) — без транзакции
        if need_transcript:
            transcript = self.transcriber.transcribe(audio_path)
            with self.session_factory() as s:
                values = {"status": db.EVALUATING, "transcript_json": transcript.to_json()}
                if transcript.duration:
                    values["duration_sec"] = transcript.duration
                _claim(s, call_id, db.TRANSCRIBING, **values)
                s.commit()
        else:
            transcript = Transcript.from_json(transcript_json)

        # 3. настройки клиники и данные звонка на момент оценки
        with self.session_factory() as s:
            call = s.get(db.Call, call_id)
            if call is None:
                raise Superseded(call_id)
            clinic = s.get(db.Clinic, call.clinic_id)
            min_seconds = clinic.min_call_seconds if clinic.min_call_seconds is not None else self.settings.min_call_seconds
            checklists = load_clinic_checklists(s, call.clinic_id)
            source, direction, started_at = call.source, call.direction, call.started_at
            manual_type = call.call_type if call.call_type_manual and call.call_type else None
            model_override = call.llm_model_override
            employee_known = call.employee_id is not None

        if source != "text" and transcript.duration and transcript.duration < min_seconds:
            return self._finish(call_id, skip_reason=f"Слишком короткий звонок (меньше {min_seconds} с)")
        if transcript.is_empty:
            return self._finish(call_id, skip_reason="В записи не распознана речь")
        if manual_type and manual_type not in checklists.scenarios:
            raise RuntimeError("Выбранного вручную сценария больше нет в чек-листе клиники — выберите другой тип")

        # 4. запросы к ИИ — без транзакции. При ручном типе классификация нужна, только если неизвестно,
        # где реплики администратора (запись «в одну дорожку»), или кто администратор.
        usage = Usage()
        evaluator = Evaluator(checklists, self.llm, self.settings.llm_classify_model, self.settings.llm_model)
        if manual_type is None:
            cls = evaluator.classify(transcript, direction, started_at, usage)
        elif transcript.roles_known and employee_known:
            cls = None
        else:
            try:
                cls = evaluator.classify(transcript, direction, started_at, usage, strict=False)
            except LLMError as e:  # при ручном типе без классификации можно обойтись
                log.warning("Звонок %s: не удалось определить администратора: %s", call_id, e)
                cls = None
        speakers_changed = bool(cls) and apply_admin_speaker(transcript, cls.admin_speaker)
        if manual_type is None and cls.call_type == NOT_TARGET:
            return self._finish(
                call_id, cls=cls, usage=usage, transcript=transcript if speakers_changed else None,
                skip_reason=f"Нецелевой звонок: {cls.reason}",
            )
        call_type = manual_type or cls.call_type
        evaluation = evaluator.evaluate(transcript, call_type, direction, started_at, usage, model=model_override)
        self._finish(
            call_id, cls=cls, usage=usage, transcript=transcript if speakers_changed else None,
            evaluation=evaluation, checklists=checklists, manual=manual_type is not None,
        )

    def _finish(
        self,
        call_id: int,
        *,
        cls=None,
        usage: Usage | None = None,
        transcript: Transcript | None = None,
        evaluation: Evaluation | None = None,
        checklists: ChecklistSet | None = None,
        manual: bool = False,
        skip_reason: str = "",
    ) -> None:
        """Записывает итог одной короткой транзакцией, если звонок всё ещё обрабатывается."""
        with self.session_factory() as s:
            _claim(s, call_id, db.EVALUATING, status=db.EVALUATING)
            call = s.get(db.Call, call_id)
            if cls is not None:
                if not manual:
                    call.call_type = cls.call_type
                    call.classification_reason = cls.reason
                if call.employee_id is None:
                    employee = match_employee(s, call.clinic_id, cls.admin_name)
                    if employee is not None:
                        call.employee_id = employee.id
            if transcript is not None:
                call.transcript_json = transcript.to_json()
            if usage is not None:
                self._save_usage(call, usage)
            if skip_reason:
                return self._skip(s, call, skip_reason)
            self._save_evaluation(call, evaluation, checklists)
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

    @staticmethod
    def _save_evaluation(call: db.Call, evaluation: Evaluation, checklists: ChecklistSet) -> None:
        general = checklists.general
        scenario = checklists.scenario(evaluation.call_type)
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
                        criterion_num=c.num,
                        criterion_text=c.text,
                        checklist_title=checklist.title,
                        weight_key=c.weight_key,
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
        self._last_clinic = 0

    def stop(self) -> None:
        self._stop_event.set()
        self._wake.set()

    def wake(self) -> None:
        self._wake.set()

    def _next_call_id(self) -> int | None:
        """Следующий звонок: сначала загруженные вручную, затем по очереди из каждой клиники
        (большая загрузка из Mango одной клиники не задерживает остальные). Звонки отключённых клиник ждут."""
        with self.processor.session_factory() as s:
            queued = (
                select(db.Call.id, db.Call.clinic_id)
                .join(db.Clinic, db.Clinic.id == db.Call.clinic_id)
                .where(db.Call.status == db.QUEUED, db.Clinic.active.is_(True))
            )
            row = s.execute(queued.where(db.Call.source != "mango").order_by(db.Call.id).limit(1)).first()
            if row is None:
                row = s.execute(
                    queued.where(db.Call.clinic_id > self._last_clinic).order_by(db.Call.clinic_id, db.Call.id).limit(1)
                ).first() or s.execute(queued.order_by(db.Call.clinic_id, db.Call.id).limit(1)).first()
            if row is None:
                return None
            self._last_clinic = row.clinic_id
            return row.id

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
            try:
                self.processor.process(call_id)
            except Exception:  # поток очереди не должен останавливаться ни при какой ошибке
                log.exception("Сбой обработки звонка %s", call_id)
                self._wake.wait(self.poll_sec)


def requeue_interrupted(session_factory: sessionmaker) -> None:
    """После перезапуска возвращает в очередь звонки, обработка которых была прервана."""
    with session_factory() as s:
        s.execute(update(db.Call).where(db.Call.status.in_(db.IN_PROGRESS)).values(status=db.QUEUED))
        s.commit()
