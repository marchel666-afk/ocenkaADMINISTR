"""Хранилище: SQLite через SQLAlchemy (при росте до сети клиник легко переводится на PostgreSQL)."""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    event,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

# Статусы обработки звонка
QUEUED = "queued"
TRANSCRIBING = "transcribing"
EVALUATING = "evaluating"
DONE = "done"
SKIPPED = "skipped"
ERROR = "error"
IN_PROGRESS = (TRANSCRIBING, EVALUATING)

STATUS_LABELS = {
    QUEUED: "В очереди",
    TRANSCRIBING: "Расшифровка",
    EVALUATING: "Оценка",
    DONE: "Готово",
    SKIPPED: "Не оценивается",
    ERROR: "Ошибка",
}


class Base(DeclarativeBase):
    pass


DEFAULT_WEIGHTS = '{"high": 3, "medium": 2, "low": 1}'

ADMIN, MANAGER = "admin", "manager"
ROLE_LABELS = {ADMIN: "Администратор сети", MANAGER: "Руководитель клиники"}


class Clinic(Base):
    """Клиника или колл-центр: свои сотрудники, звонки, критерии оценки и подключение к Mango."""

    __tablename__ = "clinics"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    ai_context: Mapped[str] = mapped_column(Text, default="")  # сведения о клинике для ИИ
    weights_json: Mapped[str] = mapped_column(Text, default=DEFAULT_WEIGHTS)  # баллы за значимость пунктов
    utc_offset: Mapped[int] = mapped_column(Integer, default=3)  # местное время клиники, часов от UTC
    min_call_seconds: Mapped[int | None] = mapped_column(Integer, default=None)  # None — общая настройка

    # Mango Office
    mango_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mango_api_key: Mapped[str] = mapped_column(String(200), default="")
    mango_api_salt: Mapped[str] = mapped_column(String(200), default="")
    mango_utc_offset: Mapped[int] = mapped_column(Integer, default=3)  # часовой пояс аккаунта Mango (имена файлов)
    mango_interval_min: Mapped[int] = mapped_column(Integer, default=30)
    mango_lines: Mapped[str] = mapped_column(Text, default="")  # какие добавочные/линии забирать; пусто — все
    mango_synced_until: Mapped[int | None] = mapped_column(Integer, default=None)  # unix-время, до которого забрано
    mango_last_run: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    mango_last_count: Mapped[int] = mapped_column(Integer, default=0)
    mango_last_error: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    @property
    def weights(self) -> dict[str, float]:
        try:
            data = json.loads(self.weights_json or DEFAULT_WEIGHTS)
            return {k: float(data[k]) for k in ("high", "medium", "low")}
        except (ValueError, KeyError, TypeError):
            return {k: float(v) for k, v in json.loads(DEFAULT_WEIGHTS).items()}


class Scenario(Base):
    """Раздел чек-листа клиники: «Общие правила» (key=general) или сценарий звонка."""

    __tablename__ = "scenarios"
    __table_args__ = (Index("ix_scenarios_clinic", "clinic_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"))
    key: Mapped[str] = mapped_column(String(50))
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    direction: Mapped[str] = mapped_column(String(10), default="")  # in / out / "" (общие правила)
    prefix: Mapped[str] = mapped_column(String(10), default="")  # начало кодов пунктов: G, IN, RQ…
    position: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    criteria: Mapped[list["CriterionDef"]] = relationship(
        back_populates="scenario", order_by="CriterionDef.position"
    )


class CriterionDef(Base):
    """Пункт чек-листа клиники. Удалённые пункты не стираются (active=False), чтобы не терять историю."""

    __tablename__ = "criteria"
    __table_args__ = (Index("ix_criteria_clinic", "clinic_id"), Index("ix_criteria_scenario", "scenario_id"))

    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"))
    scenario_id: Mapped[int] = mapped_column(ForeignKey("scenarios.id"))
    code: Mapped[str] = mapped_column(String(20))
    num: Mapped[str] = mapped_column(String(20), default="")
    text: Mapped[str] = mapped_column(Text)
    weight_key: Mapped[str] = mapped_column(String(10), default="medium")
    evaluable: Mapped[bool] = mapped_column(Boolean, default=True)
    hint: Mapped[str] = mapped_column(Text, default="")
    position: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)

    scenario: Mapped[Scenario] = relationship(back_populates="criteria")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(100), unique=True)
    password_hash: Mapped[str] = mapped_column(String(300))
    role: Mapped[str] = mapped_column(String(20), default=MANAGER)  # admin / manager
    clinic_id: Mapped[int | None] = mapped_column(ForeignKey("clinics.id"), default=None)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    clinic: Mapped[Clinic | None] = relationship()


class Employee(Base):
    __tablename__ = "employees"

    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"))
    name: Mapped[str] = mapped_column(String(200))
    mango_id: Mapped[str | None] = mapped_column(String(100), default=None)  # для интеграции с Mango
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    clinic: Mapped[Clinic] = relationship()


class Call(Base):
    __tablename__ = "calls"
    __table_args__ = (
        Index("ix_calls_status", "status"),
        Index("ix_calls_employee", "employee_id"),
        Index("ix_calls_started", "started_at"),
        Index("ix_calls_clinic", "clinic_id"),
        Index("ix_calls_external", "clinic_id", "source", "external_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"))
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), default=None)
    source: Mapped[str] = mapped_column(String(20), default="upload")  # upload / text / mango
    external_id: Mapped[str | None] = mapped_column(String(200), default=None)
    original_filename: Mapped[str] = mapped_column(String(500), default="")
    audio_path: Mapped[str | None] = mapped_column(String(500), default=None)  # относительно data/audio
    direction: Mapped[str | None] = mapped_column(String(10), default=None)  # in / out
    phone: Mapped[str | None] = mapped_column(String(30), default=None)  # номер собеседника (из Mango)
    mango_line: Mapped[str | None] = mapped_column(String(200), default=None)  # линия/сотрудник в Mango
    call_type: Mapped[str | None] = mapped_column(String(50), default=None)
    call_type_manual: Mapped[bool] = mapped_column(Boolean, default=False)
    classification_reason: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    duration_sec: Mapped[float | None] = mapped_column(Float, default=None)

    status: Mapped[str] = mapped_column(String(20), default=QUEUED)
    status_message: Mapped[str] = mapped_column(Text, default="")
    transcript_json: Mapped[str | None] = mapped_column(Text, default=None)
    evaluation_json: Mapped[str | None] = mapped_column(Text, default=None)

    score_total: Mapped[float | None] = mapped_column(Float, default=None)
    score_general: Mapped[float | None] = mapped_column(Float, default=None)
    score_scenario: Mapped[float | None] = mapped_column(Float, default=None)

    llm_model: Mapped[str] = mapped_column(String(200), default="")
    llm_model_override: Mapped[str | None] = mapped_column(String(200), default=None)  # оценка другой моделью
    llm_tokens: Mapped[int] = mapped_column(Integer, default=0)
    llm_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime, default=None)

    clinic: Mapped[Clinic] = relationship()
    employee: Mapped[Employee | None] = relationship()
    results: Mapped[list["CriterionResult"]] = relationship(
        back_populates="call", cascade="all, delete-orphan", order_by="CriterionResult.id"
    )

    @property
    def when(self) -> datetime:
        return self.started_at or self.created_at


class CriterionResult(Base):
    __tablename__ = "criterion_results"
    __table_args__ = (Index("ix_results_call", "call_id"), Index("ix_results_criterion", "criterion_id"))

    id: Mapped[int] = mapped_column(primary_key=True)
    call_id: Mapped[int] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"))
    checklist: Mapped[str] = mapped_column(String(50))  # general / ключ сценария
    criterion_id: Mapped[str] = mapped_column(String(20))
    result: Mapped[str] = mapped_column(String(5))  # yes / no / na
    weight: Mapped[float] = mapped_column(Float)
    evidence: Mapped[str] = mapped_column(Text, default="")
    comment: Mapped[str] = mapped_column(Text, default="")
    # Пункт на момент оценки — история не меняется при правке чек-листа
    criterion_num: Mapped[str] = mapped_column(String(20), default="")
    criterion_text: Mapped[str] = mapped_column(Text, default="")
    checklist_title: Mapped[str] = mapped_column(String(300), default="")
    weight_key: Mapped[str] = mapped_column(String(10), default="")

    call: Mapped[Call] = relationship(back_populates="results")


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record):
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()


def _sql_literal(value) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return None


def _add_missing_columns(engine) -> None:
    """Простая миграция: добавляет в существующие таблицы колонки, появившиеся в новых версиях.

    Для колонок со значением по умолчанию (флаги, числа, строки) существующие строки получают это значение.
    """
    insp = inspect(engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for column in table.columns:
                if column.name in existing:
                    continue
                ddl = f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {column.type.compile(dialect=engine.dialect)}'
                default = column.default.arg if column.default is not None and column.default.is_scalar else None
                literal = _sql_literal(default)
                if literal is not None:
                    ddl += f" DEFAULT {literal}"
                conn.execute(text(ddl))


def make_session_factory(db_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    _add_missing_columns(engine)
    return sessionmaker(engine, expire_on_commit=False)


def ensure_default_clinic(session_factory: sessionmaker, name: str, utc_offset: int = 3) -> int:
    """Создаёт первую клинику при первом запуске. Возвращает id первой клиники."""
    with session_factory() as s:
        clinic = s.scalars(select(Clinic).order_by(Clinic.id)).first()
        if clinic is None:
            clinic = Clinic(name=name, utc_offset=utc_offset)
            s.add(clinic)
            s.commit()
        return clinic.id
