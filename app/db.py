"""Хранилище: SQLite через SQLAlchemy (при росте до сети клиник легко переводится на PostgreSQL)."""

from __future__ import annotations

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
    select,
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


class Clinic(Base):
    __tablename__ = "clinics"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))


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
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    clinic_id: Mapped[int] = mapped_column(ForeignKey("clinics.id"))
    employee_id: Mapped[int | None] = mapped_column(ForeignKey("employees.id"), default=None)
    source: Mapped[str] = mapped_column(String(20), default="upload")  # upload / text / mango
    external_id: Mapped[str | None] = mapped_column(String(200), default=None)
    original_filename: Mapped[str] = mapped_column(String(500), default="")
    audio_path: Mapped[str | None] = mapped_column(String(500), default=None)  # относительно data/audio
    direction: Mapped[str | None] = mapped_column(String(10), default=None)  # in / out
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

    call: Mapped[Call] = relationship(back_populates="results")


@event.listens_for(Engine, "connect")
def _sqlite_pragmas(dbapi_connection, _record):
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()


def make_session_factory(db_url: str) -> sessionmaker:
    connect_args = {"check_same_thread": False, "timeout": 30} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def ensure_default_clinic(session_factory: sessionmaker, name: str) -> int:
    with session_factory() as s:
        clinic = s.scalars(select(Clinic).order_by(Clinic.id)).first()
        if clinic is None:
            clinic = Clinic(name=name)
            s.add(clinic)
            s.commit()
        return clinic.id
