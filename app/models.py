"""Persistence schema.

Three tables, each with a distinct job:

  campaigns          the enriched, queryable output — one row per *unique* campaign id
  ingest_runs        one row per POST /campaigns/ingest, for auditability
  ingest_row_results one row per *input* row, including the ones we rejected

Keeping rejected rows out of `campaigns` but inside `ingest_row_results` is the
core modelling decision: the campaigns table stays clean enough to trust in a
query, while nothing from the raw export is silently lost.
"""
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Campaign(Base):
    __tablename__ = "campaigns"
    __table_args__ = (
        # The natural key from the export. Makes re-ingest an UPSERT, not a duplicate.
        UniqueConstraint("external_id", name="uq_campaigns_external_id"),
        Index("ix_campaigns_channel", "channel"),
        Index("ix_campaigns_health_score", "health_score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[str] = mapped_column(String(64), nullable=False)

    # ---- cleaned source fields -------------------------------------------------
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    raw_channel: Mapped[str | None] = mapped_column(String(128))
    currency: Mapped[str] = mapped_column(String(8), default="INR")

    spend: Mapped[float | None] = mapped_column(Float)
    impressions: Mapped[int | None] = mapped_column(BigInteger)
    clicks: Mapped[int | None] = mapped_column(BigInteger)
    conversions: Mapped[int | None] = mapped_column(BigInteger)
    revenue: Mapped[float | None] = mapped_column(Float)

    # ---- derived metrics (computed in code, stored so filters stay cheap) -------
    ctr: Mapped[float | None] = mapped_column(Float)
    cvr: Mapped[float | None] = mapped_column(Float)
    cpa: Mapped[float | None] = mapped_column(Float)
    roas: Mapped[float | None] = mapped_column(Float)

    # ---- LLM enrichment --------------------------------------------------------
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    objective: Mapped[str] = mapped_column(String(32), nullable=False)
    objective_rationale: Mapped[str | None] = mapped_column(Text)
    health_score: Mapped[int] = mapped_column(Integer, nullable=False)
    health_rationale: Mapped[str | None] = mapped_column(Text)
    enrichment_source: Mapped[str] = mapped_column(String(16), nullable=False)  # llm | fallback
    enrichment_error: Mapped[str | None] = mapped_column(Text)

    # ---- self-check (stretch: lightweight eval) --------------------------------
    channel_check: Mapped[str | None] = mapped_column(String(16))      # agree | disagree
    channel_expected: Mapped[str | None] = mapped_column(String(32))
    health_check: Mapped[str | None] = mapped_column(String(16))       # agree | disagree
    health_score_rule_based: Mapped[int | None] = mapped_column(Integer)

    # ---- data-quality trail ----------------------------------------------------
    flags: Mapped[list] = mapped_column(JSON, default=list)

    first_ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class IngestRun(Base):
    __tablename__ = "ingest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(64))
    llm_enabled: Mapped[bool] = mapped_column(default=False)

    total_rows: Mapped[int] = mapped_column(Integer, default=0)
    ingested: Mapped[int] = mapped_column(Integer, default=0)
    rejected: Mapped[int] = mapped_column(Integer, default=0)
    enriched_by_llm: Mapped[int] = mapped_column(Integer, default=0)
    enriched_by_fallback: Mapped[int] = mapped_column(Integer, default=0)

    rows: Mapped[list["IngestRowResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class IngestRowResult(Base):
    """One row per *input* record — the per-row success/failure summary the brief asks for."""

    __tablename__ = "ingest_row_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("ingest_runs.id", ondelete="CASCADE"))
    row_index: Mapped[int] = mapped_column(Integer)
    external_id: Mapped[str | None] = mapped_column(String(64))

    status: Mapped[str] = mapped_column(String(16))        # ingested | rejected
    action: Mapped[str | None] = mapped_column(String(16)) # created | updated | skipped
    reason: Mapped[str | None] = mapped_column(Text)
    flags: Mapped[list] = mapped_column(JSON, default=list)
    enrichment_source: Mapped[str | None] = mapped_column(String(16))

    run: Mapped[IngestRun] = relationship(back_populates="rows")
