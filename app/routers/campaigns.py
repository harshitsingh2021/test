"""Campaign ingest + retrieval endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_db
from app.ingest import RawDataError, ingest
from app.insights import generate_insights
from app.models import Campaign, IngestRowResult, IngestRun
from app.normalize import CANONICAL_CHANNELS, OBJECTIVES
from app.schemas import CampaignOut, IngestSummary, InsightsOut, RowResultOut

router = APIRouter(prefix="/campaigns", tags=["campaigns"])

SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]


class IngestRequest(BaseModel):
    """All fields optional: a bare POST ingests the configured campaigns_raw.json."""

    path: str | None = Field(
        default=None, description="Override the raw JSON file to ingest."
    )
    payload: dict[str, Any] | None = Field(
        default=None, description="Ingest this document directly instead of reading a file."
    )


@router.post(
    "/ingest",
    response_model=IngestSummary,
    summary="Ingest raw campaigns, enrich via LLM, and persist",
)
def ingest_campaigns(
    db: DbDep,
    settings: SettingsDep,
    body: IngestRequest | None = None,
) -> IngestSummary:
    body = body or IngestRequest()
    source: Path | dict[str, Any] | None = None
    if body.payload is not None:
        source = body.payload
    elif body.path:
        source = Path(body.path)
    try:
        return ingest(db, settings, source)
    except RawDataError as exc:
        # The document is unusable — distinct from a row being unusable.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc


# NOTE: the two static paths below MUST be declared before "/{external_id}",
# otherwise FastAPI matches "insights" as a campaign id.
@router.get("/insights", response_model=InsightsOut, summary="LLM portfolio observations")
def campaign_insights(db: DbDep, settings: SettingsDep) -> Any:
    return generate_insights(db, settings)


@router.get("/runs", response_model=list[dict], summary="Ingest run history")
def list_runs(db: DbDep, limit: Annotated[int, Query(ge=1, le=100)] = 20) -> Any:
    runs = db.scalars(select(IngestRun).order_by(IngestRun.id.desc()).limit(limit)).all()
    return [
        {
            "run_id": r.id,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "source": r.source,
            "model": r.model,
            "llm_enabled": r.llm_enabled,
            "total_rows": r.total_rows,
            "ingested": r.ingested,
            "rejected": r.rejected,
            "enriched_by_llm": r.enriched_by_llm,
            "enriched_by_fallback": r.enriched_by_fallback,
        }
        for r in runs
    ]


@router.get(
    "/runs/{run_id}/rows",
    response_model=list[RowResultOut],
    summary="Per-row outcome for one ingest run",
)
def run_rows(run_id: int, db: DbDep) -> Any:
    if db.get(IngestRun, run_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no ingest run {run_id}")
    return db.scalars(
        select(IngestRowResult)
        .where(IngestRowResult.run_id == run_id)
        .order_by(IngestRowResult.row_index)
    ).all()


@router.get("", response_model=list[CampaignOut], summary="List campaigns, with filters")
def list_campaigns(
    db: DbDep,
    channel: Annotated[
        str | None, Query(description=f"Canonical channel. One of: {', '.join(CANONICAL_CHANNELS)}")
    ] = None,
    objective: Annotated[
        str | None, Query(description=f"One of: {', '.join(OBJECTIVES)}")
    ] = None,
    min_health_score: Annotated[int | None, Query(ge=0, le=100)] = None,
    max_health_score: Annotated[int | None, Query(ge=0, le=100)] = None,
    min_spend: Annotated[float | None, Query(ge=0)] = None,
    flagged_only: Annotated[
        bool, Query(description="Only campaigns that raised a data-quality flag")
    ] = False,
    needs_review: Annotated[
        bool, Query(description="Only campaigns where the LLM and the rule-based check disagree")
    ] = False,
    sort: Literal["health_score", "spend", "roas", "external_id"] = "health_score",
    order: Literal["asc", "desc"] = "desc",
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    if channel and channel not in CANONICAL_CHANNELS:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown channel '{channel}'. Valid: {', '.join(CANONICAL_CHANNELS)}",
        )
    if objective and objective not in OBJECTIVES:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"unknown objective '{objective}'. Valid: {', '.join(OBJECTIVES)}",
        )

    stmt = select(Campaign)
    if channel:
        stmt = stmt.where(Campaign.channel == channel)
    if objective:
        stmt = stmt.where(Campaign.objective == objective)
    if min_health_score is not None:
        stmt = stmt.where(Campaign.health_score >= min_health_score)
    if max_health_score is not None:
        stmt = stmt.where(Campaign.health_score <= max_health_score)
    if min_spend is not None:
        stmt = stmt.where(Campaign.spend >= min_spend)
    if needs_review:
        stmt = stmt.where(
            (Campaign.channel_check == "disagree") | (Campaign.health_check == "disagree")
        )

    column = getattr(Campaign, sort)
    stmt = stmt.order_by(column.desc() if order == "desc" else column.asc())

    if flagged_only:
        # JSON emptiness tests are dialect-specific, so this one predicate is
        # applied in Python. It must happen BEFORE pagination, or `limit` would
        # silently return fewer rows than asked for whenever an unflagged
        # campaign fell inside the page.
        rows = [c for c in db.scalars(stmt).all() if c.flags]
        return rows[offset : offset + limit]

    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/{external_id}", response_model=CampaignOut, summary="Fetch one campaign by id")
def get_campaign(external_id: str, db: DbDep) -> Any:
    campaign = db.scalar(select(Campaign).where(Campaign.external_id == external_id))
    if campaign is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no campaign with id '{external_id}'")
    return campaign
