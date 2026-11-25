"""The ingest pipeline: read -> clean -> enrich (concurrently) -> self-check -> upsert."""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.enrich import enrich_campaign, self_check
from app.llm import EnrichmentClient
from app.models import Campaign, IngestRun, IngestRowResult
from app.normalize import CleanedRow, clean_row
from app.schemas import IngestSummary, RowResultOut

logger = logging.getLogger(__name__)


class RawDataError(ValueError):
    """The input document itself is unusable (not just one of its rows)."""


def load_raw(source: Path | dict[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(source, dict):
        return source, "inline payload"
    if not source.exists():
        raise RawDataError(f"raw data file not found: {source}")
    try:
        return json.loads(source.read_text()), str(source)
    except json.JSONDecodeError as exc:
        raise RawDataError(f"raw data file is not valid JSON: {exc}") from exc


def ingest(
    db: Session,
    settings: Settings,
    source: Path | dict[str, Any] | None = None,
) -> IngestSummary:
    started = time.perf_counter()
    document, source_label = load_raw(source if source is not None else settings.raw_data_path)

    if not isinstance(document, dict) or not isinstance(document.get("campaigns"), list):
        raise RawDataError("expected an object with a 'campaigns' array")

    raw_rows: list[Any] = document["campaigns"]
    currency = document.get("currency") or "INR"

    client = EnrichmentClient(settings)
    run = IngestRun(
        source=source_label,
        model=settings.llm_model if client.enabled else None,
        llm_enabled=client.enabled,
        total_rows=len(raw_rows),
    )
    db.add(run)
    db.flush()

    # ---- stage 1: clean every row (cheap, deterministic, no network) ----------
    cleaned: list[CleanedRow] = [clean_row(r, i, currency) for i, r in enumerate(raw_rows)]

    # ---- stage 1b: dedupe within the batch -----------------------------------
    # The export contains the same id twice with different numbers. First
    # occurrence wins and the collision is recorded rather than silently
    # overwritten — "which of these two is right?" is a question for ad-ops, and
    # last-write-wins would quietly destroy the evidence that it was ever asked.
    seen: dict[str, int] = {}
    for row in cleaned:
        if row.rejected or not row.external_id:
            continue
        if row.external_id in seen:
            row.reject(f"duplicate id — already supplied at row {seen[row.external_id]}")
            row.flags.append("duplicate_id")
        else:
            seen[row.external_id] = row.row_index

    valid = [r for r in cleaned if not r.rejected]

    # ---- stage 2: enrich, bounded fan-out ------------------------------------
    # One call per campaign keeps failures isolated to a single row; a thread pool
    # keeps the batch fast. Each task is already exception-safe (enrich_campaign
    # degrades to the rule-based path), so nothing here can raise.
    if valid:
        with ThreadPoolExecutor(max_workers=max(1, settings.llm_concurrency)) as pool:
            outcomes = list(
                pool.map(lambda r: enrich_campaign(r, client, settings), valid)
            )
    else:
        outcomes = []

    created = updated = disagreements = 0
    by_llm = by_fallback = 0

    # ---- stage 3: self-check + upsert ----------------------------------------
    for row, outcome in zip(valid, outcomes):
        payload = dict(row.payload)
        payload["flags"] = row.flags
        check = self_check(payload, outcome.enrichment, settings)
        flags = row.flags + check.pop("flags")
        if check["channel_check"] == "disagree" or check["health_check"] == "disagree":
            disagreements += 1

        existing = db.scalar(
            select(Campaign).where(Campaign.external_id == row.external_id)
        )
        values: dict[str, Any] = {
            **row.payload,
            **outcome.enrichment.model_dump(),
            **check,
            "flags": flags,
            "enrichment_source": outcome.source,
            "enrichment_error": outcome.error,
        }

        if existing is None:
            db.add(Campaign(**values))
            created += 1
            action = "created"
        else:
            # Idempotent re-ingest: same id updates in place, never duplicates.
            for key, value in values.items():
                setattr(existing, key, value)
            updated += 1
            action = "updated"

        by_llm += outcome.source == "llm"
        by_fallback += outcome.source == "fallback"

        db.add(IngestRowResult(
            run_id=run.id,
            row_index=row.row_index,
            external_id=row.external_id,
            status="ingested",
            action=action,
            reason=outcome.error,
            flags=flags,
            enrichment_source=outcome.source,
        ))

    for row in (r for r in cleaned if r.rejected):
        db.add(IngestRowResult(
            run_id=run.id,
            row_index=row.row_index,
            external_id=row.external_id,
            status="rejected",
            action="skipped",
            reason=row.reason,
            flags=row.flags,
            enrichment_source=None,
        ))

    rejected = sum(1 for r in cleaned if r.rejected)
    run.ingested = len(valid)
    run.rejected = rejected
    run.enriched_by_llm = by_llm
    run.enriched_by_fallback = by_fallback
    run.finished_at = datetime.now(timezone.utc)
    db.commit()

    rows = db.scalars(
        select(IngestRowResult)
        .where(IngestRowResult.run_id == run.id)
        .order_by(IngestRowResult.row_index)
    ).all()

    return IngestSummary(
        run_id=run.id,
        source=source_label,
        model=run.model,
        llm_enabled=run.llm_enabled,
        duration_seconds=round(time.perf_counter() - started, 2),
        total_rows=len(raw_rows),
        ingested=len(valid),
        rejected=rejected,
        created=created,
        updated=updated,
        enriched_by_llm=by_llm,
        enriched_by_fallback=by_fallback,
        disagreements=disagreements,
        rows=[RowResultOut.model_validate(r) for r in rows],
    )
