"""Pydantic contracts.

`CampaignEnrichment` is the single source of truth for the LLM's output: it
generates the tool's JSON Schema *and* re-validates whatever comes back. One
definition, so the prompt and the parser can never drift apart.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Channel = Literal["meta", "google_ads", "youtube", "email", "sms", "influencer", "other"]
Objective = Literal["awareness", "consideration", "conversion", "retention"]


class CampaignEnrichment(BaseModel):
    """What we require back from the model for every valid campaign."""

    model_config = ConfigDict(extra="forbid")

    channel: Channel = Field(
        description=(
            "The advertising channel, normalized to the canonical set. Map every vendor "
            "spelling onto it: fb/Facebook Ads/IG/Instagram/meta ads -> meta; "
            "google/adwords/Google Ads/Performance Max -> google_ads; yt/YouTube -> youtube; "
            "Klaviyo email/email -> email; sms/WhatsApp -> sms; influencer/creator/UGC buys -> "
            "influencer. Use 'other' only when the channel genuinely does not fit any of these."
        )
    )
    objective: Objective = Field(
        description=(
            "The campaign's primary marketing objective, inferred from its name and "
            "description — not from its performance. "
            "awareness: reach/impressions-led, no direct-response optimization. "
            "consideration: driving engagement or research among a warm-ish audience. "
            "conversion: optimized for purchases, including retargeting and prospecting. "
            "retention: aimed at existing or lapsed customers (winback, loyalty, VIP)."
        )
    )
    objective_rationale: str = Field(
        max_length=200,
        description="One short clause citing the words in the name/description that decided it.",
    )
    health_score: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100 judgement of how healthy this campaign is, read against the norms of ITS OWN "
            "channel and objective — not against the portfolio. An awareness campaign is not "
            "unhealthy for having a low ROAS; an owned email flow with zero spend is not "
            "unhealthy for having no ROAS. Reserve scores below 25 for campaigns that are "
            "actively wasting money or cannot be measured at all."
        ),
    )
    health_rationale: str = Field(
        max_length=240,
        description="One line a marketer could act on. Cite the specific metric that drove the score.",
    )


class EnrichmentOutcome(BaseModel):
    """Internal transport between the LLM layer and the persistence layer."""

    enrichment: CampaignEnrichment
    source: Literal["llm", "fallback"]
    error: str | None = None
    attempts: int = 1


# --------------------------------------------------------------------------- #
# API response shapes
# --------------------------------------------------------------------------- #
class CampaignOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    external_id: str
    name: str
    description: str | None
    raw_channel: str | None
    channel: str
    objective: str
    objective_rationale: str | None
    health_score: int
    health_rationale: str | None

    currency: str
    spend: float | None
    impressions: int | None
    clicks: int | None
    conversions: int | None
    revenue: float | None

    ctr: float | None
    cvr: float | None
    cpa: float | None
    roas: float | None

    enrichment_source: str
    enrichment_error: str | None
    channel_check: str | None
    channel_expected: str | None
    health_check: str | None
    health_score_rule_based: int | None
    flags: list[str]
    first_ingested_at: datetime
    updated_at: datetime


class RowResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    row_index: int
    external_id: str | None
    status: str
    action: str | None
    reason: str | None
    flags: list[str]
    enrichment_source: str | None


class IngestSummary(BaseModel):
    """The per-row success/failure summary the brief requires."""

    run_id: int
    source: str
    model: str | None
    llm_enabled: bool
    duration_seconds: float
    total_rows: int
    ingested: int
    rejected: int
    created: int
    updated: int
    enriched_by_llm: int
    enriched_by_fallback: int
    disagreements: int
    rows: list[RowResultOut]


class InsightsOut(BaseModel):
    generated_by: str
    campaigns_considered: int
    totals: dict[str, Any]
    observations: list[str]
