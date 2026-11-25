"""Stretch: portfolio-level observations.

Aggregation happens in SQL/Python (facts), the LLM is handed only those
aggregates and asked for interpretation (judgement). It never sees raw rows, so
it cannot invent a number — every figure in an observation came from the DB.
"""
from __future__ import annotations

import json
import logging

import anthropic
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.models import Campaign

logger = logging.getLogger(__name__)

INSIGHTS_SYSTEM = """You are a performance marketing analyst reviewing a D2C skincare brand's paid portfolio (currency INR).

You are given aggregates computed from the database — not raw rows. Every number you cite must come from the data provided; never estimate, extrapolate or invent one.

A null ROAS or revenue means it was never recorded. It does NOT mean zero. Never describe an unmeasured channel as underperforming — say that it cannot be measured, because the fix is instrumentation, not budget.

Return 2-3 observations. Each must name the specific campaigns or channels involved, quote the figure that proves it, and state the action it implies. Prioritise money being wasted or left on the table over restating what is going well. One sentence each, no preamble, no bullets."""


def _f(value: float | None) -> float | None:
    return None if value is None else round(float(value), 2)


def collect_aggregates(db: Session) -> dict:
    campaigns = db.scalars(select(Campaign)).all()
    if not campaigns:
        return {"campaign_count": 0}

    total_spend = sum(c.spend or 0 for c in campaigns)
    total_revenue = sum(c.revenue or 0 for c in campaigns)
    # Spend whose return is unknown, not zero. Reported separately so the
    # blended ROAS below can be read honestly rather than looking like
    # underperformance — the same NULL-is-not-zero rule the cleaner enforces.
    spend_unknown_revenue = sum(c.spend or 0 for c in campaigns if c.revenue is None)

    by_channel: dict[str, dict] = {}
    for c in campaigns:
        bucket = by_channel.setdefault(c.channel, {
            "campaigns": 0, "spend": 0.0, "revenue": 0.0,
            "conversions": 0, "campaigns_missing_revenue": 0,
        })
        bucket["campaigns"] += 1
        bucket["spend"] += c.spend or 0
        bucket["conversions"] += c.conversions or 0
        if c.revenue is None:
            bucket["campaigns_missing_revenue"] += 1
        else:
            bucket["revenue"] += c.revenue
    for bucket in by_channel.values():
        # A channel whose every campaign is missing revenue has UNKNOWN ROAS.
        # Reporting 0.0x there would invent a catastrophic result out of a gap
        # in the data, and would read as "cut this channel" instead of
        # "instrument this channel".
        measurable = bucket["campaigns"] - bucket["campaigns_missing_revenue"]
        bucket["roas"] = (
            _f(bucket["revenue"] / bucket["spend"])
            if bucket["spend"] and measurable else None
        )
        bucket["spend"] = _f(bucket["spend"])
        bucket["revenue"] = _f(bucket["revenue"]) if measurable else None

    ranked = sorted(campaigns, key=lambda c: c.health_score)
    unattributed = [c for c in campaigns if "spend_without_attribution" in (c.flags or [])]

    return {
        "campaign_count": len(campaigns),
        "currency": campaigns[0].currency,
        "total_spend": _f(total_spend),
        "total_revenue": _f(total_revenue),
        "blended_roas": _f(total_revenue / total_spend) if total_spend else None,
        "spend_with_unknown_revenue": _f(spend_unknown_revenue),
        "by_channel": by_channel,
        "by_objective": {
            obj: count
            for obj, count in db.execute(
                select(Campaign.objective, func.count()).group_by(Campaign.objective)
            ).all()
        },
        "weakest_campaigns": [
            {
                "id": c.external_id, "name": c.name, "channel": c.channel,
                "objective": c.objective, "spend": _f(c.spend), "roas": _f(c.roas),
                "health_score": c.health_score, "why": c.health_rationale,
            }
            for c in ranked[:5]
        ],
        "strongest_campaigns": [
            {
                "id": c.external_id, "name": c.name, "channel": c.channel,
                "spend": _f(c.spend), "roas": _f(c.roas), "health_score": c.health_score,
            }
            for c in ranked[-3:]
        ],
        "spend_without_attribution": {
            "campaigns": [c.external_id for c in unattributed],
            "total_spend": _f(sum(c.spend or 0 for c in unattributed)),
        },
        "rows_with_quality_flags": sum(1 for c in campaigns if c.flags),
        "enriched_by_fallback": sum(1 for c in campaigns if c.enrichment_source == "fallback"),
    }


def _deterministic_observations(agg: dict) -> list[str]:
    """Used when the LLM is unavailable, so the endpoint still returns something true."""
    out: list[str] = []
    unattributed = agg.get("spend_without_attribution", {})
    if unattributed.get("total_spend"):
        out.append(
            f"{unattributed['total_spend']:,.0f} {agg['currency']} is spent on "
            f"{', '.join(unattributed['campaigns'])} with no conversion or revenue tracking — "
            f"that spend cannot be evaluated at all until tracking is fixed."
        )
    # Only channels with measurable returns can be ranked against each other.
    channels = {
        k: v for k, v in agg.get("by_channel", {}).items()
        if v.get("roas") is not None and v.get("spend")
    }
    if len(channels) >= 2:
        worst = min(channels.items(), key=lambda kv: kv[1]["roas"])
        best = max(channels.items(), key=lambda kv: kv[1]["roas"])
        out.append(
            f"{worst[0]} returns {worst[1]['roas']}x on {worst[1]['spend']:,.0f} "
            f"{agg['currency']} while {best[0]} returns {best[1]['roas']}x on "
            f"{best[1]['spend']:,.0f} — wide enough to justify shifting budget, "
            f"though this rule-based view does not account for objective mix."
        )
    if agg.get("blended_roas"):
        caveat = ""
        if agg.get("spend_with_unknown_revenue"):
            caveat = (
                f", though {agg['spend_with_unknown_revenue']:,.0f} of that spend has no "
                f"revenue recorded against it and drags the figure down artificially"
            )
        out.append(
            f"Blended portfolio ROAS is {agg['blended_roas']}x across "
            f"{agg['campaign_count']} campaigns{caveat}."
        )
    return out[:3]


def generate_insights(db: Session, settings: Settings) -> dict:
    agg = collect_aggregates(db)
    if agg.get("campaign_count", 0) == 0:
        return {
            "generated_by": "none",
            "campaigns_considered": 0,
            "totals": {},
            "observations": ["No campaigns ingested yet — POST /campaigns/ingest first."],
        }

    totals = {
        "total_spend": agg["total_spend"],
        "total_revenue": agg["total_revenue"],
        "blended_roas": agg["blended_roas"],
        "currency": agg["currency"],
        "by_channel": agg["by_channel"],
        "rows_with_quality_flags": agg["rows_with_quality_flags"],
        "spend_with_unknown_revenue": agg["spend_with_unknown_revenue"],
    }

    if not settings.llm_enabled:
        return {
            "generated_by": "rules",
            "campaigns_considered": agg["campaign_count"],
            "totals": totals,
            "observations": _deterministic_observations(agg),
        }

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=1)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            system=INSIGHTS_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    "Portfolio aggregates:\n\n"
                    f"{json.dumps(agg, indent=2, default=str)}\n\n"
                    "Give 2-3 observations."
                ),
            }],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "observations": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["observations"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        text = next(b.text for b in response.content if b.type == "text")
        observations = json.loads(text)["observations"]
        return {
            "generated_by": settings.llm_model,
            "campaigns_considered": agg["campaign_count"],
            "totals": totals,
            "observations": observations[:3],
        }
    except Exception as exc:  # insights are a nice-to-have; never 500 the endpoint
        logger.error("insights generation failed, serving rule-based: %s", exc)
        return {
            "generated_by": "rules (llm failed)",
            "campaigns_considered": agg["campaign_count"],
            "totals": totals,
            "observations": _deterministic_observations(agg),
        }
