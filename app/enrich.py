"""Orchestration: clean -> enrich -> self-check, with a deterministic safety net.

The failure ladder, in order. A row only moves down a rung when the rung above fails:

  1. Deterministic validation   junk never reaches the model              (normalize.py)
  2. LLM, strict tool call      retried 3x: transport, behavioural, semantic   (llm.py)
  3. Deterministic fallback     rule-based enrichment, marked as such      (this module)

Rung 3 is why one bad row — or an LLM outage, or a missing key — cannot sink the
batch. Every campaign gets a channel, an objective and a score no matter what the
model does; the `enrichment_source` column always says which rung produced it, so
nobody mistakes a fallback guess for a model judgement.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.config import Settings
from app.llm import EnrichmentClient, LLMUnavailable
from app.normalize import CleanedRow, normalize_channel_rule_based
from app.schemas import CampaignEnrichment, EnrichmentOutcome

logger = logging.getLogger(__name__)

# Keyword rules for the fallback objective classifier. First match wins, so the
# order encodes precedence: a winback promo is retention before it is a promo, and
# a discount-led sitewide push is a conversion play even when the copy says
# "top-of-funnel". Direct-response signals therefore outrank awareness vocabulary.
_OBJECTIVE_RULES: list[tuple[str, str]] = [
    (r"winback|lapsed|loyalty|vip\b|retention|repeat|existing customer|early access",
     "retention"),
    (r"\d+ ?% ?off|sitewide|promo|blowout|\bsale\b|retarget|abandon|cart|checkout|"
     r"purchase|prospect|acquisition|pmax|performance max|lookalike|\blal\b|"
     r"branded terms|non-brand",
     "conversion"),
    (r"awareness|\breach\b|bumper|brand lift|brand awareness|top-of-funnel|\btof\b|"
     r"impression",
     "awareness"),
    (r"conversion|search|shopping", "conversion"),
    (r"consideration|ugc|creator|reels|engagement|video view|traffic", "consideration"),
]


def fallback_objective(name: str, description: str | None) -> str:
    text = f"{name} {description or ''}".lower()
    for pattern, objective in _OBJECTIVE_RULES:
        if re.search(pattern, text):
            return objective
    return "consideration"  # the least consequential wrong answer of the four


def _score_with_reason(campaign: dict[str, Any], objective: str) -> tuple[int, str]:
    """A transparent, auditable score plus the branch that produced it.

    Serves two jobs:
      * the fallback value when the LLM is unavailable, and
      * the independent yardstick the LLM's own score is checked against.

    It is intentionally crude — a formula cannot read a campaign's intent. That is
    precisely what makes it a useful check: when a blunt rule and a model that read
    the brief disagree sharply, one of them is wrong and a human should look.
    """
    spend = campaign.get("spend")
    revenue = campaign.get("revenue")
    conversions = campaign.get("conversions")
    roas = campaign.get("roas")
    ctr = campaign.get("ctr")

    # Real money out, nothing measurable back. The worst state to be in, because
    # you cannot even tell whether it worked.
    if spend and conversions is None and revenue is None:
        return 12, (
            f"{spend:,.0f} spent with no conversion or revenue data — unmeasurable, "
            f"so the tracking is the problem before the budget is."
        )

    # Owned channels (email/SMS flows) carry no media cost: ROAS is undefined, so
    # judge them on whether they actually produced revenue.
    if spend == 0:
        if revenue and revenue > 0:
            return 88, (
                f"Zero media cost returning {revenue:,.0f} in revenue — an owned flow "
                f"with no acquisition cost to recover."
            )
        return 45, "Zero media cost but no recorded revenue."

    if spend is None:
        return 35, "Spend was missing or invalid, so efficiency cannot be computed."

    if objective == "awareness":
        # Awareness is bought in impressions, not purchases. Score on reach efficiency.
        impressions = campaign.get("impressions")
        if not impressions:
            return 40, "Awareness campaign with no impression data to judge reach efficiency."
        cpm = spend / (impressions / 1000)
        base = 85 if cpm <= 25 else 70 if cpm <= 60 else 50 if cpm <= 120 else 30
        note = f"Awareness buy at a CPM of {cpm:,.0f}"
        if ctr is not None and ctr < 0.001:
            base -= 10
            note += f", but a {ctr * 100:.3f}% CTR shows almost no engagement"
        return max(0, min(100, base)), note + "."

    if roas is None:
        return 35, "No revenue recorded against spend, so ROAS cannot be computed."

    if roas >= 6:
        base = 92
    elif roas >= 4:
        base = 80
    elif roas >= 3:
        base = 68
    elif roas >= 2:
        base = 54
    elif roas >= 1:
        base = 36
    else:
        base = 14
    note = f"{roas:.2f}x ROAS on {spend:,.0f} spend for a {objective} campaign"
    if ctr is not None and ctr >= 0.05:
        base += 3
        note += f" with a strong {ctr * 100:.1f}% CTR"
    return max(0, min(100, base)), note + "."


def rule_based_health_score(campaign: dict[str, Any], objective: str) -> int:
    return _score_with_reason(campaign, objective)[0]


def fallback_enrichment(campaign: dict[str, Any]) -> CampaignEnrichment:
    channel = normalize_channel_rule_based(campaign.get("raw_channel")) or "other"
    objective = fallback_objective(campaign.get("name") or "", campaign.get("description"))
    score, reason = _score_with_reason(campaign, objective)
    return CampaignEnrichment(
        channel=channel,
        objective=objective,
        objective_rationale=f"Keyword rule matched '{objective}' (rule-based, LLM unavailable).",
        health_score=score,
        health_rationale=f"{reason} [rule-based, LLM unavailable]"[:240],
    )


def enrich_campaign(
    cleaned: CleanedRow, client: EnrichmentClient, settings: Settings
) -> EnrichmentOutcome:
    """Enrich one campaign, degrading gracefully instead of raising."""
    payload = dict(cleaned.payload)
    payload["flags"] = cleaned.flags

    if not client.enabled:
        return EnrichmentOutcome(
            enrichment=fallback_enrichment(payload),
            source="fallback",
            error="ANTHROPIC_API_KEY not set — deterministic enrichment used",
            attempts=0,
        )

    try:
        enrichment, attempts = client.enrich(payload)
        return EnrichmentOutcome(enrichment=enrichment, source="llm", attempts=attempts)
    except (LLMUnavailable, RuntimeError) as exc:
        # The whole point of this except: one campaign's model failure degrades
        # that campaign, and nothing else in the batch notices.
        logger.error("falling back to rules for %s: %s", cleaned.external_id, exc)
        return EnrichmentOutcome(
            enrichment=fallback_enrichment(payload),
            source="fallback",
            error=str(exc)[:500],
            attempts=settings.llm_max_attempts,
        )


# --------------------------------------------------------------------------- #
# self-check (stretch: lightweight eval)
# --------------------------------------------------------------------------- #
def self_check(
    campaign: dict[str, Any], enrichment: CampaignEnrichment, settings: Settings
) -> dict[str, Any]:
    """Verify the model against deterministic rules and flag disagreements.

    Channel is the right field to check: it is the one enriched value with an
    objective ground truth (the alias map), so a mismatch is unambiguously a
    model error rather than a difference of opinion. Where the alias map has no
    opinion ("???"), we record 'unverified' rather than pretending to grade it.

    Health score gets a looser band check — the two are *meant* to differ a
    little, so only a gap wider than the configured threshold is worth a human.
    """
    expected_channel = normalize_channel_rule_based(campaign.get("raw_channel"))
    if expected_channel is None:
        channel_check = "unverified"
    elif expected_channel == enrichment.channel:
        channel_check = "agree"
    else:
        channel_check = "disagree"

    rule_score = rule_based_health_score(campaign, enrichment.objective)
    gap = abs(rule_score - enrichment.health_score)
    health_check = "disagree" if gap > settings.health_score_disagreement_threshold else "agree"

    flags: list[str] = []
    if channel_check == "disagree":
        flags.append(f"channel_disagreement_expected_{expected_channel}")
    if health_check == "disagree":
        flags.append(f"health_score_disagreement_rule_{rule_score}")

    return {
        "channel_check": channel_check,
        "channel_expected": expected_channel,
        "health_check": health_check,
        "health_score_rule_based": rule_score,
        "flags": flags,
    }
