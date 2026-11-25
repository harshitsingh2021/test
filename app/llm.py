"""Anthropic enrichment via strict tool calling.

Why tool calling rather than "return JSON" in a prompt:
  * `strict: true` + `additionalProperties: false` makes the API itself reject a
    malformed call before it ever reaches us — the shape is guaranteed, not hoped for.
  * There is no prose to strip, no ```json fence to regex, no "Sure, here's the JSON!"
    preamble. Free-text parsing is the failure mode the brief explicitly calls out.

Two layers of guarantee, deliberately:
  * The WIRE schema guarantees structure (right keys, right types, valid enums).
  * Pydantic re-validates on arrival and guarantees RANGE (0 <= health_score <= 100,
    rationale length). Numeric bounds are not part of the strict-tool subset, so a
    model could legally return health_score = 150. Pydantic catches it, and a range
    failure is retried with corrective feedback rather than silently stored.
"""
from __future__ import annotations

import json
import logging
import random
import time
from typing import Any

import anthropic
from pydantic import ValidationError

from app.config import Settings
from app.schemas import CampaignEnrichment

logger = logging.getLogger(__name__)

TOOL_NAME = "record_campaign_enrichment"

SYSTEM_PROMPT = """You are a marketing analytics engine for Aster & Oak, a D2C skincare brand in India (currency INR).

You receive ONE campaign from a messy ad-ops export, already cleaned and with its rate metrics pre-computed. Classify it and score its health by calling the record_campaign_enrichment tool exactly once.

Rules that matter more than anything else:

1. Judge every campaign against the norms of ITS OWN channel and objective. A YouTube awareness buy with 0.01 ROAS is doing its job; a Performance Max campaign with 0.01 ROAS is on fire. Never score a campaign badly for failing at something it was not trying to do.

2. NULL is not zero. A null metric means it was missing or unreadable in the export, not that the value was zero. Never treat a null as a zero, and never infer a rate we did not give you.

3. Unmeasurable is its own failure, and a serious one. A campaign with real spend but no conversion or revenue data is not "average" and it is not "bad performance" — it is money that cannot be evaluated. Score it low and say exactly that in the rationale, so a human knows to fix the tracking rather than cut the budget.

4. Zero spend is normal on owned channels. Email and SMS flows carry no media cost; their ROAS is undefined, not infinite and not zero. Judge them on revenue and conversion volume.

5. Infer objective from the NAME and DESCRIPTION only — never from performance. A conversion campaign that converted badly is still a conversion campaign.

Write the rationales for a performance marketer who will act on them: name the specific metric that drove your score. No hedging, no restating the inputs."""


def _wire_schema() -> dict[str, Any]:
    """Pydantic's schema, trimmed to the keywords strict tool use accepts.

    Bounds (`minimum`/`maximum`/`maxLength`) are dropped from the wire and enforced
    by Pydantic on the way back in — see the module docstring.
    """
    schema = CampaignEnrichment.model_json_schema()
    drop = {"minimum", "maximum", "maxLength", "minLength", "title", "default"}

    def prune(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: prune(v) for k, v in node.items() if k not in drop}
        if isinstance(node, list):
            return [prune(v) for v in node]
        return node

    pruned = prune(schema)
    pruned["additionalProperties"] = False
    return pruned


def build_tool(strict: bool = True) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": TOOL_NAME,
        "description": (
            "Record the normalized channel, inferred objective and health score for one "
            "campaign. Call this exactly once."
        ),
        "input_schema": _wire_schema(),
    }
    if strict:
        tool["strict"] = True
    return tool


ENRICHMENT_TOOL: dict[str, Any] = build_tool()


def build_user_prompt(campaign: dict[str, Any]) -> str:
    """Render one campaign as a compact, unambiguous brief for the model.

    Rates are pre-computed here rather than left to the model: arithmetic is the
    one thing we can do perfectly and it cannot, and an undefined ratio is stated
    as 'undefined (reason)' so the model is never left to guess what a null means.
    """
    def money(v: float | None) -> str:
        return "not recorded" if v is None else f"INR {v:,.0f}"

    def count(v: int | None) -> str:
        return "not recorded" if v is None else f"{v:,}"

    def pct(v: float | None, undefined: str) -> str:
        return undefined if v is None else f"{v * 100:.2f}%"

    spend, revenue = campaign.get("spend"), campaign.get("revenue")
    roas = campaign.get("roas")
    if roas is not None:
        roas_line = f"{roas:.2f}x"
    elif spend == 0:
        roas_line = "undefined (zero media cost — owned channel)"
    elif revenue is None:
        roas_line = "undefined (revenue was not recorded)"
    else:
        roas_line = "undefined"

    cpa = campaign.get("cpa")
    cpa_line = (
        f"INR {cpa:,.0f}" if cpa is not None
        else "undefined (no conversions recorded)" if campaign.get("conversions") in (None, 0)
        else "undefined"
    )

    flags = campaign.get("flags") or []
    flag_line = ", ".join(flags) if flags else "none"

    return f"""Campaign to classify and score.

Name:          {campaign.get('name')}
Description:   {campaign.get('description') or '(none provided)'}
Channel as written in the export: "{campaign.get('raw_channel') or '(blank)'}"

Spend:         {money(spend)}
Revenue:       {money(revenue)}
Impressions:   {count(campaign.get('impressions'))}
Clicks:        {count(campaign.get('clicks'))}
Conversions:   {count(campaign.get('conversions'))}

CTR:           {pct(campaign.get('ctr'), 'undefined (impressions not recorded)')}
CVR:           {pct(campaign.get('cvr'), 'undefined (clicks or conversions not recorded)')}
CPA:           {cpa_line}
ROAS:          {roas_line}

Data-quality flags raised during cleaning: {flag_line}

"not recorded" means the field was missing or unreadable in the export. It does NOT mean zero."""


class LLMUnavailable(RuntimeError):
    """No API key configured — callers fall back to deterministic enrichment."""


class EnrichmentClient:
    """Thin, retrying wrapper around one Messages call per campaign."""

    # Transport-level failures worth retrying. 4xx (bad request, auth) are not.
    RETRYABLE = (
        anthropic.APIConnectionError,
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.InternalServerError,
    )

    # A 400 that means "I don't like your tool schema" rather than "your request
    # was malformed". Matched loosely on purpose — the exact wording is not a
    # stable contract, and guessing wrong only costs one extra non-strict retry.
    _SCHEMA_REJECTION_HINTS = ("strict", "input_schema", "schema", "tool")

    def __init__(self, settings: Settings):
        self.settings = settings
        self.enabled = settings.llm_enabled
        self._strict = True
        self._client = (
            anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=0)
            if self.enabled
            else None
        )

    def _is_schema_rejection(self, exc: anthropic.APIStatusError) -> bool:
        return exc.status_code == 400 and any(
            hint in str(exc).lower() for hint in self._SCHEMA_REJECTION_HINTS
        )

    def enrich(self, campaign: dict[str, Any]) -> tuple[CampaignEnrichment, int]:
        """Return validated enrichment, or raise after exhausting attempts.

        Retries cover three distinct failure modes, which is the point:
          * transport  — connection reset, 429, 5xx                -> plain backoff
          * behavioural— model answered in prose instead of calling -> re-ask
          * semantic   — tool called but arguments out of range     -> re-ask WITH the
                         validation error, so attempt 2 is better informed than attempt 1
        """
        if self._client is None:
            raise LLMUnavailable("ANTHROPIC_API_KEY is not set")

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_user_prompt(campaign)}
        ]
        last_error: Exception | None = None

        for attempt in range(1, self.settings.llm_max_attempts + 1):
            try:
                response = self._client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=self.settings.llm_max_tokens,
                    system=SYSTEM_PROMPT,
                    tools=[build_tool(strict=self._strict)],
                    tool_choice={"type": "tool", "name": TOOL_NAME},
                    messages=messages,
                )
            except self.RETRYABLE as exc:
                last_error = exc
                logger.warning(
                    "llm transport failure (attempt %s/%s) for %s: %s",
                    attempt, self.settings.llm_max_attempts, campaign.get("external_id"), exc,
                )
                self._sleep(attempt)
                continue
            except anthropic.APIStatusError as exc:
                # One escape hatch before giving up on a 400: if the API rejected
                # the *schema* rather than the request, drop `strict` and try
                # again. Without this, a single incompatible keyword would push
                # every row in the batch onto the fallback path at once — the one
                # failure mode that defeats the point of per-row isolation.
                # Pydantic still validates the result, so correctness is unchanged;
                # only the server-side guarantee is lost, and that is recorded.
                if self._strict and self._is_schema_rejection(exc):
                    logger.warning(
                        "API rejected the strict tool schema (%s); retrying without "
                        "strict for the rest of this run. Validation still enforced "
                        "client-side by Pydantic.", exc,
                    )
                    self._strict = False
                    last_error = exc   # so the final message is accurate if we still fail
                    continue
                # Otherwise a 401/403/400 will still be the same on attempt 3.
                raise RuntimeError(f"non-retryable API error {exc.status_code}: {exc}") from exc

            block = next((b for b in response.content if b.type == "tool_use"), None)
            if block is None:
                # Forced tool_choice makes this near-impossible, but "near" is not
                # "never" and the brief asks us to handle the model misbehaving.
                last_error = RuntimeError("model returned no tool_use block")
                logger.warning("no tool_use for %s (stop_reason=%s)",
                               campaign.get("external_id"), response.stop_reason)
                messages = [
                    {"role": "user", "content": build_user_prompt(campaign)},
                    {"role": "assistant", "content": response.content},
                    {"role": "user", "content": f"You must call the {TOOL_NAME} tool. Call it now."},
                ]
                self._sleep(attempt)
                continue

            try:
                # json round-trip: never string-match raw tool input (escaping varies).
                return CampaignEnrichment.model_validate(json.loads(json.dumps(block.input))), attempt
            except ValidationError as exc:
                last_error = exc
                logger.warning("schema validation failed for %s: %s",
                               campaign.get("external_id"), exc)
                messages = [
                    {"role": "user", "content": build_user_prompt(campaign)},
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "is_error": True,
                            "content": (
                                f"Your arguments failed validation:\n{exc}\n\n"
                                f"Call {TOOL_NAME} again with values inside the documented bounds."
                            ),
                        }],
                    },
                ]
                self._sleep(attempt)

        raise RuntimeError(
            f"enrichment failed after {self.settings.llm_max_attempts} attempts: {last_error}"
        )

    def _sleep(self, attempt: int) -> None:
        """Exponential backoff with jitter — avoids retry stampedes across the batch."""
        delay = self.settings.llm_backoff_base_seconds * (2 ** (attempt - 1))
        time.sleep(delay + random.uniform(0, 0.25))
