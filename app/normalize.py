"""Deterministic cleaning + validation.

This runs BEFORE the LLM, on purpose. Two reasons:

  1. Junk rows should never cost a token. cmp_013 ("N/A" / "lots" / "many")
     is caught here and never reaches the model.
  2. The model is asked to make judgement calls (objective, health), not to do
     arithmetic or parse "42,500". Anything a regex can do reliably, a regex does.

The cleaning philosophy, stated once:
  * Repair a field, reject a row. A bad value kills the value, not the record.
  * Never invent a number. Unparseable -> NULL + a flag, never 0.
  * NULL and 0 are different facts. SMS has no impressions (NULL); an email flow
    genuinely costs 0 (zero). Collapsing them would corrupt every downstream rate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Strings that mean "no value here" rather than a number we failed to parse.
NULL_TOKENS = {"", "-", "--", "n/a", "na", "none", "null", "nil", "tbd", "?", "unknown"}

# Channel alias map. This is the *deterministic* ground truth we later check the
# LLM's normalization against (stretch: lightweight eval / self-check).
CANONICAL_CHANNELS = ("meta", "google_ads", "youtube", "email", "sms", "influencer", "other")

CHANNEL_ALIASES: dict[str, str] = {
    "fb": "meta", "facebook": "meta", "facebook ads": "meta", "meta": "meta",
    "meta ads": "meta", "ig": "meta", "instagram": "meta", "instagram ads": "meta",
    "google": "google_ads", "google ads": "google_ads", "adwords": "google_ads",
    "google adwords": "google_ads", "pmax": "google_ads", "performance max": "google_ads",
    "search": "google_ads",
    "youtube": "youtube", "yt": "youtube", "youtube ads": "youtube",
    "email": "email", "klaviyo": "email", "klaviyo email": "email", "mailchimp": "email",
    "crm email": "email",
    "sms": "sms", "text": "sms", "whatsapp": "sms",
    "influencer": "influencer", "creator": "influencer", "ugc": "influencer",
}

OBJECTIVES = ("awareness", "consideration", "conversion", "retention")

NUMERIC_FIELDS = ("spend", "impressions", "clicks", "conversions", "revenue")
INTEGER_FIELDS = {"impressions", "clicks", "conversions"}
# Fields that carry enough signal to say "this row is a real campaign".
SIGNAL_FIELDS = ("spend", "conversions", "revenue")

_NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")


# --------------------------------------------------------------------------- #
# value coercion
# --------------------------------------------------------------------------- #
@dataclass
class Coerced:
    value: float | None
    status: str  # ok | missing | unparseable


def coerce_number(raw: Any) -> Coerced:
    """Turn a raw cell into a number, a deliberate NULL, or a flagged failure.

    '42,500' -> 42500.0 (ok)      None  -> None (missing)
    'N/A'    -> None (missing)    'lots'-> None (unparseable)  <- the important distinction
    """
    if raw is None:
        return Coerced(None, "missing")
    if isinstance(raw, bool):           # bools are ints in Python; never a metric
        return Coerced(None, "unparseable")
    if isinstance(raw, (int, float)):
        return Coerced(float(raw), "ok")
    if isinstance(raw, str):
        token = raw.strip()
        if token.lower() in NULL_TOKENS:
            return Coerced(None, "missing")
        # strip thousands separators, currency symbols and stray whitespace
        cleaned = re.sub(r"[,\s₹$€£]", "", token)
        if _NUMERIC_RE.match(cleaned):
            return Coerced(float(cleaned), "ok")
        return Coerced(None, "unparseable")
    return Coerced(None, "unparseable")


def normalize_channel_rule_based(raw: Any) -> str | None:
    """Alias-map lookup. Returns None when no rule matches — that is a *real* answer:
    it means 'a human (or the LLM) has to decide', not 'other'."""
    if not isinstance(raw, str):
        return None
    token = re.sub(r"\s+", " ", raw.strip().lower())
    if not token:
        return None
    if token in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[token]
    # containment pass so "Klaviyo email" and "Facebook Ads - Prospecting" resolve
    for alias, canonical in CHANNEL_ALIASES.items():
        if re.search(rf"(^|\W){re.escape(alias)}($|\W)", token):
            return canonical
    return None


# --------------------------------------------------------------------------- #
# derived metrics — every division guarded
# --------------------------------------------------------------------------- #
def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """None (not 0, not inf) whenever the ratio is undefined.

    An email flow with spend=0 has *undefined* ROAS, not infinite ROAS. Returning
    0 would make it look like the worst campaign in the portfolio; returning inf
    would make it the best. Both are lies, so we return None and let the caller
    reason about it explicitly.
    """
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 4)


def derive_metrics(c: dict[str, float | None]) -> dict[str, float | None]:
    return {
        "ctr": _ratio(c["clicks"], c["impressions"]),
        "cvr": _ratio(c["conversions"], c["clicks"]),
        "cpa": _ratio(c["spend"], c["conversions"]),
        "roas": _ratio(c["revenue"], c["spend"]),
    }


# --------------------------------------------------------------------------- #
# row-level validation
# --------------------------------------------------------------------------- #
@dataclass
class CleanedRow:
    row_index: int
    external_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    rejected: bool = False
    reason: str | None = None

    def reject(self, reason: str) -> "CleanedRow":
        self.rejected = True
        self.reason = reason
        return self


def clean_row(raw: dict[str, Any], row_index: int, currency: str = "INR") -> CleanedRow:
    """Clean one raw record. Never raises — a bad row comes back marked, not thrown."""
    if not isinstance(raw, dict):
        return CleanedRow(row_index, None).reject("row is not a JSON object")

    external_id = raw.get("id")
    external_id = external_id.strip() if isinstance(external_id, str) else None
    row = CleanedRow(row_index, external_id)

    # An id is the one field we cannot synthesise: it is the dedupe key.
    if not external_id:
        return row.reject("missing id — cannot dedupe or address this campaign")

    # ---- numbers -----------------------------------------------------------
    numbers: dict[str, float | None] = {}
    unparseable: list[str] = []
    for fname in NUMERIC_FIELDS:
        res = coerce_number(raw.get(fname))
        if res.status == "unparseable":
            unparseable.append(fname)
            row.flags.append(f"unparseable_{fname}")
        elif res.status == "missing" and raw.get(fname) is not None:
            row.flags.append(f"null_token_{fname}")
        numbers[fname] = res.value

    if isinstance(raw.get("spend"), str) and numbers["spend"] is not None:
        row.flags.append("spend_was_string")

    # Negative spend/metrics are physically impossible. Drop the *value*, keep
    # the row — cmp_014 has five good fields and one corrupt one.
    for fname in NUMERIC_FIELDS:
        if numbers[fname] is not None and numbers[fname] < 0:
            row.flags.append(f"negative_{fname}")
            numbers[fname] = None

    for fname in INTEGER_FIELDS:
        if numbers[fname] is not None:
            numbers[fname] = int(numbers[fname])

    # ---- rejection rules ---------------------------------------------------
    # Rule 1: no usable signal. If we cannot read spend, conversions *or*
    # revenue, there is nothing to score and nothing to learn. This is what
    # catches the draft row — on its data, not on the words "DO NOT USE",
    # which is the kind of heuristic that breaks the moment someone renames it.
    if all(numbers[f] is None for f in SIGNAL_FIELDS):
        return row.reject(
            "no usable signal — spend, conversions and revenue are all missing or unparseable"
        )

    # Rule 2: structural junk. Three or more cells that are neither a number nor
    # a recognised null means this is not a campaign record.
    if len(unparseable) >= 3:
        return row.reject(f"{len(unparseable)} unparseable metric fields: {', '.join(unparseable)}")

    # ---- text fields -------------------------------------------------------
    description = raw.get("description")
    description = description.strip() if isinstance(description, str) else None

    name = raw.get("name")
    name = name.strip() if isinstance(name, str) else ""
    if not name:
        # A blank name is cosmetic — it does not invalidate ₹40k of spend.
        # Synthesise something addressable and flag it for a human.
        row.flags.append("missing_name")
        hint = (description or "").split(".")[0].strip()
        name = f"[unnamed {external_id}]" + (f" — {hint}" if hint else "")

    raw_channel = raw.get("channel")
    raw_channel = raw_channel.strip() if isinstance(raw_channel, str) else None
    if not raw_channel or normalize_channel_rule_based(raw_channel) is None:
        row.flags.append("unmapped_channel")

    # ---- semantic flags for the ad-ops team --------------------------------
    if numbers["spend"] == 0:
        row.flags.append("zero_spend")  # owned channel (email/SMS) — not an error
    if (
        numbers["spend"]
        and numbers["conversions"] is None
        and numbers["revenue"] is None
    ):
        # The expensive one: real money out, no attribution back.
        row.flags.append("spend_without_attribution")
    if numbers["impressions"] is None:
        row.flags.append("missing_impressions")

    row.payload = {
        "external_id": external_id,
        "name": name,
        "description": description,
        "raw_channel": raw_channel,
        "currency": currency,
        **numbers,
        **derive_metrics(numbers),
    }
    return row
