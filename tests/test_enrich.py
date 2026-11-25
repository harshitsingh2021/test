"""Tests for enrichment and, above all, for what happens when the model misbehaves.

The brief's hardest requirement is that a malformed model response must not sink
the batch. These tests drive each failure mode with a fake client rather than a
live key, so they run in CI, cost nothing, and are deterministic.
"""
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.config import Settings
from app.enrich import (
    enrich_campaign, fallback_enrichment, fallback_objective, rule_based_health_score, self_check,
)
from app.llm import EnrichmentClient
from app.normalize import clean_row
from app.schemas import CampaignEnrichment


@pytest.fixture
def settings():
    return Settings(
        anthropic_api_key="test-key-not-real",
        llm_max_attempts=3,
        llm_backoff_base_seconds=0.0,   # keep the suite fast
        database_url="sqlite://",
    )


def campaign(**overrides):
    base = {
        "id": "cmp_1", "name": "Retarget cart abandoners", "channel": "Meta",
        "description": "Dynamic product ads.", "spend": 1000, "impressions": 10000,
        "clicks": 500, "conversions": 50, "revenue": 5000,
    }
    row = clean_row({**base, **overrides}, 0)
    payload = dict(row.payload)
    payload["flags"] = row.flags
    return row, payload


# --------------------------------------------------------------------------- #
# fake Anthropic plumbing
# --------------------------------------------------------------------------- #
def tool_use(payload: dict):
    return SimpleNamespace(type="tool_use", id="tu_1", input=payload)


def text_block(body: str):
    return SimpleNamespace(type="text", text=body)


VALID = {
    "channel": "meta",
    "objective": "conversion",
    "objective_rationale": "Retargeting cart abandoners.",
    "health_score": 80,
    "health_rationale": "5x ROAS on retargeting is healthy.",
}


class FakeMessages:
    """Replays a scripted sequence of responses/exceptions, one per attempt."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(content=item, stop_reason="tool_use")


def client_with(settings, script):
    client = EnrichmentClient(settings)
    client._client = SimpleNamespace(messages=FakeMessages(script))
    return client


# --------------------------------------------------------------------------- #
class TestHappyPath:
    def test_valid_tool_call_is_accepted_first_try(self, settings):
        client = client_with(settings, [[tool_use(VALID)]])
        enrichment, attempts = client.enrich(campaign()[1])
        assert enrichment.channel == "meta"
        assert enrichment.health_score == 80
        assert attempts == 1

    def test_tool_call_is_forced_and_strict(self, settings):
        """The request itself must pin the model down — that is what makes this
        structured output rather than prose we hope parses."""
        client = client_with(settings, [[tool_use(VALID)]])
        client.enrich(campaign()[1])
        sent = client._client.messages.calls[0]
        assert sent["tool_choice"] == {"type": "tool", "name": "record_campaign_enrichment"}
        assert sent["tools"][0]["strict"] is True
        assert sent["tools"][0]["input_schema"]["additionalProperties"] is False


class TestModelMisbehaviour:
    def test_out_of_range_score_is_retried_with_the_validation_error(self, settings):
        """health_score=150 is schema-valid on the wire (bounds are not part of
        the strict subset) but invalid to us. Pydantic catches it, and attempt 2
        is told exactly what was wrong instead of blindly repeating attempt 1."""
        client = client_with(settings, [
            [tool_use({**VALID, "health_score": 150})],
            [tool_use(VALID)],
        ])
        enrichment, attempts = client.enrich(campaign()[1])
        assert enrichment.health_score == 80
        assert attempts == 2
        retry = client._client.messages.calls[1]["messages"][-1]["content"][0]
        assert retry["is_error"] is True
        assert "validation" in retry["content"].lower()

    def test_invalid_enum_is_retried(self, settings):
        client = client_with(settings, [
            [tool_use({**VALID, "channel": "tiktok"})],
            [tool_use(VALID)],
        ])
        enrichment, _ = client.enrich(campaign()[1])
        assert enrichment.channel == "meta"

    def test_prose_instead_of_a_tool_call_is_retried(self, settings):
        client = client_with(settings, [
            [text_block("Sure! Here is the JSON: {...}")],
            [tool_use(VALID)],
        ])
        enrichment, attempts = client.enrich(campaign()[1])
        assert enrichment.channel == "meta" and attempts == 2

    def test_gives_up_after_max_attempts(self, settings):
        client = client_with(settings, [[tool_use({**VALID, "health_score": 150})]] * 3)
        with pytest.raises(RuntimeError, match="after 3 attempts"):
            client.enrich(campaign()[1])

    def test_non_retryable_status_error_is_not_retried(self, settings):
        """A 401 will still be a 401 on attempt 3. Burning retries on it just
        makes the batch slower."""
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com"))
        err = anthropic.AuthenticationError("bad key", response=response, body=None)
        client = client_with(settings, [err])
        with pytest.raises(RuntimeError, match="non-retryable"):
            client.enrich(campaign()[1])
        assert len(client._client.messages.calls) == 1


class TestGracefulDegradation:
    def test_persistent_failure_falls_back_instead_of_raising(self, settings):
        """The core requirement: one campaign's model failure degrades that
        campaign and nothing else."""
        row, _ = campaign()
        client = client_with(settings, [[text_block("nope")]] * 3)
        outcome = enrich_campaign(row, client, settings)
        assert outcome.source == "fallback"
        assert outcome.enrichment.channel == "meta"     # alias map still resolves it
        assert 0 <= outcome.enrichment.health_score <= 100
        assert outcome.error

    def test_no_api_key_still_produces_enrichment(self):
        s = Settings(anthropic_api_key=None, database_url="sqlite://")
        row, _ = campaign()
        outcome = enrich_campaign(row, EnrichmentClient(s), s)
        assert outcome.source == "fallback"
        assert outcome.enrichment.objective == "conversion"


class TestFallbackRules:
    @pytest.mark.parametrize("name,expected", [
        ("Winback — lapsed 90d customers", "retention"),
        ("Retention — VIP early access", "retention"),
        ("Diwali Dhamaka Blowout — Sitewide 40% OFF", "conversion"),
        ("Retarget cart abandoners", "conversion"),
        ("Performance Max — full catalog", "conversion"),
        ("Search — Non-brand 'best vitamin c serum india'", "conversion"),
        ("Brand Awareness — Hero SKU", "awareness"),
        ("IG Reels — UGC creator batch #4", "consideration"),
    ])
    def test_objective_keywords(self, name, expected):
        assert fallback_objective(name, "") == expected

    def test_non_brand_is_not_read_as_brand_awareness(self):
        """Guards a real bug: a naive \\bbrand\\b matches inside 'non-brand'."""
        assert fallback_objective("Search — Non-brand terms", "Generic non-brand.") == "conversion"

    def test_unmeasurable_spend_scores_near_zero(self):
        _, c = campaign(spend=350000, conversions=None, revenue=None,
                        impressions=None, clicks=None)
        assert rule_based_health_score(c, "awareness") < 25

    def test_zero_spend_with_revenue_scores_well(self):
        _, c = campaign(spend=0, revenue=1080000)
        assert rule_based_health_score(c, "retention") > 70

    def test_awareness_is_not_punished_for_low_roas(self):
        """A YouTube awareness buy with 0.69x ROAS is doing its job."""
        _, c = campaign(spend=96000, impressions=5100000, clicks=8100,
                        conversions=44, revenue=66000)
        assert rule_based_health_score(c, "awareness") > rule_based_health_score(c, "conversion")

    def test_score_is_always_in_range(self):
        for obj in ("awareness", "consideration", "conversion", "retention"):
            for kwargs in [{}, {"spend": 0}, {"revenue": None}, {"impressions": None},
                           {"spend": -5}, {"clicks": 0}, {"conversions": 0}]:
                _, c = campaign(**kwargs)
                assert 0 <= rule_based_health_score(c, obj) <= 100


class TestSelfCheck:
    def test_agrees_when_llm_matches_the_alias_map(self, settings):
        _, c = campaign(channel="Facebook Ads")
        check = self_check(c, CampaignEnrichment(**VALID), settings)
        assert check["channel_check"] == "agree"
        assert check["flags"] == []

    def test_catches_a_wrong_channel(self, settings):
        _, c = campaign(channel="Klaviyo email")
        check = self_check(c, CampaignEnrichment(**VALID), settings)   # model said meta
        assert check["channel_check"] == "disagree"
        assert check["channel_expected"] == "email"
        assert any("channel_disagreement" in f for f in check["flags"])

    def test_unmappable_channel_is_unverified_not_failed(self, settings):
        """We cannot grade what the rules have no opinion about — saying so is
        more useful than inventing a verdict."""
        _, c = campaign(channel="???")
        check = self_check(c, CampaignEnrichment(**VALID), settings)
        assert check["channel_check"] == "unverified"

    def test_flags_a_wild_health_score(self, settings):
        _, c = campaign()
        check = self_check(c, CampaignEnrichment(**{**VALID, "health_score": 5}), settings)
        assert check["health_check"] == "disagree"

    def test_tolerates_a_small_difference_of_opinion(self, settings):
        _, c = campaign()
        rule = rule_based_health_score(c, "conversion")
        check = self_check(c, CampaignEnrichment(**{**VALID, "health_score": rule - 10}), settings)
        assert check["health_check"] == "agree"


class TestStrictSchemaDegradation:
    """If the API rejects the tool schema, every row in the batch would fail the
    same way at the same moment — the one failure mode that defeats per-row
    isolation. The client drops `strict` once and carries on; Pydantic still
    enforces the contract client-side, so nothing invalid gets stored."""

    def _schema_400(self):
        response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
        return anthropic.BadRequestError(
            "tools.0.input_schema: unsupported keyword for strict tool use",
            response=response, body=None,
        )

    def test_recovers_by_dropping_strict(self, settings):
        client = client_with(settings, [self._schema_400(), [tool_use(VALID)]])
        enrichment, attempts = client.enrich(campaign()[1])
        assert enrichment.health_score == 80
        assert attempts == 2
        assert client._strict is False
        assert "strict" not in client._client.messages.calls[1]["tools"][0]

    def test_validation_is_still_enforced_without_strict(self, settings):
        """Losing the server-side guarantee must not lose the guarantee."""
        client = client_with(settings, [
            self._schema_400(),
            [tool_use({**VALID, "health_score": 150})],
            [tool_use(VALID)],
        ])
        enrichment, _ = client.enrich(campaign()[1])
        assert enrichment.health_score == 80

    def test_an_unrelated_400_is_still_fatal(self, settings):
        response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com"))
        err = anthropic.BadRequestError("credit balance is too low", response=response, body=None)
        client = client_with(settings, [err])
        with pytest.raises(RuntimeError, match="non-retryable"):
            client.enrich(campaign()[1])

    def test_a_401_is_still_fatal(self, settings):
        response = httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com"))
        err = anthropic.AuthenticationError("invalid x-api-key", response=response, body=None)
        client = client_with(settings, [err])
        with pytest.raises(RuntimeError, match="non-retryable"):
            client.enrich(campaign()[1])
        assert client._strict is True
