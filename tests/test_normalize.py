"""Tests for the deterministic cleaning layer — the part that decides what the
LLM never gets to see."""
import pytest

from app.normalize import (
    clean_row, coerce_number, derive_metrics, normalize_channel_rule_based,
)


class TestCoerceNumber:
    @pytest.mark.parametrize("raw,expected", [
        (185000, 185000.0),
        (1.5, 1.5),
        ("42,500", 42500.0),        # the thousands separator in the real export
        ("  42500  ", 42500.0),
        ("₹42,500", 42500.0),
        ("-1200", -1200.0),         # parsed here; rejected later as impossible
    ])
    def test_parses_numbers(self, raw, expected):
        result = coerce_number(raw)
        assert result.status == "ok"
        assert result.value == expected

    @pytest.mark.parametrize("raw", [None, "", "N/A", "n/a", "-", "null", "unknown"])
    def test_null_tokens_are_missing_not_junk(self, raw):
        """'N/A' is the export saying 'no value', which is different from junk."""
        assert coerce_number(raw).status == "missing"
        assert coerce_number(raw).value is None

    @pytest.mark.parametrize("raw", ["lots", "many", "a few", "asdf", {}, []])
    def test_junk_is_unparseable(self, raw):
        """'lots' is not a number and not a null — it is a broken cell, and the
        distinction is what lets us reject the draft row on evidence."""
        assert coerce_number(raw).status == "unparseable"
        assert coerce_number(raw).value is None

    def test_booleans_are_never_metrics(self):
        assert coerce_number(True).status == "unparseable"


class TestChannelNormalization:
    @pytest.mark.parametrize("raw,expected", [
        ("fb", "meta"), ("Meta", "meta"), ("IG", "meta"),
        ("Facebook Ads", "meta"), ("meta ads", "meta"),
        ("google", "google_ads"), ("adwords", "google_ads"), ("Google Ads", "google_ads"),
        ("YouTube", "youtube"), ("yt", "youtube"),
        ("Klaviyo email", "email"), ("email", "email"),
        ("sms", "sms"), ("influencer", "influencer"),
    ])
    def test_maps_every_spelling_in_the_export(self, raw, expected):
        assert normalize_channel_rule_based(raw) == expected

    def test_unmappable_returns_none_not_other(self):
        """None means 'the rules have no opinion', which is honest. Silently
        answering 'other' would make the LLM self-check meaningless."""
        assert normalize_channel_rule_based("???") is None
        assert normalize_channel_rule_based("") is None
        assert normalize_channel_rule_based(None) is None


class TestDerivedMetrics:
    def test_computes_rates(self):
        m = derive_metrics({
            "spend": 100.0, "impressions": 1000, "clicks": 100,
            "conversions": 10, "revenue": 500.0,
        })
        assert m == {"ctr": 0.1, "cvr": 0.1, "cpa": 10.0, "roas": 5.0}

    def test_zero_spend_gives_undefined_roas_not_infinity(self):
        """An email flow costing nothing has undefined ROAS. Returning 0 would
        rank it as the worst campaign; returning inf, the best. Both are lies."""
        m = derive_metrics({
            "spend": 0.0, "impressions": 1000, "clicks": 100,
            "conversions": 10, "revenue": 500.0,
        })
        assert m["roas"] is None
        assert m["cpa"] == 0.0

    def test_null_impressions_gives_undefined_ctr(self):
        m = derive_metrics({
            "spend": 10.0, "impressions": None, "clicks": 100,
            "conversions": 10, "revenue": 50.0,
        })
        assert m["ctr"] is None


class TestRowValidation:
    def _row(self, **overrides):
        base = {
            "id": "cmp_x", "name": "Test", "channel": "fb", "description": "d",
            "spend": 1000, "impressions": 10000, "clicks": 500,
            "conversions": 50, "revenue": 5000,
        }
        return {**base, **overrides}

    def test_clean_row_happy_path(self):
        row = clean_row(self._row(), 0)
        assert not row.rejected
        assert row.payload["roas"] == 5.0

    def test_rejects_row_with_no_id(self):
        row = clean_row(self._row(id=""), 0)
        assert row.rejected and "id" in row.reason

    def test_rejects_the_draft_row_on_evidence_not_on_its_name(self):
        """cmp_013 from the real export. Rejected because nothing in it can be
        read — not because it is called 'DO NOT USE'."""
        row = clean_row({
            "id": "cmp_013", "name": "Test — DO NOT USE (draft row)", "channel": "???",
            "description": "asdf test test ignore", "spend": "N/A",
            "impressions": "lots", "clicks": -5, "conversions": "many", "revenue": "N/A",
        }, 12)
        assert row.rejected
        assert "no usable signal" in row.reason

    def test_a_row_named_test_with_real_numbers_survives(self):
        """The counterpart to the test above: the rule keys off data, so a real
        campaign is not thrown away for having an unlucky name."""
        row = clean_row(self._row(id="cmp_z", name="Test — new creative"), 0)
        assert not row.rejected

    def test_negative_spend_nulls_the_field_and_keeps_the_row(self):
        """cmp_014: one corrupt cell must not discard five good ones."""
        row = clean_row(self._row(id="cmp_014", spend=-1200), 0)
        assert not row.rejected
        assert row.payload["spend"] is None
        assert "negative_spend" in row.flags
        assert row.payload["conversions"] == 50      # the good fields survive

    def test_string_spend_is_coerced_and_flagged(self):
        row = clean_row(self._row(spend="42,500"), 0)
        assert row.payload["spend"] == 42500.0
        assert "spend_was_string" in row.flags

    def test_blank_name_is_synthesised_not_rejected(self):
        """cmp_011: a cosmetic gap must not discard ₹40k of real spend."""
        row = clean_row(self._row(name="", description="Bumper ads, reach buy."), 10)
        assert not row.rejected
        assert "missing_name" in row.flags
        assert row.payload["name"].startswith("[unnamed cmp_x]")

    def test_zero_spend_is_valid_and_flagged(self):
        row = clean_row(self._row(spend=0), 0)
        assert not row.rejected
        assert row.payload["spend"] == 0
        assert "zero_spend" in row.flags
        assert row.payload["roas"] is None

    def test_spend_without_attribution_is_flagged(self):
        """cmp_012: the expensive failure — real money, no way to evaluate it."""
        row = clean_row(
            self._row(spend=350000, conversions=None, revenue=None, impressions=None, clicks=None),
            11,
        )
        assert not row.rejected
        assert "spend_without_attribution" in row.flags

    def test_never_raises_on_garbage_input(self):
        for bad in [{}, {"id": None}, "not a dict", 42, None]:
            assert clean_row(bad, 0).rejected is True
