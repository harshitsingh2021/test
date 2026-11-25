"""Tests for portfolio aggregation.

The regression at the centre of this file: summing a NULL revenue as 0 invents a
0.0x ROAS for a channel that was simply never measured — which reads as "cut this
channel" when the correct reading is "instrument this channel". It is the same
NULL-is-not-zero rule the cleaner enforces, and it is easy to reintroduce.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db import Base
from app.insights import collect_aggregates, generate_insights
from app.models import Campaign


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def make(db, **overrides):
    defaults = dict(
        external_id="cmp_1", name="n", channel="meta", objective="conversion",
        health_score=50, enrichment_source="llm", flags=[], currency="INR",
        spend=1000.0, revenue=5000.0, conversions=10, impressions=1000, clicks=100,
    )
    db.add(Campaign(**{**defaults, **overrides}))
    db.commit()


class TestAggregates:
    def test_unmeasured_channel_has_null_roas_not_zero(self, db):
        """The regression: an influencer buy with no revenue recorded must not
        be reported as returning 0.0x."""
        make(db, external_id="a", channel="influencer", spend=350000.0,
             revenue=None, conversions=None)
        agg = collect_aggregates(db)
        bucket = agg["by_channel"]["influencer"]
        assert bucket["roas"] is None
        assert bucket["revenue"] is None
        assert bucket["campaigns_missing_revenue"] == 1

    def test_partially_measured_channel_still_reports_roas(self, db):
        make(db, external_id="a", channel="meta", spend=1000.0, revenue=4000.0)
        make(db, external_id="b", channel="meta", spend=1000.0, revenue=None)
        bucket = collect_aggregates(db)["by_channel"]["meta"]
        assert bucket["roas"] == 2.0     # 4000 revenue over 2000 total spend
        assert bucket["campaigns_missing_revenue"] == 1

    def test_unknown_revenue_spend_is_reported_separately(self, db):
        make(db, external_id="a", spend=1000.0, revenue=5000.0)
        make(db, external_id="b", spend=350000.0, revenue=None)
        agg = collect_aggregates(db)
        assert agg["spend_with_unknown_revenue"] == 350000.0

    def test_zero_spend_channel_has_null_roas(self, db):
        make(db, external_id="a", channel="email", spend=0.0, revenue=1080000.0)
        assert collect_aggregates(db)["by_channel"]["email"]["roas"] is None

    def test_empty_database_does_not_explode(self, db):
        assert collect_aggregates(db) == {"campaign_count": 0}


class TestInsightsEndpointBehaviour:
    def test_no_campaigns_returns_guidance_not_an_error(self, db):
        result = generate_insights(db, Settings(anthropic_api_key=None, database_url="sqlite://"))
        assert result["campaigns_considered"] == 0
        assert "ingest" in result["observations"][0].lower()

    def test_without_a_key_it_still_returns_true_observations(self, db):
        make(db, external_id="a", channel="influencer", spend=350000.0,
             revenue=None, conversions=None, flags=["spend_without_attribution"])
        make(db, external_id="b", channel="meta", spend=1000.0, revenue=5000.0)
        result = generate_insights(db, Settings(anthropic_api_key=None, database_url="sqlite://"))
        assert result["generated_by"] == "rules"
        assert result["observations"]
        # and never describes the unmeasured channel as a 0x performer
        assert not any("0.0x" in o or "0x on" in o for o in result["observations"])
