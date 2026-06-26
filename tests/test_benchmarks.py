import pytest

from kargo_media_recommender.benchmarks import calculate_benchmarks, estimate_impressions
from kargo_media_recommender.schemas import CampaignHistoryRow


def test_calculate_benchmarks_aggregates_by_product_and_vertical():
    rows = [
        CampaignHistoryRow("P001", "Retail", 1000, 10, 700),
        CampaignHistoryRow("P001", "Retail", 3000, 30, 2400),
        CampaignHistoryRow("P002", "Retail", 2000, 20, 1000),
    ]

    benchmarks = calculate_benchmarks(rows)

    p001_retail = benchmarks[("P001", "Retail")]
    assert p001_retail.impressions == 4000
    assert p001_retail.clicks == 40
    assert p001_retail.viewable_impressions == 3100
    assert p001_retail.ctr == pytest.approx(0.01)
    assert p001_retail.in_view_rate == pytest.approx(0.775)


def test_estimate_impressions_uses_budget_and_cpm():
    assert estimate_impressions(30_000, 12) == pytest.approx(2_500_000)


def test_estimate_impressions_rejects_invalid_values():
    with pytest.raises(ValueError):
        estimate_impressions(0, 12)

    with pytest.raises(ValueError):
        estimate_impressions(30_000, 0)
