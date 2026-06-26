from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

from kargo_media_recommender.recommender import RecommendationEngine
from kargo_media_recommender.schemas import ClientRequirements


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SyntheticBriefCase:
    name: str
    brief: str
    requirements: ClientRequirements


SYNTHETIC_BRIEF_CASES = (
    SyntheticBriefCase(
        "retail_us_ctr_open",
        "Acme Shoes is a Retail advertiser running in the US with a $30,000 budget. They care most about click-through rate and do not have a hard impression goal.",
        ClientRequirements("Acme Shoes", "Retail", "ctr", "US", 30_000, None, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "retail_emea_view_goal",
        "Northstar Apparel is a Retail advertiser running in EMEA with a $24,000 budget. They care most about in-view rate and want at least 850,000 impressions.",
        ClientRequirements("Northstar Apparel", "Retail", "in_view_rate", "EMEA", 24_000, 850_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "retail_apac_ctr_goal",
        "BrightCart is a Retail advertiser running in APAC with a $21,000 budget. They care most about click-through rate and want at least 700,000 impressions.",
        ClientRequirements("BrightCart", "Retail", "ctr", "APAC", 21_000, 700_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "retail_us_view_simple",
        "FitStreet wants a simple Retail recommendation in the US with a $22,000 budget focused on in-view rate.",
        ClientRequirements("FitStreet", "Retail", "in_view_rate", "US", 22_000, None, "single_product_preferred"),
    ),
    SyntheticBriefCase(
        "retail_emea_ctr_max",
        "MarketLane is a Retail advertiser in EMEA with a $32,000 budget. Maximize full budget delivery while optimizing click-through rate.",
        ClientRequirements("MarketLane", "Retail", "ctr", "EMEA", 32_000, None, "maximize_budget_delivery"),
    ),
    SyntheticBriefCase(
        "retail_apac_view_open",
        "StyleLoop is a Retail advertiser in APAC with an $18,000 budget. They care most about in-view rate and have no hard impression goal.",
        ClientRequirements("StyleLoop", "Retail", "in_view_rate", "APAC", 18_000, None, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "finance_us_ctr_goal",
        "Stellar Bank is a Finance advertiser running in the US with a $20,000 budget. They care most about click-through rate and want at least 800,000 impressions.",
        ClientRequirements("Stellar Bank", "Finance", "ctr", "US", 20_000, 800_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "finance_emea_view_open",
        "Atlas Card is a Finance advertiser running in EMEA with a $25,000 budget. They care most about in-view rate and do not have a hard impression goal.",
        ClientRequirements("Atlas Card", "Finance", "in_view_rate", "EMEA", 25_000, None, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "finance_apac_ctr_max",
        "LumenPay is a Finance advertiser in APAC with a $28,000 budget. Spend as much budget as possible while prioritizing click-through rate.",
        ClientRequirements("LumenPay", "Finance", "ctr", "APAC", 28_000, None, "maximize_budget_delivery"),
    ),
    SyntheticBriefCase(
        "finance_us_view_simple",
        "Harbor Bank wants one straightforward Finance recommendation in the US with an $18,000 budget focused on in-view rate.",
        ClientRequirements("Harbor Bank", "Finance", "in_view_rate", "US", 18_000, None, "single_product_preferred"),
    ),
    SyntheticBriefCase(
        "finance_emea_ctr_goal",
        "Pinnacle Credit is a Finance advertiser running in EMEA with a $26,000 budget. They care most about click-through rate and want at least 900,000 impressions.",
        ClientRequirements("Pinnacle Credit", "Finance", "ctr", "EMEA", 26_000, 900_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "finance_apac_view_goal",
        "Summit Wallet is a Finance advertiser running in APAC with a $19,000 budget. They care most about in-view rate and want at least 650,000 impressions.",
        ClientRequirements("Summit Wallet", "Finance", "in_view_rate", "APAC", 19_000, 650_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "travel_apac_view_goal",
        "Wanderlust Air is a Travel advertiser running in APAC with an $18,000 budget. They care most about in-view rate and want at least 850,000 impressions.",
        ClientRequirements("Wanderlust Air", "Travel", "in_view_rate", "APAC", 18_000, 850_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "travel_us_ctr_open",
        "Vista Hotels is a Travel advertiser running in the US with a $27,000 budget. They care most about click-through rate and have no hard impression goal.",
        ClientRequirements("Vista Hotels", "Travel", "ctr", "US", 27_000, None, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "travel_emea_view_simple",
        "EuroStay wants a straightforward Travel recommendation in EMEA with a $16,000 budget focused on in-view rate.",
        ClientRequirements("EuroStay", "Travel", "in_view_rate", "EMEA", 16_000, None, "single_product_preferred"),
    ),
    SyntheticBriefCase(
        "travel_apac_ctr_max",
        "JetPath is a Travel advertiser in APAC with a $34,000 budget. Maximize delivery while optimizing click-through rate.",
        ClientRequirements("JetPath", "Travel", "ctr", "APAC", 34_000, None, "maximize_budget_delivery"),
    ),
    SyntheticBriefCase(
        "travel_us_view_goal",
        "TrailQuest is a Travel advertiser running in the US with a $23,000 budget. They care most about in-view rate and want at least 1,000,000 impressions.",
        ClientRequirements("TrailQuest", "Travel", "in_view_rate", "US", 23_000, 1_000_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "travel_emea_ctr_goal",
        "CityHop is a Travel advertiser running in EMEA with a $20,000 budget. They care most about click-through rate and want at least 750,000 impressions.",
        ClientRequirements("CityHop", "Travel", "ctr", "EMEA", 20_000, 750_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "qsr_us_view_goal",
        "Burger Bazaar is a QSR advertiser running in the US with a $15,000 budget. They care most about in-view rate and want at least 800,000 impressions.",
        ClientRequirements("Burger Bazaar", "QSR", "in_view_rate", "US", 15_000, 800_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "qsr_emea_ctr_open",
        "SnackBox is a QSR advertiser running in EMEA with a $22,000 budget. They care most about click-through rate and have no hard impression goal.",
        ClientRequirements("SnackBox", "QSR", "ctr", "EMEA", 22_000, None, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "qsr_apac_view_max",
        "NoodleGo is a QSR advertiser in APAC with a $25,000 budget. Use the full budget while prioritizing in-view rate.",
        ClientRequirements("NoodleGo", "QSR", "in_view_rate", "APAC", 25_000, None, "maximize_budget_delivery"),
    ),
    SyntheticBriefCase(
        "qsr_us_ctr_simple",
        "Taco Tower wants a simple QSR recommendation in the US with a $17,000 budget focused on click-through rate.",
        ClientRequirements("Taco Tower", "QSR", "ctr", "US", 17_000, None, "single_product_preferred"),
    ),
    SyntheticBriefCase(
        "qsr_emea_view_goal",
        "FreshBowl is a QSR advertiser running in EMEA with a $19,000 budget. They care most about in-view rate and want at least 700,000 impressions.",
        ClientRequirements("FreshBowl", "QSR", "in_view_rate", "EMEA", 19_000, 700_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "qsr_apac_ctr_goal",
        "TeaDash is a QSR advertiser running in APAC with an $18,000 budget. They care most about click-through rate and want at least 650,000 impressions.",
        ClientRequirements("TeaDash", "QSR", "ctr", "APAC", 18_000, 650_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "entertainment_emea_ctr_simple",
        "CineVerse is an Entertainment advertiser running in EMEA with a $40,000 budget. They care most about click-through rate and want a straightforward recommendation.",
        ClientRequirements("CineVerse", "Entertainment", "ctr", "EMEA", 40_000, None, "single_product_preferred"),
    ),
    SyntheticBriefCase(
        "entertainment_us_view_open",
        "StreamHouse is an Entertainment advertiser running in the US with a $24,000 budget. They care most about in-view rate and do not have a hard impression goal.",
        ClientRequirements("StreamHouse", "Entertainment", "in_view_rate", "US", 24_000, None, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "entertainment_apac_ctr_max",
        "GameOn is an Entertainment advertiser in APAC with a $31,000 budget. Maximize full budget delivery while optimizing click-through rate.",
        ClientRequirements("GameOn", "Entertainment", "ctr", "APAC", 31_000, None, "maximize_budget_delivery"),
    ),
    SyntheticBriefCase(
        "entertainment_emea_view_goal",
        "Showtime Plus is an Entertainment advertiser running in EMEA with a $21,000 budget. They care most about in-view rate and want at least 700,000 impressions.",
        ClientRequirements("Showtime Plus", "Entertainment", "in_view_rate", "EMEA", 21_000, 700_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "entertainment_us_ctr_goal",
        "MusicBox is an Entertainment advertiser running in the US with a $26,000 budget. They care most about click-through rate and want at least 900,000 impressions.",
        ClientRequirements("MusicBox", "Entertainment", "ctr", "US", 26_000, 900_000, "bundle_allowed"),
    ),
    SyntheticBriefCase(
        "entertainment_apac_view_simple",
        "AnimeNow wants one clear Entertainment recommendation in APAC with a $15,000 budget focused on in-view rate.",
        ClientRequirements("AnimeNow", "Entertainment", "in_view_rate", "APAC", 15_000, None, "single_product_preferred"),
    ),
)


def test_synthetic_brief_cases_cover_all_core_categories():
    verticals = Counter(case.requirements.vertical for case in SYNTHETIC_BRIEF_CASES)
    geos = Counter(case.requirements.geo for case in SYNTHETIC_BRIEF_CASES)
    kpis = Counter(case.requirements.primary_kpi for case in SYNTHETIC_BRIEF_CASES)
    styles = Counter(case.requirements.recommendation_style for case in SYNTHETIC_BRIEF_CASES)

    assert verticals == {
        "Retail": 6,
        "Finance": 6,
        "Travel": 6,
        "QSR": 6,
        "Entertainment": 6,
    }
    assert set(geos) == {"US", "EMEA", "APAC"}
    assert set(kpis) == {"ctr", "in_view_rate"}
    assert set(styles) == {
        "bundle_allowed",
        "single_product_preferred",
        "maximize_budget_delivery",
    }
    assert sum(case.requirements.impression_goal is not None for case in SYNTHETIC_BRIEF_CASES) >= 12
    assert all(case.brief for case in SYNTHETIC_BRIEF_CASES)


@pytest.mark.parametrize(
    "case",
    SYNTHETIC_BRIEF_CASES,
    ids=lambda case: case.name,
)
def test_synthetic_brief_recommendations_follow_planning_invariants(
    case: SyntheticBriefCase,
):
    engine = RecommendationEngine.from_data_dir(ROOT)

    result = engine.recommend(case.requirements)

    assert result.missing_fields == ()
    assert result.candidates

    if case.requirements.recommendation_style == "single_product_preferred":
        assert len(result.recommended_products) <= 1
        assert result.status in {"single_product", "single_product_budget_caveat"}

    if result.status == "single_product":
        assert len(result.recommended_products) == 1
        selected = result.recommended_products[0]
        assert selected.budget == pytest.approx(case.requirements.budget)
        if case.requirements.impression_goal is not None:
            assert selected.forecasted_impressions >= case.requirements.impression_goal

    if result.status == "single_product_budget_caveat":
        assert len(result.recommended_products) == 1
        assert result.recommended_products[0].budget <= case.requirements.budget

    if result.status == "bundle":
        assert 2 <= len(result.recommended_products) <= engine.max_bundle_size
        assert sum(product.budget for product in result.recommended_products) == pytest.approx(
            case.requirements.budget
        )
        if case.requirements.impression_goal is not None:
            assert (
                sum(product.forecasted_impressions for product in result.recommended_products)
                >= case.requirements.impression_goal
            )
        for product in result.recommended_products:
            assert (
                product.forecasted_impressions
                <= product.confidence_adjusted_available_impressions + 0.5
            )

    assert result.status != "no_viable_option"
