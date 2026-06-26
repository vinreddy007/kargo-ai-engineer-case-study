from pathlib import Path

import pytest

from kargo_media_recommender import ClientRequirements, RecommendationEngine
from kargo_media_recommender.schemas import (
    CampaignHistoryRow,
    InventoryForecast,
    MediaData,
    Product,
)


ROOT = Path(__file__).resolve().parents[1]


def test_recommendation_requests_missing_required_fields():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="New Mobile App",
        geo="US",
        budget=25_000,
        primary_kpi="in-view rate",
    )

    result = engine.recommend(requirements)

    assert result.status == "clarification_needed"
    assert result.missing_fields == ("vertical",)
    assert result.recommended_products == ()


def test_default_style_allows_bundle_when_it_improves_kpi_and_budget_fit():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="Acme Shoes",
        vertical="Retail",
        geo="US",
        budget=30_000,
        primary_kpi="click-through rate",
    )

    result = engine.recommend(requirements)

    assert result.status == "bundle"
    assert [product.product_id for product in result.recommended_products] == ["P003", "P006"]


def test_single_product_preferred_style_recommends_one_clear_product():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="Acme Shoes",
        vertical="Retail",
        geo="US",
        budget=30_000,
        primary_kpi="click-through rate",
        recommendation_style="single_product_preferred",
    )

    result = engine.recommend(requirements)

    assert result.status == "single_product"
    assert [product.product_id for product in result.recommended_products] == ["P005"]
    assert result.recommended_products[0].product_name == "Premium Takeover"


def test_bundle_allowed_style_recommends_kpi_ranked_bundle_when_it_beats_single_product():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="Acme Shoes",
        vertical="Retail",
        geo="US",
        budget=30_000,
        primary_kpi="click-through rate",
        recommendation_style="bundle_allowed",
    )

    result = engine.recommend(requirements)

    assert result.status == "bundle"
    assert [product.product_id for product in result.recommended_products] == ["P003", "P006"]
    assert [product.product_name for product in result.recommended_products] == [
        "Commerce Connect",
        "Attention Builder",
    ]
    assert sum(product.budget for product in result.recommended_products) == 30_000
    assert result.recommended_products[0].forecasted_impressions == 1_427_600
    assert result.recommended_products[1].forecasted_impressions > 613_000
    assert result.rejected_alternatives
    assert result.rejected_alternatives[0].product_id == "P005"
    assert result.rejected_alternatives[0].product_name == "Premium Takeover"
    assert result.rejected_alternatives[0].reasons[0] == (
        "Best single-product alternative, but lower click-through rate than the "
        "selected bundle (0.9% vs. 1.2%)."
    )
    assert result.rationale == (
        "Recommend Commerce Connect + Attention Builder. This bundle has the strongest "
        "blended click-through rate at 1.2% and uses the full $30,000 budget."
    )
    assert "Budget used: $30,000 across Commerce Connect + Attention Builder." in result.tradeoffs


def test_bundle_rejected_alternatives_explain_lower_kpi_instead_of_capacity():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="Finance Brand",
        vertical="Finance",
        geo="EMEA",
        budget=25_000,
        primary_kpi="in_view_rate",
        recommendation_style="bundle_allowed",
    )

    result = engine.recommend(requirements)

    assert result.status == "bundle"
    premium_takeover = next(
        product for product in result.rejected_alternatives if product.product_id == "P005"
    )
    assert premium_takeover.reasons[0] == (
        "Lower in-view rate than the selected bundle (72.8% vs. 75.3%)."
    )
    assert len(premium_takeover.reasons) == 1


def test_inventory_constraints_can_force_a_two_product_bundle():
    media_data = MediaData(
        products={
            "P001": Product("P001", "High CTR", "Best historical CTR.", 10),
            "P002": Product("P002", "Scale Helper", "Additional scale.", 10),
        },
        history_rows=[
            CampaignHistoryRow("P001", "Retail", 100_000, 1_000, 70_000),
            CampaignHistoryRow("P002", "Retail", 100_000, 800, 80_000),
        ],
        inventory={
            ("P001", "Retail", "US"): InventoryForecast("P001", "Retail", "US", 500_000, 1),
            ("P002", "Retail", "US"): InventoryForecast("P002", "Retail", "US", 500_000, 1),
        },
    )
    engine = RecommendationEngine(media_data)
    requirements = ClientRequirements(
        advertiser_name="Acme Shoes",
        vertical="Retail",
        geo="US",
        budget=10_000,
        primary_kpi="ctr",
        impression_goal=900_000,
        recommendation_style="bundle_allowed",
    )

    result = engine.recommend(requirements)

    assert result.status == "bundle"
    assert [product.product_id for product in result.recommended_products] == ["P001", "P002"]
    assert sum(product.budget for product in result.recommended_products) == 10_000
    assert sum(product.forecasted_impressions for product in result.recommended_products) >= 900_000


def test_goal_aware_bundle_mixes_high_ctr_product_with_scale_product():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="Stellar Bank",
        vertical="Finance",
        geo="US",
        budget=20_000,
        primary_kpi="ctr",
        impression_goal=800_000,
        recommendation_style="bundle_allowed",
    )

    result = engine.recommend(requirements)

    assert result.status == "bundle"
    assert [product.product_id for product in result.recommended_products] == ["P005", "P003"]
    assert sum(product.budget for product in result.recommended_products) == pytest.approx(20_000)
    assert sum(product.forecasted_impressions for product in result.recommended_products) == pytest.approx(
        800_000
    )
    assert result.recommended_products[0].budget == pytest.approx(15_058.82, abs=0.01)
    assert result.recommended_products[1].budget == pytest.approx(4_941.18, abs=0.01)
    assert RecommendationEngine._plan_kpi(
        list(result.recommended_products),
        requirements.primary_kpi,
    ) > 0.011024409908975822


def test_straightforward_recommendation_selects_best_product_with_budget_caveat():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="CineVerse",
        vertical="Entertainment",
        geo="EMEA",
        budget=40_000,
        primary_kpi="ctr",
        recommendation_style="single_product_preferred",
    )

    result = engine.recommend(requirements)

    assert result.status == "single_product_budget_caveat"
    assert [product.product_id for product in result.recommended_products] == ["P006"]
    assert result.recommended_products[0].product_name == "Attention Builder"
    assert result.recommended_products[0].budget < 40_000
    assert "strongest single-product fit" in result.rationale


def test_maximize_budget_delivery_can_use_three_product_bundle():
    engine = RecommendationEngine.from_data_dir(ROOT)
    requirements = ClientRequirements(
        advertiser_name="CineVerse",
        vertical="Entertainment",
        geo="EMEA",
        budget=40_000,
        primary_kpi="ctr",
        recommendation_style="maximize_budget_delivery",
    )

    result = engine.recommend(requirements)

    assert result.status == "bundle"
    assert [product.product_id for product in result.recommended_products] == ["P006", "P005", "P003"]
    assert sum(product.budget for product in result.recommended_products) == 40_000


def test_no_viable_option_when_no_matching_product_data_exists():
    media_data = MediaData(
        products={"P001": Product("P001", "High CTR", "Best historical CTR.", 10)},
        history_rows=[CampaignHistoryRow("P001", "Retail", 100_000, 1_000, 70_000)],
        inventory={},
    )
    engine = RecommendationEngine(media_data)
    requirements = ClientRequirements(
        advertiser_name="No Inventory",
        vertical="Retail",
        geo="US",
        budget=10_000,
        primary_kpi="ctr",
    )

    result = engine.recommend(requirements)

    assert result.status == "no_viable_option"
    assert result.recommended_products == ()
    assert result.rejected_alternatives == ()
