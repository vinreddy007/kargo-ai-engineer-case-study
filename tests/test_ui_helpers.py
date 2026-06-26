from dataclasses import replace
import json
from pathlib import Path

import pytest

from kargo_media_recommender import RecommendationEngine
from kargo_media_recommender.schemas import ClientRequirements
from kargo_media_recommender.ui_helpers import (
    assistant_chat_summary,
    campaign_summary_items,
    decision_summary_rows,
    escape_markdown_dollars,
    load_sample_briefs,
    recommendation_title,
    rejected_product_rows,
    requirements_rows,
    sample_brief_label,
)


ROOT = Path(__file__).resolve().parents[1]


def test_load_sample_briefs_reads_json_array(tmp_path):
    path = tmp_path / "briefs.json"
    path.write_text(json.dumps(["Brief one", "Brief two"]), encoding="utf-8")

    assert load_sample_briefs(path) == ["Brief one", "Brief two"]


def test_load_sample_briefs_rejects_non_string_array(tmp_path):
    path = tmp_path / "briefs.json"
    path.write_text(json.dumps(["Brief one", 2]), encoding="utf-8")

    with pytest.raises(ValueError):
        load_sample_briefs(path)


def test_requirements_rows_formats_missing_and_present_values():
    requirements = ClientRequirements(
        advertiser_name="Acme Shoes",
        vertical="Retail",
        primary_kpi="click-through rate",
        geo="US",
        budget=30000,
    )

    rows = requirements_rows(requirements)

    assert {"Field": "Advertiser", "Value": "Acme Shoes"} in rows
    assert {"Field": "Primary KPI", "Value": "click-through rate"} in rows
    assert {"Field": "Budget", "Value": "$30,000"} in rows
    assert {"Field": "Impression goal", "Value": "None"} in rows
    assert {"Field": "Recommendation style", "Value": "Bundle allowed"} in rows


def test_recommendation_title_handles_empty_result():
    assert recommendation_title(None) == ""


def test_sample_brief_label_adds_index_and_truncates_long_brief():
    label = sample_brief_label(1, "Acme Shoes is a Retail advertiser running in the US.")

    assert label == "1. Acme Shoes is a Retail advertis..."


def test_assistant_chat_summary_uses_fallback_when_no_recommendation():
    assert assistant_chat_summary(None, "Follow-up question") == "Follow-up question"


def test_assistant_chat_summary_points_to_reviewable_recommendation_details():
    engine = RecommendationEngine.from_data_dir(ROOT)
    result = engine.recommend(
        ClientRequirements(
            advertiser_name="Stellar Bank",
            vertical="Finance",
            geo="US",
            budget=20_000,
            primary_kpi="ctr",
            impression_goal=800_000,
            recommendation_style="bundle_allowed",
        )
    )

    summary = assistant_chat_summary(result, "fallback")

    assert summary.startswith("Recommendation: Premium Takeover, Commerce Connect.")
    assert "Review the setup, delivery, and alternatives below." in summary


def test_campaign_summary_items_use_media_strategy_labels():
    requirements = ClientRequirements(
        advertiser_name="Stellar Bank",
        vertical="Finance",
        primary_kpi="ctr",
        geo="US",
        budget=20_000,
        impression_goal=800_000,
        recommendation_style="bundle_allowed",
    )

    items = campaign_summary_items(requirements)

    assert {"Label": "Advertiser", "Value": "Stellar Bank"} in items
    assert {"Label": "Primary KPI", "Value": "click-through rate"} in items
    assert {"Label": "Budget", "Value": "$20,000"} in items
    assert {"Label": "Impression goal", "Value": "800K"} in items
    assert {"Label": "Planning intent", "Value": "Bundle eligible"} in items


def test_escape_markdown_dollars_prevents_currency_from_rendering_as_math():
    text = "It can use about $15,572 of the $40,000 budget."

    assert escape_markdown_dollars(text) == (
        r"It can use about \$15,572 of the \$40,000 budget."
    )


def test_decision_summary_labels_bundle_items_and_single_product_alternative():
    engine = RecommendationEngine.from_data_dir(ROOT)
    result = engine.recommend(
        ClientRequirements(
            advertiser_name="Acme Shoes",
            vertical="Retail",
            geo="US",
            budget=30_000,
            primary_kpi="ctr",
            recommendation_style="bundle_allowed",
        )
    )

    rows = decision_summary_rows(result)

    assert len(rows) == len(result.candidates)
    assert len(rows) > len(result.recommended_products) + len(result.rejected_alternatives)
    assert rows[0]["Product"] == "Commerce Connect"
    assert "Role" not in rows[0]
    assert rows[0]["Badge"] == "Selected"
    assert rows[0]["Tone"] == "selected"
    assert rows[0]["Decision"] == "Selected"
    assert rows[1]["Product"] == "Attention Builder"
    assert "Role" not in rows[1]
    assert rows[2]["Product"] == "Premium Takeover"
    assert "Role" not in rows[2]
    assert rows[2]["Badge"] == "Best single"
    assert rows[2]["Tone"] == "alternative"
    assert rows[2]["Budget Basis"] == "$30,000 as single product"
    assert rows[2]["Decision"] == "Best single-product alternative"
    assert {row["Product"] for row in rows} == {
        candidate.product.product_name for candidate in result.candidates
    }


def test_decision_summary_prioritizes_lower_kpi_over_inventory_constraints():
    engine = RecommendationEngine.from_data_dir(ROOT)
    result = engine.recommend(
        ClientRequirements(
            advertiser_name="Finance Brand",
            vertical="Finance",
            geo="EMEA",
            budget=25_000,
            primary_kpi="in_view_rate",
            recommendation_style="bundle_allowed",
        )
    )

    rows = decision_summary_rows(result)

    premium_takeover = next(row for row in rows if row["Product"] == "Premium Takeover")
    social_amplifier = next(row for row in rows if row["Product"] == "Social Amplifier")
    assert premium_takeover["Badge"] == "Lower in-view"
    assert premium_takeover["Tone"] == "alternative"
    assert premium_takeover["Decision"] == "Lower in-view rate"
    assert social_amplifier["Badge"] == "Lower in-view"
    assert social_amplifier["Decision"] == "Lower in-view rate"


def test_rejected_cards_normalize_stale_capacity_reasons_for_bundle():
    engine = RecommendationEngine.from_data_dir(ROOT)
    result = engine.recommend(
        ClientRequirements(
            advertiser_name="Finance Brand",
            vertical="Finance",
            geo="EMEA",
            budget=25_000,
            primary_kpi="ctr",
            recommendation_style="bundle_allowed",
        )
    )
    stale_rejected = tuple(
        replace(
            product,
            reasons=(
                "Insufficient usable inventory for the full budget "
                "(929,660 usable vs. 1,785,714 needed).",
            ),
        )
        for product in result.rejected_alternatives
    )

    rows = rejected_product_rows(
        stale_rejected,
        result.requirements.primary_kpi,
        result.recommended_products,
    )

    attention_builder = next(row for row in rows if row["Product"] == "Attention Builder")
    assert attention_builder["Badge"] == "Lower CTR"
    assert attention_builder["Tone"] == "alternative"
    assert attention_builder["Reason"] == (
        "Lower click-through rate than the selected bundle (0.8% vs. 1.2%)."
    )
