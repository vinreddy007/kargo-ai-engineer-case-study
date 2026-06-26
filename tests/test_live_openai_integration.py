from __future__ import annotations

import json
import os
import sys
import tomllib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

from kargo_media_recommender.llm_parser import OpenAIRequirementParser
from kargo_media_recommender.recommender import RecommendationEngine
from kargo_media_recommender.schemas import ClientRequirements
from kargo_media_recommender.workflow import RecommendationWorkflow


ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.live_openai,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_OPENAI_TESTS") != "1",
        reason="Set RUN_LIVE_OPENAI_TESTS=1 to call the live OpenAI API.",
    ),
]


SAMPLE_BRIEF_EXPECTATIONS = (
    (
        "Acme Shoes is a Retail advertiser running in the US with a $30,000 budget. They care most about click-through rate and do not have a hard impression goal.",
        ClientRequirements("Acme Shoes", "Retail", "ctr", "US", 30_000, None, "bundle_allowed"),
    ),
    (
        "Burger Bazaar is a QSR advertiser running in the US with a $15,000 budget. They care most about in-view rate and want at least 800,000 impressions.",
        ClientRequirements("Burger Bazaar", "QSR", "in_view_rate", "US", 15_000, 800_000, "bundle_allowed"),
    ),
    (
        "CineVerse is an Entertainment advertiser running in EMEA with a $40,000 budget. They care most about click-through rate and want a straightforward recommendation.",
        ClientRequirements("CineVerse", "Entertainment", "ctr", "EMEA", 40_000, None, "single_product_preferred"),
    ),
    (
        "Stellar Bank is a Finance advertiser running in the US with a $20,000 budget. They care most about click-through rate and want at least 800,000 impressions.",
        ClientRequirements("Stellar Bank", "Finance", "ctr", "US", 20_000, 800_000, "bundle_allowed"),
    ),
    (
        "Wanderlust Air is a Travel advertiser running in APAC with an $18,000 budget. They care most about in-view rate and want at least 850,000 impressions.",
        ClientRequirements("Wanderlust Air", "Travel", "in_view_rate", "APAC", 18_000, 850_000, "bundle_allowed"),
    ),
    (
        "A new mobile app wants efficient awareness in the US with a $25,000 budget and strong in-view performance, but they have not provided their vertical.",
        ClientRequirements(None, None, "in_view_rate", "US", 25_000, None, "bundle_allowed"),
    ),
    (
        "A Finance brand wants a recommendation for EMEA with a $25,000 budget, but they have not said whether they care more about click-through rate or in-view rate.",
        ClientRequirements(None, "Finance", None, "EMEA", 25_000, None, "bundle_allowed"),
    ),
)


@pytest.fixture(scope="session")
def live_openai_parser() -> OpenAIRequirementParser:
    _configure_openai_key()
    return OpenAIRequirementParser(model=os.getenv("OPENAI_LIVE_TEST_MODEL", "gpt-5.5"))


def test_live_openai_parses_existing_sample_briefs(live_openai_parser: OpenAIRequirementParser):
    with (ROOT / "client_briefs.json").open(encoding="utf-8") as file:
        sample_briefs = json.load(file)

    assert sample_briefs == [brief for brief, _expected in SAMPLE_BRIEF_EXPECTATIONS]

    for brief, expected in SAMPLE_BRIEF_EXPECTATIONS:
        parsed = live_openai_parser.parse(brief)
        _assert_requirements_match(parsed, expected)


def test_live_openai_parser_merges_clarification_follow_up(
    live_openai_parser: OpenAIRequirementParser,
):
    initial = live_openai_parser.parse(
        "A new mobile app wants efficient awareness in the US with a $25,000 budget and strong in-view performance, but they have not provided their vertical."
    )
    follow_up = live_openai_parser.parse("Entertainment", existing_requirements=initial)

    assert initial.missing_required_fields == ("vertical",)
    assert follow_up == ClientRequirements(
        advertiser_name=initial.advertiser_name,
        vertical="Entertainment",
        primary_kpi="in_view_rate",
        geo="US",
        budget=25_000,
        impression_goal=None,
        recommendation_style="bundle_allowed",
    )


def test_live_openai_workflow_recommends_for_sample_brief(
    live_openai_parser: OpenAIRequirementParser,
):
    workflow = RecommendationWorkflow.from_data_dir(ROOT, parser=live_openai_parser)

    output = workflow.run(
        "Stellar Bank is a Finance advertiser running in the US with a $20,000 budget. They care most about click-through rate and want at least 800,000 impressions."
    )

    assert output.needs_clarification is False
    assert output.recommendation is not None
    assert output.recommendation.status == "bundle"
    assert [product.product_id for product in output.recommendation.recommended_products] == [
        "P005",
        "P003",
    ]


def test_live_openai_parses_synthetic_brief_mix(live_openai_parser: OpenAIRequirementParser):
    synthetic_cases = _load_synthetic_cases()
    engine = RecommendationEngine.from_data_dir(ROOT)

    for case in synthetic_cases:
        parsed = live_openai_parser.parse(case.brief)
        _assert_requirements_match(parsed, case.requirements)

        result = engine.recommend(parsed)
        assert result.status != "no_viable_option"
        assert result.missing_fields == ()
        if parsed.recommendation_style == "single_product_preferred":
            assert len(result.recommended_products) <= 1
        if result.status == "bundle" and parsed.impression_goal is not None:
            assert (
                sum(product.forecasted_impressions for product in result.recommended_products)
                >= parsed.impression_goal
            )


def _configure_openai_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return

    secrets_path = ROOT / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        with secrets_path.open("rb") as file:
            secrets = tomllib.load(file)
        api_key = secrets.get("OPENAI_API_KEY")
        if api_key:
            os.environ["OPENAI_API_KEY"] = str(api_key)
            return

    pytest.skip("OPENAI_API_KEY is not set and .streamlit/secrets.toml did not provide one.")


def _load_synthetic_cases():
    module_path = ROOT / "tests" / "test_synthetic_brief_scenarios.py"
    spec = spec_from_file_location("synthetic_brief_scenarios", module_path)
    module = module_from_spec(spec)
    sys.modules["synthetic_brief_scenarios"] = module
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SYNTHETIC_BRIEF_CASES


def _assert_requirements_match(
    parsed: ClientRequirements,
    expected: ClientRequirements,
) -> None:
    assert parsed.vertical == expected.vertical
    assert parsed.primary_kpi == expected.primary_kpi
    assert parsed.geo == expected.geo
    assert parsed.budget == pytest.approx(expected.budget)
    assert parsed.impression_goal == expected.impression_goal
    assert parsed.recommendation_style == expected.recommendation_style
