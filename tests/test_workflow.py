from pathlib import Path

from kargo_media_recommender import ClientRequirements, RecommendationWorkflow


ROOT = Path(__file__).resolve().parents[1]


class MappingParser:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def parse(self, message, existing_requirements=None):
        self.calls.append((message, existing_requirements))
        value = self.mapping[message]
        if callable(value):
            return value(existing_requirements)
        return value


def test_workflow_recommends_for_complete_brief():
    parser = MappingParser(
        {
            "complete": ClientRequirements(
                advertiser_name="Acme Shoes",
                vertical="Retail",
                geo="US",
                budget=30000,
                primary_kpi="ctr",
            )
        }
    )
    workflow = RecommendationWorkflow.from_data_dir(ROOT, parser=parser)

    output = workflow.run("complete")

    assert output.needs_clarification is False
    assert output.recommendation.status == "bundle"
    assert [product.product_id for product in output.recommendation.recommended_products] == [
        "P003",
        "P006",
    ]
    assert "Campaign setup:" in output.response_text
    assert "Recommended bundle:" in output.response_text
    assert "Why this recommendation:" in output.response_text
    assert "Planning notes:" in output.response_text
    assert "Commerce Connect" in output.response_text


def test_workflow_asks_clarification_when_required_field_is_missing():
    parser = MappingParser(
        {
            "missing vertical": ClientRequirements(
                advertiser_name="New Mobile App",
                geo="US",
                budget=25000,
                primary_kpi="in_view_rate",
            )
        }
    )
    workflow = RecommendationWorkflow.from_data_dir(ROOT, parser=parser)

    output = workflow.run("missing vertical")

    assert output.needs_clarification is True
    assert output.missing_fields == ("vertical",)
    assert output.recommendation is None
    assert output.response_text == (
        "Which vertical is this advertiser in: Retail, Finance, Travel, QSR, or Entertainment?"
    )


def test_workflow_merges_follow_up_answer_through_parser_state():
    partial = ClientRequirements(
        advertiser_name="New Mobile App",
        geo="US",
        budget=25000,
        primary_kpi="in_view_rate",
    )

    def complete_from_follow_up(existing):
        return ClientRequirements(
            advertiser_name=existing.advertiser_name,
            vertical="Entertainment",
            geo=existing.geo,
            budget=existing.budget,
            primary_kpi=existing.primary_kpi,
        )

    parser = MappingParser({"initial": partial, "Entertainment": complete_from_follow_up})
    workflow = RecommendationWorkflow.from_data_dir(ROOT, parser=parser)

    first = workflow.run("initial")
    second = workflow.run("Entertainment", current_requirements=first.requirements)

    assert first.needs_clarification is True
    assert second.needs_clarification is False
    assert second.requirements.vertical == "Entertainment"
    assert second.recommendation.status in {
        "single_product",
        "single_product_budget_caveat",
        "bundle",
    }
    assert parser.calls[1][1] == partial
