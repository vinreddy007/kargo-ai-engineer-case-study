import json

from kargo_media_recommender.llm_parser import OpenAIRequirementParser
from kargo_media_recommender.schemas import ClientRequirements


class FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class FakeResponses:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse(json.dumps(self.payload))


class FakeClient:
    def __init__(self, payload):
        self.responses = FakeResponses(payload)


def test_openai_requirement_parser_uses_structured_outputs_and_normalizes_result():
    client = FakeClient(
        {
            "advertiser_name": "Acme Shoes",
            "vertical": "Retail",
            "primary_kpi": "ctr",
            "geo": "US",
            "budget": 30000,
            "impression_goal": None,
            "recommendation_style": "bundle_allowed",
        }
    )
    parser = OpenAIRequirementParser(client=client)

    requirements = parser.parse("Acme Shoes is a Retail advertiser in the US with $30,000.")

    assert requirements == ClientRequirements(
        advertiser_name="Acme Shoes",
        vertical="Retail",
        primary_kpi="ctr",
        geo="US",
        budget=30000,
    )

    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["reasoning"] == {"effort": "low"}
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True


def test_openai_requirement_parser_sends_existing_requirements_for_follow_up_merge():
    client = FakeClient(
        {
            "advertiser_name": "New Mobile App",
            "vertical": "Entertainment",
            "primary_kpi": "in_view_rate",
            "geo": "US",
            "budget": 25000,
            "impression_goal": None,
            "recommendation_style": "bundle_allowed",
        }
    )
    parser = OpenAIRequirementParser(client=client)
    existing = ClientRequirements(
        advertiser_name="New Mobile App",
        geo="US",
        budget=25000,
        primary_kpi="in_view_rate",
    )

    parser.parse("Entertainment", existing_requirements=existing)

    user_payload = json.loads(client.responses.calls[0]["input"][1]["content"])
    assert user_payload["existing_requirements"]["geo"] == "US"
    assert user_payload["latest_user_message"] == "Entertainment"
