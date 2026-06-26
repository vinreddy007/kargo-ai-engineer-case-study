from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Protocol

from kargo_media_recommender.schemas import ClientRequirements


class RequirementParser(Protocol):
    def parse(
        self,
        message: str,
        existing_requirements: ClientRequirements | None = None,
    ) -> ClientRequirements:
        ...


class RequirementParseError(ValueError):
    pass


class OpenAIRequirementParser:
    def __init__(
        self,
        model: str = "gpt-5.5",
        reasoning_effort: str = "low",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client

    def parse(
        self,
        message: str,
        existing_requirements: ClientRequirements | None = None,
    ) -> ClientRequirements:
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            input=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "existing_requirements": _requirements_to_json(existing_requirements),
                            "latest_user_message": message,
                        },
                        separators=(",", ":"),
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "client_requirements",
                    "schema": _CLIENT_REQUIREMENTS_SCHEMA,
                    "strict": True,
                }
            },
        )

        try:
            payload = json.loads(response.output_text)
        except json.JSONDecodeError as exc:
            raise RequirementParseError("LLM did not return valid JSON.") from exc

        try:
            return ClientRequirements(
                advertiser_name=_clean_optional_string(payload["advertiser_name"]),
                vertical=payload["vertical"],
                primary_kpi=payload["primary_kpi"],
                geo=payload["geo"],
                budget=payload["budget"],
                impression_goal=payload["impression_goal"],
                recommendation_style=payload["recommendation_style"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RequirementParseError("LLM returned invalid client requirements.") from exc


def _requirements_to_json(requirements: ClientRequirements | None) -> dict[str, Any] | None:
    if requirements is None:
        return None
    return asdict(requirements)


def _clean_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


_SYSTEM_PROMPT = """Extract campaign requirements for a Kargo media recommendation assistant.

Return only fields in the supplied schema.

Rules:
- Preserve existing_requirements unless the latest user message clearly updates or corrects a field.
- Use null for unknown values.
- advertiser_name is optional. Do not invent one.
- vertical must be one of Retail, Finance, Travel, QSR, Entertainment, or null.
- primary_kpi must be ctr for click-through rate/clicks, in_view_rate for in-view/viewability/awareness visibility, or null.
- geo must be US, EMEA, APAC, or null.
- budget is a dollar amount as a number, or null.
- impression_goal is an integer number of impressions, or null. Phrases like "no hard impression goal" mean null.
- recommendation_style must be:
  - single_product_preferred only when the client asks for a straightforward/simple/one-product/single-product/best product recommendation, or otherwise clearly implies they want one product.
  - bundle_allowed when the client asks for a recommendation but does not clearly limit the answer to one product.
  - maximize_budget_delivery when the client explicitly asks to use the full budget, maximize delivery, maximize scale, or spend as much budget as possible.
- Do not infer a vertical from a vague advertiser description such as "new mobile app" unless a supported vertical is explicit.
"""


_CLIENT_REQUIREMENTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "advertiser_name": {
            "type": ["string", "null"],
            "description": "Client or advertiser name when explicitly available.",
        },
        "vertical": {
            "type": ["string", "null"],
            "enum": ["Retail", "Finance", "Travel", "QSR", "Entertainment", None],
        },
        "primary_kpi": {
            "type": ["string", "null"],
            "enum": ["ctr", "in_view_rate", None],
        },
        "geo": {
            "type": ["string", "null"],
            "enum": ["US", "EMEA", "APAC", None],
        },
        "budget": {
            "type": ["number", "null"],
            "description": "Campaign budget in dollars.",
        },
        "impression_goal": {
            "type": ["integer", "null"],
            "description": "Minimum desired impressions when specified.",
        },
        "recommendation_style": {
            "type": "string",
            "enum": [
                "single_product_preferred",
                "bundle_allowed",
                "maximize_budget_delivery",
            ],
            "description": "Planning intent inferred from the brief.",
        },
    },
    "required": [
        "advertiser_name",
        "vertical",
        "primary_kpi",
        "geo",
        "budget",
        "impression_goal",
        "recommendation_style",
    ],
    "additionalProperties": False,
}
