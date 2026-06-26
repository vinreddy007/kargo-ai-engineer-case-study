from __future__ import annotations

FIELD_QUESTIONS = {
    "vertical": "Which vertical is this advertiser in: Retail, Finance, Travel, QSR, or Entertainment?",
    "primary_kpi": "Should I optimize for click-through rate or in-view rate?",
    "geo": "Which geo should the campaign run in: US, EMEA, or APAC?",
    "budget": "What is the campaign budget in dollars?",
}


def build_clarification_question(missing_fields: tuple[str, ...]) -> str:
    if not missing_fields:
        return ""
    if len(missing_fields) == 1:
        return FIELD_QUESTIONS[missing_fields[0]]

    questions = [FIELD_QUESTIONS[field] for field in missing_fields]
    return "I need a few details before recommending: " + " ".join(questions)
