from __future__ import annotations

from kargo_media_recommender.display import (
    format_currency,
    format_impressions,
    format_percent,
)
from kargo_media_recommender.schemas import KPI_LABELS, KPI_SHORT_LABELS, RecommendationResult


def format_recommendation_response(result: RecommendationResult) -> str:
    if result.status == "clarification_needed":
        return result.rationale

    requirements = result.requirements
    lines = [
        "Campaign setup:",
        f"- Advertiser: {requirements.advertiser_name or 'Not provided'}",
        f"- Vertical: {requirements.vertical}",
        f"- KPI: {KPI_LABELS[requirements.primary_kpi]}",
        f"- Geo: {requirements.geo}",
        f"- Budget: {format_currency(requirements.budget)}",
    ]
    if requirements.impression_goal is not None:
        lines.append(f"- Impression goal: {format_impressions(requirements.impression_goal)}")

    if result.recommended_products:
        heading = "Recommended product:" if len(result.recommended_products) == 1 else "Recommended bundle:"
        lines.extend(["", heading])
        for product in result.recommended_products:
            kpi_label = KPI_SHORT_LABELS[requirements.primary_kpi]
            kpi_value = (
                product.benchmark_ctr
                if requirements.primary_kpi == "ctr"
                else product.benchmark_in_view_rate
            )
            lines.append(
                "- "
                f"{product.product_name}: {format_currency(product.budget)} budget, "
                f"{format_impressions(product.forecasted_impressions)} forecasted impressions, "
                f"{kpi_label} {format_percent(kpi_value)}, "
                "usable inventory "
                f"{format_impressions(product.confidence_adjusted_available_impressions)}."
            )

    lines.extend(["", "Why this recommendation:", result.rationale])

    if result.tradeoffs:
        lines.extend(["", "Planning notes:"])
        for tradeoff in result.tradeoffs:
            lines.append(f"- {tradeoff}")

    if result.rejected_alternatives:
        lines.extend(["", "Rejected alternatives:"])
        for alternative in result.rejected_alternatives:
            lines.append(f"- {alternative.product_name}: {' '.join(alternative.reasons)}")

    return "\n".join(lines)
