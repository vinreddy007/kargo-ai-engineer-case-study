from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kargo_media_recommender.display import (
    format_currency,
    format_impressions,
    format_percent,
)
from kargo_media_recommender.schemas import (
    ClientRequirements,
    KPI_LABELS,
    KPI_SHORT_LABELS,
    ProductCandidate,
    RecommendationResult,
    RecommendedProduct,
    RejectedProduct,
)


def load_sample_briefs(path: str | Path) -> list[str]:
    with Path(path).open(encoding="utf-8") as file:
        briefs = json.load(file)
    if not isinstance(briefs, list) or not all(isinstance(brief, str) for brief in briefs):
        raise ValueError("Sample briefs file must contain a JSON array of strings.")
    return briefs


def sample_brief_label(index: int, brief: str, max_chars: int = 34) -> str:
    normalized = " ".join(brief.split())
    if len(normalized) > max_chars:
        normalized = normalized[: max_chars - 3].rstrip() + "..."
    return f"{index}. {normalized}"


def assistant_chat_summary(result: RecommendationResult | None, fallback: str) -> str:
    if result is None:
        return fallback

    if result.status in {"single_product", "single_product_budget_caveat"} and result.recommended_products:
        product = result.recommended_products[0]
        return (
            f"Recommendation: {product.product_name}. "
            f"Review the setup, delivery, and alternatives below."
        )

    if result.status == "bundle" and result.recommended_products:
        product_names = ", ".join(product.product_name for product in result.recommended_products)
        return (
            f"Recommendation: {product_names}. "
            f"Review the setup, delivery, and alternatives below."
        )

    if result.status == "no_viable_option":
        return "I could not find a viable product or bundle. See the rejected alternatives below."

    return fallback


def escape_markdown_dollars(text: str) -> str:
    return text.replace("$", r"\$")


def campaign_summary_items(requirements: ClientRequirements | None) -> list[dict[str, str]]:
    if requirements is None:
        return []

    items = [
        {"Label": "Advertiser", "Value": requirements.advertiser_name or "Not provided"},
        {"Label": "Vertical", "Value": requirements.vertical or "Missing"},
        {
            "Label": "Primary KPI",
            "Value": KPI_LABELS.get(requirements.primary_kpi, "Missing"),
        },
        {"Label": "Geo", "Value": requirements.geo or "Missing"},
        {"Label": "Budget", "Value": _format_currency(requirements.budget)},
        {
            "Label": "Impression goal",
            "Value": _format_impressions(requirements.impression_goal)
            if requirements.impression_goal is not None
            else "No hard goal",
        },
    ]

    if requirements.recommendation_style == "single_product_preferred":
        items.append({"Label": "Planning intent", "Value": "Single product requested"})
    elif requirements.recommendation_style == "maximize_budget_delivery":
        items.append({"Label": "Planning intent", "Value": "Maximize delivery"})
    else:
        items.append({"Label": "Planning intent", "Value": "Bundle eligible"})

    return items


def requirements_rows(requirements: ClientRequirements | None) -> list[dict[str, str]]:
    if requirements is None:
        return []

    rows = [
        {"Field": "Advertiser", "Value": requirements.advertiser_name or "Not provided"},
        {"Field": "Vertical", "Value": requirements.vertical or "Missing"},
        {
            "Field": "Primary KPI",
            "Value": KPI_LABELS.get(requirements.primary_kpi, "Missing"),
        },
        {"Field": "Geo", "Value": requirements.geo or "Missing"},
        {"Field": "Budget", "Value": _format_currency(requirements.budget)},
        {"Field": "Impression goal", "Value": _format_impressions(requirements.impression_goal)},
        {
            "Field": "Recommendation style",
            "Value": _format_recommendation_style(requirements.recommendation_style),
        },
    ]
    return rows


def recommended_product_rows(
    products: tuple[RecommendedProduct, ...],
    primary_kpi: str,
) -> list[dict[str, str]]:
    return [
        {
            "Product": product.product_name,
            "Badge": "Selected",
            "Tone": "selected",
            "Budget": _format_currency(product.budget),
            "CPM": _format_currency(product.cpm),
            "Forecasted Imps": _format_impressions(product.forecasted_impressions),
            "KPI Label": KPI_SHORT_LABELS[primary_kpi],
            "KPI Value": _format_percent(_recommended_product_kpi_value(product, primary_kpi)),
            "Inventory Confidence": _format_percent(product.inventory_confidence),
            "Usable Inventory": _format_impressions(
                product.confidence_adjusted_available_impressions
            ),
        }
        for product in products
    ]


def rejected_product_rows(
    products: tuple[RejectedProduct, ...],
    primary_kpi: str,
    recommended_products: tuple[RecommendedProduct, ...] = (),
) -> list[dict[str, str]]:
    selected_plan_kpi = _recommended_plan_kpi(recommended_products, primary_kpi)
    selected_plan_label = (
        "selected bundle" if len(recommended_products) > 1 else "selected product"
    )
    rows = []
    for product in products:
        reason = _rejected_product_reason_text(
            product,
            primary_kpi,
            selected_plan_kpi,
            selected_plan_label,
        )
        rows.append(
            {
                "Product": product.product_name,
                "Badge": _badge_from_reason(reason),
                "Tone": _tone_from_reason(reason),
                "Forecasted Imps": _format_impressions(product.estimated_impressions),
                "KPI Label": KPI_SHORT_LABELS[primary_kpi],
                "KPI Value": _format_percent(_rejected_product_kpi_value(product, primary_kpi)),
                "Reason": reason,
            }
        )
    return rows


def decision_summary_rows(result: RecommendationResult) -> list[dict[str, str]]:
    primary_kpi = result.requirements.primary_kpi
    if primary_kpi is None:
        return []

    kpi_label = KPI_SHORT_LABELS[primary_kpi]
    recommended_by_id = {
        product.product_id: product for product in result.recommended_products
    }
    rejected_reasons_by_id = {
        product.product_id: product.reasons for product in result.rejected_alternatives
    }
    selected_plan_kpi = _recommended_plan_kpi(result.recommended_products, primary_kpi)
    selected_plan_label = (
        "selected bundle" if len(result.recommended_products) > 1 else "selected product"
    )
    rows: list[dict[str, str]] = []

    for candidate in result.candidates:
        recommended_product = recommended_by_id.get(candidate.product.product_id)
        if recommended_product is not None:
            rows.append(
                _selected_decision_row(
                    recommended_product,
                    result.status,
                    kpi_label,
                    primary_kpi,
                )
            )
            continue

        reasons = rejected_reasons_by_id.get(
            candidate.product.product_id,
            candidate.rejection_reasons,
        )
        rows.append(
            _candidate_decision_row(
                candidate,
                reasons,
                result.requirements.budget,
                kpi_label,
                primary_kpi,
                selected_plan_kpi,
                selected_plan_label,
            )
        )

    return rows


def recommendation_title(result: RecommendationResult | None) -> str:
    if result is None:
        return ""
    if result.status in {"single_product", "single_product_budget_caveat"}:
        return "Recommended Product"
    if result.status == "bundle":
        return "Recommended Bundle"
    if result.status == "no_viable_option":
        return "No Viable Option"
    return "Clarification Needed"


def _format_currency(value: float | int | None) -> str:
    return format_currency(value)


def _format_impressions(value: float | int | None) -> str:
    return format_impressions(value)


def _format_percent(value: Any) -> str:
    return format_percent(value)


def _format_recommendation_style(value: str) -> str:
    labels = {
        "single_product_preferred": "Single product preferred",
        "bundle_allowed": "Bundle allowed",
        "maximize_budget_delivery": "Maximize budget delivery",
    }
    return labels.get(value, value)


def _recommended_product_kpi_value(product: RecommendedProduct, primary_kpi: str) -> float:
    if primary_kpi == "ctr":
        return product.benchmark_ctr
    if primary_kpi == "in_view_rate":
        return product.benchmark_in_view_rate
    raise ValueError(f"Unsupported primary KPI: {primary_kpi!r}")


def _rejected_product_kpi_value(product: RejectedProduct, primary_kpi: str) -> float:
    if primary_kpi == "ctr":
        return product.benchmark_ctr
    if primary_kpi == "in_view_rate":
        return product.benchmark_in_view_rate
    raise ValueError(f"Unsupported primary KPI: {primary_kpi!r}")


def _candidate_kpi_value(candidate: ProductCandidate, primary_kpi: str) -> float:
    if primary_kpi == "ctr":
        return candidate.benchmark.ctr
    if primary_kpi == "in_view_rate":
        return candidate.benchmark.in_view_rate
    raise ValueError(f"Unsupported primary KPI: {primary_kpi!r}")


def _selected_decision_row(
    product: RecommendedProduct,
    status: str,
    kpi_label: str,
    primary_kpi: str,
) -> dict[str, str]:
    decision = "Selected with budget caveat" if status == "single_product_budget_caveat" else "Selected"
    return {
        "Product": product.product_name,
        "Badge": "Selected",
        "Tone": "selected",
        "Budget Basis": f"{_format_currency(product.budget)} allocated",
        "Forecasted Imps": _format_impressions(product.forecasted_impressions),
        kpi_label: _format_percent(_recommended_product_kpi_value(product, primary_kpi)),
        "Decision": decision,
    }


def _candidate_decision_row(
    candidate: ProductCandidate,
    reasons: tuple[str, ...],
    budget: float | None,
    kpi_label: str,
    primary_kpi: str,
    selected_plan_kpi: float | None,
    selected_plan_label: str,
) -> dict[str, str]:
    reason = _candidate_reason_text(
        candidate,
        reasons,
        primary_kpi,
        selected_plan_kpi,
        selected_plan_label,
    )
    return {
        "Product": candidate.product.product_name,
        "Badge": _badge_from_reason(reason),
        "Tone": _tone_from_reason(reason),
        "Budget Basis": f"{_format_currency(budget)} as single product",
        "Forecasted Imps": _format_impressions(candidate.estimated_impressions),
        kpi_label: _format_percent(_candidate_kpi_value(candidate, primary_kpi)),
        "Decision": _decision_from_reason(reason),
    }


def _candidate_reason_text(
    candidate: ProductCandidate,
    reasons: tuple[str, ...],
    primary_kpi: str,
    selected_plan_kpi: float | None,
    selected_plan_label: str,
) -> str:
    if reasons and _has_kpi_comparison_reason(reasons[0]):
        return " ".join(reasons)

    comparison = _kpi_comparison_reason(
        candidate,
        primary_kpi,
        selected_plan_kpi,
        selected_plan_label,
    )
    if comparison:
        if selected_plan_label == "selected bundle":
            return comparison
        return " ".join((comparison, *reasons))

    if reasons:
        return " ".join(reasons)

    if selected_plan_kpi is not None:
        return (
            f"Viable, but did not improve {KPI_LABELS[primary_kpi]} over the "
            f"{selected_plan_label} ({_format_percent(_candidate_kpi_value(candidate, primary_kpi))} "
            f"vs. {_format_percent(selected_plan_kpi)})."
        )
    return "Not selected."


def _has_kpi_comparison_reason(reason: str) -> bool:
    return (
        reason.startswith("Lower ")
        or reason.startswith("Best single-product alternative")
        or reason.startswith("Viable, but did not improve")
    )


def _kpi_comparison_reason(
    candidate: ProductCandidate,
    primary_kpi: str,
    selected_plan_kpi: float | None,
    selected_plan_label: str,
) -> str | None:
    if selected_plan_kpi is None:
        return None

    candidate_kpi = _candidate_kpi_value(candidate, primary_kpi)
    if candidate_kpi >= selected_plan_kpi - 1e-12:
        return None

    return (
        f"Lower {KPI_LABELS[primary_kpi]} than the {selected_plan_label} "
        f"({_format_percent(candidate_kpi)} vs. {_format_percent(selected_plan_kpi)})."
    )


def _rejected_product_reason_text(
    product: RejectedProduct,
    primary_kpi: str,
    selected_plan_kpi: float | None,
    selected_plan_label: str,
) -> str:
    if product.reasons and _has_kpi_comparison_reason(product.reasons[0]):
        if selected_plan_label == "selected bundle":
            return product.reasons[0]
        return " ".join(product.reasons)

    comparison = _rejected_product_kpi_comparison_reason(
        product,
        primary_kpi,
        selected_plan_kpi,
        selected_plan_label,
    )
    if comparison:
        if selected_plan_label == "selected bundle":
            return comparison
        return " ".join((comparison, *product.reasons))

    if product.reasons:
        return " ".join(product.reasons)
    return "Not selected."


def _rejected_product_kpi_comparison_reason(
    product: RejectedProduct,
    primary_kpi: str,
    selected_plan_kpi: float | None,
    selected_plan_label: str,
) -> str | None:
    if selected_plan_kpi is None:
        return None

    product_kpi = _rejected_product_kpi_value(product, primary_kpi)
    if product_kpi >= selected_plan_kpi - 1e-12:
        return None

    return (
        f"Lower {KPI_LABELS[primary_kpi]} than the {selected_plan_label} "
        f"({_format_percent(product_kpi)} vs. {_format_percent(selected_plan_kpi)})."
    )


def _rejected_product_decision(product: RejectedProduct) -> str:
    return _decision_from_reason(" ".join(product.reasons))


def _decision_from_reason(reason: str) -> str:
    if reason.startswith("Best single-product alternative"):
        return "Best single-product alternative"
    if reason.startswith("Lower in-view rate"):
        return "Lower in-view rate"
    if reason.startswith("Lower click-through rate"):
        return "Lower CTR"
    if reason.startswith("Viable, but did not improve"):
        return "Lower KPI than selected recommendation"
    if "Insufficient usable inventory" in reason:
        return "Insufficient usable inventory"
    if "impression goal" in reason:
        return "Below impression goal"
    return "Not selected"


def _rejected_product_badge(product: RejectedProduct) -> str:
    return _badge_from_reason(" ".join(product.reasons))


def _badge_from_reason(reason: str) -> str:
    if reason.startswith("Best single-product alternative"):
        return "Best single"
    if reason.startswith("Lower in-view rate"):
        return "Lower in-view"
    if reason.startswith("Lower click-through rate"):
        return "Lower CTR"
    if reason.startswith("Viable, but did not improve"):
        return "Lower KPI"
    if "Insufficient usable inventory" in reason:
        return "Capacity limit"
    if "impression goal" in reason:
        return "Below goal"
    return "Not selected"


def _rejected_product_tone(product: RejectedProduct) -> str:
    return _tone_from_reason(" ".join(product.reasons))


def _tone_from_reason(reason: str) -> str:
    if reason.startswith("Best single-product alternative"):
        return "alternative"
    if reason.startswith("Lower "):
        return "alternative"
    if reason.startswith("Viable, but did not improve"):
        return "alternative"
    return "blocked"


def _recommended_plan_kpi(
    products: tuple[RecommendedProduct, ...],
    primary_kpi: str,
) -> float | None:
    if not products:
        return None
    total_impressions = sum(product.forecasted_impressions for product in products)
    if total_impressions <= 0:
        return None

    return (
        sum(
            product.forecasted_impressions
            * _recommended_product_kpi_value(product, primary_kpi)
            for product in products
        )
        / total_impressions
    )
