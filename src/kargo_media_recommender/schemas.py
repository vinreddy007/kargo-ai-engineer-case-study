from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Vertical = Literal["Retail", "Finance", "Travel", "QSR", "Entertainment"]
PrimaryKpi = Literal["ctr", "in_view_rate"]
Geo = Literal["US", "EMEA", "APAC"]
RecommendationStyle = Literal[
    "single_product_preferred",
    "bundle_allowed",
    "maximize_budget_delivery",
]
RecommendationStatus = Literal[
    "clarification_needed",
    "single_product",
    "single_product_budget_caveat",
    "bundle",
    "no_viable_option",
]

VALID_VERTICALS = {"Retail", "Finance", "Travel", "QSR", "Entertainment"}
VALID_PRIMARY_KPIS = {"ctr", "in_view_rate"}
VALID_GEOS = {"US", "EMEA", "APAC"}
VALID_RECOMMENDATION_STYLES = {
    "single_product_preferred",
    "bundle_allowed",
    "maximize_budget_delivery",
}

KPI_LABELS = {
    "ctr": "click-through rate",
    "in_view_rate": "in-view rate",
}

KPI_SHORT_LABELS = {
    "ctr": "CTR",
    "in_view_rate": "In-view rate",
}


def _normalize_vertical(value: str | None) -> str | None:
    if value is None:
        return None
    aliases = {vertical.lower(): vertical for vertical in VALID_VERTICALS}
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"Unsupported vertical: {value!r}")
    return normalized


def _normalize_geo(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized not in VALID_GEOS:
        raise ValueError(f"Unsupported geo: {value!r}")
    return normalized


def _normalize_kpi(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ctr": "ctr",
        "click_through_rate": "ctr",
        "clickthrough_rate": "ctr",
        "clicks": "ctr",
        "in_view_rate": "in_view_rate",
        "in_view": "in_view_rate",
        "viewability": "in_view_rate",
        "viewable_rate": "in_view_rate",
        "strong_in_view_performance": "in_view_rate",
    }
    kpi = aliases.get(normalized)
    if kpi is None:
        raise ValueError(f"Unsupported primary KPI: {value!r}")
    return kpi


def _normalize_recommendation_style(value: str | None) -> str:
    if value is None:
        return "bundle_allowed"
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single_product_preferred": "single_product_preferred",
        "single_product": "single_product_preferred",
        "one_product": "single_product_preferred",
        "best_product": "single_product_preferred",
        "straightforward": "single_product_preferred",
        "simple": "single_product_preferred",
        "bundle_allowed": "bundle_allowed",
        "bundle": "bundle_allowed",
        "mix": "bundle_allowed",
        "product_mix": "bundle_allowed",
        "maximize_budget_delivery": "maximize_budget_delivery",
        "maximize_budget": "maximize_budget_delivery",
        "full_budget": "maximize_budget_delivery",
        "maximize_delivery": "maximize_budget_delivery",
    }
    recommendation_style = aliases.get(normalized)
    if recommendation_style is None:
        raise ValueError(f"Unsupported recommendation style: {value!r}")
    return recommendation_style


@dataclass(frozen=True)
class ClientRequirements:
    advertiser_name: str | None = None
    vertical: str | None = None
    primary_kpi: str | None = None
    geo: str | None = None
    budget: float | None = None
    impression_goal: int | None = None
    recommendation_style: str = "bundle_allowed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertical", _normalize_vertical(self.vertical))
        object.__setattr__(self, "primary_kpi", _normalize_kpi(self.primary_kpi))
        object.__setattr__(self, "geo", _normalize_geo(self.geo))
        object.__setattr__(
            self,
            "recommendation_style",
            _normalize_recommendation_style(self.recommendation_style),
        )
        if self.budget is not None and self.budget <= 0:
            raise ValueError("Budget must be greater than zero.")
        if self.impression_goal is not None and self.impression_goal <= 0:
            raise ValueError("Impression goal must be greater than zero.")

    @property
    def missing_required_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if self.vertical is None:
            missing.append("vertical")
        if self.primary_kpi is None:
            missing.append("primary_kpi")
        if self.geo is None:
            missing.append("geo")
        if self.budget is None:
            missing.append("budget")
        return tuple(missing)


@dataclass(frozen=True)
class Product:
    product_id: str
    product_name: str
    product_description: str
    cpm: float


@dataclass(frozen=True)
class CampaignHistoryRow:
    product_id: str
    vertical: str
    impressions: int
    clicks: int
    viewable_impressions: int


@dataclass(frozen=True)
class ProductBenchmark:
    product_id: str
    vertical: str
    impressions: int
    clicks: int
    viewable_impressions: int
    ctr: float
    in_view_rate: float

    def metric_value(self, primary_kpi: str) -> float:
        if primary_kpi == "ctr":
            return self.ctr
        if primary_kpi == "in_view_rate":
            return self.in_view_rate
        raise ValueError(f"Unsupported primary KPI: {primary_kpi!r}")


@dataclass(frozen=True)
class InventoryForecast:
    product_id: str
    vertical: str
    geo: str
    available_imps: int
    inventory_confidence: float

    @property
    def confidence_adjusted_available_imps(self) -> float:
        return self.available_imps * self.inventory_confidence


@dataclass(frozen=True)
class MediaData:
    products: dict[str, Product]
    history_rows: list[CampaignHistoryRow]
    inventory: dict[tuple[str, str, str], InventoryForecast]


@dataclass(frozen=True)
class ProductCandidate:
    product: Product
    benchmark: ProductBenchmark
    inventory: InventoryForecast
    estimated_impressions: float
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def viable(self) -> bool:
        return not self.rejection_reasons

    def score(self, primary_kpi: str) -> float:
        return self.benchmark.metric_value(primary_kpi)


@dataclass(frozen=True)
class RecommendedProduct:
    product_id: str
    product_name: str
    budget: float
    cpm: float
    forecasted_impressions: float
    benchmark_ctr: float
    benchmark_in_view_rate: float
    available_impressions: int
    inventory_confidence: float
    confidence_adjusted_available_impressions: float


@dataclass(frozen=True)
class RejectedProduct:
    product_id: str
    product_name: str
    benchmark_ctr: float
    benchmark_in_view_rate: float
    estimated_impressions: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecommendationResult:
    status: RecommendationStatus
    requirements: ClientRequirements
    missing_fields: tuple[str, ...]
    recommended_products: tuple[RecommendedProduct, ...]
    rejected_alternatives: tuple[RejectedProduct, ...]
    candidates: tuple[ProductCandidate, ...]
    rationale: str
    tradeoffs: tuple[str, ...] = field(default_factory=tuple)
