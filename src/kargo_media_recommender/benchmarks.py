from __future__ import annotations

from collections import defaultdict

from kargo_media_recommender.schemas import CampaignHistoryRow, ProductBenchmark


def calculate_benchmarks(
    history_rows: list[CampaignHistoryRow],
) -> dict[tuple[str, str], ProductBenchmark]:
    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"impressions": 0, "clicks": 0, "viewable_impressions": 0}
    )

    for row in history_rows:
        key = (row.product_id, row.vertical)
        totals[key]["impressions"] += row.impressions
        totals[key]["clicks"] += row.clicks
        totals[key]["viewable_impressions"] += row.viewable_impressions

    benchmarks: dict[tuple[str, str], ProductBenchmark] = {}
    for (product_id, vertical), values in totals.items():
        impressions = values["impressions"]
        if impressions <= 0:
            raise ValueError(f"No impressions for benchmark {product_id}/{vertical}.")

        benchmarks[(product_id, vertical)] = ProductBenchmark(
            product_id=product_id,
            vertical=vertical,
            impressions=impressions,
            clicks=values["clicks"],
            viewable_impressions=values["viewable_impressions"],
            ctr=values["clicks"] / impressions,
            in_view_rate=values["viewable_impressions"] / impressions,
        )

    return benchmarks


def estimate_impressions(budget: float, cpm: float) -> float:
    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")
    if cpm <= 0:
        raise ValueError("CPM must be greater than zero.")
    return budget / cpm * 1000
