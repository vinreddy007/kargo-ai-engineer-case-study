from __future__ import annotations

import csv
from pathlib import Path

from kargo_media_recommender.schemas import (
    CampaignHistoryRow,
    InventoryForecast,
    MediaData,
    Product,
)


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def load_products(path: str | Path) -> dict[str, Product]:
    products: dict[str, Product] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            product = Product(
                product_id=row["product_id"],
                product_name=row["product_name"],
                product_description=row["product_description"],
                cpm=float(row["cpm"]),
            )
            products[product.product_id] = product
    return products


def load_campaign_history(path: str | Path) -> list[CampaignHistoryRow]:
    rows: list[CampaignHistoryRow] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                CampaignHistoryRow(
                    product_id=row["product_id"],
                    vertical=row["vertical"],
                    impressions=int(row["impressions"]),
                    clicks=int(row["clicks"]),
                    viewable_impressions=int(row["viewable_impressions"]),
                )
            )
    return rows


def load_inventory(path: str | Path) -> dict[tuple[str, str, str], InventoryForecast]:
    forecasts: dict[tuple[str, str, str], InventoryForecast] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        for row in reader:
            forecast = InventoryForecast(
                product_id=row["product_id"],
                vertical=row["vertical"],
                geo=row["geo"],
                available_imps=int(row["available_imps"]),
                inventory_confidence=float(row["inventory_risk"]),
            )
            forecasts[(forecast.product_id, forecast.vertical, forecast.geo)] = forecast
    return forecasts


def load_media_data(data_dir: str | Path | None = None) -> MediaData:
    directory = Path(data_dir) if data_dir is not None else default_data_dir()
    return MediaData(
        products=load_products(directory / "product_catalog.csv"),
        history_rows=load_campaign_history(directory / "campaign_history.csv"),
        inventory=load_inventory(directory / "inventory_forecaster.csv"),
    )
