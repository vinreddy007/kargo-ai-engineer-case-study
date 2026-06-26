from pathlib import Path

from kargo_media_recommender.data import load_media_data


ROOT = Path(__file__).resolve().parents[1]


def test_load_media_data_reads_all_case_study_files():
    data = load_media_data(ROOT)

    assert len(data.products) == 6
    assert len(data.history_rows) == 1200
    assert len(data.inventory) == 90

    display_plus = data.products["P001"]
    assert display_plus.product_name == "Display Plus"
    assert display_plus.cpm == 12

    retail_us = data.inventory[("P001", "Retail", "US")]
    assert retail_us.available_imps == 1_466_000
    assert retail_us.inventory_confidence == 0.93
    assert retail_us.confidence_adjusted_available_imps == 1_363_380
