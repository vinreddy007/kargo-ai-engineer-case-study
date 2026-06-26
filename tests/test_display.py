from kargo_media_recommender.display import (
    format_currency,
    format_impressions,
    format_percent,
)


def test_format_impressions_uses_compact_media_scale():
    assert format_impressions(800_000) == "800K"
    assert format_impressions(1_427_600) == "1.4M"
    assert format_impressions(2_040_886) == "2M"
    assert format_impressions(9_500) == "9,500"
    assert format_impressions(None) == "None"


def test_format_percent_uses_one_decimal_place():
    assert format_percent(0.012355) == "1.2%"
    assert format_percent(0.75308) == "75.3%"
    assert format_percent(None) == "None"


def test_format_currency_keeps_whole_dollars():
    assert format_currency(21_414.49) == "$21,414"
    assert format_currency(None) == "Missing"
