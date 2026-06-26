from __future__ import annotations

from typing import Any


def format_currency(value: float | int | None) -> str:
    if value is None:
        return "Missing"
    return f"${float(value):,.0f}"


def format_impressions(value: float | int | None, none_label: str = "None") -> str:
    if value is None:
        return none_label

    numeric = float(value)
    sign = "-" if numeric < 0 else ""
    absolute = abs(numeric)

    if absolute >= 1_000_000:
        return f"{sign}{_format_one_decimal(absolute / 1_000_000)}M"

    if absolute >= 10_000:
        rounded_thousands = round(absolute / 1_000)
        if rounded_thousands >= 1_000:
            return f"{sign}1M"
        return f"{sign}{rounded_thousands:,.0f}K"

    return f"{numeric:,.0f}"


def format_percent(value: Any, none_label: str = "None") -> str:
    if value is None:
        return none_label
    return f"{float(value):.1%}"


def _format_one_decimal(value: float) -> str:
    formatted = f"{value:.1f}"
    return formatted.removesuffix(".0")
