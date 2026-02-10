"""Orderbook processing for Kalshi markets."""

from typing import Optional

import pandas as pd

from ..models import OrderbookSummary


def kalshi_orderbook_to_df(orderbook: dict, ticker: str) -> pd.DataFrame:
    """
    Convert Kalshi order book API response to DataFrame.

    Kalshi format: {"orderbook": {"yes": [[price, size], ...], "no": [[price, size], ...]}}
    """
    ob_data = orderbook.get("orderbook", orderbook) or {}

    yes_levels = ob_data.get("yes") or []
    no_levels = ob_data.get("no") or []

    max_levels = max(len(yes_levels), len(no_levels), 1)

    rows = []
    for i in range(max_levels):
        row = {
            "ticker": ticker,
            "level": i,
            "yes_price": yes_levels[i][0] if i < len(yes_levels) else None,
            "yes_size": yes_levels[i][1] if i < len(yes_levels) else None,
            "no_price": no_levels[i][0] if i < len(no_levels) else None,
            "no_size": no_levels[i][1] if i < len(no_levels) else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_orderbook_summary(orderbook: dict, ticker: str, near_touch_cents: int = 5) -> OrderbookSummary:
    """
    Compute orderbook summary with total depth and near-touch liquidity.

    Prices are in cents. Depth in dollars = sum(price * size) / 100.
    Near-touch = levels within `near_touch_cents` of best price.
    """
    ob_data = orderbook.get("orderbook", orderbook) or {}

    yes_levels = ob_data.get("yes") or []
    no_levels = ob_data.get("no") or []

    # YES side
    total_yes = 0.0
    near_yes = 0.0
    best_yes = yes_levels[0][0] if yes_levels else None

    for price, size in yes_levels:
        dollar_value = (price * size) / 100.0
        total_yes += dollar_value
        if best_yes is not None and abs(price - best_yes) <= near_touch_cents:
            near_yes += dollar_value

    # NO side
    total_no = 0.0
    near_no = 0.0
    best_no = no_levels[0][0] if no_levels else None

    for price, size in no_levels:
        dollar_value = (price * size) / 100.0
        total_no += dollar_value
        if best_no is not None and abs(price - best_no) <= near_touch_cents:
            near_no += dollar_value

    return OrderbookSummary(
        ticker=ticker,
        total_yes_depth_dollars=round(total_yes, 2),
        total_no_depth_dollars=round(total_no, 2),
        near_touch_yes_dollars=round(near_yes, 2),
        near_touch_no_dollars=round(near_no, 2),
        yes_levels=len(yes_levels),
        no_levels=len(no_levels),
        best_yes_price=best_yes,
        best_no_price=best_no,
    )


def get_best_prices(df: pd.DataFrame) -> dict:
    """Extract best bid/ask prices from an order book DataFrame."""
    best_bid = df["yes_price"].dropna().min() if "yes_price" in df.columns else None
    best_ask = df["no_price"].dropna().min() if "no_price" in df.columns else None

    mid_price = None
    spread = None

    if pd.notna(best_bid) and pd.notna(best_ask):
        mid_price = (best_bid + best_ask) / 2
        spread = best_ask - best_bid

    return {
        "best_bid": best_bid if pd.notna(best_bid) else None,
        "best_ask": best_ask if pd.notna(best_ask) else None,
        "mid_price": mid_price,
        "spread": spread,
    }
