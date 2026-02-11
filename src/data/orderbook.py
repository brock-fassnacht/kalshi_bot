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


def compute_orderbook_summary(
    orderbook: dict,
    ticker: str,
    near_touch_cents: int = 5,
    near_mid_range_cents: int = 20,
    yes_bid: Optional[int] = None,
    yes_ask: Optional[int] = None,
) -> OrderbookSummary:
    """
    Compute orderbook summary with total depth, near-touch, and near-mid liquidity.

    Prices are in cents. Depth in dollars = sum(price * size) / 100.
    Near-touch = levels within `near_touch_cents` of best price.
    Near-mid = levels within `near_mid_range_cents` of the midpoint between bid and ask.
    """
    ob_data = orderbook.get("orderbook", orderbook) or {}

    yes_levels = ob_data.get("yes") or []
    no_levels = ob_data.get("no") or []

    # Compute midpoint from market bid/ask if provided, else from orderbook best prices
    mid = None
    if yes_bid is not None and yes_ask is not None:
        mid = (yes_bid + yes_ask) / 2.0
    elif yes_levels:
        # Orderbook YES shows ask prices (prices to buy YES)
        mid = yes_levels[0][0]

    # YES side
    total_yes = 0.0
    near_yes = 0.0
    near_mid_yes = 0.0
    best_yes = yes_levels[0][0] if yes_levels else None

    for price, size in yes_levels:
        dollar_value = size  # Each contract is $1, size = number of contracts at this price
        total_yes += dollar_value
        if best_yes is not None and abs(price - best_yes) <= near_touch_cents:
            near_yes += dollar_value
        if mid is not None and abs(price - mid) <= near_mid_range_cents:
            near_mid_yes += dollar_value

    # NO side
    total_no = 0.0
    near_no = 0.0
    near_mid_no = 0.0
    best_no = no_levels[0][0] if no_levels else None

    # NO mid is complementary: if YES mid is 60, NO mid is 40
    no_mid = (100 - mid) if mid is not None else None

    for price, size in no_levels:
        dollar_value = size
        total_no += dollar_value
        if best_no is not None and abs(price - best_no) <= near_touch_cents:
            near_no += dollar_value
        if no_mid is not None and abs(price - no_mid) <= near_mid_range_cents:
            near_mid_no += dollar_value

    near_mid_total = near_mid_yes + near_mid_no

    return OrderbookSummary(
        ticker=ticker,
        total_yes_depth_dollars=round(total_yes, 2),
        total_no_depth_dollars=round(total_no, 2),
        near_touch_yes_dollars=round(near_yes, 2),
        near_touch_no_dollars=round(near_no, 2),
        near_mid_depth_dollars=round(near_mid_total, 2),
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
