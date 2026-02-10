"""Market table component - sortable table with all market data."""

import pandas as pd
import streamlit as st


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to the market DataFrame."""
    if df.empty:
        return df

    filtered = df.copy()

    # Search filter
    if filters.get("search"):
        search = filters["search"]
        mask = (
            filtered["ticker"].str.lower().str.contains(search, na=False)
            | filtered["title"].str.lower().str.contains(search, na=False)
        )
        filtered = filtered[mask]

    # Category filter
    if filters.get("categories"):
        filtered = filtered[filtered["category"].isin(filters["categories"])]

    # Volume filter
    if filters.get("min_volume", 0) > 0:
        filtered = filtered[filtered["volume"] >= filters["min_volume"]]

    # Active only
    if filters.get("show_active_only"):
        filtered = filtered[filtered["status"].isin(["open", "active"])]

    return filtered


def render_market_table(df: pd.DataFrame, filters: dict) -> str | None:
    """
    Render the main sortable market table.

    Returns the selected ticker if a row is clicked, else None.
    """
    filtered = apply_filters(df, filters)

    if filtered.empty:
        st.info("No markets match the current filters.")
        return None

    # Select and rename columns for display
    display_cols = {
        "ticker": "Ticker",
        "title": "Title",
        "yes_bid": "Yes Bid",
        "yes_ask": "Yes Ask",
        "spread": "Spread",
        "volume": "Volume",
        "open_interest": "OI",
        "last_price": "Last",
    }

    # Add orderbook columns if available
    if "total_yes_depth" in filtered.columns:
        display_cols["total_yes_depth"] = "Depth $ (Y)"
        display_cols["total_no_depth"] = "Depth $ (N)"
        display_cols["near_touch_yes"] = "Near $ (Y)"
        display_cols["near_touch_no"] = "Near $ (N)"

    # Add price change if available
    if "price_change_24h" in filtered.columns:
        display_cols["price_change_24h"] = "24h Chg"

    if "open_time" in filtered.columns:
        display_cols["open_time"] = "Opened"

    # Only keep columns that exist
    available_cols = [c for c in display_cols.keys() if c in filtered.columns]
    display_df = filtered[available_cols].copy()
    display_df.columns = [display_cols[c] for c in available_cols]

    # Sort by volume descending by default
    if "Volume" in display_df.columns:
        display_df = display_df.sort_values("Volume", ascending=False)

    st.caption(f"Showing {len(display_df)} of {len(df)} markets")

    # Use dataframe with selection
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Title": st.column_config.TextColumn(width="large"),
            "Ticker": st.column_config.TextColumn(width="small"),
            "Volume": st.column_config.NumberColumn(format="%d"),
            "OI": st.column_config.NumberColumn(format="%d"),
            "24h Chg": st.column_config.NumberColumn(format="%+d"),
            "Depth $ (Y)": st.column_config.NumberColumn(format="$%.2f"),
            "Depth $ (N)": st.column_config.NumberColumn(format="$%.2f"),
            "Near $ (Y)": st.column_config.NumberColumn(format="$%.2f"),
            "Near $ (N)": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    # Get selected ticker
    if event and event.selection and event.selection.rows:
        row_idx = event.selection.rows[0]
        selected_ticker = filtered.iloc[row_idx]["ticker"]
        return selected_ticker

    return None
