"""New markets tab - recently opened events."""

import pandas as pd
import streamlit as st

from src.dashboard.utils import compute_days_to_expiry


def render_new_markets(df: pd.DataFrame):
    """Render the new events table (events opened in lookback period)."""
    if df.empty:
        st.info("No new events found in the lookback period.")
        return

    st.caption(f"{len(df)} new events")

    # Add days-to-expiry column (prefer ticker-derived date, fall back to close_time)
    if "event_ticker" in df.columns:
        df = df.copy()
        df["days_to_expiry"] = df.apply(
            lambda row: compute_days_to_expiry(
                row["event_ticker"],
                row.get("close_time"),
            ),
            axis=1,
        )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "event_ticker": None,
            "title": st.column_config.TextColumn("Event", width="large"),
            "category": "Category",
            "market_count": st.column_config.NumberColumn("Markets", format="%d"),
            "total_volume": st.column_config.NumberColumn("Total Volume", format="%d"),
            "total_oi": st.column_config.NumberColumn("Total OI", format="%d"),
            "open_time": st.column_config.DatetimeColumn("Opened", format="MMM DD, HH:mm"),
            "close_time": st.column_config.DatetimeColumn("Closes", format="MMM DD, HH:mm"),
            "days_to_expiry": st.column_config.NumberColumn("Days to Expiry", format="%.1f"),
        },
    )
