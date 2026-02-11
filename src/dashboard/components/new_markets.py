"""New markets tab - recently opened events."""

from datetime import datetime

import pandas as pd
import streamlit as st


def render_new_markets(df: pd.DataFrame):
    """Render the new events table (events opened in lookback period)."""
    if df.empty:
        st.info("No new events found in the lookback period.")
        return

    st.caption(f"{len(df)} new events")

    # Add days-to-expiry column
    if "close_time" in df.columns:
        now = datetime.utcnow()
        df = df.copy()
        df["days_to_expiry"] = df["close_time"].apply(
            lambda ct: round((ct - now).total_seconds() / 86400, 1) if pd.notna(ct) else None
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
