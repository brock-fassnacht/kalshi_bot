"""New markets tab - recently opened markets."""

import pandas as pd
import streamlit as st


def render_new_markets(df: pd.DataFrame):
    """Render the new markets table (markets opened in last 24h)."""
    if df.empty:
        st.info("No new markets found in the lookback period.")
        return

    st.caption(f"{len(df)} new markets")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": "Ticker",
            "title": st.column_config.TextColumn("Title", width="large"),
            "category": "Category",
            "yes_bid": "Yes Bid",
            "yes_ask": "Yes Ask",
            "spread": "Spread",
            "volume": st.column_config.NumberColumn("Volume", format="%d"),
            "open_interest": st.column_config.NumberColumn("OI", format="%d"),
            "open_time": st.column_config.DatetimeColumn("Opened", format="MMM DD, HH:mm"),
        },
    )
