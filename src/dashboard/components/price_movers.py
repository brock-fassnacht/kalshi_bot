"""Price movers tab - top gainers/losers by 24h price change."""

import pandas as pd
import streamlit as st


def render_price_movers(df: pd.DataFrame):
    """Render top gainers and losers using st.metric cards."""
    if df.empty or "price_change_24h" not in df.columns:
        st.info(
            "Price change data requires snapshots over time. "
            "Keep the dashboard running and this tab will populate after the lookback period."
        )
        return

    # Filter to rows with valid price changes
    movers = df[df["price_change_24h"].notna()].copy()
    if movers.empty:
        st.info("No price change data available yet.")
        return

    movers = movers.sort_values("price_change_24h", ascending=False)

    # Top gainers
    gainers = movers[movers["price_change_24h"] > 0].head(10)
    losers = movers[movers["price_change_24h"] < 0].tail(10).sort_values("price_change_24h")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top Gainers")
        if gainers.empty:
            st.caption("No gainers in this period")
        else:
            for _, row in gainers.iterrows():
                title = row["title"][:50] if len(row["title"]) > 50 else row["title"]
                change = int(row["price_change_24h"])
                current = row.get("yes_bid")
                current_str = f"{current}c" if pd.notna(current) else "N/A"
                st.metric(
                    label=f"{row['ticker']}",
                    value=current_str,
                    delta=f"{change:+d}c",
                    help=title,
                )

    with col2:
        st.subheader("Top Losers")
        if losers.empty:
            st.caption("No losers in this period")
        else:
            for _, row in losers.iterrows():
                title = row["title"][:50] if len(row["title"]) > 50 else row["title"]
                change = int(row["price_change_24h"])
                current = row.get("yes_bid")
                current_str = f"{current}c" if pd.notna(current) else "N/A"
                st.metric(
                    label=f"{row['ticker']}",
                    value=current_str,
                    delta=f"{change:+d}c",
                    help=title,
                )
