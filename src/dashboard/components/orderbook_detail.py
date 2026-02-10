"""Orderbook detail view - expandable depth chart for a selected market."""

import json

import pandas as pd
import streamlit as st

from ...data.orderbook import kalshi_orderbook_to_df
from ...models import OrderbookSummary


def render_orderbook_detail(summary: OrderbookSummary, raw_orderbook: dict):
    """Render orderbook detail with depth chart for a selected market."""
    st.subheader(f"Orderbook: {summary.ticker}")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("YES Depth", f"${summary.total_yes_depth_dollars:.2f}")
    with col2:
        st.metric("NO Depth", f"${summary.total_no_depth_dollars:.2f}")
    with col3:
        st.metric("YES Near-Touch", f"${summary.near_touch_yes_dollars:.2f}")
    with col4:
        st.metric("NO Near-Touch", f"${summary.near_touch_no_dollars:.2f}")

    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("YES Levels", summary.yes_levels)
    with col6:
        st.metric("NO Levels", summary.no_levels)
    with col7:
        best = f"{summary.best_yes_price}c" if summary.best_yes_price else "N/A"
        st.metric("Best YES", best)
    with col8:
        best = f"{summary.best_no_price}c" if summary.best_no_price else "N/A"
        st.metric("Best NO", best)

    # Depth chart
    ob_data = raw_orderbook.get("orderbook", raw_orderbook) or {}
    yes_levels = ob_data.get("yes") or []
    no_levels = ob_data.get("no") or []

    if yes_levels or no_levels:
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if yes_levels:
                st.caption("YES Side (buy YES)")
                yes_df = pd.DataFrame(yes_levels, columns=["Price (c)", "Size"])
                yes_df["Dollar Value"] = (yes_df["Price (c)"] * yes_df["Size"]) / 100
                st.bar_chart(yes_df.set_index("Price (c)")["Dollar Value"])
                st.dataframe(yes_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No YES orders")

        with chart_col2:
            if no_levels:
                st.caption("NO Side (buy NO)")
                no_df = pd.DataFrame(no_levels, columns=["Price (c)", "Size"])
                no_df["Dollar Value"] = (no_df["Price (c)"] * no_df["Size"]) / 100
                st.bar_chart(no_df.set_index("Price (c)")["Dollar Value"])
                st.dataframe(no_df, hide_index=True, use_container_width=True)
            else:
                st.caption("No NO orders")
    else:
        st.info("No orderbook data available.")
