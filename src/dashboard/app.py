"""Main Streamlit dashboard for Kalshi market monitoring."""

import time

import streamlit as st

from ..config import get_settings
from .data_service import DashboardDataService
from .components.sidebar import render_sidebar
from .components.market_table import render_market_table
from .components.new_markets import render_new_markets
from .components.price_movers import render_price_movers
from .components.orderbook_detail import render_orderbook_detail

st.set_page_config(
    page_title="Kalshi Market Dashboard",
    page_icon="📊",
    layout="wide",
)


def get_data_service() -> DashboardDataService:
    """Get or create the data service singleton in session state."""
    if "data_service" not in st.session_state:
        settings = get_settings()
        st.session_state.data_service = DashboardDataService(settings)
    return st.session_state.data_service


def main():
    service = get_data_service()

    # Initialize data on first load
    if "master_df" not in st.session_state:
        with st.spinner("Loading markets..."):
            service.refresh_markets()
            st.session_state.master_df = service.build_master_dataframe()
            st.session_state.last_refresh = time.time()

    # Sidebar
    categories = service.get_categories()
    filters = render_sidebar(categories)

    # Handle refresh
    should_refresh = filters["refresh_clicked"]
    if filters["auto_refresh"] and filters["refresh_interval"]:
        elapsed = time.time() - st.session_state.get("last_refresh", 0)
        if elapsed >= filters["refresh_interval"]:
            should_refresh = True

    if should_refresh:
        with st.spinner("Refreshing markets..."):
            service.refresh_markets()
            st.session_state.master_df = service.build_master_dataframe()
            st.session_state.last_refresh = time.time()

    df = st.session_state.get("master_df")
    if df is None or df.empty:
        st.warning("No market data loaded. Click 'Refresh Now' in the sidebar.")
        return

    # KPI cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Markets", len(df))
    with col2:
        total_vol = int(df["volume"].sum()) if "volume" in df.columns else 0
        st.metric("Total Volume", f"{total_vol:,}")
    with col3:
        active = len(df[df["status"].isin(["open", "active"])]) if "status" in df.columns else 0
        st.metric("Active Markets", active)

    # Tabs
    tab_all, tab_new, tab_movers = st.tabs(["All Markets", "New Markets", "Price Movers"])

    with tab_all:
        selected_ticker = render_market_table(df, filters)

        # Orderbook detail on row select
        if selected_ticker:
            st.divider()
            with st.spinner(f"Fetching orderbook for {selected_ticker}..."):
                try:
                    summary, raw_ob = service.fetch_orderbook(selected_ticker)
                    render_orderbook_detail(summary, raw_ob)
                except Exception as e:
                    st.error(f"Failed to fetch orderbook: {e}")

    with tab_new:
        new_df = service.get_new_markets()
        render_new_markets(new_df)

    with tab_movers:
        render_price_movers(df)

    # Auto-refresh via rerun
    if filters["auto_refresh"] and filters["refresh_interval"]:
        time.sleep(filters["refresh_interval"])
        st.rerun()


if __name__ == "__main__":
    main()
