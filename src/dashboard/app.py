"""Main Streamlit dashboard for Kalshi market monitoring."""

import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st

from src.config import get_settings
from src.dashboard.data_service import DashboardDataService
from src.dashboard.components.sidebar import render_sidebar
from src.dashboard.components.market_table import render_market_table
from src.dashboard.components.new_markets import render_new_markets
from src.dashboard.components.price_movers import render_price_movers
from src.dashboard.components.orderbook_detail import render_orderbook_detail

st.set_page_config(
    page_title="Kalshi Market Dashboard",
    page_icon="📊",
    layout="wide",
)


def get_service() -> DashboardDataService:
    if "service" not in st.session_state:
        st.session_state.service = DashboardDataService(get_settings())
    return st.session_state.service


def show_worker_progress(service: DashboardDataService):
    """Poll worker status and show a progress bar. Rerun until done."""
    status = service.get_worker_status()

    if status and status["is_running"]:
        st.progress(status["progress"], text=status["message"])
        time.sleep(1)
        st.rerun()
    elif status and status["stage"] == "error":
        st.error(f"Refresh failed: {status['message']}")
    elif status and status["stage"] == "done":
        st.toast(f"Refresh complete: {status['qualified_markets']} qualified markets")


def main():
    service = get_service()

    # Auto-start first refresh if DB is empty
    df = service.get_qualified_dataframe()
    if df.empty and not service.is_refreshing():
        service.start_refresh()

    # If worker is running, show progress and keep polling
    if service.is_refreshing():
        st.title("Kalshi Market Dashboard")
        st.info("Background data refresh in progress...")
        show_worker_progress(service)
        return

    # Worker finished but we haven't loaded data yet
    df = service.get_qualified_dataframe()

    # Sidebar
    categories = service.get_categories()
    filters = render_sidebar(categories)

    # Handle manual refresh
    if filters["refresh_clicked"] and not service.is_refreshing():
        service.start_refresh()
        st.rerun()

    # Show status
    status = service.get_worker_status()

    if df.empty:
        st.warning("No markets in database yet. Click 'Refresh Now' to start.")
        if status:
            st.info(f"Last status: {status['message']}")
        return

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Qualified Markets", len(df))
    with col2:
        total_vol = int(df["volume"].sum()) if "volume" in df.columns else 0
        st.metric("Total Volume", f"{total_vol:,}")
    with col3:
        if status:
            st.metric("Total Scanned", status["total_markets"])
    with col4:
        if "near_mid_depth" in df.columns:
            total_depth = int(df["near_mid_depth"].sum())
            st.metric("Total Near-Mid $", f"${total_depth:,}")

    # Show last refresh time
    if status and status.get("updated_at"):
        st.caption(f"Last refreshed: {status['updated_at'].strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Tabs
    tab_all, tab_new, tab_movers = st.tabs(["All Markets", "New Markets", "Price Movers"])

    with tab_all:
        selected_ticker = render_market_table(df, filters)

        if selected_ticker:
            st.divider()
            with st.spinner(f"Loading orderbook for {selected_ticker}..."):
                try:
                    summary, raw_ob = service.fetch_orderbook_detail(selected_ticker)
                    if summary and raw_ob:
                        render_orderbook_detail(summary, raw_ob)
                    else:
                        st.warning("No orderbook data available.")
                except Exception as e:
                    st.error(f"Failed to fetch orderbook: {e}")

    with tab_new:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=get_settings().new_market_hours)
        new_events_df = service.get_new_events_dataframe(cutoff)
        # Apply days-to-expiry filter (keep only events expiring within N days)
        if not new_events_df.empty and filters["max_days_to_expiry"] > 0:
            expiry_cutoff = datetime.utcnow() + timedelta(days=filters["max_days_to_expiry"])
            new_events_df = new_events_df[
                new_events_df["close_time"].notna() & (new_events_df["close_time"] <= expiry_cutoff)
            ]
        render_new_markets(new_events_df)

    with tab_movers:
        render_price_movers(df)

    # Auto-refresh: kick off background worker, then schedule a rerun
    if filters["auto_refresh"] and filters["refresh_interval"]:
        if status and status.get("updated_at"):
            elapsed = (datetime.utcnow() - status["updated_at"]).total_seconds()
            if elapsed >= filters["refresh_interval"] and not service.is_refreshing():
                service.start_refresh()
        time.sleep(filters["refresh_interval"])
        st.rerun()


# Need this import at module level for the auto-refresh datetime check
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    main()
