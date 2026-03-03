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


def _load_data(service: DashboardDataService):
    """Load data from DB into session state."""
    st.session_state.df = service.get_qualified_dataframe()
    st.session_state.categories = service.get_categories()
    st.session_state.worker_status = service.get_worker_status()


def main():
    service = get_service()

    # Auto-start first refresh if DB is empty
    if "df" not in st.session_state:
        _load_data(service)
    if st.session_state.df.empty and not service.is_refreshing():
        service.start_refresh()

    # Sidebar
    filters = render_sidebar(st.session_state.categories)

    # Handle manual refresh
    if filters["refresh_clicked"] and not service.is_refreshing():
        service.start_refresh()
        st.rerun()

    # Show progress if refreshing, then poll
    if service.is_refreshing():
        status = service.get_worker_status()
        if status and status["is_running"]:
            st.progress(status["progress"], text=status["message"])
            time.sleep(1)
            st.rerun()
        else:
            # Refresh just finished — reload data from DB
            _load_data(service)

    # Check for errors
    status = st.session_state.worker_status
    if status and status["stage"] == "error":
        st.error(f"Refresh failed: {status['message']}")

    df = st.session_state.df
    status = st.session_state.worker_status

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

    # Market table
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

    # Filter thresholds legend
    s = get_settings()
    st.divider()
    st.caption(
        f"**Market filter thresholds** — "
        f"Markets scanned: up to {getattr(s, 'max_market_pages', 50) * 200:,} ({getattr(s, 'max_market_pages', 50)} pages × 200) · "
        f"Min expiry: {s.min_expiry_hours}h · "
        f"Min OI: {s.min_oi_prefilter:,} · "
        f"Min Yes Ask: {s.min_yes_ask_prefilter}¢ · "
        f"Max orderbook fetches: {s.max_orderbook_fetches:,} · "
        f"Min near-mid depth: ${s.min_near_mid_depth_dollars:,.0f} (±{s.near_mid_range_cents}¢ of mid) · "
        f"Min YES depth: ${s.min_yes_depth_dollars:,.0f} · "
        f"Min NO depth: ${s.min_no_depth_dollars:,.0f}"
    )

    # Auto-refresh: kick off background worker, then schedule a rerun
    if filters["auto_refresh"] and filters["refresh_interval"]:
        if status and status.get("updated_at"):
            elapsed = (datetime.utcnow() - status["updated_at"]).total_seconds()
            if elapsed >= filters["refresh_interval"] and not service.is_refreshing():
                service.start_refresh()
                st.rerun()
        # Poll while a refresh is in progress so we pick up the result
        if service.is_refreshing():
            time.sleep(2)
            st.rerun()


# Need this import at module level for the auto-refresh datetime check
from datetime import datetime

if __name__ == "__main__":
    main()
