"""Sidebar component with filters and controls."""

import streamlit as st


def render_sidebar(categories: list[str]) -> dict:
    """
    Render sidebar with refresh button, auto-refresh toggle, and filters.

    Returns dict with filter settings.
    """
    with st.sidebar:
        st.title("Kalshi Dashboard")
        st.divider()

        # Refresh controls
        refresh_clicked = st.button("Refresh Now", use_container_width=True, type="primary")
        auto_refresh = st.toggle("Auto-refresh", value=True)

        if auto_refresh:
            refresh_interval = st.slider(
                "Refresh interval (seconds)",
                min_value=30,
                max_value=300,
                value=60,
                step=10,
            )
        else:
            refresh_interval = None

        st.divider()

        # Filters
        st.subheader("Filters")

        search = st.text_input("Search markets", placeholder="Ticker or title...")

        selected_categories = st.multiselect(
            "Categories",
            options=categories,
            default=[],
            placeholder="All categories",
        )

        min_volume = st.number_input(
            "Min volume",
            min_value=0,
            value=0,
            step=100,
        )

        show_active_only = st.checkbox("Active markets only", value=True)

    return {
        "refresh_clicked": refresh_clicked,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "search": search.strip().lower() if search else "",
        "categories": selected_categories,
        "min_volume": min_volume,
        "show_active_only": show_active_only,
    }
