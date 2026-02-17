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

        st.subheader("Title Keywords")
        include_keywords = st.text_input(
            "Include",
            placeholder="e.g. bitcoin, fed rate",
            help="Show only titles containing any of these (comma-separated)",
        )
        exclude_keywords = st.text_input(
            "Exclude",
            placeholder="e.g. crypto, sports",
            help="Hide titles containing any of these (comma-separated)",
        )

        show_active_only = st.checkbox("Active markets only", value=True)

        min_yes_ask = st.number_input(
            "Min Yes Ask (cents)",
            min_value=0,
            value=0,
            step=1,
            help="Only show markets with Yes Ask >= this value",
        )

        max_days_to_expiry = st.number_input(
            "Max days until expiry",
            min_value=0,
            value=0,
            step=1,
            help="Only show events expiring within this many days (0 = no filter)",
        )

    return {
        "refresh_clicked": refresh_clicked,
        "auto_refresh": auto_refresh,
        "refresh_interval": refresh_interval,
        "search": search.strip().lower() if search else "",
        "categories": selected_categories,
        "include_keywords": include_keywords.strip() if include_keywords else "",
        "exclude_keywords": exclude_keywords.strip() if exclude_keywords else "",
        "show_active_only": show_active_only,
        "min_yes_ask": min_yes_ask,
        "max_days_to_expiry": max_days_to_expiry,
    }
