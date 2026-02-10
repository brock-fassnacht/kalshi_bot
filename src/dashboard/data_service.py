"""Central data service for the dashboard."""

import json
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from ..api.client import KalshiClient
from ..config import Settings
from ..data.aggregator import DataAggregator
from ..data.database import Database
from ..data.orderbook import compute_orderbook_summary
from ..models import OrderbookSummary
from .async_bridge import get_bridge


class DashboardDataService:
    """
    Central orchestrator for dashboard data.

    Manages market fetching, caching, price change computation,
    orderbook fetching, and building the master DataFrame.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.bridge = get_bridge()

        # These get created lazily via async init
        self._client: Optional[KalshiClient] = None
        self._database: Optional[Database] = None
        self._aggregator: Optional[DataAggregator] = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazily initialize async resources."""
        if self._initialized:
            return

        self._client = KalshiClient(self.settings)
        self._database = Database(self.settings)
        self._aggregator = DataAggregator(self.settings, self._client, self._database)

        # Open the client and init DB
        self.bridge.run(self._client.__aenter__())
        self.bridge.run(self._database.init_db())
        self._initialized = True

    def refresh_markets(self) -> pd.DataFrame:
        """Fetch all markets from API, save snapshots, return DataFrame."""
        self._ensure_initialized()

        self.bridge.run(self._aggregator.fetch_and_cache_markets())
        self.bridge.run(self._aggregator.save_snapshots())

        return self._aggregator.to_dataframe()

    def get_market_dataframe(self) -> pd.DataFrame:
        """Get current cached markets as DataFrame (no API call)."""
        self._ensure_initialized()
        return self._aggregator.to_dataframe()

    def get_categories(self) -> list[str]:
        """Get unique categories from cached markets."""
        self._ensure_initialized()
        cats = set()
        for m in self._aggregator.markets.values():
            if m.category:
                cats.add(m.category)
        return sorted(cats)

    def compute_price_changes(self) -> dict[str, Optional[float]]:
        """
        Compute 24h price changes for all markets.

        Returns dict of ticker -> change in cents (current yes_bid - old yes_bid).
        """
        self._ensure_initialized()

        lookback = datetime.utcnow() - timedelta(hours=self.settings.price_change_lookback_hours)
        old_snapshots = self.bridge.run(
            self._database.get_oldest_snapshots_per_ticker(lookback)
        )

        changes: dict[str, Optional[float]] = {}
        for ticker, market in self._aggregator.markets.items():
            old = old_snapshots.get(ticker)
            if old and old.yes_bid is not None and market.yes_bid is not None:
                changes[ticker] = market.yes_bid - old.yes_bid
            else:
                changes[ticker] = None

        return changes

    def fetch_orderbook(self, ticker: str) -> tuple[OrderbookSummary, dict]:
        """
        Fetch orderbook for a single market, compute summary, cache in DB.

        Returns (summary, raw_orderbook_dict).
        """
        self._ensure_initialized()

        raw_ob = self.bridge.run(self._aggregator.fetch_orderbook(ticker))
        summary = compute_orderbook_summary(raw_ob, ticker)

        # Cache in DB
        self.bridge.run(self._database.upsert_orderbook_cache(
            ticker=ticker,
            total_yes_depth=summary.total_yes_depth_dollars,
            total_no_depth=summary.total_no_depth_dollars,
            near_touch_yes=summary.near_touch_yes_dollars,
            near_touch_no=summary.near_touch_no_dollars,
            yes_levels=summary.yes_levels,
            no_levels=summary.no_levels,
            best_yes_price=summary.best_yes_price,
            best_no_price=summary.best_no_price,
            raw_json=json.dumps(raw_ob),
            updated_at=datetime.utcnow(),
        ))

        return summary, raw_ob

    def get_cached_orderbooks(self) -> pd.DataFrame:
        """Get all cached orderbook summaries as DataFrame."""
        self._ensure_initialized()

        caches = self.bridge.run(self._database.get_all_orderbook_cache())
        if not caches:
            return pd.DataFrame()

        records = []
        for c in caches:
            records.append({
                "ticker": c.ticker,
                "total_yes_depth": c.total_yes_depth,
                "total_no_depth": c.total_no_depth,
                "near_touch_yes": c.near_touch_yes,
                "near_touch_no": c.near_touch_no,
                "yes_levels": c.yes_levels,
                "no_levels": c.no_levels,
                "ob_updated_at": c.updated_at,
            })

        return pd.DataFrame(records)

    def build_master_dataframe(self) -> pd.DataFrame:
        """
        Build the master DataFrame by joining markets + orderbook cache + price changes.
        """
        df = self.get_market_dataframe()
        if df.empty:
            return df

        # Join orderbook cache
        ob_df = self.get_cached_orderbooks()
        if not ob_df.empty:
            df = df.merge(ob_df, on="ticker", how="left")

        # Join price changes
        changes = self.compute_price_changes()
        df["price_change_24h"] = df["ticker"].map(changes)

        return df

    def get_new_markets(self, hours: Optional[int] = None) -> pd.DataFrame:
        """Get markets opened in the last N hours."""
        self._ensure_initialized()

        if hours is None:
            hours = self.settings.new_market_hours

        cutoff = datetime.utcnow() - timedelta(hours=hours)
        new = []
        for m in self._aggregator.markets.values():
            if m.open_time and m.open_time.replace(tzinfo=None) >= cutoff:
                new.append(m)

        if not new:
            return pd.DataFrame()

        new.sort(key=lambda m: m.open_time, reverse=True)

        records = []
        for m in new:
            records.append({
                "ticker": m.ticker,
                "title": m.title,
                "category": m.category,
                "yes_bid": m.yes_bid,
                "yes_ask": m.yes_ask,
                "spread": m.spread,
                "volume": m.volume,
                "open_interest": m.open_interest,
                "open_time": m.open_time,
            })

        return pd.DataFrame(records)

    def cleanup_old_data(self, days: int = 7) -> int:
        """Delete snapshots older than N days."""
        self._ensure_initialized()
        before = datetime.utcnow() - timedelta(days=days)
        return self.bridge.run(self._database.cleanup_old_snapshots(before))
