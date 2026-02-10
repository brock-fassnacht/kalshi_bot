"""Data aggregation pipeline for collecting and processing market data."""

import asyncio
from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger

from ..api.client import KalshiClient
from ..config import Settings
from ..models import Event, Market
from .database import Database


class DataAggregator:
    """
    Aggregates market data from Kalshi API.

    Fetches all open markets, caches in-memory, and persists snapshots to DB.
    """

    def __init__(
        self,
        settings: Settings,
        client: KalshiClient,
        database: Database,
    ):
        self.settings = settings
        self.client = client
        self.database = database

        self._markets: dict[str, Market] = {}
        self._events: dict[str, Event] = {}
        self._last_update: Optional[datetime] = None

    @property
    def markets(self) -> dict[str, Market]:
        return self._markets

    @property
    def events(self) -> dict[str, Event]:
        return self._events

    @property
    def last_update(self) -> Optional[datetime]:
        return self._last_update

    async def fetch_and_cache_markets(self) -> dict[str, Market]:
        """Fetch all open markets and update cache."""
        logger.info("Fetching all open markets...")

        markets = await self.client.get_all_markets(status="open")

        self._markets = {m.ticker: m for m in markets}
        self._last_update = datetime.utcnow()

        logger.info(f"Cached {len(self._markets)} markets")
        return self._markets

    async def fetch_and_cache_events(self) -> dict[str, Event]:
        """Fetch all open events with nested markets."""
        logger.info("Fetching all open events...")

        events = await self.client.get_all_events(status="open")

        self._events = {e.event_ticker: e for e in events}

        for event in events:
            for market in event.markets:
                self._markets[market.ticker] = market

        self._last_update = datetime.utcnow()

        logger.info(f"Cached {len(self._events)} events with {len(self._markets)} markets")
        return self._events

    async def save_snapshots(self) -> None:
        """Save current market data to database."""
        if not self._markets:
            return

        logger.debug(f"Saving {len(self._markets)} market snapshots")
        now = datetime.utcnow()

        for market in self._markets.values():
            await self.database.save_market_snapshot(
                ticker=market.ticker,
                event_ticker=market.event_ticker,
                title=market.title,
                yes_bid=market.yes_bid,
                yes_ask=market.yes_ask,
                no_bid=market.no_bid,
                no_ask=market.no_ask,
                last_price=market.last_price,
                volume=market.volume,
                open_interest=market.open_interest,
                fetched_at=now,
            )

    async def fetch_orderbook(self, ticker: str) -> dict:
        """Fetch orderbook for a single market."""
        return await self.client.get_orderbook(ticker, depth=self.settings.orderbook_depth)

    def get_markets_by_event(self, event_ticker: str) -> list[Market]:
        return [m for m in self._markets.values() if m.event_ticker == event_ticker]

    def get_markets_by_category(self, category: str) -> list[Market]:
        return [m for m in self._markets.values() if m.category == category]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert current market cache to pandas DataFrame."""
        if not self._markets:
            return pd.DataFrame()

        records = []
        for m in self._markets.values():
            records.append({
                "ticker": m.ticker,
                "event_ticker": m.event_ticker,
                "title": m.title,
                "status": m.status.value,
                "yes_bid": m.yes_bid,
                "yes_ask": m.yes_ask,
                "no_bid": m.no_bid,
                "no_ask": m.no_ask,
                "yes_mid": m.yes_mid,
                "no_mid": m.no_mid,
                "spread": m.spread,
                "last_price": m.last_price,
                "volume": m.volume,
                "open_interest": m.open_interest,
                "category": m.category,
                "open_time": m.open_time,
                "close_time": m.close_time,
                "expiration_time": m.expiration_time,
            })

        return pd.DataFrame(records)
