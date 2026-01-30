"""Async HTTP client for Kalshi API."""

from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from ..config import Settings
from ..models import Event, Market, MarketStatus
from .auth import get_auth_headers


class KalshiClient:
    """Async client for Kalshi Trade API v2."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = settings.kalshi_base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "KalshiClient":
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    def _get_auth_path(self, path: str) -> str:
        """
        Get the full path for authentication signing.

        Kalshi requires signing the full path (including /trade-api/v2)
        but WITHOUT query parameters.
        """
        # Ensure path starts with /trade-api/v2
        if not path.startswith("/trade-api/v2"):
            path = "/trade-api/v2" + path
        return path

    def _get_headers(self, method: str, path: str) -> dict[str, str]:
        """Generate headers with authentication."""
        # Get full path for signing (without query params)
        auth_path = self._get_auth_path(path)

        auth_headers = get_auth_headers(
            api_key=self.settings.kalshi_api_key,
            private_key_pem=self.settings.private_key,
            method=method,
            path=auth_path,
        )
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **auth_headers,
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Make an authenticated API request."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Build full URL
        url = f"{self.base_url}{path}"

        # Get headers (signing uses path without query params)
        headers = self._get_headers(method, path)

        logger.debug(f"Request: {method} {url}")

        response = await self._client.request(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json,
        )

        if response.status_code >= 400:
            logger.error(f"API error {response.status_code}: {response.text}")
            response.raise_for_status()

        return response.json()

    # -------------------------------------------------------------------------
    # Market Data Endpoints
    # -------------------------------------------------------------------------

    async def get_markets(
        self,
        cursor: Optional[str] = None,
        limit: int = 100,
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> tuple[list[Market], Optional[str]]:
        """
        Fetch markets with optional filters.

        Returns:
            Tuple of (list of markets, next cursor for pagination)
        """
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status

        data = await self._request("GET", "/markets", params=params)

        markets = []
        for m in data.get("markets", []):
            market = self._parse_market(m)
            if market:
                markets.append(market)

        next_cursor = data.get("cursor")
        return markets, next_cursor

    async def get_all_markets(
        self,
        status: str = "open",
        event_ticker: Optional[str] = None,
    ) -> list[Market]:
        """Fetch all markets, handling pagination automatically."""
        all_markets = []
        cursor = None

        while True:
            markets, cursor = await self.get_markets(
                cursor=cursor,
                limit=200,
                status=status,
                event_ticker=event_ticker,
            )
            all_markets.extend(markets)
            logger.debug(f"Fetched {len(markets)} markets, total: {len(all_markets)}")

            if not cursor:
                break

        return all_markets

    async def get_market(self, ticker: str) -> Optional[Market]:
        """Fetch a single market by ticker."""
        data = await self._request("GET", f"/markets/{ticker}")
        return self._parse_market(data.get("market", {}))

    async def get_events(
        self,
        cursor: Optional[str] = None,
        limit: int = 100,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = True,
    ) -> tuple[list[Event], Optional[str]]:
        """Fetch events with optional filters."""
        params = {"limit": limit}
        if with_nested_markets:
            params["with_nested_markets"] = "true"
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker

        data = await self._request("GET", "/events", params=params)

        events = []
        for e in data.get("events", []):
            event = self._parse_event(e)
            if event:
                events.append(event)

        next_cursor = data.get("cursor")
        return events, next_cursor

    async def get_all_events(self, status: str = "open") -> list[Event]:
        """Fetch all events with nested markets."""
        all_events = []
        cursor = None

        while True:
            events, cursor = await self.get_events(
                cursor=cursor,
                limit=100,
                status=status,
            )
            all_events.extend(events)

            if not cursor:
                break

        return all_events

    async def get_event(self, event_ticker: str) -> Optional[Event]:
        """Fetch a single event by ticker."""
        data = await self._request("GET", f"/events/{event_ticker}")
        return self._parse_event(data.get("event", {}))

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict[str, Any]:
        """Fetch orderbook for a market."""
        params = {"depth": depth}
        return await self._request("GET", f"/markets/{ticker}/orderbook", params=params)

    async def get_btc_markets(self) -> list[Market]:
        """
        Fetch all open Bitcoin-related markets.

        Searches for markets with series tickers like KXBTC, KXBTCD, BTC.
        """
        btc_series = ["KXBTC", "KXBTCD", "BTC", "BITCOIN"]
        all_markets = []

        for series in btc_series:
            try:
                markets, _ = await self.get_markets(series_ticker=series, status="open")
                all_markets.extend(markets)
                logger.debug(f"Found {len(markets)} markets for series {series}")
            except httpx.HTTPStatusError:
                # Series might not exist
                continue

        # Also search by title containing "bitcoin" if no series markets found
        if not all_markets:
            logger.info("No BTC series found, searching all markets...")
            markets, _ = await self.get_markets(status="open", limit=200)
            for m in markets:
                if "bitcoin" in m.title.lower() or "btc" in m.title.lower():
                    all_markets.append(m)

        return all_markets

    # -------------------------------------------------------------------------
    # Trading Endpoints
    # -------------------------------------------------------------------------

    async def create_order(
        self,
        ticker: str,
        side: str,  # "yes" or "no"
        action: str,  # "buy" or "sell"
        count: int,
        type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a new order.

        Args:
            ticker: Market ticker
            side: "yes" or "no"
            action: "buy" or "sell"
            count: Number of contracts
            type: "limit" or "market"
            yes_price: Limit price for yes (1-99 cents)
            no_price: Limit price for no (1-99 cents)
            client_order_id: Optional client-defined order ID
        """
        payload = {
            "ticker": ticker,
            "side": side,
            "action": action,
            "count": count,
            "type": type,
        }

        if yes_price is not None:
            payload["yes_price"] = yes_price
        if no_price is not None:
            payload["no_price"] = no_price
        if client_order_id:
            payload["client_order_id"] = client_order_id

        return await self._request("POST", "/portfolio/orders", json=payload)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        """Cancel an order by ID."""
        return await self._request("DELETE", f"/portfolio/orders/{order_id}")

    async def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get orders with optional filters."""
        params = {}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status

        data = await self._request("GET", "/portfolio/orders", params=params)
        return data.get("orders", [])

    async def get_positions(self) -> list[dict[str, Any]]:
        """Get all current positions."""
        data = await self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    async def get_balance(self) -> dict[str, Any]:
        """Get account balance."""
        return await self._request("GET", "/portfolio/balance")

    # -------------------------------------------------------------------------
    # Parsing Helpers
    # -------------------------------------------------------------------------

    def _parse_market(self, data: dict) -> Optional[Market]:
        """Parse API response into Market model."""
        if not data:
            return None

        try:
            return Market(
                ticker=data["ticker"],
                event_ticker=data.get("event_ticker", ""),
                title=data.get("title", ""),
                subtitle=data.get("subtitle"),
                status=MarketStatus(data.get("status", "open")),
                yes_bid=data.get("yes_bid"),
                yes_ask=data.get("yes_ask"),
                no_bid=data.get("no_bid"),
                no_ask=data.get("no_ask"),
                last_price=data.get("last_price"),
                volume=data.get("volume", 0),
                open_interest=data.get("open_interest", 0),
                category=data.get("category"),
                open_time=self._parse_datetime(data.get("open_time")),
                close_time=self._parse_datetime(data.get("close_time")),
                expiration_time=self._parse_datetime(data.get("expiration_time")),
            )
        except Exception as e:
            logger.warning(f"Failed to parse market {data.get('ticker')}: {e}")
            return None

    def _parse_event(self, data: dict) -> Optional[Event]:
        """Parse API response into Event model."""
        if not data:
            return None

        try:
            markets = []
            for m in data.get("markets", []):
                market = self._parse_market(m)
                if market:
                    markets.append(market)

            return Event(
                event_ticker=data["event_ticker"],
                title=data.get("title", ""),
                category=data.get("category"),
                markets=markets,
            )
        except Exception as e:
            logger.warning(f"Failed to parse event {data.get('event_ticker')}: {e}")
            return None

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
