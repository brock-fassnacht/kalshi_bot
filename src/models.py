"""Data models for Kalshi market dashboard."""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MarketStatus(str, Enum):
    """Market trading status."""
    ACTIVE = "active"
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    FINALIZED = "finalized"
    INITIALIZED = "initialized"


class Market(BaseModel):
    """Kalshi market data model."""

    ticker: str
    event_ticker: str
    title: str
    subtitle: Optional[str] = None
    status: MarketStatus

    # Pricing (in cents, 1-99)
    yes_bid: Optional[int] = None
    yes_ask: Optional[int] = None
    no_bid: Optional[int] = None
    no_ask: Optional[int] = None

    last_price: Optional[int] = None
    volume: int = 0
    open_interest: int = 0

    # Metadata
    category: Optional[str] = None
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None

    fetched_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def yes_mid(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return None

    @property
    def no_mid(self) -> Optional[float]:
        if self.no_bid is not None and self.no_ask is not None:
            return (self.no_bid + self.no_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None

    @property
    def duration_days(self) -> Optional[float]:
        if self.open_time is None:
            return None
        end_time = self.close_time or self.expiration_time
        if end_time is None:
            return None
        delta = end_time - self.open_time
        return delta.total_seconds() / 86400

    def is_long_duration(self, min_days: float = 7.0) -> bool:
        duration = self.duration_days
        if duration is None:
            return False
        return duration >= min_days


class Event(BaseModel):
    """Kalshi event containing multiple markets."""

    event_ticker: str
    title: str
    category: Optional[str] = None
    markets: list[Market] = Field(default_factory=list)

    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class OrderbookSummary(BaseModel):
    """Summary of orderbook depth and liquidity."""

    ticker: str
    total_yes_depth_dollars: float = 0.0  # Total $ on YES side
    total_no_depth_dollars: float = 0.0   # Total $ on NO side
    near_touch_yes_dollars: float = 0.0   # YES $ within 5c of best
    near_touch_no_dollars: float = 0.0    # NO $ within 5c of best
    yes_levels: int = 0
    no_levels: int = 0
    best_yes_price: Optional[int] = None
    best_no_price: Optional[int] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
