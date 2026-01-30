"""Data models for Kalshi markets and arbitrage opportunities."""

from datetime import datetime
from decimal import Decimal
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
    yes_bid: Optional[int] = None  # Best bid for Yes
    yes_ask: Optional[int] = None  # Best ask for Yes
    no_bid: Optional[int] = None   # Best bid for No
    no_ask: Optional[int] = None   # Best ask for No

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
        """Calculate mid price for Yes contracts."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return (self.yes_bid + self.yes_ask) / 2
        return None

    @property
    def no_mid(self) -> Optional[float]:
        """Calculate mid price for No contracts."""
        if self.no_bid is not None and self.no_ask is not None:
            return (self.no_bid + self.no_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread for Yes contracts."""
        if self.yes_bid is not None and self.yes_ask is not None:
            return self.yes_ask - self.yes_bid
        return None

    @property
    def duration_days(self) -> Optional[float]:
        """Calculate total contract duration in days (from open to close/expiry)."""
        if self.open_time is None:
            return None
        end_time = self.close_time or self.expiration_time
        if end_time is None:
            return None
        delta = end_time - self.open_time
        return delta.total_seconds() / 86400  # Convert to days

    def is_long_duration(self, min_days: float = 7.0) -> bool:
        """Check if contract duration is at least min_days (default 7 days)."""
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


class ArbitrageType(str, Enum):
    """Type of arbitrage opportunity."""
    YES_NO_MISPRICING = "yes_no_mispricing"  # Yes + No != 100
    MULTI_OUTCOME = "multi_outcome"  # Sum of mutually exclusive outcomes != 100
    CORRELATED_MARKETS = "correlated_markets"  # Mispricing between related markets
    CALENDAR_SPREAD = "calendar_spread"  # Same event, different expiry dates


class ArbitrageOpportunity(BaseModel):
    """Detected arbitrage opportunity."""

    id: str  # Unique identifier
    arb_type: ArbitrageType
    markets: list[str]  # List of market tickers involved

    # Profit metrics (in cents)
    expected_profit: float  # Expected profit per contract set
    max_profit: float  # Maximum possible profit
    risk: float  # Potential loss if arb fails

    # Suggested trades
    legs: list[dict]  # List of trades to execute

    # Confidence and validity
    confidence: float = Field(ge=0, le=1)  # 0-1 confidence score
    valid_until: Optional[datetime] = None

    detected_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def profit_ratio(self) -> float:
        """Return profit as a ratio of risk."""
        if self.risk == 0:
            return float("inf")
        return self.expected_profit / self.risk


class OrderSide(str, Enum):
    """Order side."""
    YES = "yes"
    NO = "no"


class OrderAction(str, Enum):
    """Order action."""
    BUY = "buy"
    SELL = "sell"


class TradeLeg(BaseModel):
    """Single leg of an arbitrage trade."""

    market_ticker: str
    side: OrderSide
    action: OrderAction
    quantity: int
    limit_price: int  # In cents

    @property
    def notional_value(self) -> int:
        """Calculate notional value in cents."""
        return self.quantity * self.limit_price


# =============================================================================
# Interactive Brokers Models
# =============================================================================


class IBContract(BaseModel):
    """Interactive Brokers contract specification."""

    symbol: str
    sec_type: str  # FUT, FOP (futures option), STK, etc.
    exchange: str  # CME, NYMEX, etc.
    currency: str  # USD
    last_trade_date: Optional[str] = None  # For futures expiration (YYYYMMDD)
    strike: Optional[float] = None  # For options
    right: Optional[str] = None  # C/P for options
    multiplier: Optional[str] = None  # Contract multiplier
    con_id: Optional[int] = None  # IB contract ID


class IBOrderBookLevel(BaseModel):
    """Single level in IB order book (market depth)."""

    price: float
    size: int
    market_maker: Optional[str] = None


class IBOrderBook(BaseModel):
    """IB order book (market depth) for a contract."""

    contract: IBContract
    bids: list[IBOrderBookLevel] = Field(default_factory=list)
    asks: list[IBOrderBookLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class IBTicker(BaseModel):
    """IB ticker/quote data."""

    contract: IBContract
    bid: Optional[float] = None
    bid_size: Optional[int] = None
    ask: Optional[float] = None
    ask_size: Optional[int] = None
    last: Optional[float] = None
    last_size: Optional[int] = None
    volume: Optional[int] = None
    close: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
