"""Database management for market data persistence."""

import json
from datetime import datetime
from typing import Optional

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    delete,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from ..config import Settings


class Base(DeclarativeBase):
    pass


class MarketSnapshot(Base):
    """Historical market data snapshot."""

    __tablename__ = "market_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(100), nullable=False, index=True)
    event_ticker = Column(String(100), nullable=False, index=True)
    title = Column(Text)

    yes_bid = Column(Integer)
    yes_ask = Column(Integer)
    no_bid = Column(Integer)
    no_ask = Column(Integer)
    last_price = Column(Integer)
    volume = Column(Integer)
    open_interest = Column(Integer)

    fetched_at = Column(DateTime, nullable=False, index=True)


class QualifiedMarket(Base):
    """Markets that passed all filters - the display set."""

    __tablename__ = "qualified_markets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(100), nullable=False, unique=True, index=True)
    event_ticker = Column(String(100), nullable=False)
    title = Column(Text)
    status = Column(String(50))
    category = Column(String(100))

    yes_bid = Column(Integer)
    yes_ask = Column(Integer)
    no_bid = Column(Integer)
    no_ask = Column(Integer)
    last_price = Column(Integer)
    volume = Column(Integer)
    open_interest = Column(Integer)
    spread = Column(Float)

    open_time = Column(DateTime)
    close_time = Column(DateTime)
    expiration_time = Column(DateTime)

    # Orderbook data
    total_yes_depth = Column(Float, default=0.0)
    total_no_depth = Column(Float, default=0.0)
    near_mid_depth = Column(Float, default=0.0)

    # Price change
    price_change_24h = Column(Float)

    updated_at = Column(DateTime, nullable=False, index=True)


class EventSummary(Base):
    """Event-level summary aggregated from all sub-markets."""

    __tablename__ = "event_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_ticker = Column(String(100), nullable=False, unique=True, index=True)
    title = Column(Text)
    category = Column(String(100))
    market_count = Column(Integer, default=0)
    total_volume = Column(Integer, default=0)
    total_oi = Column(Integer, default=0)
    open_time = Column(DateTime)
    close_time = Column(DateTime)
    updated_at = Column(DateTime, nullable=False, index=True)


class OrderbookCache(Base):
    """Cached orderbook summary for each market."""

    __tablename__ = "orderbook_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(100), nullable=False, unique=True, index=True)

    total_yes_depth = Column(Float, default=0.0)
    total_no_depth = Column(Float, default=0.0)
    near_touch_yes = Column(Float, default=0.0)
    near_touch_no = Column(Float, default=0.0)
    near_mid_depth = Column(Float, default=0.0)
    yes_levels = Column(Integer, default=0)
    no_levels = Column(Integer, default=0)
    best_yes_price = Column(Integer)
    best_no_price = Column(Integer)

    # Market-level bid/ask at time of cache (for incremental refresh comparison)
    market_yes_bid = Column(Integer)
    market_yes_ask = Column(Integer)

    raw_json = Column(Text)
    updated_at = Column(DateTime, nullable=False, index=True)


class WorkerStatus(Base):
    """Background worker progress tracking."""

    __tablename__ = "worker_status"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stage = Column(String(100), nullable=False)  # e.g. "fetching_markets", "fetching_orderbooks"
    message = Column(Text)
    progress = Column(Float, default=0.0)  # 0.0 - 1.0
    total_markets = Column(Integer, default=0)
    filtered_markets = Column(Integer, default=0)
    qualified_markets = Column(Integer, default=0)
    is_running = Column(Integer, default=0)  # 0=idle, 1=running
    updated_at = Column(DateTime, nullable=False)


class Database:
    """Async database manager."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.engine = create_async_engine(
            settings.database_url,
            echo=False,
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Create all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")

    # ---- Worker Status ----

    async def update_worker_status(
        self,
        stage: str,
        message: str,
        progress: float = 0.0,
        total_markets: int = 0,
        filtered_markets: int = 0,
        qualified_markets: int = 0,
        is_running: bool = True,
    ) -> None:
        async with self.session_factory() as session:
            result = await session.execute(select(WorkerStatus).limit(1))
            existing = result.scalar_one_or_none()
            now = datetime.utcnow()

            if existing:
                existing.stage = stage
                existing.message = message
                existing.progress = progress
                existing.total_markets = total_markets
                existing.filtered_markets = filtered_markets
                existing.qualified_markets = qualified_markets
                existing.is_running = 1 if is_running else 0
                existing.updated_at = now
            else:
                session.add(WorkerStatus(
                    stage=stage, message=message, progress=progress,
                    total_markets=total_markets, filtered_markets=filtered_markets,
                    qualified_markets=qualified_markets,
                    is_running=1 if is_running else 0, updated_at=now,
                ))
            await session.commit()

    async def get_worker_status(self) -> Optional[WorkerStatus]:
        async with self.session_factory() as session:
            result = await session.execute(select(WorkerStatus).limit(1))
            return result.scalar_one_or_none()

    # ---- Qualified Markets ----

    async def replace_qualified_markets(self, records: list[dict]) -> None:
        """Replace all qualified markets with a new set."""
        async with self.session_factory() as session:
            await session.execute(delete(QualifiedMarket))
            for r in records:
                session.add(QualifiedMarket(**r))
            await session.commit()

    async def get_all_qualified_markets(self) -> list[QualifiedMarket]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(QualifiedMarket).order_by(QualifiedMarket.volume.desc())
            )
            return list(result.scalars().all())

    # ---- Event Summaries ----

    async def replace_event_summaries(self, records: list[dict]) -> None:
        """Replace all event summaries with a new set."""
        async with self.session_factory() as session:
            await session.execute(delete(EventSummary))
            for r in records:
                session.add(EventSummary(**r))
            await session.commit()

    async def get_new_events(self, since: datetime) -> list[EventSummary]:
        """Return events with open_time >= since, ordered by open_time desc."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(EventSummary)
                .where(EventSummary.open_time >= since)
                .order_by(EventSummary.open_time.desc())
            )
            return list(result.scalars().all())

    # ---- Market Snapshots ----

    async def save_market_snapshot(
        self,
        ticker: str,
        event_ticker: str,
        title: str,
        yes_bid: Optional[int],
        yes_ask: Optional[int],
        no_bid: Optional[int],
        no_ask: Optional[int],
        last_price: Optional[int],
        volume: int,
        open_interest: int,
        fetched_at: datetime,
    ) -> None:
        async with self.session_factory() as session:
            snapshot = MarketSnapshot(
                ticker=ticker, event_ticker=event_ticker, title=title,
                yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask,
                last_price=last_price, volume=volume, open_interest=open_interest,
                fetched_at=fetched_at,
            )
            session.add(snapshot)
            await session.commit()

    async def save_market_snapshots_batch(self, snapshots: list[dict]) -> None:
        """Save multiple snapshots in one transaction."""
        async with self.session_factory() as session:
            for s in snapshots:
                session.add(MarketSnapshot(**s))
            await session.commit()

    async def get_snapshots_since(self, since: datetime) -> list[MarketSnapshot]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(MarketSnapshot)
                .where(MarketSnapshot.fetched_at >= since)
                .order_by(MarketSnapshot.fetched_at.asc())
            )
            return list(result.scalars().all())

    async def get_oldest_snapshots_per_ticker(self, since: datetime) -> dict[str, MarketSnapshot]:
        snapshots = await self.get_snapshots_since(since)
        oldest: dict[str, MarketSnapshot] = {}
        for s in snapshots:
            if s.ticker not in oldest:
                oldest[s.ticker] = s
        return oldest

    # ---- Orderbook Cache ----

    async def upsert_orderbook_cache(
        self,
        ticker: str,
        total_yes_depth: float,
        total_no_depth: float,
        near_touch_yes: float,
        near_touch_no: float,
        near_mid_depth: float,
        yes_levels: int,
        no_levels: int,
        best_yes_price: Optional[int],
        best_no_price: Optional[int],
        raw_json: str,
        updated_at: datetime,
        market_yes_bid: Optional[int] = None,
        market_yes_ask: Optional[int] = None,
    ) -> None:
        async with self.session_factory() as session:
            result = await session.execute(
                select(OrderbookCache).where(OrderbookCache.ticker == ticker)
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.total_yes_depth = total_yes_depth
                existing.total_no_depth = total_no_depth
                existing.near_touch_yes = near_touch_yes
                existing.near_touch_no = near_touch_no
                existing.near_mid_depth = near_mid_depth
                existing.yes_levels = yes_levels
                existing.no_levels = no_levels
                existing.best_yes_price = best_yes_price
                existing.best_no_price = best_no_price
                existing.market_yes_bid = market_yes_bid
                existing.market_yes_ask = market_yes_ask
                existing.raw_json = raw_json
                existing.updated_at = updated_at
            else:
                session.add(OrderbookCache(
                    ticker=ticker,
                    total_yes_depth=total_yes_depth, total_no_depth=total_no_depth,
                    near_touch_yes=near_touch_yes, near_touch_no=near_touch_no,
                    near_mid_depth=near_mid_depth,
                    yes_levels=yes_levels, no_levels=no_levels,
                    best_yes_price=best_yes_price, best_no_price=best_no_price,
                    market_yes_bid=market_yes_bid, market_yes_ask=market_yes_ask,
                    raw_json=raw_json, updated_at=updated_at,
                ))
            await session.commit()

    async def get_all_orderbook_caches(self) -> dict[str, OrderbookCache]:
        """Get all cached orderbooks, keyed by ticker."""
        async with self.session_factory() as session:
            result = await session.execute(select(OrderbookCache))
            return {row.ticker: row for row in result.scalars().all()}

    async def get_orderbook_cache(self, ticker: str) -> Optional[OrderbookCache]:
        async with self.session_factory() as session:
            result = await session.execute(
                select(OrderbookCache).where(OrderbookCache.ticker == ticker)
            )
            return result.scalar_one_or_none()

    async def cleanup_old_snapshots(self, before: datetime) -> int:
        async with self.session_factory() as session:
            result = await session.execute(
                delete(MarketSnapshot).where(MarketSnapshot.fetched_at < before)
            )
            await session.commit()
            return result.rowcount

    async def close(self) -> None:
        await self.engine.dispose()
