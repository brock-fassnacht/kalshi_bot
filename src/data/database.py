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
    create_engine,
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


class OrderbookCache(Base):
    """Cached orderbook summary for each market."""

    __tablename__ = "orderbook_cache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(100), nullable=False, unique=True, index=True)

    total_yes_depth = Column(Float, default=0.0)
    total_no_depth = Column(Float, default=0.0)
    near_touch_yes = Column(Float, default=0.0)
    near_touch_no = Column(Float, default=0.0)
    yes_levels = Column(Integer, default=0)
    no_levels = Column(Integer, default=0)
    best_yes_price = Column(Integer)
    best_no_price = Column(Integer)

    raw_json = Column(Text)  # Raw orderbook JSON for detail views
    updated_at = Column(DateTime, nullable=False, index=True)


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
        """Save a market data snapshot."""
        async with self.session_factory() as session:
            snapshot = MarketSnapshot(
                ticker=ticker,
                event_ticker=event_ticker,
                title=title,
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=no_bid,
                no_ask=no_ask,
                last_price=last_price,
                volume=volume,
                open_interest=open_interest,
                fetched_at=fetched_at,
            )
            session.add(snapshot)
            await session.commit()

    async def get_market_history(
        self,
        ticker: str,
        limit: int = 1000,
    ) -> list[MarketSnapshot]:
        """Get historical snapshots for a market."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(MarketSnapshot)
                .where(MarketSnapshot.ticker == ticker)
                .order_by(MarketSnapshot.fetched_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_snapshots_since(self, since: datetime) -> list[MarketSnapshot]:
        """Get all snapshots since a given time."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(MarketSnapshot)
                .where(MarketSnapshot.fetched_at >= since)
                .order_by(MarketSnapshot.fetched_at.asc())
            )
            return list(result.scalars().all())

    async def get_oldest_snapshots_per_ticker(self, since: datetime) -> dict[str, MarketSnapshot]:
        """Get the oldest snapshot for each ticker since a given time (for price change calc)."""
        snapshots = await self.get_snapshots_since(since)
        oldest: dict[str, MarketSnapshot] = {}
        for s in snapshots:
            if s.ticker not in oldest:
                oldest[s.ticker] = s
        return oldest

    async def upsert_orderbook_cache(
        self,
        ticker: str,
        total_yes_depth: float,
        total_no_depth: float,
        near_touch_yes: float,
        near_touch_no: float,
        yes_levels: int,
        no_levels: int,
        best_yes_price: Optional[int],
        best_no_price: Optional[int],
        raw_json: str,
        updated_at: datetime,
    ) -> None:
        """Insert or update orderbook cache for a ticker."""
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
                existing.yes_levels = yes_levels
                existing.no_levels = no_levels
                existing.best_yes_price = best_yes_price
                existing.best_no_price = best_no_price
                existing.raw_json = raw_json
                existing.updated_at = updated_at
            else:
                session.add(OrderbookCache(
                    ticker=ticker,
                    total_yes_depth=total_yes_depth,
                    total_no_depth=total_no_depth,
                    near_touch_yes=near_touch_yes,
                    near_touch_no=near_touch_no,
                    yes_levels=yes_levels,
                    no_levels=no_levels,
                    best_yes_price=best_yes_price,
                    best_no_price=best_no_price,
                    raw_json=raw_json,
                    updated_at=updated_at,
                ))

            await session.commit()

    async def get_all_orderbook_cache(self) -> list[OrderbookCache]:
        """Get all cached orderbook summaries."""
        async with self.session_factory() as session:
            result = await session.execute(select(OrderbookCache))
            return list(result.scalars().all())

    async def get_orderbook_cache(self, ticker: str) -> Optional[OrderbookCache]:
        """Get cached orderbook for a specific ticker."""
        async with self.session_factory() as session:
            result = await session.execute(
                select(OrderbookCache).where(OrderbookCache.ticker == ticker)
            )
            return result.scalar_one_or_none()

    async def cleanup_old_snapshots(self, before: datetime) -> int:
        """Delete snapshots older than the given datetime. Returns count deleted."""
        async with self.session_factory() as session:
            result = await session.execute(
                delete(MarketSnapshot).where(MarketSnapshot.fetched_at < before)
            )
            await session.commit()
            return result.rowcount

    async def close(self) -> None:
        """Close database connection."""
        await self.engine.dispose()
