"""Central data service - background worker writes to DB, Streamlit reads from DB."""

import asyncio
import json
import threading
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from loguru import logger

from src.api.client import KalshiClient
from src.config import Settings
from src.data.database import Database
from src.data.orderbook import compute_orderbook_summary
from src.models import Market, OrderbookSummary


class DashboardDataService:
    """
    Background worker fetches data -> writes to DB.
    Streamlit reads from DB only (instant).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._db: Optional[Database] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._db_initialized = False

    def _get_db(self) -> Database:
        if self._db is None:
            self._db = Database(self.settings)
        return self._db

    def _ensure_db(self):
        if not self._db_initialized:
            db = self._get_db()
            loop = asyncio.new_event_loop()
            loop.run_until_complete(db.init_db())
            loop.close()
            self._db_initialized = True

    # ---- Background worker ----

    def start_refresh(self):
        """Kick off a background refresh if not already running."""
        if self._worker_thread and self._worker_thread.is_alive():
            return  # Already running
        self._ensure_db()
        self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
        self._worker_thread.start()

    def is_refreshing(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    def _run_worker(self):
        """The actual refresh pipeline, runs in a background thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_refresh_pipeline())
        except Exception as e:
            logger.error(f"Worker error: {e}")
            # Write error status
            db = self._get_db()
            loop.run_until_complete(db.update_worker_status(
                stage="error", message=str(e), is_running=False,
            ))
        finally:
            loop.close()

    async def _async_refresh_pipeline(self):
        """Full async pipeline: fetch markets -> filter -> fetch orderbooks -> save."""
        db = self._get_db()
        settings = self.settings

        # Use a fresh client for this worker thread
        async with KalshiClient(settings) as client:

            # Step 1: Fetch open markets with server-side close time filter
            # This eliminates all short-duration recurring markets (hourly crypto, etc.)
            cutoff = datetime.utcnow() + timedelta(hours=settings.min_expiry_hours)
            min_close_ts = int(cutoff.timestamp())

            await db.update_worker_status(
                stage="fetching_markets",
                message="Fetching markets (filtering short-duration server-side)...",
                progress=0.0, is_running=True,
            )

            all_markets_list = await client.get_all_markets(
                status="open",
                min_close_ts=min_close_ts,
            )
            all_markets = {m.ticker: m for m in all_markets_list}
            total = len(all_markets)

            await db.update_worker_status(
                stage="filtering",
                message=f"Fetched {total} markets, applying OI filter...",
                progress=0.2, total_markets=total, is_running=True,
            )

            # Step 2: Pre-filter by OI, sort by OI desc, cap at max_orderbook_fetches
            after_oi_list = [
                m for m in all_markets.values()
                if m.open_interest >= settings.min_oi_prefilter
            ]
            after_oi_list.sort(key=lambda m: m.open_interest, reverse=True)
            after_oi_list = after_oi_list[:settings.max_orderbook_fetches]
            after_oi = {m.ticker: m for m in after_oi_list}

            await db.update_worker_status(
                stage="fetching_orderbooks",
                message=f"Fetching orderbooks for {len(after_oi)} markets "
                        f"(top by OI, filtered from {total})...",
                progress=0.3, total_markets=total,
                filtered_markets=len(after_oi), is_running=True,
            )

            # Step 3: Parallel orderbook fetches with incremental refresh
            summaries: dict[str, OrderbookSummary] = {}
            ob_total = len(after_oi)

            # Load cached orderbooks to detect unchanged markets
            cached_orderbooks = await db.get_all_orderbook_caches()

            # Split into markets that need re-fetch vs those we can skip
            tickers_to_fetch = []
            for ticker, market in after_oi.items():
                cached = cached_orderbooks.get(ticker)
                if cached and cached.market_yes_bid == market.yes_bid and cached.market_yes_ask == market.yes_ask:
                    # Price unchanged — reuse cached orderbook
                    raw_ob = json.loads(cached.raw_json) if cached.raw_json else None
                    if raw_ob:
                        summary = compute_orderbook_summary(
                            raw_ob, ticker,
                            near_mid_range_cents=settings.near_mid_range_cents,
                            yes_bid=market.yes_bid,
                            yes_ask=market.yes_ask,
                        )
                        summaries[ticker] = summary
                        continue
                tickers_to_fetch.append(ticker)

            skipped = ob_total - len(tickers_to_fetch)
            logger.info(f"Incremental refresh: {skipped} cached, {len(tickers_to_fetch)} to fetch")

            await db.update_worker_status(
                stage="fetching_orderbooks",
                message=f"Fetching {len(tickers_to_fetch)} orderbooks "
                        f"({skipped} cached)...",
                progress=0.3, total_markets=total,
                filtered_markets=ob_total, is_running=True,
            )

            # Parallel fetch with semaphore for rate limiting
            semaphore = asyncio.Semaphore(settings.orderbook_concurrency)
            fetch_results: dict[str, tuple] = {}  # ticker -> (summary, raw_ob)

            async def _fetch_one(ticker: str):
                async with semaphore:
                    try:
                        market = after_oi[ticker]
                        raw_ob = await client.get_orderbook(ticker, depth=settings.orderbook_depth)
                        summary = compute_orderbook_summary(
                            raw_ob, ticker,
                            near_mid_range_cents=settings.near_mid_range_cents,
                            yes_bid=market.yes_bid,
                            yes_ask=market.yes_ask,
                        )
                        fetch_results[ticker] = (summary, raw_ob)
                    except Exception as e:
                        logger.warning(f"Orderbook fetch failed for {ticker}: {e}")

            # Launch all fetches concurrently (semaphore limits parallelism)
            if tickers_to_fetch:
                await asyncio.gather(*[_fetch_one(t) for t in tickers_to_fetch])

            # Write results to DB sequentially to avoid SQLite locking
            for ticker, (summary, raw_ob) in fetch_results.items():
                summaries[ticker] = summary
                market = after_oi[ticker]
                try:
                    await db.upsert_orderbook_cache(
                        ticker=ticker,
                        total_yes_depth=summary.total_yes_depth_dollars,
                        total_no_depth=summary.total_no_depth_dollars,
                        near_touch_yes=summary.near_touch_yes_dollars,
                        near_touch_no=summary.near_touch_no_dollars,
                        near_mid_depth=summary.near_mid_depth_dollars,
                        yes_levels=summary.yes_levels,
                        no_levels=summary.no_levels,
                        best_yes_price=summary.best_yes_price,
                        best_no_price=summary.best_no_price,
                        raw_json=json.dumps(raw_ob),
                        updated_at=datetime.utcnow(),
                        market_yes_bid=market.yes_bid,
                        market_yes_ask=market.yes_ask,
                    )
                except Exception as e:
                    logger.warning(f"Cache write failed for {ticker}: {e}")

            await db.update_worker_status(
                stage="fetching_orderbooks",
                message=f"Orderbooks: {ob_total}/{ob_total} ({skipped} cached, {len(tickers_to_fetch)} fetched)",
                progress=0.9, total_markets=total,
                filtered_markets=ob_total, is_running=True,
            )

            # Step 5: Filter by near-mid depth
            qualified_tickers = {
                t for t, s in summaries.items()
                if s.near_mid_depth_dollars >= settings.min_near_mid_depth_dollars
            }

            # Step 6: Compute price changes
            lookback = datetime.utcnow() - timedelta(hours=settings.price_change_lookback_hours)
            old_snapshots = await db.get_oldest_snapshots_per_ticker(lookback)

            # Step 7: Build qualified market records and write to DB
            now = datetime.utcnow()
            records = []
            snapshot_records = []

            for ticker in qualified_tickers:
                m = after_oi[ticker]
                s = summaries[ticker]

                # Price change
                old = old_snapshots.get(ticker)
                price_change = None
                if old and old.yes_bid is not None and m.yes_bid is not None:
                    price_change = float(m.yes_bid - old.yes_bid)

                records.append({
                    "ticker": m.ticker,
                    "event_ticker": m.event_ticker,
                    "title": m.title,
                    "status": m.status.value,
                    "category": m.category,
                    "yes_bid": m.yes_bid,
                    "yes_ask": m.yes_ask,
                    "no_bid": m.no_bid,
                    "no_ask": m.no_ask,
                    "last_price": m.last_price,
                    "volume": m.volume,
                    "open_interest": m.open_interest,
                    "spread": m.spread,
                    "open_time": m.open_time,
                    "close_time": m.close_time,
                    "expiration_time": m.expiration_time,
                    "total_yes_depth": s.total_yes_depth_dollars,
                    "total_no_depth": s.total_no_depth_dollars,
                    "near_mid_depth": s.near_mid_depth_dollars,
                    "price_change_24h": price_change,
                    "updated_at": now,
                })

                snapshot_records.append({
                    "ticker": m.ticker,
                    "event_ticker": m.event_ticker,
                    "title": m.title,
                    "yes_bid": m.yes_bid,
                    "yes_ask": m.yes_ask,
                    "no_bid": m.no_bid,
                    "no_ask": m.no_ask,
                    "last_price": m.last_price,
                    "volume": m.volume,
                    "open_interest": m.open_interest,
                    "fetched_at": now,
                })

            await db.replace_qualified_markets(records)
            await db.save_market_snapshots_batch(snapshot_records)

            await db.update_worker_status(
                stage="done",
                message=f"Done: {len(qualified_tickers)} qualified markets",
                progress=1.0, total_markets=total,
                filtered_markets=ob_total,
                qualified_markets=len(qualified_tickers),
                is_running=False,
            )

            logger.info(
                f"Refresh complete: {total} total (server-filtered) "
                f"-> {ob_total} after OI -> {len(qualified_tickers)} qualified"
            )

    # ---- Read from DB (called by Streamlit, instant) ----

    def get_qualified_dataframe(self) -> pd.DataFrame:
        """Read qualified markets from DB. No API calls."""
        self._ensure_db()
        db = self._get_db()
        loop = asyncio.new_event_loop()
        rows = loop.run_until_complete(db.get_all_qualified_markets())
        loop.close()

        if not rows:
            return pd.DataFrame()

        records = []
        for r in rows:
            records.append({
                "ticker": r.ticker,
                "event_ticker": r.event_ticker,
                "title": r.title,
                "status": r.status,
                "category": r.category,
                "yes_bid": r.yes_bid,
                "yes_ask": r.yes_ask,
                "no_bid": r.no_bid,
                "no_ask": r.no_ask,
                "last_price": r.last_price,
                "volume": r.volume,
                "open_interest": r.open_interest,
                "spread": r.spread,
                "open_time": r.open_time,
                "close_time": r.close_time,
                "expiration_time": r.expiration_time,
                "total_yes_depth": r.total_yes_depth,
                "total_no_depth": r.total_no_depth,
                "near_mid_depth": r.near_mid_depth,
                "price_change_24h": r.price_change_24h,
                "updated_at": r.updated_at,
            })

        return pd.DataFrame(records)

    def get_worker_status(self) -> Optional[dict]:
        """Read worker status from DB."""
        self._ensure_db()
        db = self._get_db()
        loop = asyncio.new_event_loop()
        status = loop.run_until_complete(db.get_worker_status())
        loop.close()

        if not status:
            return None

        return {
            "stage": status.stage,
            "message": status.message,
            "progress": status.progress,
            "total_markets": status.total_markets,
            "filtered_markets": status.filtered_markets,
            "qualified_markets": status.qualified_markets,
            "is_running": status.is_running == 1,
            "updated_at": status.updated_at,
        }

    def get_categories(self) -> list[str]:
        df = self.get_qualified_dataframe()
        if df.empty or "category" not in df.columns:
            return []
        return sorted(df["category"].dropna().unique().tolist())

    def fetch_orderbook_detail(self, ticker: str) -> tuple[Optional[OrderbookSummary], Optional[dict]]:
        """Fetch fresh orderbook for detail view (single API call)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._async_fetch_orderbook(ticker))
        finally:
            loop.close()

    async def _async_fetch_orderbook(self, ticker: str) -> tuple[Optional[OrderbookSummary], Optional[dict]]:
        db = self._get_db()

        # Try cache first
        cached = await db.get_orderbook_cache(ticker)
        if cached and cached.raw_json:
            raw_ob = json.loads(cached.raw_json)
            summary = OrderbookSummary(
                ticker=ticker,
                total_yes_depth_dollars=cached.total_yes_depth,
                total_no_depth_dollars=cached.total_no_depth,
                near_touch_yes_dollars=cached.near_touch_yes,
                near_touch_no_dollars=cached.near_touch_no,
                near_mid_depth_dollars=cached.near_mid_depth,
                yes_levels=cached.yes_levels,
                no_levels=cached.no_levels,
                best_yes_price=cached.best_yes_price,
                best_no_price=cached.best_no_price,
            )
            return summary, raw_ob

        # Fetch fresh if no cache
        async with KalshiClient(self.settings) as client:
            raw_ob = await client.get_orderbook(ticker, depth=self.settings.orderbook_depth)
            summary = compute_orderbook_summary(
                raw_ob, ticker,
                near_mid_range_cents=self.settings.near_mid_range_cents,
            )
            await db.upsert_orderbook_cache(
                ticker=ticker,
                total_yes_depth=summary.total_yes_depth_dollars,
                total_no_depth=summary.total_no_depth_dollars,
                near_touch_yes=summary.near_touch_yes_dollars,
                near_touch_no=summary.near_touch_no_dollars,
                near_mid_depth=summary.near_mid_depth_dollars,
                yes_levels=summary.yes_levels,
                no_levels=summary.no_levels,
                best_yes_price=summary.best_yes_price,
                best_no_price=summary.best_no_price,
                raw_json=json.dumps(raw_ob),
                updated_at=datetime.utcnow(),
            )
            return summary, raw_ob
