"""
Arbitrage Scanner - Main entry point for detecting arbitrage opportunities.

Usage:
    python -m src.scanner              # Run once
    python -m src.scanner --continuous # Run continuously
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from .analysis.arbitrage import ArbitrageAnalyzer
from .api.client import KalshiClient
from .config import get_settings
from .data.aggregator import DataAggregator
from .data.database import Database

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
)
logger.add(
    "logs/scanner_{time}.log",
    rotation="1 day",
    retention="7 days",
    level="DEBUG",
)


async def run_scan(continuous: bool = False, save_to_db: bool = True) -> None:
    """
    Run the arbitrage scanner.

    Args:
        continuous: If True, run continuously. Otherwise scan once.
        save_to_db: If True, save snapshots and opportunities to database.
    """
    settings = get_settings()
    database = Database(settings)
    analyzer = ArbitrageAnalyzer(settings)

    # Ensure data directory exists
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    if save_to_db:
        await database.init_db()

    async with KalshiClient(settings) as client:
        aggregator = DataAggregator(settings, client, database)

        if continuous:
            # Register callback to analyze on each update
            def on_update(markets):
                opportunities = analyzer.find_all_opportunities(
                    markets, aggregator.events
                )
                if opportunities:
                    logger.info(f"Found {len(opportunities)} arbitrage opportunities")
                    for opp in opportunities[:5]:  # Show top 5
                        logger.info(
                            f"  {opp.arb_type.value}: {opp.markets} | "
                            f"Profit: {opp.expected_profit}¢ | "
                            f"Confidence: {opp.confidence:.0%}"
                        )

            aggregator.on_update(on_update)
            await aggregator.run_continuous(save_to_db=save_to_db)

        else:
            # Single scan
            await aggregator.fetch_and_cache_events()

            logger.info(f"Loaded {len(aggregator.markets)} markets")
            logger.info(f"Loaded {len(aggregator.events)} events")

            # Find opportunities
            opportunities = analyzer.find_all_opportunities(
                aggregator.markets, aggregator.events
            )

            if opportunities:
                logger.info(f"\n{'='*60}")
                logger.info(f"FOUND {len(opportunities)} ARBITRAGE OPPORTUNITIES")
                logger.info(f"{'='*60}\n")

                for i, opp in enumerate(opportunities, 1):
                    logger.info(f"#{i} | {opp.arb_type.value}")
                    logger.info(f"    Markets: {', '.join(opp.markets)}")
                    logger.info(f"    Expected Profit: {opp.expected_profit}¢")
                    logger.info(f"    Risk: {opp.risk}¢")
                    logger.info(f"    Confidence: {opp.confidence:.0%}")
                    logger.info(f"    Legs: {json.dumps(opp.legs, indent=6)}")
                    logger.info("")

                    # Save to database
                    if save_to_db:
                        await database.save_arbitrage(
                            arb_id=opp.id,
                            arb_type=opp.arb_type.value,
                            markets=json.dumps(opp.markets),
                            expected_profit=opp.expected_profit,
                            max_profit=opp.max_profit,
                            risk=opp.risk,
                            confidence=opp.confidence,
                            legs=json.dumps(opp.legs),
                            detected_at=opp.detected_at,
                        )
            else:
                logger.info("No arbitrage opportunities found at this time.")

            # Show market summary
            df = aggregator.to_dataframe()
            if not df.empty:
                logger.info(f"\nMarket Summary:")
                logger.info(f"  Total markets: {len(df)}")
                logger.info(f"  Markets with bids: {df['yes_bid'].notna().sum()}")
                logger.info(f"  Avg spread: {df['spread'].mean():.1f}¢")
                logger.info(f"  Total volume: {df['volume'].sum():,}")

            if save_to_db:
                await aggregator.save_snapshots()

    await database.close()


def main():
    """CLI entry point."""
    continuous = "--continuous" in sys.argv or "-c" in sys.argv
    no_db = "--no-db" in sys.argv

    logger.info(f"Starting Kalshi Arbitrage Scanner")
    logger.info(f"Mode: {'Continuous' if continuous else 'Single scan'}")

    asyncio.run(run_scan(continuous=continuous, save_to_db=not no_db))


if __name__ == "__main__":
    main()
