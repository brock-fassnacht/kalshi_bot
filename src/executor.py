"""
Trade Executor - Executes arbitrage opportunities.

This module handles the actual trade execution with safety checks.

WARNING: This is a template. Review carefully before using with real money.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Optional

from loguru import logger

from .api.client import KalshiClient
from .config import Settings
from .models import ArbitrageOpportunity, OrderAction, OrderSide


class TradeExecutor:
    """
    Executes arbitrage trades on Kalshi.

    Safety features:
    - Position size limits
    - Dry-run mode for testing
    - Order confirmation
    - Execution logging
    """

    def __init__(
        self,
        client: KalshiClient,
        settings: Settings,
        dry_run: bool = True,
    ):
        self.client = client
        self.settings = settings
        self.dry_run = dry_run
        self._execution_log: list[dict] = []

    async def execute_opportunity(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: int = 1,
    ) -> dict:
        """
        Execute an arbitrage opportunity.

        Args:
            opportunity: The arbitrage opportunity to execute
            quantity: Number of contract sets to trade

        Returns:
            Execution result dict
        """
        # Safety checks
        if quantity > self.settings.max_position_size:
            logger.warning(
                f"Quantity {quantity} exceeds max position size "
                f"{self.settings.max_position_size}"
            )
            quantity = self.settings.max_position_size

        if opportunity.confidence < 0.5:
            logger.warning(f"Low confidence opportunity: {opportunity.confidence:.0%}")
            return {"status": "rejected", "reason": "low_confidence"}

        logger.info(f"Executing arbitrage: {opportunity.id}")
        logger.info(f"  Type: {opportunity.arb_type.value}")
        logger.info(f"  Quantity: {quantity}")
        logger.info(f"  Expected profit: {opportunity.expected_profit * quantity}¢")

        if self.dry_run:
            logger.info("  [DRY RUN - No orders placed]")
            return await self._simulate_execution(opportunity, quantity)

        # Execute each leg
        results = []
        for leg in opportunity.legs:
            try:
                result = await self._execute_leg(leg, quantity)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute leg: {e}")
                # In production, implement rollback logic here
                return {
                    "status": "partial_failure",
                    "completed_legs": results,
                    "error": str(e),
                }

        execution_result = {
            "status": "success",
            "opportunity_id": opportunity.id,
            "quantity": quantity,
            "legs": results,
            "executed_at": datetime.utcnow().isoformat(),
        }

        self._execution_log.append(execution_result)
        return execution_result

    async def _execute_leg(self, leg: dict, quantity: int) -> dict:
        """Execute a single trade leg."""
        ticker = leg["ticker"]
        side = leg["side"]
        action = leg["action"]
        price = leg["price"]

        logger.info(f"    {action.upper()} {quantity} {side.upper()} @ {price}¢ on {ticker}")

        # Create order
        order_result = await self.client.create_order(
            ticker=ticker,
            side=side,
            action=action,
            count=quantity,
            type="limit",
            yes_price=price if side == "yes" else None,
            no_price=price if side == "no" else None,
            client_order_id=f"arb_{uuid.uuid4().hex[:8]}",
        )

        return {
            "ticker": ticker,
            "side": side,
            "action": action,
            "quantity": quantity,
            "price": price,
            "order_id": order_result.get("order", {}).get("order_id"),
        }

    async def _simulate_execution(
        self,
        opportunity: ArbitrageOpportunity,
        quantity: int,
    ) -> dict:
        """Simulate execution for dry run mode."""
        simulated_legs = []
        for leg in opportunity.legs:
            simulated_legs.append({
                "ticker": leg["ticker"],
                "side": leg["side"],
                "action": leg["action"],
                "quantity": quantity,
                "price": leg["price"],
                "order_id": f"SIM_{uuid.uuid4().hex[:8]}",
            })
            logger.info(
                f"    [SIM] {leg['action'].upper()} {quantity} "
                f"{leg['side'].upper()} @ {leg['price']}¢ on {leg['ticker']}"
            )

        return {
            "status": "simulated",
            "opportunity_id": opportunity.id,
            "quantity": quantity,
            "legs": simulated_legs,
            "simulated_profit": opportunity.expected_profit * quantity,
            "executed_at": datetime.utcnow().isoformat(),
        }

    async def get_current_positions(self) -> list[dict]:
        """Get current positions from Kalshi."""
        return await self.client.get_positions()

    async def get_balance(self) -> dict:
        """Get current account balance."""
        return await self.client.get_balance()

    def get_execution_log(self) -> list[dict]:
        """Get log of all executions."""
        return self._execution_log.copy()


async def run_executor_demo():
    """Demo the executor in dry-run mode."""
    from .analysis.arbitrage import ArbitrageAnalyzer
    from .config import get_settings
    from .data.aggregator import DataAggregator
    from .data.database import Database

    settings = get_settings()
    database = Database(settings)

    async with KalshiClient(settings) as client:
        aggregator = DataAggregator(settings, client, database)
        analyzer = ArbitrageAnalyzer(settings)
        executor = TradeExecutor(client, settings, dry_run=True)

        # Fetch data
        await aggregator.fetch_and_cache_events()

        # Find opportunities
        opportunities = analyzer.find_all_opportunities(
            aggregator.markets, aggregator.events
        )

        if opportunities:
            # Execute best opportunity (dry run)
            best = opportunities[0]
            result = await executor.execute_opportunity(best, quantity=10)
            logger.info(f"Execution result: {result}")
        else:
            logger.info("No opportunities to execute")


if __name__ == "__main__":
    asyncio.run(run_executor_demo())
