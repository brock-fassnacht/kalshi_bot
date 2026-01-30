"""Cross-market arbitrage detection and analysis."""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..config import Settings
from ..models import (
    ArbitrageOpportunity,
    ArbitrageType,
    Event,
    Market,
    OrderAction,
    OrderSide,
    TradeLeg,
)


class ArbitrageAnalyzer:
    """
    Analyzes Kalshi markets for cross-market arbitrage opportunities.

    Arbitrage Types Detected:
    1. Yes/No Mispricing: When yes_ask + no_ask < 100 (guaranteed profit)
    2. Multi-outcome: Sum of mutually exclusive outcomes in an event != 100
    3. Correlated Markets: Statistical mispricing between related markets
    4. Calendar Spreads: Same underlying, different expiration dates
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.min_spread = settings.min_arbitrage_spread * 100  # Convert to cents

    def find_all_opportunities(
        self,
        markets: dict[str, Market],
        events: dict[str, Event],
    ) -> list[ArbitrageOpportunity]:
        """
        Scan all markets for arbitrage opportunities.

        Args:
            markets: Dict of ticker -> Market
            events: Dict of event_ticker -> Event

        Returns:
            List of detected arbitrage opportunities
        """
        opportunities = []

        # 1. Check Yes/No mispricing on individual markets
        yes_no_opps = self._find_yes_no_mispricing(markets)
        opportunities.extend(yes_no_opps)

        # 2. Check multi-outcome arbitrage within events
        multi_outcome_opps = self._find_multi_outcome_arbitrage(events)
        opportunities.extend(multi_outcome_opps)

        # 3. Check correlated market mispricing
        correlated_opps = self._find_correlated_arbitrage(markets, events)
        opportunities.extend(correlated_opps)

        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit, reverse=True)

        return opportunities

    def _find_yes_no_mispricing(
        self,
        markets: dict[str, Market],
    ) -> list[ArbitrageOpportunity]:
        """
        Find markets where Yes + No prices create arbitrage.

        In a perfect market: yes_ask + no_ask >= 100 and yes_bid + no_bid <= 100
        Arbitrage exists when: yes_ask + no_ask < 100 (buy both, guaranteed profit)
        """
        opportunities = []

        for ticker, market in markets.items():
            if market.yes_ask is None or market.no_ask is None:
                continue

            total_ask = market.yes_ask + market.no_ask
            spread = 100 - total_ask  # Positive = arbitrage opportunity

            if spread >= self.min_spread:
                # Calculate profit: buying both Yes and No guarantees $1 payout
                # Cost = yes_ask + no_ask, Payout = 100, Profit = spread
                profit = spread

                opp = ArbitrageOpportunity(
                    id=self._generate_id("yes_no", [ticker]),
                    arb_type=ArbitrageType.YES_NO_MISPRICING,
                    markets=[ticker],
                    expected_profit=profit,
                    max_profit=profit,
                    risk=0,  # Risk-free if both orders fill
                    legs=[
                        {
                            "ticker": ticker,
                            "side": "yes",
                            "action": "buy",
                            "price": market.yes_ask,
                        },
                        {
                            "ticker": ticker,
                            "side": "no",
                            "action": "buy",
                            "price": market.no_ask,
                        },
                    ],
                    confidence=1.0,  # Deterministic arbitrage
                    valid_until=datetime.utcnow() + timedelta(seconds=30),
                )
                opportunities.append(opp)

                logger.info(
                    f"Yes/No arbitrage: {ticker} | "
                    f"Yes@{market.yes_ask} + No@{market.no_ask} = {total_ask}¢ | "
                    f"Profit: {profit}¢"
                )

        return opportunities

    def _find_multi_outcome_arbitrage(
        self,
        events: dict[str, Event],
    ) -> list[ArbitrageOpportunity]:
        """
        Find events where mutually exclusive outcomes don't sum to 100.

        For mutually exclusive outcomes, the sum of all yes prices should = 100.
        If sum of yes_asks < 100: buy all outcomes for guaranteed profit.
        If sum of yes_bids > 100: sell all outcomes for guaranteed profit.
        """
        opportunities = []

        for event_ticker, event in events.items():
            if len(event.markets) < 2:
                continue

            # Filter markets with valid pricing
            valid_markets = [
                m for m in event.markets
                if m.yes_ask is not None and m.yes_bid is not None
            ]

            if len(valid_markets) < 2:
                continue

            # Check if this looks like mutually exclusive outcomes
            # (Usually event markets are structured this way)
            sum_yes_ask = sum(m.yes_ask for m in valid_markets)
            sum_yes_bid = sum(m.yes_bid for m in valid_markets)

            # Buy-side arbitrage: sum of asks < 100
            buy_spread = 100 - sum_yes_ask
            if buy_spread >= self.min_spread:
                legs = [
                    {
                        "ticker": m.ticker,
                        "side": "yes",
                        "action": "buy",
                        "price": m.yes_ask,
                    }
                    for m in valid_markets
                ]

                opp = ArbitrageOpportunity(
                    id=self._generate_id("multi_buy", [m.ticker for m in valid_markets]),
                    arb_type=ArbitrageType.MULTI_OUTCOME,
                    markets=[m.ticker for m in valid_markets],
                    expected_profit=buy_spread,
                    max_profit=buy_spread,
                    risk=0,
                    legs=legs,
                    confidence=0.95,  # Slight uncertainty about mutual exclusivity
                    valid_until=datetime.utcnow() + timedelta(seconds=30),
                )
                opportunities.append(opp)

                logger.info(
                    f"Multi-outcome BUY arbitrage: {event_ticker} | "
                    f"Sum of asks: {sum_yes_ask}¢ | Profit: {buy_spread}¢"
                )

            # Sell-side arbitrage: sum of bids > 100
            sell_spread = sum_yes_bid - 100
            if sell_spread >= self.min_spread:
                legs = [
                    {
                        "ticker": m.ticker,
                        "side": "yes",
                        "action": "sell",
                        "price": m.yes_bid,
                    }
                    for m in valid_markets
                ]

                opp = ArbitrageOpportunity(
                    id=self._generate_id("multi_sell", [m.ticker for m in valid_markets]),
                    arb_type=ArbitrageType.MULTI_OUTCOME,
                    markets=[m.ticker for m in valid_markets],
                    expected_profit=sell_spread,
                    max_profit=sell_spread,
                    risk=0,
                    legs=legs,
                    confidence=0.95,
                    valid_until=datetime.utcnow() + timedelta(seconds=30),
                )
                opportunities.append(opp)

                logger.info(
                    f"Multi-outcome SELL arbitrage: {event_ticker} | "
                    f"Sum of bids: {sum_yes_bid}¢ | Profit: {sell_spread}¢"
                )

        return opportunities

    def _find_correlated_arbitrage(
        self,
        markets: dict[str, Market],
        events: dict[str, Event],
    ) -> list[ArbitrageOpportunity]:
        """
        Find statistical arbitrage between correlated markets.

        Examples:
        - "Will X win State A?" vs "Will X win nationally?"
        - "Temperature > 80F on Monday?" vs "Temperature > 80F this week?"

        This is probabilistic arbitrage based on logical relationships.
        """
        opportunities = []

        # Group markets by event for correlation analysis
        event_markets = defaultdict(list)
        for ticker, market in markets.items():
            event_markets[market.event_ticker].append(market)

        # Look for hierarchical relationships
        # Example: If market A implies market B, then P(A) <= P(B)
        for event_ticker, event_market_list in event_markets.items():
            if len(event_market_list) < 2:
                continue

            # Check for calendar-style relationships (by expiration)
            sorted_by_expiry = sorted(
                [m for m in event_market_list if m.expiration_time],
                key=lambda x: x.expiration_time,
            )

            for i in range(len(sorted_by_expiry) - 1):
                earlier = sorted_by_expiry[i]
                later = sorted_by_expiry[i + 1]

                if earlier.yes_ask is None or later.yes_bid is None:
                    continue

                # If earlier expires first, P(earlier=yes) <= P(later=yes)
                # So earlier.yes_ask should be <= later.yes_bid
                # Arbitrage if earlier.yes_ask > later.yes_bid
                if earlier.yes_ask > later.yes_bid:
                    spread = earlier.yes_ask - later.yes_bid

                    if spread >= self.min_spread:
                        opp = ArbitrageOpportunity(
                            id=self._generate_id(
                                "calendar", [earlier.ticker, later.ticker]
                            ),
                            arb_type=ArbitrageType.CALENDAR_SPREAD,
                            markets=[earlier.ticker, later.ticker],
                            expected_profit=spread * 0.7,  # Discount for risk
                            max_profit=spread,
                            risk=spread * 0.3,
                            legs=[
                                {
                                    "ticker": earlier.ticker,
                                    "side": "yes",
                                    "action": "sell",
                                    "price": earlier.yes_bid,
                                },
                                {
                                    "ticker": later.ticker,
                                    "side": "yes",
                                    "action": "buy",
                                    "price": later.yes_ask,
                                },
                            ],
                            confidence=0.7,
                            valid_until=datetime.utcnow() + timedelta(seconds=30),
                        )
                        opportunities.append(opp)

                        logger.info(
                            f"Calendar spread: {earlier.ticker} vs {later.ticker} | "
                            f"Spread: {spread}¢"
                        )

        return opportunities

    def analyze_correlation(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        price_col: str = "last_price",
    ) -> dict:
        """
        Analyze statistical correlation between two market price series.

        Args:
            df1: Historical data for market 1
            df2: Historical data for market 2
            price_col: Column to use for correlation

        Returns:
            Dict with correlation metrics
        """
        if df1.empty or df2.empty:
            return {"error": "Insufficient data"}

        # Align on timestamp
        merged = pd.merge(
            df1[[price_col]].rename(columns={price_col: "price1"}),
            df2[[price_col]].rename(columns={price_col: "price2"}),
            left_index=True,
            right_index=True,
            how="inner",
        )

        if len(merged) < 10:
            return {"error": "Insufficient overlapping data"}

        # Calculate metrics
        correlation = merged["price1"].corr(merged["price2"])

        # Calculate spread statistics
        spread = merged["price1"] - merged["price2"]
        spread_mean = spread.mean()
        spread_std = spread.std()
        current_spread = spread.iloc[-1]
        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Cointegration test (simplified)
        # In production, use statsmodels.tsa.stattools.coint
        spread_adf = self._adf_test(spread)

        return {
            "correlation": correlation,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "current_spread": current_spread,
            "z_score": z_score,
            "is_cointegrated": spread_adf < -2.5,  # Simplified threshold
            "sample_size": len(merged),
        }

    def _adf_test(self, series: pd.Series) -> float:
        """Simplified ADF test statistic."""
        if len(series) < 20:
            return 0

        # Simple approximation - in production use statsmodels
        diff = series.diff().dropna()
        if diff.std() == 0:
            return 0
        return (series.iloc[-1] - series.mean()) / diff.std()

    def _generate_id(self, prefix: str, tickers: list[str]) -> str:
        """Generate unique ID for an arbitrage opportunity."""
        key = f"{prefix}:{'|'.join(sorted(tickers))}:{datetime.utcnow().isoformat()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def get_opportunity_summary(
        self,
        opportunities: list[ArbitrageOpportunity],
    ) -> pd.DataFrame:
        """Convert opportunities to summary DataFrame."""
        if not opportunities:
            return pd.DataFrame()

        records = []
        for opp in opportunities:
            records.append({
                "id": opp.id,
                "type": opp.arb_type.value,
                "markets": ", ".join(opp.markets),
                "expected_profit": opp.expected_profit,
                "max_profit": opp.max_profit,
                "risk": opp.risk,
                "profit_ratio": opp.profit_ratio,
                "confidence": opp.confidence,
                "num_legs": len(opp.legs),
                "detected_at": opp.detected_at,
            })

        return pd.DataFrame(records)
