"""Cross-market arbitrage analyzer for Kalshi vs IB futures."""
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from ..config import Settings
from ..models import ArbitrageOpportunity, ArbitrageType, Market

logger = logging.getLogger(__name__)


class CrossMarketArbitrageAnalyzer:
    """Detects arbitrage between Kalshi prediction markets and IB futures."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.roi_threshold = settings.roi_threshold

    def find_btc_arbitrage(
        self,
        kalshi_markets: list[Market],
        ib_orderbook: pd.DataFrame,
        btc_strike_price: Optional[float] = None,
    ) -> list[ArbitrageOpportunity]:
        """
        Find arbitrage between Kalshi "BTC above/below X" markets
        and actual MBT futures pricing.

        Logic:
        - Kalshi market: "Will BTC be above $100k by March 31?"
          - Yes price = 65 cents (implies 65% probability)
        - MBT futures: Trading at $98,500 for March expiry

        If futures price + volatility suggests >65% chance of hitting $100k,
        but Kalshi yes is at 65 cents, there may be an edge.

        Simpler approach for direct arb:
        - If you can buy Kalshi Yes at 65 cents
        - And the futures are already above strike price
        - That's a clear mispricing (100% payout for 65 cent cost = 53% ROI)

        Args:
            kalshi_markets: List of Kalshi markets (Bitcoin-related)
            ib_orderbook: DataFrame with IB futures order book
            btc_strike_price: Optional override strike price

        Returns:
            List of ArbitrageOpportunity objects
        """
        opportunities = []

        # Get IB best prices
        ib_best_bid = ib_orderbook["bid_price"].dropna().iloc[0] if len(ib_orderbook) > 0 else None
        ib_best_ask = ib_orderbook["ask_price"].dropna().iloc[0] if len(ib_orderbook) > 0 else None

        if ib_best_bid is None or ib_best_ask is None:
            logger.warning("No IB order book data available")
            return opportunities

        ib_mid = (ib_best_bid + ib_best_ask) / 2
        logger.info(f"IB MBT mid price: ${ib_mid:,.2f}")

        for market in kalshi_markets:
            # Parse strike price from market title
            strike = self._extract_strike_from_market(market)
            if strike is None:
                continue

            logger.debug(
                f"Analyzing {market.ticker}: strike=${strike:,.0f}, "
                f"yes_ask={market.yes_ask}, yes_bid={market.yes_bid}"
            )

            # Case 1: Futures already above strike, Kalshi Yes is cheap
            # If BTC is trading above the strike, YES should resolve to 100
            if ib_mid > strike and market.yes_ask is not None:
                opp = self._analyze_above_strike_opportunity(
                    market, strike, ib_mid, ib_best_bid, ib_best_ask
                )
                if opp:
                    opportunities.append(opp)

            # Case 2: Futures below strike, Kalshi No is cheap
            # If BTC is trading below the strike, NO should resolve to 100
            if ib_mid < strike and market.no_ask is not None:
                opp = self._analyze_below_strike_opportunity(
                    market, strike, ib_mid, ib_best_bid, ib_best_ask
                )
                if opp:
                    opportunities.append(opp)

            # Case 3: Check for put-call parity violations (YES + NO should = ~100)
            parity_opp = self._analyze_parity_violation(market, ib_mid)
            if parity_opp:
                opportunities.append(parity_opp)

        return opportunities

    def _analyze_above_strike_opportunity(
        self,
        market: Market,
        strike: float,
        ib_mid: float,
        ib_bid: float,
        ib_ask: float,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Analyze opportunity when futures are above Kalshi strike.

        If BTC futures > strike price, buying YES at discount is profitable.
        """
        if market.yes_ask is None:
            return None

        # Buying Yes should be near 100 cents if futures > strike
        implied_value = 100  # Will resolve to Yes
        cost = market.yes_ask
        profit = implied_value - cost
        roi = profit / cost

        # Higher confidence if futures are well above strike
        buffer_pct = (ib_mid - strike) / strike
        confidence = min(0.95, 0.5 + buffer_pct * 2)

        # Risk score (lower if futures are much higher than strike)
        risk = max(5, 50 - buffer_pct * 100)

        if roi >= self.roi_threshold:
            return ArbitrageOpportunity(
                id=self._generate_id("btc_above", [market.ticker, "MBT"]),
                arb_type=ArbitrageType.CORRELATED_MARKETS,
                markets=[market.ticker],
                expected_profit=profit,
                max_profit=profit,
                risk=risk,
                legs=[
                    {
                        "ticker": market.ticker,
                        "side": "yes",
                        "action": "buy",
                        "price": cost,
                    },
                    {
                        "symbol": "MBT",
                        "action": "reference",
                        "price": ib_mid,
                    },
                ],
                confidence=confidence,
                valid_until=datetime.utcnow() + timedelta(seconds=60),
            )
        return None

    def _analyze_below_strike_opportunity(
        self,
        market: Market,
        strike: float,
        ib_mid: float,
        ib_bid: float,
        ib_ask: float,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Analyze opportunity when futures are below Kalshi strike.

        If BTC futures < strike price, buying NO at discount is profitable.
        """
        if market.no_ask is None:
            return None

        # Buying No should be near 100 cents if futures < strike
        implied_value = 100  # Will resolve to No
        cost = market.no_ask
        profit = implied_value - cost
        roi = profit / cost

        # Higher confidence if futures are well below strike
        buffer_pct = (strike - ib_mid) / strike
        confidence = min(0.95, 0.5 + buffer_pct * 2)

        # Risk score (lower if futures are much lower than strike)
        risk = max(5, 50 - buffer_pct * 100)

        if roi >= self.roi_threshold:
            return ArbitrageOpportunity(
                id=self._generate_id("btc_below", [market.ticker, "MBT"]),
                arb_type=ArbitrageType.CORRELATED_MARKETS,
                markets=[market.ticker],
                expected_profit=profit,
                max_profit=profit,
                risk=risk,
                legs=[
                    {
                        "ticker": market.ticker,
                        "side": "no",
                        "action": "buy",
                        "price": cost,
                    },
                    {
                        "symbol": "MBT",
                        "action": "reference",
                        "price": ib_mid,
                    },
                ],
                confidence=confidence,
                valid_until=datetime.utcnow() + timedelta(seconds=60),
            )
        return None

    def _analyze_parity_violation(
        self,
        market: Market,
        ib_mid: float,
    ) -> Optional[ArbitrageOpportunity]:
        """
        Check for YES + NO parity violations.

        In a binary market, YES_price + NO_price should equal ~100 cents
        (minus spread). If they don't, there's an arbitrage opportunity.
        """
        if market.yes_ask is None or market.no_ask is None:
            return None

        # Cost to buy both YES and NO
        total_cost = market.yes_ask + market.no_ask

        # One of them will pay 100 cents
        guaranteed_payout = 100

        if total_cost < guaranteed_payout:
            profit = guaranteed_payout - total_cost
            roi = profit / total_cost

            if roi >= self.roi_threshold:
                return ArbitrageOpportunity(
                    id=self._generate_id("parity", [market.ticker]),
                    arb_type=ArbitrageType.YES_NO_MISPRICING,
                    markets=[market.ticker],
                    expected_profit=profit,
                    max_profit=profit,
                    risk=1,  # Very low risk - guaranteed payout
                    legs=[
                        {
                            "ticker": market.ticker,
                            "side": "yes",
                            "action": "buy",
                            "price": market.yes_ask,
                        },
                        {
                            "ticker": market.ticker,
                            "side": "no",
                            "action": "buy",
                            "price": market.no_ask,
                        },
                    ],
                    confidence=0.99,  # Nearly certain
                    valid_until=datetime.utcnow() + timedelta(seconds=30),
                )
        return None

    def _extract_strike_from_market(self, market: Market) -> Optional[float]:
        """
        Extract strike price from Kalshi market title.

        Examples:
        - "Will Bitcoin be above $100,000 on March 31?" -> 100000
        - "BTC >= 95000 by EOD" -> 95000
        - "Bitcoin above 100k" -> 100000
        """
        title = market.title.lower()

        # Pattern 1: "$X,XXX" or "$XXX,XXX"
        match = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", title)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                return float(price_str)
            except ValueError:
                pass

        # Pattern 2: "XXXk" (thousands shorthand)
        match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", title)
        if match:
            try:
                return float(match.group(1)) * 1000
            except ValueError:
                pass

        # Pattern 3: Plain number after "above", "below", ">=", "<="
        match = re.search(r"(?:above|below|>=|<=|>|<)\s*(\d{4,})", title)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

    @staticmethod
    def _generate_id(prefix: str, components: list[str]) -> str:
        """Generate a unique ID for an arbitrage opportunity."""
        combined = "_".join([prefix] + components + [str(datetime.utcnow().timestamp())])
        return hashlib.md5(combined.encode()).hexdigest()[:12]
