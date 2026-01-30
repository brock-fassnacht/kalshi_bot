"""
Options-hedged arbitrage analyzer for Kalshi prediction markets.

Strategy:
- Find Kalshi markets like "When will BTC hit $150K?"
- Buy NO contracts (betting BTC won't hit target by expiry)
- Hedge with bull call spread (buy lower strike call, sell higher strike call)

Example:
- Kalshi: "Will BTC hit $150K by June?" - NO @ 90 cents (11% ROI if BTC < $150K)
- Hedge: Buy $130K call, Sell $150K call (profits if BTC rises above $130K)

The combined position profits in ALL scenarios:
1. BTC < $130K: Kalshi NO wins (+10c), spread expires worthless (lose premium)
2. $130K < BTC < $150K: Kalshi NO wins (+10c), spread has value
3. BTC >= $150K: Kalshi NO loses (-90c), spread at max value (offsets loss)

CALCULATION LOCATION: All profit/loss calculations are in this file:
- `calculate_spread_payoff()`: Options spread P&L at various BTC prices
- `calculate_combined_position()`: Combined Kalshi + options P&L
- `find_hedged_arbitrage()`: Main entry point that finds opportunities
- `_calculate_arbitrage_metrics()`: ROI and position sizing calculations
"""
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from ..config import Settings
from ..models import ArbitrageOpportunity, ArbitrageType, Market

logger = logging.getLogger(__name__)


@dataclass
class OptionQuote:
    """Quote data for a single option."""
    strike: float
    expiry: str  # YYYYMMDD format
    right: str  # 'C' for call, 'P' for put
    bid: Optional[float]
    ask: Optional[float]
    last: Optional[float]
    volume: int = 0
    open_interest: int = 0

    @property
    def mid(self) -> Optional[float]:
        """Mid price of the option."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread_width(self) -> Optional[float]:
        """Bid-ask spread in dollars."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def spread_percent(self) -> Optional[float]:
        """Bid-ask spread as percentage of mid price."""
        if self.mid and self.spread_width is not None:
            return (self.spread_width / self.mid) * 100
        return None

    def is_tight_spread(self, max_percent: float = 10.0) -> bool:
        """Check if spread is tight enough to use bid/ask pricing."""
        pct = self.spread_percent
        if pct is None:
            return False
        return pct <= max_percent


@dataclass
class OptionsChain:
    """Full options chain for MBT futures."""
    underlying_price: float
    underlying_expiry: str
    calls: list[OptionQuote]
    puts: list[OptionQuote]
    timestamp: datetime

    def get_call(self, strike: float) -> Optional[OptionQuote]:
        """Get call option at specific strike."""
        for c in self.calls:
            if abs(c.strike - strike) < 0.01:
                return c
        return None

    def get_put(self, strike: float) -> Optional[OptionQuote]:
        """Get put option at specific strike."""
        for p in self.puts:
            if abs(p.strike - strike) < 0.01:
                return p
        return None

    def get_available_strikes(self, min_expiry: Optional[str] = None) -> list[float]:
        """Get all available strike prices, optionally filtered by minimum expiry."""
        strikes = set()
        for c in self.calls:
            if min_expiry is None or c.expiry >= min_expiry:
                strikes.add(c.strike)
        for p in self.puts:
            if min_expiry is None or p.expiry >= min_expiry:
                strikes.add(p.strike)
        return sorted(strikes)

    def get_call_with_min_expiry(self, strike: float, min_expiry: str) -> Optional[OptionQuote]:
        """Get call option at specific strike with expiry >= min_expiry."""
        # Find all calls at this strike with valid expiry, return the nearest one
        valid_calls = [
            c for c in self.calls
            if abs(c.strike - strike) < 0.01 and c.expiry >= min_expiry
        ]
        if not valid_calls:
            return None
        # Return the one with nearest expiry (to minimize time premium)
        return min(valid_calls, key=lambda c: c.expiry)

    def get_available_expiries(self) -> list[str]:
        """Get all available option expiry dates sorted."""
        expiries = set()
        for c in self.calls:
            expiries.add(c.expiry)
        for p in self.puts:
            expiries.add(p.expiry)
        return sorted(expiries)


@dataclass
class SpreadPosition:
    """A call or put spread position."""
    long_strike: float  # Strike of the option we buy
    short_strike: float  # Strike of the option we sell
    long_premium: float  # Premium paid for long option
    short_premium: float  # Premium received for short option
    expiry: str
    is_call_spread: bool = True  # True for call spread, False for put spread
    contracts: int = 1  # Number of spread contracts
    pricing_method: str = "bid/ask"  # "bid/ask" for tight spreads, "mid" for wide spreads

    @property
    def net_debit(self) -> float:
        """Net premium paid to enter the spread (per contract)."""
        return (self.long_premium - self.short_premium) * self.contracts

    @property
    def max_profit(self) -> float:
        """Maximum profit if underlying goes past short strike."""
        spread_width = abs(self.short_strike - self.long_strike)
        # MBT is 0.1 BTC per contract
        return (spread_width * 0.1 * self.contracts) - self.net_debit

    @property
    def max_loss(self) -> float:
        """Maximum loss (net debit paid)."""
        return self.net_debit


@dataclass
class HedgedArbitrageOpportunity:
    """
    A complete hedged arbitrage opportunity.

    Combines a Kalshi prediction market position with an options spread hedge.
    """
    # Kalshi position
    kalshi_ticker: str
    kalshi_title: str
    kalshi_side: str  # 'yes' or 'no'
    kalshi_price: float  # Price in cents
    kalshi_strike: float  # The price target from the market title
    kalshi_expiry: Optional[datetime]

    # Options hedge
    spread: SpreadPosition

    # Analysis results
    scenarios: dict  # BTC price -> combined P&L
    min_profit: float  # Worst case profit
    max_profit: float  # Best case profit
    expected_profit: float  # Probability-weighted profit

    # Position sizing
    kalshi_contracts: int
    options_contracts: int
    total_capital_required: float
    roi: float  # Return on invested capital

    # Metadata
    btc_current_price: float
    confidence: float
    valid_until: datetime


class OptionsHedgeArbitrageAnalyzer:
    """
    Analyzes arbitrage opportunities between Kalshi prediction markets
    and BTC futures options spreads.

    CALCULATION DETAILS:
    -------------------
    All calculations assume:
    - Kalshi prices in CENTS (1-99)
    - Kalshi payout is $1.00 per contract
    - Options premiums in USD
    - Contract multiplier configurable (BFF=0.01, MBT=0.1 BTC per contract)
    """

    def __init__(self, settings: Settings, tight_spread_threshold: float = 10.0):
        self.settings = settings
        self.min_roi = settings.roi_threshold
        # Get multiplier from settings (BFF=0.01, MBT=0.1)
        # For IBIT, this will be overridden dynamically
        self.btc_multiplier = settings.btc_multiplier
        # Spread threshold: if bid-ask spread > this % of mid, use mid pricing
        self.tight_spread_threshold = tight_spread_threshold
        logger.info(f"Default multiplier: {self.btc_multiplier} BTC/contract")
        logger.info(f"Tight spread threshold: {tight_spread_threshold}% (use mid if wider)")

    def set_ibit_multiplier(self, btc_ibit_ratio: float):
        """
        Set BTC multiplier for IBIT options based on current ratio.

        IBIT options: 100 shares per contract
        BTC equivalent = 100 / btc_ibit_ratio

        Example: If BTC=$100K, IBIT=$55, ratio=1818
        Then 1 IBIT contract = 100/$55 = ~0.055 BTC
        """
        self.btc_multiplier = 100.0 / btc_ibit_ratio
        logger.info(f"IBIT multiplier set to {self.btc_multiplier:.6f} BTC/contract (ratio: {btc_ibit_ratio:.1f})")

    def find_hedged_arbitrage(
        self,
        kalshi_markets: list[Market],
        options_chain: OptionsChain,
    ) -> list[HedgedArbitrageOpportunity]:
        """
        Find all hedged arbitrage opportunities.

        MAIN ENTRY POINT for finding opportunities.

        Args:
            kalshi_markets: List of Kalshi BTC prediction markets
            options_chain: MBT options chain with current prices

        Returns:
            List of profitable hedged arbitrage opportunities
        """
        opportunities = []
        btc_price = options_chain.underlying_price
        available_strikes = options_chain.get_available_strikes()

        logger.info(f"Analyzing {len(kalshi_markets)} Kalshi markets against "
                   f"{len(available_strikes)} option strikes")
        logger.info(f"Current BTC price: ${btc_price:,.2f}")

        for market in kalshi_markets:
            # Extract strike price from market title
            kalshi_strike = self._extract_strike_from_market(market)
            if kalshi_strike is None:
                continue

            # Only analyze markets where strike is reasonably close
            # (within 50% of current price)
            if kalshi_strike < btc_price * 0.5 or kalshi_strike > btc_price * 2:
                continue

            # Strategy: Buy NO on "will BTC hit $X" markets
            # This profits if BTC doesn't hit the target
            if market.no_ask is None or market.no_ask <= 0:
                continue

            # Find suitable hedge strikes
            # For NO position (betting BTC won't hit target):
            # - Buy a call below the target
            # - Sell a call at the target
            # This profits if BTC rises toward but doesn't exceed target

            opp = self._analyze_no_with_call_spread(
                market=market,
                kalshi_strike=kalshi_strike,
                options_chain=options_chain,
                btc_price=btc_price,
            )

            if opp and opp.min_profit > 0 and opp.roi >= self.min_roi:
                opportunities.append(opp)

        # Sort by ROI descending
        opportunities.sort(key=lambda x: x.roi, reverse=True)

        return opportunities

    def _analyze_no_with_call_spread(
        self,
        market: Market,
        kalshi_strike: float,
        options_chain: OptionsChain,
        btc_price: float,
    ) -> Optional[HedgedArbitrageOpportunity]:
        """
        Analyze buying Kalshi NO + bull call spread hedge.

        CALCULATION LOGIC:
        -----------------
        1. Buy NO on "will BTC hit $X?" at price P cents
           - If BTC < X at expiry: Win $1.00, profit = $1.00 - $P
           - If BTC >= X at expiry: Lose $P

        2. Buy bull call spread (buy lower strike call, sell higher strike call)
           - Net debit = premium_paid - premium_received
           - If BTC < lower_strike: Lose net_debit
           - If lower_strike < BTC < higher_strike: Partial profit
           - If BTC >= higher_strike: Max profit = (higher - lower) * 0.1 - net_debit

        3. Combined position must profit in ALL scenarios
        """
        no_price_cents = market.no_ask
        no_price_dollars = no_price_cents / 100.0

        # Calculate Kalshi profit/loss ratio
        kalshi_profit = 1.0 - no_price_dollars  # Profit if BTC < strike
        kalshi_loss = no_price_dollars  # Loss if BTC >= strike
        kalshi_ratio = kalshi_profit / kalshi_loss if kalshi_loss > 0 else float('inf')

        logger.debug(f"Kalshi {market.ticker}: NO@{no_price_cents}c, ratio={kalshi_ratio:.3f}")

        # Get Kalshi expiration date and convert to YYYYMMDD for comparison
        # Options must expire ON or AFTER the Kalshi market settles
        min_option_expiry = None
        if market.expiration_time:
            min_option_expiry = market.expiration_time.strftime("%Y%m%d")
            logger.debug(f"Kalshi expires: {min_option_expiry}, filtering options accordingly")

        # Check if we have any options with valid expiry
        available_expiries = options_chain.get_available_expiries()
        if min_option_expiry:
            valid_expiries = [e for e in available_expiries if e >= min_option_expiry]
            if not valid_expiries:
                logger.debug(
                    f"No options expire after Kalshi ({min_option_expiry}). "
                    f"Available: {available_expiries[:3]}..."
                )
                return None
            logger.debug(f"Valid option expiries: {valid_expiries}")

        # Find best call spread strikes (filtered by expiry)
        # Lower strike should be below the Kalshi target
        # Upper strike should be at or near the Kalshi target

        available_strikes = options_chain.get_available_strikes(min_expiry=min_option_expiry)

        # Filter to relevant strikes (around the Kalshi target)
        lower_candidates = [s for s in available_strikes if s < kalshi_strike]
        upper_candidates = [s for s in available_strikes if s >= kalshi_strike * 0.95]

        if not lower_candidates or not upper_candidates:
            return None

        best_opp = None
        best_roi = -float('inf')

        # Try different spread combinations
        for lower_strike in lower_candidates[-5:]:  # Last 5 (closest to target)
            for upper_strike in upper_candidates[:5]:  # First 5 (closest to target)
                if upper_strike <= lower_strike:
                    continue

                # Get option quotes (filtered by expiry if Kalshi has expiration)
                if min_option_expiry:
                    long_call = options_chain.get_call_with_min_expiry(lower_strike, min_option_expiry)
                    short_call = options_chain.get_call_with_min_expiry(upper_strike, min_option_expiry)
                else:
                    long_call = options_chain.get_call(lower_strike)
                    short_call = options_chain.get_call(upper_strike)

                if long_call is None or short_call is None:
                    continue
                if long_call.ask is None or short_call.bid is None:
                    continue

                # Verify both options have same expiry (for proper spread)
                if long_call.expiry != short_call.expiry:
                    logger.debug(f"Skipping mismatched expiries: {long_call.expiry} vs {short_call.expiry}")
                    continue

                # Determine pricing method based on bid-ask spread width
                # If spreads are tight, use conservative bid/ask pricing
                # If spreads are wide, use mid pricing (more realistic fill assumption)
                long_tight = long_call.is_tight_spread(self.tight_spread_threshold)
                short_tight = short_call.is_tight_spread(self.tight_spread_threshold)

                if long_tight and short_tight:
                    # Both spreads tight - use conservative bid/ask pricing
                    long_price = long_call.ask
                    short_price = short_call.bid
                    pricing_method = "bid/ask"
                else:
                    # Wide spread(s) - use mid pricing
                    long_price = long_call.mid if long_call.mid else long_call.ask
                    short_price = short_call.mid if short_call.mid else short_call.bid
                    pricing_method = "mid"

                long_spread_pct = long_call.spread_percent or 0
                short_spread_pct = short_call.spread_percent or 0
                logger.debug(
                    f"Pricing method: {pricing_method} | "
                    f"Long ${lower_strike}: spread={long_spread_pct:.1f}% | "
                    f"Short ${upper_strike}: spread={short_spread_pct:.1f}%"
                )

                # Calculate spread debit and max profit
                spread_debit = long_price - short_price
                spread_width = upper_strike - lower_strike
                spread_max_value = spread_width * self.btc_multiplier
                spread_max_profit = spread_max_value - spread_debit

                if spread_max_profit <= 0:
                    # Spread has no profit potential
                    continue

                # KEY CHECK: Arbitrage ratio condition
                # For arbitrage: kalshi_ratio >= spread_debit / spread_max_profit
                spread_ratio = spread_debit / spread_max_profit
                if kalshi_ratio < spread_ratio:
                    # Math doesn't work - skip this combination
                    logger.debug(
                        f"Skipping {lower_strike}/{upper_strike}: "
                        f"kalshi_ratio {kalshi_ratio:.3f} < spread_ratio {spread_ratio:.3f}"
                    )
                    continue

                logger.debug(
                    f"Valid spread {lower_strike}/{upper_strike}: "
                    f"debit=${spread_debit:.0f}, max_profit=${spread_max_profit:.0f}, "
                    f"ratio={spread_ratio:.3f}"
                )

                # Calculate spread position using determined prices
                spread = SpreadPosition(
                    long_strike=lower_strike,
                    short_strike=upper_strike,
                    long_premium=long_price,  # Price we pay (ask or mid)
                    short_premium=short_price,  # Price we receive (bid or mid)
                    expiry=long_call.expiry,
                    is_call_spread=True,
                    contracts=1,  # Will be sized later
                    pricing_method=pricing_method,
                )

                # Calculate combined P&L at various BTC prices
                metrics = self._calculate_arbitrage_metrics(
                    no_price_dollars=no_price_dollars,
                    kalshi_strike=kalshi_strike,
                    spread=spread,
                    btc_price=btc_price,
                )

                if metrics is None:
                    continue

                if metrics['roi'] > best_roi and metrics['min_profit'] > 0:
                    best_roi = metrics['roi']
                    best_opp = HedgedArbitrageOpportunity(
                        kalshi_ticker=market.ticker,
                        kalshi_title=market.title,
                        kalshi_side='no',
                        kalshi_price=no_price_cents,
                        kalshi_strike=kalshi_strike,
                        kalshi_expiry=market.expiration_time,
                        spread=spread,
                        scenarios=metrics['scenarios'],
                        min_profit=metrics['min_profit'],
                        max_profit=metrics['max_profit'],
                        expected_profit=metrics['expected_profit'],
                        kalshi_contracts=metrics['kalshi_contracts'],
                        options_contracts=metrics['options_contracts'],
                        total_capital_required=metrics['capital_required'],
                        roi=metrics['roi'],
                        btc_current_price=btc_price,
                        confidence=metrics['confidence'],
                        valid_until=datetime.utcnow() + timedelta(minutes=5),
                    )

        return best_opp

    def _calculate_arbitrage_metrics(
        self,
        no_price_dollars: float,
        kalshi_strike: float,
        spread: SpreadPosition,
        btc_price: float,
    ) -> Optional[dict]:
        """
        Calculate profit/loss metrics for the combined position.

        THIS IS WHERE THE CORE CALCULATIONS HAPPEN.

        CALCULATION DETAILS:
        -------------------
        For each BTC price scenario:
        1. Calculate Kalshi P&L:
           - If BTC < kalshi_strike: Win $1.00 - cost = $1.00 - no_price_dollars
           - If BTC >= kalshi_strike: Lose no_price_dollars

        2. Calculate spread P&L:
           - If BTC < long_strike: Both expire worthless, lose net_debit
           - If long_strike < BTC < short_strike:
             Long call value = (BTC - long_strike) * 0.1
             P&L = long_call_value - net_debit
           - If BTC >= short_strike:
             P&L = (short_strike - long_strike) * 0.1 - net_debit (max profit)

        3. Combined P&L = Kalshi P&L * kalshi_contracts + Spread P&L * spread_contracts

        Position Sizing:
        ---------------
        We need to size positions so that the spread max profit covers Kalshi max loss.

        Kalshi max loss = no_price_dollars per contract
        Spread max profit = (short_strike - long_strike) * 0.1 - net_debit per contract

        To neutralize: spread_contracts * spread_max_profit >= kalshi_contracts * kalshi_max_loss
        """
        # Calculate per-contract values
        kalshi_win = 1.00 - no_price_dollars  # Profit if BTC < strike
        kalshi_loss = no_price_dollars  # Loss if BTC >= strike

        spread_max_profit = spread.max_profit
        spread_max_loss = spread.max_loss  # = net_debit

        if spread_max_profit <= 0:
            # Spread doesn't have positive max profit (mispriced or inverted)
            return None

        # POSITION SIZING CALCULATION
        # To be fully hedged: spread_contracts * spread_max_profit = kalshi_loss
        # For 1 Kalshi contract, we need:
        # spread_contracts = kalshi_loss / spread_max_profit

        kalshi_contracts = 100  # Start with 100 Kalshi contracts ($100 notional)

        # Required spread contracts to fully hedge the Kalshi loss
        kalshi_total_loss = kalshi_contracts * kalshi_loss
        spread_contracts_needed = kalshi_total_loss / spread_max_profit

        # Round up to whole contracts
        import math
        options_contracts = math.ceil(spread_contracts_needed)

        if options_contracts == 0:
            options_contracts = 1

        # Update spread with actual contract count
        spread.contracts = options_contracts

        # Calculate scenarios
        # Key price points to analyze
        scenarios = {}

        test_prices = [
            btc_price * 0.8,  # 20% below current
            spread.long_strike * 0.95,  # Just below long strike
            spread.long_strike,  # At long strike
            (spread.long_strike + spread.short_strike) / 2,  # Mid spread
            spread.short_strike,  # At short strike (kalshi strike)
            kalshi_strike,  # Kalshi target
            kalshi_strike * 1.05,  # Just above Kalshi target
            btc_price * 1.2,  # 20% above current
        ]

        test_prices = sorted(set(test_prices))

        for price in test_prices:
            combined_pnl = self._calculate_combined_pnl(
                btc_price=price,
                kalshi_contracts=kalshi_contracts,
                kalshi_win=kalshi_win,
                kalshi_loss=kalshi_loss,
                kalshi_strike=kalshi_strike,
                spread=spread,
            )
            scenarios[price] = combined_pnl

        # Find min/max profit
        min_profit = min(scenarios.values())
        max_profit = max(scenarios.values())

        # Expected profit (simple average - could weight by probability)
        expected_profit = sum(scenarios.values()) / len(scenarios)

        # Total capital required
        kalshi_capital = kalshi_contracts * no_price_dollars
        spread_capital = spread.net_debit
        total_capital = kalshi_capital + spread_capital

        # ROI calculation
        if total_capital <= 0:
            return None

        roi = min_profit / total_capital

        # Confidence based on how robust the arbitrage is
        # Higher confidence if min_profit is closer to max_profit (more consistent)
        profit_range = max_profit - min_profit
        confidence = 0.9 if profit_range < max_profit * 0.5 else 0.7

        return {
            'scenarios': scenarios,
            'min_profit': min_profit,
            'max_profit': max_profit,
            'expected_profit': expected_profit,
            'kalshi_contracts': kalshi_contracts,
            'options_contracts': options_contracts,
            'capital_required': total_capital,
            'roi': roi,
            'confidence': confidence,
        }

    def _calculate_combined_pnl(
        self,
        btc_price: float,
        kalshi_contracts: int,
        kalshi_win: float,
        kalshi_loss: float,
        kalshi_strike: float,
        spread: SpreadPosition,
    ) -> float:
        """
        Calculate combined P&L for a specific BTC price.

        DETAILED P&L CALCULATION:
        ------------------------
        """
        # Kalshi P&L
        if btc_price < kalshi_strike:
            kalshi_pnl = kalshi_contracts * kalshi_win
        else:
            kalshi_pnl = -kalshi_contracts * kalshi_loss

        # Spread P&L
        spread_pnl = self.calculate_spread_payoff(spread, btc_price)

        return kalshi_pnl + spread_pnl

    def calculate_spread_payoff(
        self,
        spread: SpreadPosition,
        btc_price: float,
    ) -> float:
        """
        Calculate the payoff of a call/put spread at a given BTC price.

        SPREAD PAYOFF CALCULATION:
        -------------------------
        For a BULL CALL SPREAD (buy lower strike, sell higher strike):

        1. If BTC < long_strike:
           - Long call expires worthless: 0
           - Short call expires worthless: 0
           - P&L = -net_debit (lose what we paid)

        2. If long_strike <= BTC < short_strike:
           - Long call value = (BTC - long_strike) * multiplier
           - Short call expires worthless: 0
           - P&L = long_call_value - net_debit

        3. If BTC >= short_strike:
           - Long call value = (BTC - long_strike) * multiplier
           - Short call value = (BTC - short_strike) * multiplier (we owe this)
           - Net option value = (short_strike - long_strike) * multiplier
           - P&L = net_option_value - net_debit (MAX PROFIT)

        For MBT: multiplier = 0.1 (each contract is 0.1 BTC)
        """
        multiplier = self.btc_multiplier * spread.contracts

        if spread.is_call_spread:
            # Bull call spread
            if btc_price < spread.long_strike:
                # Both expire worthless
                option_value = 0
            elif btc_price < spread.short_strike:
                # Long call has value, short call worthless
                option_value = (btc_price - spread.long_strike) * multiplier
            else:
                # Both have value, but capped at spread width
                option_value = (spread.short_strike - spread.long_strike) * multiplier

            return option_value - spread.net_debit
        else:
            # Bear put spread (not used in current strategy but included for completeness)
            if btc_price > spread.long_strike:
                option_value = 0
            elif btc_price > spread.short_strike:
                option_value = (spread.long_strike - btc_price) * multiplier
            else:
                option_value = (spread.long_strike - spread.short_strike) * multiplier

            return option_value - spread.net_debit

    def _extract_strike_from_market(self, market: Market) -> Optional[float]:
        """
        Extract strike price from Kalshi market title or ticker.

        Supports multiple formats:
        1. Title-based: "When will Bitcoin hit $150,000?" -> 150000
        2. Ticker-based: KXBTC-26JAN2921-B97875 -> 97875 (B=below, T=top/above)
        3. Title: "Bitcoin above 100k" -> 100000
        """
        # Try ticker-based extraction first (KXBTC format)
        # Format: KXBTC-26JAN2921-B97875 or KXBTC-26JAN2921-T98499.99
        ticker_match = re.search(r'-([BT])(\d+(?:\.\d+)?)', market.ticker)
        if ticker_match:
            direction = ticker_match.group(1)  # B=below, T=top
            strike_str = ticker_match.group(2)
            try:
                strike = float(strike_str)
                # Store direction for later use
                market._strike_direction = direction
                return strike
            except ValueError:
                pass

        title = market.title.lower()

        # Pattern 1: "$X,XXX" or "$XXX,XXX" or "$X,XXX,XXX"
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

        # Pattern 3: Plain number after "hit", "above", "below", ">=", "<="
        match = re.search(r"(?:hit|above|below|>=|<=|>|<)\s*\$?(\d{4,})", title)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass

        return None

    def _get_market_direction(self, market: Market) -> Optional[str]:
        """
        Get the direction of a KXBTC market (below/above).

        Returns 'below' or 'above' based on ticker pattern.
        """
        ticker_match = re.search(r'-([BT])\d', market.ticker)
        if ticker_match:
            direction = ticker_match.group(1)
            return 'below' if direction == 'B' else 'above'

        title = market.title.lower()
        if 'below' in title:
            return 'below'
        if 'above' in title or 'hit' in title:
            return 'above'

        return None

    def convert_to_arbitrage_opportunity(
        self,
        hedged_opp: HedgedArbitrageOpportunity
    ) -> ArbitrageOpportunity:
        """Convert to standard ArbitrageOpportunity model for display/execution."""
        return ArbitrageOpportunity(
            id=self._generate_id(hedged_opp),
            arb_type=ArbitrageType.CORRELATED_MARKETS,
            markets=[hedged_opp.kalshi_ticker],
            expected_profit=hedged_opp.expected_profit * 100,  # Convert to cents
            max_profit=hedged_opp.max_profit * 100,
            risk=abs(hedged_opp.min_profit) * 100 if hedged_opp.min_profit < 0 else 0,
            legs=[
                {
                    "type": "kalshi",
                    "ticker": hedged_opp.kalshi_ticker,
                    "side": hedged_opp.kalshi_side,
                    "action": "buy",
                    "price": hedged_opp.kalshi_price,
                    "quantity": hedged_opp.kalshi_contracts,
                },
                {
                    "type": "mbt_option",
                    "action": "buy_call",
                    "strike": hedged_opp.spread.long_strike,
                    "premium": hedged_opp.spread.long_premium,
                    "quantity": hedged_opp.options_contracts,
                },
                {
                    "type": "mbt_option",
                    "action": "sell_call",
                    "strike": hedged_opp.spread.short_strike,
                    "premium": hedged_opp.spread.short_premium,
                    "quantity": hedged_opp.options_contracts,
                },
            ],
            confidence=hedged_opp.confidence,
            valid_until=hedged_opp.valid_until,
        )

    @staticmethod
    def _generate_id(opp: HedgedArbitrageOpportunity) -> str:
        """Generate unique ID for the opportunity."""
        components = [
            opp.kalshi_ticker,
            str(opp.spread.long_strike),
            str(opp.spread.short_strike),
            str(datetime.utcnow().timestamp()),
        ]
        combined = "_".join(components)
        return hashlib.md5(combined.encode()).hexdigest()[:12]
