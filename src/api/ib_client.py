"""Interactive Brokers client using ib_insync."""
import asyncio
import logging
import math
from datetime import datetime
from typing import Optional

# Fix for Python 3.14+ compatibility with ib_insync
# Must set event loop before importing ib_insync
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

from ib_insync import IB, Contract, Future, FuturesOption

from ..config import Settings
from ..models import IBContract, IBOrderBook, IBOrderBookLevel, IBTicker

logger = logging.getLogger(__name__)


class IBClient:
    """Async client for Interactive Brokers using ib_insync."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ib: Optional[IB] = None
        self._connected = False

    async def __aenter__(self) -> "IBClient":
        """Connect to IB TWS/Gateway on context entry."""
        self.ib = IB()

        # ib_insync's connectAsync for async connection
        await self.ib.connectAsync(
            host=self.settings.ib_host,
            port=self.settings.ib_port,
            clientId=self.settings.ib_client_id,
            timeout=self.settings.ib_timeout,
        )
        self._connected = True

        # Request delayed market data (type 3) - works without paid subscriptions
        # Type 1 = Live, Type 2 = Frozen, Type 3 = Delayed, Type 4 = Delayed Frozen
        self.ib.reqMarketDataType(3)

        logger.info(
            f"Connected to IB at {self.settings.ib_host}:{self.settings.ib_port} (delayed data mode)"
        )
        return self

    async def __aexit__(self, *args) -> None:
        """Disconnect from IB on context exit."""
        if self.ib and self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IB")

    @property
    def is_connected(self) -> bool:
        """Check if connected to IB."""
        return self.ib is not None and self.ib.isConnected()

    async def get_mbt_contract(self, expiry: Optional[str] = None) -> Contract:
        """
        Get MBT (Micro Bitcoin) futures contract.

        Args:
            expiry: Contract expiry in YYYYMM format (e.g., "202503" for March 2025).
                   If None, gets the front-month contract.

        Returns:
            Qualified IB Contract object
        """
        contract = Future(
            symbol=self.settings.btc_futures_symbol,
            exchange="CME",
            currency="USD",
        )

        if expiry:
            contract.lastTradeDateOrContractMonth = expiry

        # Qualify contract to get full details including conId
        qualified = await self.ib.qualifyContractsAsync(contract)

        if not qualified:
            raise ValueError(f"Could not qualify MBT contract with expiry {expiry}")

        return qualified[0]

    async def get_front_month_mbt(self) -> Contract:
        """Get the front-month MBT futures contract."""
        # Request contract details to find front month
        contract = Future(
            symbol=self.settings.btc_futures_symbol,
            exchange="CME",
            currency="USD",
        )

        details = await self.ib.reqContractDetailsAsync(contract)

        if not details:
            raise ValueError("No MBT contract details found")

        # Sort by expiry and get the nearest one
        sorted_details = sorted(
            details,
            key=lambda d: d.contract.lastTradeDateOrContractMonth or "",
        )

        front_month = sorted_details[0].contract
        logger.info(
            f"Front month MBT: {front_month.lastTradeDateOrContractMonth}"
        )

        return front_month

    async def get_ticker(self, contract: Contract, wait_seconds: float = 3.0) -> IBTicker:
        """
        Get current ticker/quote data for a contract.

        Args:
            contract: IB Contract object
            wait_seconds: Time to wait for data (delayed data needs more time)

        Returns:
            IBTicker with current bid/ask/last prices
        """
        # Request market data
        self.ib.reqMktData(contract, "", False, False)

        # Wait for data to arrive (delayed data takes longer)
        await asyncio.sleep(wait_seconds)

        ticker = self.ib.ticker(contract)

        ib_contract = IBContract(
            symbol=contract.symbol,
            sec_type=contract.secType,
            exchange=contract.exchange,
            currency=contract.currency,
            last_trade_date=contract.lastTradeDateOrContractMonth,
            multiplier=contract.multiplier,
            con_id=contract.conId,
        )

        def safe_value(val):
            """Convert -1 or nan to None."""
            if val is None or val == -1:
                return None
            try:
                if math.isnan(val):
                    return None
            except (TypeError, ValueError):
                pass
            return val

        return IBTicker(
            contract=ib_contract,
            bid=safe_value(ticker.bid),
            bid_size=safe_value(ticker.bidSize),
            ask=safe_value(ticker.ask),
            ask_size=safe_value(ticker.askSize),
            last=safe_value(ticker.last),
            last_size=safe_value(ticker.lastSize),
            volume=safe_value(ticker.volume),
            close=safe_value(ticker.close),
            timestamp=datetime.utcnow(),
        )

    async def get_orderbook(
        self, contract: Contract, depth: int = 10
    ) -> IBOrderBook:
        """
        Fetch order book (market depth) for a contract.

        Falls back to Level 1 bid/ask if Level 2 depth data is not available.

        Args:
            contract: IB Contract object
            depth: Number of price levels to request

        Returns:
            IBOrderBook with bid and ask levels
        """
        bids = []
        asks = []

        # Try to get Level 2 market depth first
        try:
            self.ib.reqMktDepth(contract, numRows=depth, isSmartDepth=False)
            await asyncio.sleep(2)

            ticker = self.ib.ticker(contract)

            if ticker.domBids:
                for dom in ticker.domBids:
                    if dom.price > 0:
                        bids.append(
                            IBOrderBookLevel(
                                price=dom.price,
                                size=dom.size,
                                market_maker=dom.marketMaker if hasattr(dom, "marketMaker") else None,
                            )
                        )

            if ticker.domAsks:
                for dom in ticker.domAsks:
                    if dom.price > 0:
                        asks.append(
                            IBOrderBookLevel(
                                price=dom.price,
                                size=dom.size,
                                market_maker=dom.marketMaker if hasattr(dom, "marketMaker") else None,
                            )
                        )

            self.ib.cancelMktDepth(contract, isSmartDepth=False)

        except Exception as e:
            logger.debug(f"Level 2 depth not available: {e}")

        # Fall back to Level 1 (top of book) if no depth data
        if not bids or not asks:
            logger.info("Using Level 1 data (top of book only)")
            self.ib.reqMktData(contract, "", False, False)
            await asyncio.sleep(3)  # Delayed data needs more time

            ticker = self.ib.ticker(contract)

            def is_valid(val):
                if val is None or val == -1:
                    return False
                try:
                    return not math.isnan(val)
                except (TypeError, ValueError):
                    return True

            if is_valid(ticker.bid):
                bids = [IBOrderBookLevel(
                    price=ticker.bid,
                    size=int(ticker.bidSize) if is_valid(ticker.bidSize) else 1,
                )]
            if is_valid(ticker.ask):
                asks = [IBOrderBookLevel(
                    price=ticker.ask,
                    size=int(ticker.askSize) if is_valid(ticker.askSize) else 1,
                )]

        ib_contract = IBContract(
            symbol=contract.symbol,
            sec_type=contract.secType,
            exchange=contract.exchange,
            currency=contract.currency,
            last_trade_date=contract.lastTradeDateOrContractMonth,
            multiplier=contract.multiplier,
            con_id=contract.conId,
        )

        return IBOrderBook(
            contract=ib_contract,
            bids=bids,
            asks=asks,
            timestamp=datetime.utcnow(),
        )

    async def get_mbt_options_chain(
        self, underlying_expiry: str
    ) -> list[Contract]:
        """
        Get options chain for MBT futures.

        Args:
            underlying_expiry: Underlying futures expiry in YYYYMM format

        Returns:
            List of FuturesOption contracts
        """
        underlying = await self.get_mbt_contract(underlying_expiry)

        # Request option parameters
        chains = await self.ib.reqSecDefOptParamsAsync(
            underlyingSymbol=underlying.symbol,
            futFopExchange="CME",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )

        if not chains:
            logger.warning(f"No options chain found for MBT {underlying_expiry}")
            return []

        # Build option contracts from chain parameters
        options = []
        for chain in chains:
            for expiry in chain.expirations[:3]:  # Limit to nearest 3 expiries
                for strike in chain.strikes:
                    for right in ["C", "P"]:
                        opt = FuturesOption(
                            symbol=underlying.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike,
                            right=right,
                            exchange=chain.exchange,
                            currency="USD",
                        )
                        options.append(opt)

        return options

    async def qualify_contracts(self, contracts: list[Contract]) -> list[Contract]:
        """Qualify multiple contracts to get full details."""
        qualified = []
        for contract in contracts:
            try:
                result = await self.ib.qualifyContractsAsync(contract)
                if result:
                    qualified.extend(result)
            except Exception as e:
                logger.debug(f"Could not qualify contract: {e}")
        return qualified

    async def get_mbt_options_with_prices(
        self,
        underlying_expiry: Optional[str] = None,
        strike_range: tuple[float, float] = None,
        option_expiries: int = 2,
        wait_seconds: float = 3.0,
    ) -> dict:
        """
        Get MBT options chain with live prices.

        Args:
            underlying_expiry: Underlying futures expiry in YYYYMM format.
                             If None, uses front month.
            strike_range: (min_strike, max_strike) to filter strikes.
                         If None, gets strikes within 30% of current price.
            option_expiries: Number of option expiries to fetch (nearest N)
            wait_seconds: Time to wait for market data

        Returns:
            Dictionary with:
            - 'underlying_price': Current futures price
            - 'underlying_expiry': Futures expiry
            - 'calls': List of call option dicts with prices
            - 'puts': List of put option dicts with prices
            - 'timestamp': When data was fetched
        """
        # Get underlying futures contract and price
        if underlying_expiry:
            underlying = await self.get_mbt_contract(underlying_expiry)
        else:
            underlying = await self.get_front_month_mbt()

        underlying_ticker = await self.get_ticker(underlying, wait_seconds)
        underlying_price = underlying_ticker.last or underlying_ticker.bid or underlying_ticker.ask

        if underlying_price is None:
            raise ValueError("Could not get underlying futures price")

        logger.info(f"MBT underlying price: ${underlying_price:,.2f}")

        # Determine strike range if not provided
        if strike_range is None:
            min_strike = underlying_price * 0.7  # 30% below
            max_strike = underlying_price * 1.5  # 50% above
        else:
            min_strike, max_strike = strike_range

        # Get options chain parameters
        chains = await self.ib.reqSecDefOptParamsAsync(
            underlyingSymbol=underlying.symbol,
            futFopExchange="CME",
            underlyingSecType=underlying.secType,
            underlyingConId=underlying.conId,
        )

        if not chains:
            logger.warning(f"No options chain found for MBT {underlying.lastTradeDateOrContractMonth}")
            return {
                'underlying_price': underlying_price,
                'underlying_expiry': underlying.lastTradeDateOrContractMonth,
                'calls': [],
                'puts': [],
                'timestamp': datetime.utcnow(),
            }

        # Build option contracts within strike range
        calls = []
        puts = []

        for chain in chains:
            # Get nearest N expiries
            expiries = sorted(chain.expirations)[:option_expiries]

            # Filter strikes within range
            valid_strikes = [s for s in chain.strikes if min_strike <= s <= max_strike]

            logger.info(f"Processing {len(valid_strikes)} strikes across {len(expiries)} expiries "
                       f"on {chain.exchange}")

            for expiry in expiries:
                for strike in valid_strikes:
                    # Create call option
                    call = FuturesOption(
                        symbol=underlying.symbol,
                        lastTradeDateOrContractMonth=expiry,
                        strike=strike,
                        right="C",
                        exchange=chain.exchange,
                        currency="USD",
                    )

                    # Create put option
                    put = FuturesOption(
                        symbol=underlying.symbol,
                        lastTradeDateOrContractMonth=expiry,
                        strike=strike,
                        right="P",
                        exchange=chain.exchange,
                        currency="USD",
                    )

                    calls.append((call, expiry, strike))
                    puts.append((put, expiry, strike))

        # Qualify and get prices for options (in batches to avoid overwhelming IB)
        call_data = await self._fetch_option_prices(calls, wait_seconds)
        put_data = await self._fetch_option_prices(puts, wait_seconds)

        return {
            'underlying_price': underlying_price,
            'underlying_expiry': underlying.lastTradeDateOrContractMonth,
            'calls': call_data,
            'puts': put_data,
            'timestamp': datetime.utcnow(),
        }

    async def _fetch_option_prices(
        self,
        options: list[tuple],
        wait_seconds: float,
        batch_size: int = 50,
    ) -> list[dict]:
        """
        Fetch prices for a list of options in batches.

        Args:
            options: List of (contract, expiry, strike) tuples
            wait_seconds: Time to wait for data
            batch_size: How many options to request at once

        Returns:
            List of option dicts with price data
        """
        results = []

        for i in range(0, len(options), batch_size):
            batch = options[i:i + batch_size]
            contracts = [opt[0] for opt in batch]

            # Qualify contracts
            qualified = []
            for contract in contracts:
                try:
                    qual = await self.ib.qualifyContractsAsync(contract)
                    if qual:
                        qualified.append((qual[0], contract.strike, contract.lastTradeDateOrContractMonth))
                except Exception as e:
                    logger.debug(f"Could not qualify option {contract.strike} {contract.right}: {e}")

            if not qualified:
                continue

            # Request market data for all qualified contracts
            for qual_contract, strike, expiry in qualified:
                self.ib.reqMktData(qual_contract, "", False, False)

            # Wait for data
            await asyncio.sleep(wait_seconds)

            # Collect ticker data
            for qual_contract, strike, expiry in qualified:
                ticker = self.ib.ticker(qual_contract)

                def safe_val(v):
                    if v is None or v == -1:
                        return None
                    try:
                        if math.isnan(v):
                            return None
                    except (TypeError, ValueError):
                        pass
                    return v

                bid = safe_val(ticker.bid)
                ask = safe_val(ticker.ask)
                last = safe_val(ticker.last)

                # Only include options with some price data
                if bid is not None or ask is not None or last is not None:
                    results.append({
                        'strike': strike,
                        'expiry': expiry,
                        'right': qual_contract.right,
                        'bid': bid,
                        'ask': ask,
                        'last': last,
                        'volume': safe_val(ticker.volume) or 0,
                        'open_interest': 0,  # Would need separate request
                    })

            # Cancel market data to free up slots
            for qual_contract, _, _ in qualified:
                self.ib.cancelMktData(qual_contract)

            logger.info(f"Fetched prices for {len(qualified)} options in batch")

        return results

    async def get_mbt_futures_contracts(self, num_expiries: int = 4) -> list[tuple]:
        """
        Get multiple MBT futures contracts with their prices.

        Args:
            num_expiries: Number of futures expiries to fetch

        Returns:
            List of (contract, ticker) tuples sorted by expiry
        """
        contract = Future(
            symbol=self.settings.btc_futures_symbol,
            exchange="CME",
            currency="USD",
        )

        details = await self.ib.reqContractDetailsAsync(contract)

        if not details:
            raise ValueError("No MBT contract details found")

        # Sort by expiry
        sorted_details = sorted(
            details,
            key=lambda d: d.contract.lastTradeDateOrContractMonth or "",
        )[:num_expiries]

        results = []
        for detail in sorted_details:
            ticker = await self.get_ticker(detail.contract, wait_seconds=2.0)
            results.append((detail.contract, ticker))

        return results

    async def get_targeted_options(
        self,
        target_strike: float,
        min_option_expiry: str,
        num_strikes: int = 20,
        strike_interval: float = 2000,
        wait_seconds: float = 2.0,
    ) -> dict:
        """
        Fetch options for specific strikes around a target price.

        This is optimized for arbitrage scanning - only fetches what we need.

        Args:
            target_strike: The Kalshi strike price (e.g., 150000 for $150K)
            min_option_expiry: Minimum option expiry in YYYYMMDD format
            num_strikes: Number of strikes to fetch (below and at target)
            strike_interval: Spacing between strikes (e.g., 2000 for $2K)
            wait_seconds: Time to wait for market data

        Returns:
            Dictionary with options data for the targeted strikes
        """
        # Get front month futures for price reference
        underlying = await self.get_front_month_mbt()
        underlying_ticker = await self.get_ticker(underlying, wait_seconds=2.0)
        underlying_price = underlying_ticker.last or underlying_ticker.bid or underlying_ticker.ask

        if underlying_price is None:
            raise ValueError("Could not get underlying futures price")

        logger.info(f"BTC price: ${underlying_price:,.0f}, Target strike: ${target_strike:,.0f}")

        # Find the BFF futures contract that has options expiring AFTER min_option_expiry
        contract = Future(
            symbol=self.settings.btc_futures_symbol,
            exchange="CME",
            currency="USD",
        )

        details = await self.ib.reqContractDetailsAsync(contract)
        if not details:
            raise ValueError("No futures contracts found")

        # Sort by expiry
        sorted_details = sorted(
            details,
            key=lambda d: d.contract.lastTradeDateOrContractMonth or "",
        )

        # Find a futures contract that has options expiring >= min_option_expiry
        target_futures = None
        target_option_expiry = None

        for detail in sorted_details[:6]:  # Check up to 6 futures
            fut = detail.contract
            fut_expiry = fut.lastTradeDateOrContractMonth

            # Get options chain params for this futures
            try:
                chains = await self.ib.reqSecDefOptParamsAsync(
                    underlyingSymbol=fut.symbol,
                    futFopExchange="CME",
                    underlyingSecType=fut.secType,
                    underlyingConId=fut.conId,
                )
            except Exception:
                continue

            if not chains:
                continue

            for chain in chains:
                # Find the nearest option expiry >= min_option_expiry
                valid_expiries = sorted([e for e in chain.expirations if e >= min_option_expiry])
                if valid_expiries:
                    target_futures = fut
                    target_option_expiry = valid_expiries[0]  # Nearest valid expiry
                    logger.info(f"Using futures {fut_expiry} with option expiry {target_option_expiry}")
                    break

            if target_futures:
                break

        if not target_futures or not target_option_expiry:
            logger.warning(f"No options found expiring >= {min_option_expiry}")
            return {
                'underlying_price': underlying_price,
                'calls': [],
                'puts': [],
                'option_expiry': None,
                'timestamp': datetime.utcnow(),
            }

        # Generate strikes: target and num_strikes below it
        # Round target to nearest 500 (BFF options have $500 strike intervals)
        rounded_target = round(target_strike / 500) * 500

        # E.g., if target=$150K and interval=$2K, get $150K, $148K, $146K, etc.
        strikes = []
        for i in range(num_strikes):
            strike = rounded_target - (i * strike_interval)
            if strike > 0:
                strikes.append(float(strike))

        logger.info(f"Fetching {len(strikes)} strikes from ${min(strikes):,.0f} to ${max(strikes):,.0f}")

        # Build option contracts
        calls_to_fetch = []
        for strike in strikes:
            call = FuturesOption(
                symbol=self.settings.btc_futures_symbol,
                lastTradeDateOrContractMonth=target_option_expiry,
                strike=strike,
                right="C",
                exchange="CME",
                currency="USD",
            )
            calls_to_fetch.append((call, target_option_expiry, strike))

        # Fetch prices
        call_data = await self._fetch_option_prices(calls_to_fetch, wait_seconds, batch_size=25)

        return {
            'underlying_price': underlying_price,
            'calls': call_data,
            'puts': [],  # We only need calls for bull call spread
            'option_expiry': target_option_expiry,
            'timestamp': datetime.utcnow(),
        }

    async def get_multi_expiry_options_with_prices(
        self,
        min_option_expiry: str,
        strike_range: tuple[float, float] = None,
        max_futures_contracts: int = 4,
        wait_seconds: float = 3.0,
    ) -> dict:
        """
        Get options chain from multiple futures contracts, filtered by minimum expiry.

        This solves the problem where front-month futures options expire before
        Kalshi markets. We fetch options from multiple underlying futures to find
        options that expire AFTER the specified minimum date.

        Args:
            min_option_expiry: Minimum option expiry in YYYYMMDD format (e.g., "20260206")
            strike_range: (min_strike, max_strike) to filter strikes.
                         If None, gets strikes within 30% of current price.
            max_futures_contracts: Maximum number of futures contracts to search
            wait_seconds: Time to wait for market data

        Returns:
            Dictionary with:
            - 'underlying_price': Current futures price (from front month)
            - 'calls': List of call option dicts with prices (all expiries >= min_option_expiry)
            - 'puts': List of put option dicts with prices (all expiries >= min_option_expiry)
            - 'option_expiries': Set of available option expiries
            - 'timestamp': When data was fetched
        """
        # Get all available futures contracts
        contract = Future(
            symbol=self.settings.btc_futures_symbol,
            exchange="CME",
            currency="USD",
        )

        details = await self.ib.reqContractDetailsAsync(contract)

        if not details:
            raise ValueError("No futures contract details found")

        # Sort by expiry and take up to max_futures_contracts
        sorted_details = sorted(
            details,
            key=lambda d: d.contract.lastTradeDateOrContractMonth or "",
        )[:max_futures_contracts]

        logger.info(f"Found {len(sorted_details)} futures contracts to search for options")
        for d in sorted_details:
            logger.info(f"  Futures: {d.contract.symbol} {d.contract.lastTradeDateOrContractMonth}")

        # Get price from front month
        front_month = sorted_details[0].contract
        front_ticker = await self.get_ticker(front_month, wait_seconds)
        underlying_price = front_ticker.last or front_ticker.bid or front_ticker.ask

        if underlying_price is None:
            raise ValueError("Could not get underlying futures price")

        logger.info(f"Front month price: ${underlying_price:,.2f}")

        # Determine strike range if not provided
        if strike_range is None:
            min_strike = underlying_price * 0.7  # 30% below
            max_strike = underlying_price * 1.5  # 50% above
        else:
            min_strike, max_strike = strike_range

        all_calls = []
        all_puts = []
        all_expiries = set()

        # Search for options on each futures contract
        for detail in sorted_details:
            underlying = detail.contract
            underlying_expiry = underlying.lastTradeDateOrContractMonth

            logger.info(f"Searching options for futures expiry {underlying_expiry}...")

            # Get options chain parameters for this futures contract
            try:
                chains = await self.ib.reqSecDefOptParamsAsync(
                    underlyingSymbol=underlying.symbol,
                    futFopExchange="CME",
                    underlyingSecType=underlying.secType,
                    underlyingConId=underlying.conId,
                )
            except Exception as e:
                logger.warning(f"Could not get options chain for {underlying_expiry}: {e}")
                continue

            if not chains:
                logger.warning(f"No options chain found for futures {underlying_expiry}")
                continue

            for chain in chains:
                # Filter to expiries >= min_option_expiry
                valid_expiries = [exp for exp in chain.expirations if exp >= min_option_expiry]

                if not valid_expiries:
                    logger.debug(f"No valid expiries for {underlying_expiry} (need >= {min_option_expiry})")
                    continue

                # Filter strikes within range
                valid_strikes = [s for s in chain.strikes if min_strike <= s <= max_strike]

                logger.info(f"  Found {len(valid_expiries)} valid expiries, {len(valid_strikes)} strikes")

                for expiry in valid_expiries:
                    all_expiries.add(expiry)
                    for strike in valid_strikes:
                        # Create call option
                        call = FuturesOption(
                            symbol=underlying.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike,
                            right="C",
                            exchange=chain.exchange,
                            currency="USD",
                        )

                        # Create put option
                        put = FuturesOption(
                            symbol=underlying.symbol,
                            lastTradeDateOrContractMonth=expiry,
                            strike=strike,
                            right="P",
                            exchange=chain.exchange,
                            currency="USD",
                        )

                        all_calls.append((call, expiry, strike))
                        all_puts.append((put, expiry, strike))

        logger.info(f"Total options to fetch: {len(all_calls)} calls, {len(all_puts)} puts")
        logger.info(f"Available expiries >= {min_option_expiry}: {sorted(all_expiries)}")

        if not all_calls:
            return {
                'underlying_price': underlying_price,
                'calls': [],
                'puts': [],
                'option_expiries': all_expiries,
                'timestamp': datetime.utcnow(),
            }

        # Fetch prices
        call_data = await self._fetch_option_prices(all_calls, wait_seconds)
        put_data = await self._fetch_option_prices(all_puts, wait_seconds)

        return {
            'underlying_price': underlying_price,
            'calls': call_data,
            'puts': put_data,
            'option_expiries': all_expiries,
            'timestamp': datetime.utcnow(),
        }
