"""Cross-market arbitrage scanner for Kalshi vs IB futures."""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table

from .analysis.cross_market import CrossMarketArbitrageAnalyzer
from .api.ib_client import IBClient
from .api.client import KalshiClient
from .config import Settings, get_settings
from .data.orderbook import ib_orderbook_to_df, kalshi_orderbook_to_df
from .models import ArbitrageOpportunity, Market

logger = logging.getLogger(__name__)
console = Console()


def display_opportunities(opportunities: list[ArbitrageOpportunity]) -> None:
    """Display arbitrage opportunities in a formatted table."""
    if not opportunities:
        console.print("[yellow]No arbitrage opportunities found[/yellow]")
        return

    table = Table(title="Cross-Market Arbitrage Opportunities")
    table.add_column("ID", style="dim")
    table.add_column("Type", style="cyan")
    table.add_column("Market")
    table.add_column("Profit", justify="right", style="green")
    table.add_column("Risk", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Legs")

    for opp in opportunities:
        legs_str = ", ".join(
            f"{leg.get('side', leg.get('action'))}@{leg.get('price')}"
            for leg in opp.legs
        )

        table.add_row(
            opp.id[:8],
            opp.arb_type.value,
            opp.markets[0] if opp.markets else "N/A",
            f"{opp.expected_profit:.1f}c",
            f"{opp.risk:.0f}",
            f"{opp.confidence:.0%}",
            legs_str[:40],
        )

    console.print(table)


async def run_cross_market_scan(
    continuous: bool = False,
    settings: Optional[Settings] = None,
) -> list[ArbitrageOpportunity]:
    """
    Main entry point for cross-market arbitrage scanning.

    Args:
        continuous: If True, run continuously with scan_interval_seconds delay
        settings: Optional settings override

    Returns:
        List of detected arbitrage opportunities (from last scan if continuous)
    """
    if settings is None:
        settings = get_settings()

    analyzer = CrossMarketArbitrageAnalyzer(settings)
    all_opportunities = []

    console.print("[bold blue]Starting Cross-Market Arbitrage Scanner[/bold blue]")
    console.print(f"ROI Threshold: {settings.roi_threshold * 100:.1f}%")
    console.print(f"IB Connection: {settings.ib_host}:{settings.ib_port}")

    try:
        async with KalshiClient(settings) as kalshi, IBClient(settings) as ib:
            console.print("[green]Connected to Kalshi and IB[/green]")

            # Get MBT futures contract
            try:
                mbt_contract = await ib.get_front_month_mbt()
                console.print(
                    f"[green]MBT Contract: {mbt_contract.symbol} "
                    f"{mbt_contract.lastTradeDateOrContractMonth}[/green]"
                )
            except Exception as e:
                console.print(f"[red]Failed to get MBT contract: {e}[/red]")
                raise

            scan_count = 0
            while True:
                scan_count += 1
                scan_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                console.print(f"\n[bold]Scan #{scan_count} at {scan_time}[/bold]")

                # Fetch Kalshi Bitcoin markets
                try:
                    kalshi_markets = await kalshi.get_btc_markets()
                    console.print(f"Found {len(kalshi_markets)} Kalshi BTC markets")
                except Exception as e:
                    console.print(f"[red]Failed to fetch Kalshi markets: {e}[/red]")
                    kalshi_markets = []

                if not kalshi_markets:
                    console.print("[yellow]No Kalshi BTC markets found[/yellow]")
                    if not continuous:
                        break
                    await asyncio.sleep(settings.scan_interval_seconds)
                    continue

                # Fetch Kalshi order books for tradeable markets (mid-range prices, not settled)
                kalshi_orderbooks = {}
                # Get markets with YES bid between 5-95 cents (not already settled)
                tradeable_markets = [
                    m for m in kalshi_markets
                    if m.yes_bid and 5 < m.yes_bid < 95
                ][:20]  # Limit to 20
                console.print(f"Fetching orderbooks for {len(tradeable_markets)} tradeable markets...")

                for market in tradeable_markets:
                    try:
                        ob = await kalshi.get_orderbook(market.ticker, depth=settings.orderbook_depth)
                        kalshi_orderbooks[market.ticker] = ob

                        # Update market prices from orderbook if available
                        yes_bids = ob.get("orderbook", ob).get("yes", [])
                        yes_asks = ob.get("orderbook", ob).get("asks", []) or []
                        if yes_bids:
                            # Orderbook format: [[price, size], ...]
                            best_yes_bid = yes_bids[0][0] if yes_bids else None
                        if yes_asks:
                            best_yes_ask = yes_asks[0][0] if yes_asks else None

                    except Exception as e:
                        logger.debug(f"Failed to fetch orderbook for {market.ticker}: {e}")

                console.print(f"Kalshi Orderbooks: {len(kalshi_orderbooks)} fetched")

                # Display Kalshi orderbook stats
                if kalshi_orderbooks:
                    with_depth = 0
                    total_yes_levels = 0
                    total_no_levels = 0

                    for ticker, ob in kalshi_orderbooks.items():
                        ob_data = ob.get("orderbook", ob) or {}
                        yes_levels = ob_data.get("yes") or []
                        no_levels = ob_data.get("no") or []
                        if yes_levels or no_levels:
                            with_depth += 1
                            total_yes_levels += len(yes_levels)
                            total_no_levels += len(no_levels)

                    console.print(f"[dim]Kalshi depth: {with_depth}/{len(kalshi_orderbooks)} books have liquidity[/dim]")
                    console.print(f"[dim]  Total: {total_yes_levels} YES levels, {total_no_levels} NO levels[/dim]")

                    # Show first orderbook with actual data
                    for ticker, ob in kalshi_orderbooks.items():
                        ob_data = ob.get("orderbook", ob) or {}
                        yes_levels = ob_data.get("yes") or []
                        no_levels = ob_data.get("no") or []
                        if yes_levels or no_levels:
                            console.print(f"[dim]Sample: {ticker}[/dim]")
                            if yes_levels:
                                console.print(f"[dim]  YES: {yes_levels[0][0]}c x {yes_levels[0][1]}[/dim]")
                            if no_levels:
                                console.print(f"[dim]  NO: {no_levels[0][0]}c x {no_levels[0][1]}[/dim]")
                            break

                # Fetch IB order book
                try:
                    ib_orderbook = await ib.get_orderbook(
                        mbt_contract, depth=settings.orderbook_depth
                    )
                    ib_df = ib_orderbook_to_df(ib_orderbook)
                    console.print(
                        f"IB Orderbook: {len(ib_orderbook.bids)} bids, "
                        f"{len(ib_orderbook.asks)} asks"
                    )

                    # Show current BTC price
                    if ib_orderbook.bids and ib_orderbook.asks:
                        mid = (ib_orderbook.bids[0].price + ib_orderbook.asks[0].price) / 2
                        console.print(f"[cyan]MBT Price: ${mid:,.2f}[/cyan]")

                except Exception as e:
                    console.print(f"[red]Failed to fetch IB orderbook: {e}[/red]")
                    if not continuous:
                        break
                    await asyncio.sleep(settings.scan_interval_seconds)
                    continue

                # Find arbitrage opportunities
                opportunities = analyzer.find_btc_arbitrage(
                    kalshi_markets=kalshi_markets,
                    ib_orderbook=ib_df,
                )

                all_opportunities = opportunities
                display_opportunities(opportunities)

                if opportunities:
                    logger.info(f"Found {len(opportunities)} cross-market opportunities")

                if not continuous:
                    break

                console.print(
                    f"[dim]Next scan in {settings.scan_interval_seconds} seconds...[/dim]"
                )
                await asyncio.sleep(settings.scan_interval_seconds)

    except Exception as e:
        console.print(f"[red]Scanner error: {e}[/red]")
        logger.exception("Scanner error")
        raise

    return all_opportunities


async def test_ib_connection(settings: Optional[Settings] = None) -> bool:
    """Test IB Gateway connection."""
    if settings is None:
        settings = get_settings()

    console.print("[bold]Testing IB Connection[/bold]")
    console.print(f"Host: {settings.ib_host}:{settings.ib_port}")

    try:
        async with IBClient(settings) as ib:
            console.print(f"[green]Connected: {ib.is_connected}[/green]")

            # Get MBT contract
            contract = await ib.get_front_month_mbt()
            console.print(f"MBT Contract: {contract.symbol} {contract.lastTradeDateOrContractMonth}")

            # Get ticker
            ticker = await ib.get_ticker(contract)
            console.print(f"Bid: ${ticker.bid:,.2f}" if ticker.bid else "Bid: N/A")
            console.print(f"Ask: ${ticker.ask:,.2f}" if ticker.ask else "Ask: N/A")
            console.print(f"Last: ${ticker.last:,.2f}" if ticker.last else "Last: N/A")

            return True
    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")
        return False
