"""
Options-hedged arbitrage scanner for Kalshi vs MBT options.

This scanner finds arbitrage opportunities where you can:
1. Buy NO on Kalshi "When will BTC hit $X?" markets
2. Hedge with MBT call spreads

Example:
- Kalshi: "Will BTC hit $150K by June?" - NO @ 90 cents
- Hedge: Buy $130K call, Sell $150K call
- Result: Profit regardless of BTC price direction

Run with: python run.py options-scan
         python run.py options-scan --demo  # Run with synthetic data
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .analysis.options_hedge_arb import (
    OptionsHedgeArbitrageAnalyzer,
    OptionsChain,
    OptionQuote,
    HedgedArbitrageOpportunity,
)
from .api.ib_client import IBClient
from .api.client import KalshiClient
from .config import Settings, get_settings

logger = logging.getLogger(__name__)
console = Console()


def display_hedged_opportunities(opportunities: list[HedgedArbitrageOpportunity]) -> None:
    """Display hedged arbitrage opportunities in formatted tables."""
    if not opportunities:
        console.print("[yellow]No hedged arbitrage opportunities found[/yellow]")
        return

    console.print(f"\n[bold green]Found {len(opportunities)} Hedged Arbitrage Opportunities[/bold green]\n")

    for i, opp in enumerate(opportunities, 1):
        # Create detailed panel for each opportunity
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Label", style="dim")
        table.add_column("Value")

        # Kalshi position
        table.add_row("Kalshi Market", opp.kalshi_ticker)
        table.add_row("", f"[dim]{opp.kalshi_title}[/dim]")
        table.add_row("Kalshi Position", f"Buy {opp.kalshi_side.upper()} @ {opp.kalshi_price}c")
        table.add_row("Kalshi Contracts", str(opp.kalshi_contracts))
        table.add_row("Kalshi Target", f"${opp.kalshi_strike:,.0f}")

        # Options hedge
        table.add_row("", "")
        table.add_row("[cyan]Options Hedge[/cyan]", "")
        table.add_row(
            "Buy Call",
            f"${opp.spread.long_strike:,.0f} @ ${opp.spread.long_premium:,.2f}"
        )
        table.add_row(
            "Sell Call",
            f"${opp.spread.short_strike:,.0f} @ ${opp.spread.short_premium:,.2f}"
        )
        table.add_row("Options Contracts", str(opp.options_contracts))
        table.add_row("Net Debit", f"${opp.spread.net_debit:,.2f}")

        # P&L Analysis
        table.add_row("", "")
        table.add_row("[green]P&L Analysis[/green]", "")
        table.add_row("Min Profit", f"[green]${opp.min_profit:,.2f}[/green]" if opp.min_profit > 0 else f"[red]${opp.min_profit:,.2f}[/red]")
        table.add_row("Max Profit", f"[green]${opp.max_profit:,.2f}[/green]")
        table.add_row("Expected Profit", f"${opp.expected_profit:,.2f}")

        # Capital & ROI
        table.add_row("", "")
        table.add_row("Total Capital", f"${opp.total_capital_required:,.2f}")
        table.add_row("[bold]ROI[/bold]", f"[bold green]{opp.roi * 100:.2f}%[/bold green]")
        table.add_row("Confidence", f"{opp.confidence * 100:.0f}%")

        # Current BTC price context
        table.add_row("", "")
        table.add_row("Current BTC", f"${opp.btc_current_price:,.2f}")

        panel = Panel(
            table,
            title=f"[bold]Opportunity #{i}[/bold]",
            border_style="green" if opp.min_profit > 0 else "yellow",
        )
        console.print(panel)

        # Show scenario analysis
        if opp.scenarios:
            scenario_table = Table(title="Scenario Analysis (P&L at different BTC prices)")
            scenario_table.add_column("BTC Price", justify="right")
            scenario_table.add_column("Combined P&L", justify="right")
            scenario_table.add_column("Status")

            for price, pnl in sorted(opp.scenarios.items()):
                price_str = f"${price:,.0f}"
                pnl_str = f"${pnl:,.2f}"
                status = "[green]PROFIT[/green]" if pnl > 0 else "[red]LOSS[/red]"
                scenario_table.add_row(price_str, pnl_str, status)

            console.print(scenario_table)
            console.print("")


def display_options_chain_summary(chain_data: dict) -> None:
    """Display summary of the options chain data."""
    console.print("\n[bold]Options Chain Summary[/bold]")
    console.print(f"Underlying Price: ${chain_data['underlying_price']:,.2f}")
    if 'underlying_expiry' in chain_data:
        console.print(f"Underlying Expiry: {chain_data['underlying_expiry']}")
    console.print(f"Calls with prices: {len(chain_data['calls'])}")
    console.print(f"Puts with prices: {len(chain_data['puts'])}")

    if chain_data['calls']:
        # Show sample of available strikes
        strikes = sorted(set(c['strike'] for c in chain_data['calls']))[:10]
        console.print(f"Sample strikes: {', '.join(f'${s:,.0f}' for s in strikes)}")

        # Show a few sample quotes
        console.print("\n[dim]Sample call quotes:[/dim]")
        for call in chain_data['calls'][:5]:
            bid = f"${call['bid']:,.2f}" if call['bid'] else "N/A"
            ask = f"${call['ask']:,.2f}" if call['ask'] else "N/A"
            console.print(f"  ${call['strike']:,.0f} {call['expiry']}: bid={bid}, ask={ask}")


def convert_chain_data_to_options_chain(chain_data: dict) -> OptionsChain:
    """Convert IB chain data dict to OptionsChain object."""
    calls = []
    puts = []

    for c in chain_data['calls']:
        calls.append(OptionQuote(
            strike=c['strike'],
            expiry=c['expiry'],
            right='C',
            bid=c['bid'],
            ask=c['ask'],
            last=c['last'],
            volume=c.get('volume', 0),
            open_interest=c.get('open_interest', 0),
        ))

    for p in chain_data['puts']:
        puts.append(OptionQuote(
            strike=p['strike'],
            expiry=p['expiry'],
            right='P',
            bid=p['bid'],
            ask=p['ask'],
            last=p['last'],
            volume=p.get('volume', 0),
            open_interest=p.get('open_interest', 0),
        ))

    return OptionsChain(
        underlying_price=chain_data['underlying_price'],
        underlying_expiry=chain_data.get('underlying_expiry', ''),
        calls=calls,
        puts=puts,
        timestamp=chain_data['timestamp'],
    )


def parse_kalshi_market(market) -> dict:
    """Parse Kalshi market ticker to extract strike and expiry."""
    import re

    result = {
        'ticker': market.ticker,
        'title': market.title,
        'strike': None,
        'expiry': None,
        'direction': None,  # 'above' or 'below'
        'no_price': market.no_ask,
        'yes_price': market.yes_ask,
    }

    # Parse ticker like KXBTC-26JAN2922-T81749.99 or KXBTCD-26JAN2921-B82875
    # Format: SERIES-YYMMMDDHH-DIRECTION+STRIKE

    # Extract expiry: YY + MMM + DD + HH
    expiry_match = re.search(r'-(\d{2})([A-Z]{3})(\d{2})(\d{2})-', market.ticker)
    if expiry_match:
        year, month_str, day, hour = expiry_match.groups()
        month_map = {'JAN': '01', 'FEB': '02', 'MAR': '03', 'APR': '04',
                    'MAY': '05', 'JUN': '06', 'JUL': '07', 'AUG': '08',
                    'SEP': '09', 'OCT': '10', 'NOV': '11', 'DEC': '12'}
        month = month_map.get(month_str, '01')
        result['expiry'] = f"20{year}{month}{day}"

    # Extract strike and direction: -T81749.99 or -B82875
    strike_match = re.search(r'-([BT])(\d+(?:\.\d+)?)$', market.ticker)
    if strike_match:
        direction_char, strike_str = strike_match.groups()
        result['direction'] = 'above' if direction_char == 'T' else 'below'
        result['strike'] = float(strike_str)

    return result


async def run_options_arb_scan_fast(
    settings: Optional[Settings] = None,
    max_markets: int = 10,
) -> list[HedgedArbitrageOpportunity]:
    """
    Fast targeted arbitrage scanner.

    For each Kalshi market:
    1. Parse the strike price from ticker
    2. Find the nearest BFF option expiry after Kalshi expiry
    3. Fetch only ~20 strikes around the target
    4. Calculate arbitrage

    Args:
        settings: Optional settings override
        max_markets: Maximum number of markets to scan

    Returns:
        List of detected opportunities
    """
    if settings is None:
        settings = get_settings()

    analyzer = OptionsHedgeArbitrageAnalyzer(settings)
    all_opportunities = []

    console.print(Panel(
        "[bold]Fast Options-Hedged Arbitrage Scanner[/bold]\n\n"
        "Strategy: Buy Kalshi NO + Bull Call Spread hedge\n"
        "Optimized: Only fetches options needed for each market",
        title="Scanner Info",
    ))
    console.print(f"ROI Threshold: {settings.roi_threshold * 100:.1f}%")
    console.print(f"Max markets to scan: {max_markets}")

    try:
        async with KalshiClient(settings) as kalshi, IBClient(settings) as ib:
            console.print("[green]Connected to Kalshi and IB[/green]")

            # Step 1: Fetch Kalshi markets
            console.print("\n[cyan]Step 1: Fetching Kalshi BTC markets...[/cyan]")
            kalshi_markets = await kalshi.get_btc_markets()

            # Filter for markets with good NO pricing (potential arb targets)
            # We want markets where NO is cheap (high potential profit)
            candidate_markets = []
            for m in kalshi_markets:
                parsed = parse_kalshi_market(m)

                # Skip if can't parse
                if not parsed['strike'] or not parsed['expiry']:
                    continue

                # We want "above" markets where NO is cheap
                # (betting BTC won't go ABOVE the strike)
                if parsed['direction'] == 'above' and parsed['no_price']:
                    if 5 < parsed['no_price'] < 70:  # NO between 5c and 70c
                        parsed['market'] = m
                        candidate_markets.append(parsed)

            # Sort by NO price (cheapest first = highest potential ROI)
            candidate_markets.sort(key=lambda x: x['no_price'])

            console.print(f"Found {len(candidate_markets)} candidate markets (NO < 70c)")

            if not candidate_markets:
                console.print("[yellow]No suitable markets found[/yellow]")
                return []

            # Show top candidates
            console.print("\n[bold]Top candidates:[/bold]")
            for m in candidate_markets[:5]:
                roi = (100 - m['no_price']) / m['no_price'] * 100
                console.print(f"  ${m['strike']:,.0f} exp {m['expiry']}: NO@{m['no_price']}c ({roi:.0f}% potential ROI)")

            # Step 2: Scan each market
            console.print(f"\n[cyan]Step 2: Scanning top {max_markets} markets...[/cyan]")

            for i, candidate in enumerate(candidate_markets[:max_markets]):
                console.print(f"\n[bold]Market {i+1}/{min(max_markets, len(candidate_markets))}:[/bold] ${candidate['strike']:,.0f}")

                try:
                    # Fetch targeted options for this strike
                    console.print(f"  Fetching options around ${candidate['strike']:,.0f}, expiry >= {candidate['expiry']}...")

                    chain_data = await ib.get_targeted_options(
                        target_strike=candidate['strike'],
                        min_option_expiry=candidate['expiry'],
                        num_strikes=20,
                        strike_interval=500,  # $500 intervals (BFF option spacing)
                        wait_seconds=2.0,
                    )

                    if not chain_data['calls']:
                        console.print("  [yellow]No options data available[/yellow]")
                        continue

                    calls_with_prices = [c for c in chain_data['calls'] if c.get('bid') or c.get('ask')]
                    console.print(f"  Got {len(chain_data['calls'])} strikes, {len(calls_with_prices)} with prices")

                    if not calls_with_prices:
                        console.print("  [yellow]No options with bid/ask prices[/yellow]")
                        continue

                    # Convert to OptionsChain format
                    options_chain = convert_chain_data_to_options_chain(chain_data)

                    # Find arbitrage for just this market
                    opportunities = analyzer.find_hedged_arbitrage(
                        kalshi_markets=[candidate['market']],
                        options_chain=options_chain,
                    )

                    if opportunities:
                        console.print(f"  [green]Found {len(opportunities)} opportunity![/green]")
                        all_opportunities.extend(opportunities)
                    else:
                        console.print("  [dim]No profitable arbitrage[/dim]")

                except Exception as e:
                    console.print(f"  [red]Error: {e}[/red]")
                    logger.exception(f"Error scanning market {candidate['ticker']}")
                    continue

            # Display all opportunities
            if all_opportunities:
                console.print(f"\n[bold green]{'='*60}[/bold green]")
                display_hedged_opportunities(all_opportunities)

                console.print("\n[bold]Summary:[/bold]")
                total_capital = sum(o.total_capital_required for o in all_opportunities)
                total_min_profit = sum(o.min_profit for o in all_opportunities)
                console.print(f"Total capital required: ${total_capital:,.2f}")
                console.print(f"Total min profit: ${total_min_profit:,.2f}")
                avg_roi = sum(o.roi for o in all_opportunities) / len(all_opportunities)
                console.print(f"Average ROI: {avg_roi * 100:.2f}%")
            else:
                console.print("\n[yellow]No hedged arbitrage opportunities found[/yellow]")

    except Exception as e:
        console.print(f"[red]Scanner error: {e}[/red]")
        logger.exception("Scanner error")
        raise

    return all_opportunities


async def run_options_arb_scan(
    continuous: bool = False,
    settings: Optional[Settings] = None,
) -> list[HedgedArbitrageOpportunity]:
    """
    Main entry point for options-hedged arbitrage scanning.

    For fast scanning, use run_options_arb_scan_fast() instead.

    Args:
        continuous: If True, run continuously
        settings: Optional settings override

    Returns:
        List of detected opportunities (from last scan if continuous)
    """
    # Use fast scanner by default for single scans
    if not continuous:
        return await run_options_arb_scan_fast(settings=settings)

    if settings is None:
        settings = get_settings()

    analyzer = OptionsHedgeArbitrageAnalyzer(settings)
    all_opportunities = []

    console.print(Panel(
        "[bold]Options-Hedged Arbitrage Scanner[/bold]\n\n"
        "Strategy: Buy Kalshi NO + Bull Call Spread hedge\n"
        "Goal: Profit regardless of BTC price direction",
        title="Scanner Info",
    ))
    console.print(f"ROI Threshold: {settings.roi_threshold * 100:.1f}%")
    console.print(f"IB Connection: {settings.ib_host}:{settings.ib_port}")

    try:
        async with KalshiClient(settings) as kalshi, IBClient(settings) as ib:
            console.print("[green]Connected to Kalshi and IB[/green]")

            scan_count = 0
            while True:
                scan_count += 1
                scan_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
                console.print(f"\n[bold]{'='*60}[/bold]")
                console.print(f"[bold]Scan #{scan_count} at {scan_time}[/bold]")
                console.print(f"[bold]{'='*60}[/bold]")

                # Use fast scanner for each iteration
                opportunities = await run_options_arb_scan_fast(settings=settings)
                all_opportunities = opportunities

                console.print(
                    f"\n[dim]Next scan in {settings.scan_interval_seconds} seconds...[/dim]"
                )
                await asyncio.sleep(settings.scan_interval_seconds)

    except Exception as e:
        console.print(f"[red]Scanner error: {e}[/red]")
        logger.exception("Scanner error")
        raise

    return all_opportunities


async def test_options_fetch(settings: Optional[Settings] = None) -> None:
    """Test fetching MBT options data."""
    if settings is None:
        settings = get_settings()

    console.print("[bold]Testing MBT Options Data Fetch[/bold]")

    try:
        async with IBClient(settings) as ib:
            console.print("[green]Connected to IB[/green]")

            # Get options chain
            console.print("\nFetching options chain (this may take a minute)...")
            chain_data = await ib.get_mbt_options_with_prices(
                option_expiries=1,
                wait_seconds=4.0,
            )

            display_options_chain_summary(chain_data)

            # Show full call chain
            if chain_data['calls']:
                console.print("\n[bold]All Call Options:[/bold]")
                table = Table()
                table.add_column("Strike")
                table.add_column("Expiry")
                table.add_column("Bid")
                table.add_column("Ask")
                table.add_column("Last")

                for call in sorted(chain_data['calls'], key=lambda x: x['strike']):
                    table.add_row(
                        f"${call['strike']:,.0f}",
                        call['expiry'],
                        f"${call['bid']:,.2f}" if call['bid'] else "-",
                        f"${call['ask']:,.2f}" if call['ask'] else "-",
                        f"${call['last']:,.2f}" if call['last'] else "-",
                    )

                console.print(table)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Options test error")


def run_demo_calculation() -> None:
    """
    Run a demonstration of the arbitrage calculation with synthetic data.

    This shows exactly how the calculations work using the example:
    - Kalshi: "Will BTC hit $150K by June?" NO @ 90 cents
    - Hedge: Buy $130K call, Sell $150K call (using BFF contracts)
    """
    from .config import get_settings
    from .models import Market, MarketStatus

    settings = get_settings()
    btc_mult = settings.btc_multiplier
    symbol = settings.btc_futures_symbol

    console.print(Panel(
        f"[bold]OPTIONS-HEDGED ARBITRAGE DEMO[/bold]\n\n"
        f"Using {symbol} contracts ({btc_mult} BTC each)\n"
        f"Demonstrating proper position sizing for the arbitrage.",
        title="Demo Mode",
        border_style="yellow",
    ))

    # Parameters
    long_strike = 130000
    short_strike = 150000
    kalshi_strike = 150000
    btc_price = 105000

    # BFF options are smaller, so premiums are proportionally smaller
    # MBT premiums / 10 for BFF (since BFF is 1/10th the size)
    if symbol == "BFF":
        long_call_ask = 105   # $105 for $130K call (vs $1050 for MBT)
        short_call_bid = 20   # $20 for $150K call (vs $200 for MBT)
    else:  # MBT
        long_call_ask = 1050
        short_call_bid = 200

    spread_debit = long_call_ask - short_call_bid
    spread_width = short_strike - long_strike
    spread_max_value = spread_width * btc_mult  # Value at expiry if BTC >= short_strike
    spread_max_profit = spread_max_value - spread_debit

    console.print(f"\n[bold cyan]1. Contract Specifications[/bold cyan]")
    console.print(f"Symbol: [green]{symbol}[/green]")
    console.print(f"BTC per contract: [green]{btc_mult} BTC[/green]")
    console.print(f"Spread width: ${long_strike:,} to ${short_strike:,} = ${spread_width:,}")
    console.print(f"Spread max value at expiry: ${spread_width:,} x {btc_mult} = [green]${spread_max_value:,.0f}[/green]")

    console.print(f"\n[bold cyan]2. Kalshi Market[/bold cyan]")
    console.print(f"Market: 'When will Bitcoin hit ${kalshi_strike:,}?'")
    console.print(f"NO price: [green]90 cents[/green]")
    console.print(f"NO profit if BTC < ${kalshi_strike:,}: $0.10 per contract")
    console.print(f"NO loss if BTC >= ${kalshi_strike:,}: $0.90 per contract")

    console.print(f"\n[bold cyan]3. Options Spread ({symbol})[/bold cyan]")
    console.print(f"Buy ${long_strike:,} call @ ${long_call_ask:,.0f}")
    console.print(f"Sell ${short_strike:,} call @ ${short_call_bid:,.0f}")
    console.print(f"Net debit: [red]${spread_debit:,.0f}[/red]")
    console.print(f"Max profit (at ${short_strike:,}+): [green]${spread_max_profit:,.0f}[/green]")

    console.print(f"\n[bold cyan]4. Position Sizing - THE KEY![/bold cyan]")
    console.print("""
[yellow]Goal:[/yellow] Size positions so we profit in ALL scenarios.

[yellow]Two constraints:[/yellow]
1. If BTC < $130K: Kalshi profit must cover spread loss
2. If BTC >= $150K: Spread profit must cover Kalshi loss
""")

    # Calculate required sizing
    # Constraint 1: kalshi_profit >= spread_loss
    # kalshi_contracts * 0.10 >= spread_debit
    # kalshi_contracts >= spread_debit / 0.10
    min_kalshi_for_spread_loss = spread_debit / 0.10

    # Constraint 2: spread_profit >= kalshi_loss
    # spread_contracts * spread_max_profit >= kalshi_contracts * 0.90
    # spread_contracts >= kalshi_contracts * 0.90 / spread_max_profit

    console.print(f"[yellow]Constraint 1:[/yellow] Kalshi profit covers spread loss")
    console.print(f"  Kalshi contracts x $0.10 >= ${spread_debit}")
    console.print(f"  Kalshi contracts >= {min_kalshi_for_spread_loss:,.0f}")

    console.print(f"\n[yellow]Constraint 2:[/yellow] Spread profit covers Kalshi loss")
    console.print(f"  For K Kalshi contracts: loss = K x $0.90")
    console.print(f"  Need spread profit >= K x $0.90")
    console.print(f"  Spreads x ${spread_max_profit:,.0f} >= K x $0.90")
    console.print(f"  Spreads >= K x 0.90 / {spread_max_profit:,.0f}")

    # Find the balance point
    # K * 0.10 = spread_debit (covers spread loss exactly)
    # S * spread_max_profit = K * 0.90 (covers kalshi loss exactly)
    # From first: K = spread_debit / 0.10
    # From second: S = K * 0.90 / spread_max_profit

    K = int(spread_debit / 0.10)  # Kalshi contracts
    S = int((K * 0.90) / spread_max_profit) + 1  # Spread contracts (round up)

    console.print(f"\n[bold green]OPTIMAL SIZING:[/bold green]")
    console.print(f"  Kalshi NO contracts: [bold]{K:,}[/bold]")
    console.print(f"  Options spreads: [bold]{S:,}[/bold]")

    kalshi_cost = K * 0.90
    total_spread_cost = S * spread_debit
    total_capital = kalshi_cost + total_spread_cost

    console.print(f"\n[yellow]Capital Required:[/yellow]")
    console.print(f"  Kalshi: {K:,} x $0.90 = ${kalshi_cost:,.0f}")
    console.print(f"  Spreads: {S} x ${spread_debit:,.0f} = ${total_spread_cost:,.0f}")
    console.print(f"  [bold]Total: ${total_capital:,.0f}[/bold]")

    console.print(f"\n[bold cyan]5. P&L at Different BTC Prices[/bold cyan]")

    scenarios = [
        (100000, "Below long strike"),
        (130000, "At long strike"),
        (140000, "Between strikes"),
        (150000, "At short strike (Kalshi trigger)"),
        (175000, "Above both"),
    ]

    results_table = Table(title=f"P&L Analysis ({K:,} Kalshi + {S} {symbol} spreads)")
    results_table.add_column("BTC Price")
    results_table.add_column("Kalshi P&L")
    results_table.add_column("Spread P&L")
    results_table.add_column("Total P&L")
    results_table.add_column("ROI")

    for btc_price_scenario, desc in scenarios:
        # Kalshi P&L
        if btc_price_scenario < kalshi_strike:
            kalshi_pnl = K * 0.10
        else:
            kalshi_pnl = -kalshi_cost

        # Spread P&L
        if btc_price_scenario < long_strike:
            spread_value = 0
        elif btc_price_scenario < short_strike:
            spread_value = (btc_price_scenario - long_strike) * btc_mult * S
        else:
            spread_value = spread_max_value * S

        spread_pnl = spread_value - (spread_debit * S)
        total_pnl = kalshi_pnl + spread_pnl
        roi = total_pnl / total_capital * 100

        pnl_color = "green" if total_pnl >= 0 else "red"

        results_table.add_row(
            f"${btc_price_scenario:,}",
            f"${kalshi_pnl:+,.0f}",
            f"${spread_pnl:+,.0f}",
            f"[{pnl_color}]${total_pnl:+,.0f}[/{pnl_color}]",
            f"[{pnl_color}]{roi:+.1f}%[/{pnl_color}]",
        )

    console.print(results_table)

    # Check if it's actually profitable
    # Worst case 1: BTC < $130K (spread worthless)
    worst1_kalshi = K * 0.10
    worst1_spread = -spread_debit * S
    worst1_total = worst1_kalshi + worst1_spread

    # Worst case 2: BTC >= $150K (Kalshi loses)
    worst2_kalshi = -kalshi_cost
    worst2_spread = spread_max_profit * S
    worst2_total = worst2_kalshi + worst2_spread

    console.print(f"\n[bold cyan]6. Profit Analysis[/bold cyan]")
    console.print(f"\n[yellow]Worst Case 1: BTC stays below ${long_strike:,}[/yellow]")
    console.print(f"  Kalshi wins: +${worst1_kalshi:,.0f}")
    console.print(f"  Spread expires worthless: -${spread_debit * S:,.0f}")
    console.print(f"  [bold]Net: ${worst1_total:+,.0f}[/bold]")

    console.print(f"\n[yellow]Worst Case 2: BTC hits ${kalshi_strike:,}+[/yellow]")
    console.print(f"  Kalshi loses: -${kalshi_cost:,.0f}")
    console.print(f"  Spread max profit: +${spread_max_profit * S:,.0f}")
    console.print(f"  [bold]Net: ${worst2_total:+,.0f}[/bold]")

    min_profit = min(worst1_total, worst2_total)
    if min_profit >= 0:
        console.print(f"\n[bold green]SUCCESS! Minimum profit: ${min_profit:,.0f} ({min_profit/total_capital*100:.1f}% ROI)[/bold green]")
    else:
        console.print(f"\n[bold red]Not profitable in all scenarios. Worst case: ${min_profit:,.0f}[/bold red]")
        console.print("[dim]Need cheaper options or better Kalshi pricing[/dim]")

    console.print(f"\n[bold cyan]7. When DOES It Work?[/bold cyan]")
    console.print("""
[yellow]The math requires:[/yellow]
  Kalshi_profit_ratio / Kalshi_loss_ratio > Spread_loss / Spread_profit

With NO @ 90c: profit=$0.10, loss=$0.90, ratio = 0.111
With spread debit=$85, profit=$115: ratio = 0.739

0.111 < 0.739 = [red]NO ARBITRAGE POSSIBLE[/red]

[green]For arbitrage to exist, we need Kalshi ratio > Spread ratio:[/green]
""")

    # Calculate what NO price would make this work
    # K * NO_profit >= S * spread_debit
    # S * spread_profit >= K * NO_loss
    #
    # From these: NO_profit / NO_loss >= spread_debit / spread_profit
    # NO_profit / (1 - NO_profit) >= 85 / 115 = 0.739
    # NO_profit >= 0.739 * (1 - NO_profit)
    # NO_profit >= 0.739 - 0.739 * NO_profit
    # NO_profit + 0.739 * NO_profit >= 0.739
    # 1.739 * NO_profit >= 0.739
    # NO_profit >= 0.425

    required_no_profit = spread_debit / spread_max_profit
    required_no_price = 1 / (1 + 1/required_no_profit)

    console.print(f"[yellow]Option 1: Better Kalshi pricing[/yellow]")
    console.print(f"  Required: NO profit/loss ratio >= {spread_debit/spread_max_profit:.3f}")
    console.print(f"  This means NO price <= {(1-required_no_profit/(1+required_no_profit))*100:.0f} cents")
    console.print(f"  Example: NO @ 55c -> profit $0.45, loss $0.55, ratio = 0.82 > 0.74 [green]WORKS![/green]")

    console.print(f"\n[yellow]Option 2: Cheaper options spread[/yellow]")
    # For NO @ 90c (ratio 0.111), need spread ratio <= 0.111
    # spread_debit / spread_profit <= 0.111
    # spread_debit <= 0.111 * (spread_value - spread_debit)
    # spread_debit <= 0.111 * spread_value - 0.111 * spread_debit
    # 1.111 * spread_debit <= 0.111 * 200
    # spread_debit <= 20

    max_spread_debit = 0.10 / 0.90 * spread_max_value / (1 + 0.10/0.90)
    console.print(f"  With NO @ 90c, need spread debit <= ${max_spread_debit:.0f}")
    console.print(f"  (Current spread debit: ${spread_debit})")
    console.print(f"  Example: If spread costs $15 -> profit $185, ratio = 0.08 < 0.11 [green]WORKS![/green]")

    console.print(f"\n[bold green]Example of WORKING Arbitrage:[/bold green]")
    console.print(f"Kalshi NO @ 55c (45c profit, 55c loss)")
    console.print(f"BFF spread @ $15 debit ($185 max profit)")

    # Calculate sizing for this scenario
    # K * 0.45 >= S * 15  -> K >= S * 33.3
    # S * 185 >= K * 0.55 -> S >= K * 0.00297

    # Let's use S = 1
    # K >= 33.3, let's use K = 34
    # Check: S * 185 >= 34 * 0.55 = 18.7, so 185 >= 18.7 YES

    demo_K = 34
    demo_S = 1
    demo_kalshi_cost = demo_K * 0.55
    demo_spread_cost = demo_S * 15
    demo_total_capital = demo_kalshi_cost + demo_spread_cost

    console.print(f"\nPosition: {demo_K} Kalshi NO + {demo_S} spread")
    console.print(f"Capital: ${demo_kalshi_cost:.2f} + ${demo_spread_cost:.2f} = ${demo_total_capital:.2f}")

    # Worst case 1: BTC < $130K
    wc1 = demo_K * 0.45 - demo_S * 15
    # Worst case 2: BTC >= $150K
    wc2 = -demo_K * 0.55 + demo_S * 185

    console.print(f"\nBTC < $130K: +${demo_K * 0.45:.2f} - ${demo_S * 15} = [green]${wc1:.2f}[/green]")
    console.print(f"BTC >= $150K: -${demo_K * 0.55:.2f} + ${demo_S * 185} = [green]${wc2:.2f}[/green]")
    console.print(f"[bold green]Min profit: ${min(wc1, wc2):.2f} = {min(wc1, wc2)/demo_total_capital*100:.1f}% ROI[/bold green]")

    console.print("""
[yellow]Key insight:[/yellow]
The scanner looks for these exact conditions - where the math works out.
Real opportunities appear when:
* Kalshi is mispriced (NO too cheap relative to probability)
* Options IV is low (cheap spreads)
* Near expiry with clear price direction
""")

    console.print(f"\n[bold]Calculation files:[/bold]")
    console.print("* src/analysis/options_hedge_arb.py - Core P&L calculations")
    console.print("  - calculate_spread_payoff(): Options spread value at any BTC price")
    console.print("  - _calculate_arbitrage_metrics(): Position sizing and scenario analysis")
    console.print("  - find_hedged_arbitrage(): Main scanner that finds opportunities")
