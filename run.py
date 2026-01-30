"""
Simple entry point for the Kalshi Arbitrage Bot.

Usage:
    python run.py scan          # Single scan for opportunities
    python run.py scan -c       # Continuous scanning
    python run.py execute       # Execute opportunities (dry-run)
    python run.py cross-scan    # Cross-market scan (Kalshi + IB)
    python run.py options-scan  # Options-hedged arbitrage scan (Kalshi + MBT options)
    python run.py test-ib       # Test IB Gateway connection
"""

import sys


def main():
    if len(sys.argv) < 2:
        print("Kalshi Arbitrage Bot")
        print("=" * 40)
        print("\nUsage:")
        print("  python run.py scan          # Single Kalshi scan")
        print("  python run.py scan -c       # Continuous Kalshi scan")
        print("  python run.py scan --no-db  # Scan without database")
        print("  python run.py execute       # Execute (dry-run)")
        print()
        print("  python run.py cross-scan    # Cross-market scan (Kalshi + IB futures)")
        print("  python run.py cross-scan -c # Continuous cross-market scan")
        print()
        print("  python run.py options-scan         # Fast options-hedged scan (top 10 markets)")
        print("  python run.py options-scan --max=5 # Scan top 5 markets only")
        print("  python run.py options-scan -c      # Continuous options-hedged scan")
        print("  python run.py test-options         # Test options data fetch")
        print()
        print("  python run.py test-ib       # Test IB Gateway connection")
        print("  python run.py test-kalshi   # Test Kalshi API connection")
        print("\nSetup:")
        print("  1. Copy .env.example to .env")
        print("  2. Add your Kalshi API credentials")
        print("  3. Start IB Gateway (port 4002)")
        print("  4. Install dependencies: pip install -r requirements.txt")
        return

    command = sys.argv[1]

    if command == "scan":
        from src.scanner import main as scanner_main
        scanner_main()

    elif command == "execute":
        import asyncio
        from src.executor import run_executor_demo
        asyncio.run(run_executor_demo())

    elif command == "cross-scan":
        import asyncio
        from src.cross_scanner import run_cross_market_scan
        continuous = "-c" in sys.argv or "--continuous" in sys.argv
        asyncio.run(run_cross_market_scan(continuous=continuous))

    elif command == "test-ib":
        import asyncio
        from src.cross_scanner import test_ib_connection
        asyncio.run(test_ib_connection())

    elif command == "test-kalshi":
        import asyncio
        from src.api.client import KalshiClient
        from src.config import get_settings

        async def test():
            settings = get_settings()
            print(f"Connecting to Kalshi: {settings.kalshi_base_url}")
            async with KalshiClient(settings) as client:
                print("Connected!")
                markets, cursor = await client.get_markets(limit=10)
                print(f"Found {len(markets)} markets")
                for m in markets[:5]:
                    print(f"  - {m.ticker}: {m.title[:50]}")

        asyncio.run(test())

    elif command == "options-scan":
        import asyncio
        from src.options_arb_scanner import run_options_arb_scan, run_options_arb_scan_fast, run_demo_calculation
        continuous = "-c" in sys.argv or "--continuous" in sys.argv
        demo = "--demo" in sys.argv

        # Parse max markets argument (--max=5)
        max_markets = 10
        for arg in sys.argv:
            if arg.startswith("--max="):
                try:
                    max_markets = int(arg.split("=")[1])
                except ValueError:
                    pass

        if demo:
            run_demo_calculation()
        elif continuous:
            asyncio.run(run_options_arb_scan(continuous=True))
        else:
            asyncio.run(run_options_arb_scan_fast(max_markets=max_markets))

    elif command == "test-options":
        import asyncio
        from src.options_arb_scanner import test_options_fetch
        asyncio.run(test_options_fetch())

    else:
        print(f"Unknown command: {command}")
        print("Available commands: scan, execute, cross-scan, options-scan, test-ib, test-kalshi, test-options")


if __name__ == "__main__":
    main()
