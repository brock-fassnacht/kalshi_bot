"""
Kalshi Market Dashboard launcher.

Usage:
    python run.py              # Launch the Streamlit dashboard
    python run.py test-kalshi  # Test Kalshi API connection
"""

import sys
import subprocess


def main():
    if len(sys.argv) < 2:
        # Default: launch dashboard
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.headless", "true",
        ])
        return

    command = sys.argv[1]

    if command == "dashboard":
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.headless", "true",
        ])

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

    else:
        print(f"Unknown command: {command}")
        print("Available commands: dashboard, test-kalshi")
        print("Default (no args): launches the Streamlit dashboard")


if __name__ == "__main__":
    main()
