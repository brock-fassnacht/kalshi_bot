"""
Test script to verify Kalshi API connection.

Run this after setting up your .env file to verify credentials work.

Usage:
    python test_connection.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api.client import KalshiClient
from src.config import get_settings


async def test_connection():
    """Test basic API connectivity."""
    print("=" * 50)
    print("Kalshi Demo API Connection Test")
    print("=" * 50)

    try:
        settings = get_settings()
        print(f"\n[1] Configuration loaded")
        print(f"    Base URL: {settings.kalshi_base_url}")
        print(f"    API Key: {settings.kalshi_api_key[:8]}...")
        print(f"    Private Key Path: {settings.kalshi_private_key_path}")

        # Check if private key exists
        if not settings.kalshi_private_key_path.exists():
            print(f"\n[ERROR] Private key not found at: {settings.kalshi_private_key_path}")
            print("    Please place your Kalshi private key at the specified path.")
            return False

        print(f"    Private Key: Found ({settings.kalshi_private_key_path.stat().st_size} bytes)")

    except Exception as e:
        print(f"\n[ERROR] Failed to load configuration: {e}")
        print("    Make sure you have copied .env.example to .env and filled in your credentials.")
        return False

    try:
        async with KalshiClient(settings) as client:
            # Test 1: Get balance (tests authentication)
            print(f"\n[2] Testing authentication...")
            balance = await client.get_balance()
            print(f"    Success! Account balance: ${balance.get('balance', 0) / 100:.2f}")

            # Test 2: Fetch some markets
            print(f"\n[3] Fetching markets...")
            markets, _ = await client.get_markets(limit=5, status="open")
            print(f"    Found {len(markets)} markets")

            if markets:
                print(f"\n    Sample markets:")
                for m in markets[:3]:
                    bid = m.yes_bid if m.yes_bid else "N/A"
                    ask = m.yes_ask if m.yes_ask else "N/A"
                    print(f"      - {m.ticker}: Yes {bid}¢/{ask}¢")

            # Test 3: Fetch events
            print(f"\n[4] Fetching events...")
            events, _ = await client.get_events(limit=3)
            print(f"    Found {len(events)} events")

            if events:
                print(f"\n    Sample events:")
                for e in events[:3]:
                    print(f"      - {e.event_ticker}: {e.title} ({len(e.markets)} markets)")

            # Test 4: Get positions
            print(f"\n[5] Fetching positions...")
            positions = await client.get_positions()
            print(f"    You have {len(positions)} open positions")

            print("\n" + "=" * 50)
            print("All tests passed! Your setup is working correctly.")
            print("=" * 50)
            print("\nNext steps:")
            print("  1. Run: python run.py scan")
            print("  2. For continuous monitoring: python run.py scan -c")
            return True

    except Exception as e:
        print(f"\n[ERROR] API request failed: {e}")
        print("\nPossible issues:")
        print("  - Invalid API key or private key")
        print("  - Private key doesn't match the API key")
        print("  - Network connectivity issues")
        print("  - Demo API may be down")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_connection())
    sys.exit(0 if success else 1)
