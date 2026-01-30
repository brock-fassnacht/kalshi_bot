"""Configuration management for Kalshi arbitrage bot."""

from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Kalshi API
    kalshi_api_key: str
    kalshi_private_key_path: Path = Path("./keys/kalshi_private_key.pem")
    kalshi_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"

    # Interactive Brokers Settings
    ib_host: str = "127.0.0.1"
    ib_port: int = 4002  # 7497=TWS Paper, 7496=TWS Live, 4001=Gateway Paper, 4002=Gateway Live
    ib_client_id: int = 1
    ib_timeout: int = 30

    # Bitcoin Futures Symbol (for price reference)
    # MBT = Micro Bitcoin (0.1 BTC per contract)
    # BFF = Bitcoin Friday Futures (0.01 BTC per contract) - 10x smaller
    btc_futures_symbol: str = "BFF"  # Default to smaller BFF contracts

    # Options source: "IBIT" for ETF options, "BFF" for futures options
    options_source: str = "IBIT"

    # Contract multipliers (BTC per contract)
    BTC_MULTIPLIERS: dict = {
        "MBT": 0.1,    # Micro Bitcoin Futures - 0.1 BTC
        "BFF": 0.02,   # Bitcoin Friday Futures - 0.02 BTC (1/5 of MBT)
        "BTC": 5.0,    # Full Bitcoin Futures - 5 BTC
    }

    # IBIT options: standard 100 shares per contract
    # BTC equivalent depends on current BTC/IBIT ratio (calculated dynamically)
    IBIT_SHARES_PER_CONTRACT: int = 100

    @property
    def btc_multiplier(self) -> float:
        """Get the BTC multiplier for the configured futures symbol."""
        return self.BTC_MULTIPLIERS.get(self.btc_futures_symbol, 0.1)

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/kalshi.db"

    # Arbitrage settings
    min_arbitrage_spread: float = 0.02  # Minimum spread to trigger arbitrage (2%)
    roi_threshold: float = 0.02  # 2% minimum ROI for cross-market arbitrage
    max_position_size: int = 100  # Max contracts per position
    scan_interval_seconds: int = 30  # How often to scan for opportunities
    orderbook_depth: int = 10

    @property
    def private_key(self) -> str:
        """Load private key from file."""
        return self.kalshi_private_key_path.read_text()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
