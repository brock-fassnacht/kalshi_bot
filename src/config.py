"""Configuration management for Kalshi market dashboard."""

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

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/kalshi.db"

    # Dashboard settings
    refresh_interval_seconds: int = 60
    orderbook_depth: int = 10
    snapshot_interval_seconds: int = 300  # How often to save snapshots to DB
    new_market_hours: int = 24  # Show markets opened in last N hours
    price_change_lookback_hours: int = 24  # Lookback window for price changes

    @property
    def private_key(self) -> str:
        """Load private key from file."""
        return self.kalshi_private_key_path.read_text()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
