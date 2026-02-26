"""Configuration management for Kalshi market dashboard."""

import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env" if os.path.exists(".env") else None,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Kalshi API
    kalshi_api_key: str
    kalshi_private_key_path: Path = Path("./keys/kalshi_private_key.pem")
    kalshi_private_key: str = ""  # Direct PEM string (for cloud deployment)
    kalshi_base_url: str = "https://demo-api.kalshi.co/trade-api/v2"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/kalshi.db"

    # Dashboard settings
    refresh_interval_seconds: int = 60
    orderbook_depth: int = 10
    snapshot_interval_seconds: int = 300  # How often to save snapshots to DB
    new_market_hours: int = 24  # Show markets opened in last N hours
    price_change_lookback_hours: int = 24  # Lookback window for price changes

    # Market filters
    min_expiry_hours: int = 24  # Exclude markets expiring within N hours
    near_mid_range_cents: int = 20  # Range around midpoint for depth calc
    min_near_mid_depth_dollars: float = 500  # Min $ depth within range of mid
    min_yes_depth_dollars: float = 5000  # Min $ depth on YES side
    min_no_depth_dollars: float = 5000  # Min $ depth on NO side
    min_yes_ask_prefilter: int = 5  # Exclude markets with yes_ask below this (cents)
    min_oi_prefilter: int = 500  # Basic OI pre-filter before fetching orderbooks
    max_market_pages: int = 50        # Pages of markets to scan (200/page = 10,000 max)
    max_orderbook_fetches: int = 5000  # Cap on how many orderbooks to fetch per refresh
    orderbook_concurrency: int = 10  # Max concurrent orderbook API requests

    @property
    def private_key(self) -> str:
        """Load private key from env var string, or fall back to file."""
        if self.kalshi_private_key:
            # Handle literal \n from env vars / Streamlit secrets
            return self.kalshi_private_key.replace("\\n", "\n")
        return self.kalshi_private_key_path.read_text()


def _read_streamlit_secrets() -> dict:
    """Read Streamlit Cloud secrets as a dict for Settings overrides."""
    try:
        import streamlit as st
        overrides = {}
        for key, value in st.secrets.items():
            if isinstance(value, str):
                overrides[key.lower()] = value
        return overrides
    except Exception:
        return {}


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance, with Streamlit secrets taking priority."""
    overrides = _read_streamlit_secrets()
    s = Settings(**overrides)
    print(f"[CONFIG] base_url={s.kalshi_base_url}")
    return s
