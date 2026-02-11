"""Shared utilities for dashboard components."""

import re
from datetime import datetime


# Month abbreviations to month numbers
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Patterns to match date components after the first dash in a ticker
# e.g. -28NOV07, -29JAN20, -27DEC, -26NOV03, -26, -2028, -2030, -28JUN
_DATE_PATTERN = re.compile(
    r"-(\d{2,4})"            # year: 2 or 4 digits
    r"([A-Z]{3})?"           # optional month abbreviation
    r"(\d{1,2})?"            # optional day
    r"(?=-|$)",              # followed by another dash or end
)


def parse_expiry_from_ticker(ticker: str) -> datetime | None:
    """
    Extract an expiry date from a ticker string.

    Examples:
        KXNBA-26-OKC          -> Dec 31, 2026
        KXPRESNOMD-28-GN      -> Dec 31, 2028
        KXAKSENATE-26NOV03    -> Nov 3, 2026
        KXBTCHALF-28JUN       -> Jun 30, 2028
        KXPRESPARTY-2028      -> Dec 31, 2028
        KXTRDBAN-25DEC31      -> Dec 31, 2025
        KXCOLONIZEMARS-50     -> Dec 31, 2050

    Returns None if no date can be parsed.
    """
    match = _DATE_PATTERN.search(ticker)
    if not match:
        return None

    year_str, month_str, day_str = match.groups()

    # Parse year
    year = int(year_str)
    if year < 100:
        year += 2000

    # Parse month
    if month_str and month_str.upper() in _MONTHS:
        month = _MONTHS[month_str.upper()]
    else:
        month = 12  # default to end of year

    # Parse day
    if day_str:
        day = int(day_str)
    elif month_str:
        # Have month but no day — use last day of month
        import calendar
        day = calendar.monthrange(year, month)[1]
    else:
        # Year only — Dec 31
        day = 31

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def compute_days_to_expiry(ticker: str, close_time=None) -> float | None:
    """
    Compute days until expiry, preferring ticker-derived date, falling back to close_time.
    """
    now = datetime.utcnow()
    expiry = parse_expiry_from_ticker(ticker)
    if expiry is not None:
        return round((expiry - now).total_seconds() / 86400, 1)
    if close_time is not None and not _is_nat(close_time):
        return round((close_time - now).total_seconds() / 86400, 1)
    return None


def _is_nat(val) -> bool:
    """Check if a value is NaT/None/NaN."""
    try:
        import pandas as pd
        return pd.isna(val)
    except (TypeError, ValueError):
        return val is None
