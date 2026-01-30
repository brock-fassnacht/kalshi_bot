"""Authentication utilities for Kalshi API."""

import base64
import time
from datetime import datetime, timezone

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def sign_request(
    private_key_pem: str,
    timestamp: int,
    method: str,
    path: str,
) -> str:
    """
    Sign a Kalshi API request using RSA-PSS.

    Args:
        private_key_pem: PEM-encoded private key string
        timestamp: Unix timestamp in milliseconds
        method: HTTP method (GET, POST, etc.)
        path: API path (e.g., /trade-api/v2/markets)

    Returns:
        Base64-encoded signature
    """
    # Load the private key
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(),
        password=None,
    )

    # Create the message to sign: timestamp + method + path
    message = f"{timestamp}{method}{path}"

    # Sign with RSA-PSS
    signature = private_key.sign(
        message.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )

    return base64.b64encode(signature).decode()


def get_auth_headers(
    api_key: str,
    private_key_pem: str,
    method: str,
    path: str,
) -> dict[str, str]:
    """
    Generate authentication headers for Kalshi API request.

    Args:
        api_key: Kalshi API key
        private_key_pem: PEM-encoded private key
        method: HTTP method
        path: API path

    Returns:
        Dictionary of headers to include in request
    """
    timestamp = int(datetime.now(timezone.utc).timestamp() * 1000)
    signature = sign_request(private_key_pem, timestamp, method, path)

    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
    }
