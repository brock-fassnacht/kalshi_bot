"""Async bridge for running async code from Streamlit's sync context."""

import asyncio
import threading
from typing import Any, Coroutine


class AsyncBridge:
    """
    Singleton background thread running an asyncio event loop.
    Streamlit (sync) calls bridge.run(coroutine) to execute async Kalshi API calls.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro: Coroutine, timeout: float = 300) -> Any:
        """Submit a coroutine to the background loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


def get_bridge() -> AsyncBridge:
    """Get the singleton AsyncBridge instance."""
    return AsyncBridge()
