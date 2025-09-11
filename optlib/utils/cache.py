"""
cache.py
========
Session-level caching for market data and ticker objects to avoid redundant network calls.
"""

import time
from typing import Dict, Any, Optional
import yfinance as yf
from threading import Lock


class SessionCache:
    """Thread-safe cache for storing market data during a session."""
    
    def __init__(self, default_ttl: float = 3600):  # 1 hour TTL
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data if it exists and is not expired."""
        with self._lock:
            if key in self._data:
                entry = self._data[key]
                if time.time() < entry['expires']:
                    return entry['value']
                else:
                    # Remove expired entry
                    del self._data[key]
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Store data in cache with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        expires = time.time() + ttl
        with self._lock:
            self._data[key] = {
                'value': value,
                'expires': expires
            }
    
    def clear(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._data.clear()


class TickerRegistry:
    """Registry for reusing yfinance Ticker objects."""
    
    def __init__(self):
        self._tickers: Dict[str, yf.Ticker] = {}
        self._lock = Lock()
    
    def get_ticker(self, symbol: str) -> yf.Ticker:
        """Get or create a Ticker object for the given symbol."""
        with self._lock:
            if symbol not in self._tickers:
                self._tickers[symbol] = yf.Ticker(symbol)
            return self._tickers[symbol]
    
    def clear(self) -> None:
        """Clear all cached ticker objects."""
        with self._lock:
            self._tickers.clear()


# Global instances for session-wide caching
_session_cache = SessionCache()
_ticker_registry = TickerRegistry()


def get_session_cache() -> SessionCache:
    """Get the global session cache instance."""
    return _session_cache


def get_ticker_registry() -> TickerRegistry:
    """Get the global ticker registry instance."""
    return _ticker_registry


def clear_all_caches() -> None:
    """Clear all session caches."""
    _session_cache.clear()
    _ticker_registry.clear()