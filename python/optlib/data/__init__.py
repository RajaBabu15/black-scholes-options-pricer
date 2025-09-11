# optlib/data/__init__.py
"""
Data access module for market data, historical prices, and option chains.
"""

from .unified_loader import (
    load_historical_data,
    load_option_chain,
    get_risk_free_rate,
    get_dividend_yield,
    clear_cache,
    get_cache_info
)

__all__ = [
    'load_historical_data',
    'load_option_chain', 
    'get_risk_free_rate',
    'get_dividend_yield',
    'clear_cache',
    'get_cache_info'
]