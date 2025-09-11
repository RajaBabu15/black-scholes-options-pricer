# optlib/data/unified_loader.py
"""
Unified Data Loader for Market Data, Historical Prices, and Option Chains.

This module provides centralized data access with caching and thread safety for:
- Historical price data from Yahoo Finance
- Option chains data
- Risk-free rates and dividend yields
- File-based caching with automatic cache validation
- Thread-safe operations using file locking

The cache system stores data in organized directories and validates freshness
based on configurable expiration times. All operations are atomic to ensure
data integrity in concurrent environments.
"""

import os
import json
import pickle
import threading
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import yfinance as yf
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache configuration
DEFAULT_CACHE_DIR = os.path.expanduser("~/.optlib_cache")
CACHE_EXPIRY_HOURS = {
    'historical': 24,      # Historical data expires after 24 hours
    'options': 1,          # Option chains expire after 1 hour
    'risk_free': 24,       # Risk-free rates expire after 24 hours  
    'dividends': 24        # Dividend data expires after 24 hours
}

# Thread-safe locks for different data types
_cache_locks = {
    'historical': threading.RLock(),
    'options': threading.RLock(),
    'risk_free': threading.RLock(),
    'dividends': threading.RLock(),
    'global': threading.RLock()
}


class CacheManager:
    """
    Thread-safe cache manager with file locking and automatic expiration.
    
    Provides atomic operations for reading/writing cached data with proper
    locking mechanisms to prevent race conditions in concurrent environments.
    """
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Base directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different data types
        for data_type in CACHE_EXPIRY_HOURS.keys():
            (self.cache_dir / data_type).mkdir(exist_ok=True)
    
    def _generate_cache_key(self, data_type: str, **kwargs) -> str:
        """
        Generate a unique cache key based on parameters.
        
        Args:
            data_type: Type of data (historical, options, etc.)
            **kwargs: Parameters to include in cache key
            
        Returns:
            SHA256 hash of the parameters for unique identification
        """
        # Sort kwargs for consistent key generation
        sorted_params = sorted(kwargs.items())
        key_string = f"{data_type}_{sorted_params}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def _get_cache_path(self, data_type: str, cache_key: str) -> Path:
        """Get full path for cache file."""
        return self.cache_dir / data_type / f"{cache_key}.pkl"
    
    def _is_cache_valid(self, cache_path: Path, data_type: str) -> bool:
        """
        Check if cached data is still valid based on expiration time.
        
        Args:
            cache_path: Path to cached file
            data_type: Type of data to check expiration rules
            
        Returns:
            True if cache is valid, False if expired or doesn't exist
        """
        if not cache_path.exists():
            return False
        
        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry_hours = CACHE_EXPIRY_HOURS.get(data_type, 24)
        expiry_time = file_time + timedelta(hours=expiry_hours)
        
        return datetime.now() < expiry_time
    
    def get_cached_data(self, data_type: str, **kwargs) -> Optional[Any]:
        """
        Retrieve cached data if valid.
        
        Args:
            data_type: Type of data to retrieve
            **kwargs: Parameters for cache key generation
            
        Returns:
            Cached data if valid, None if not found or expired
        """
        cache_key = self._generate_cache_key(data_type, **kwargs)
        cache_path = self._get_cache_path(data_type, cache_key)
        
        with _cache_locks[data_type]:
            if self._is_cache_valid(cache_path, data_type):
                try:
                    with open(cache_path, 'rb') as f:
                        logger.debug(f"Cache hit for {data_type}: {cache_key}")
                        return pickle.load(f)
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_path}: {e}")
                    # Remove corrupted cache file
                    try:
                        cache_path.unlink()
                    except:
                        pass
        
        logger.debug(f"Cache miss for {data_type}: {cache_key}")
        return None
    
    def save_cached_data(self, data: Any, data_type: str, **kwargs) -> None:
        """
        Save data to cache atomically.
        
        Args:
            data: Data to cache
            data_type: Type of data being cached
            **kwargs: Parameters for cache key generation
        """
        cache_key = self._generate_cache_key(data_type, **kwargs)
        cache_path = self._get_cache_path(data_type, cache_key)
        temp_path = cache_path.with_suffix('.tmp')
        
        with _cache_locks[data_type]:
            try:
                # Write to temporary file first for atomic operation
                with open(temp_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Atomic move to final location
                temp_path.replace(cache_path)
                logger.debug(f"Cached {data_type} data: {cache_key}")
                
            except Exception as e:
                logger.error(f"Error saving cache file {cache_path}: {e}")
                # Clean up temporary file
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
    
    def clear_cache(self, data_type: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            data_type: Specific data type to clear, or None to clear all
        """
        with _cache_locks['global']:
            if data_type:
                cache_dir = self.cache_dir / data_type
                if cache_dir.exists():
                    for file_path in cache_dir.glob("*.pkl"):
                        try:
                            file_path.unlink()
                        except Exception as e:
                            logger.warning(f"Error deleting cache file {file_path}: {e}")
                    logger.info(f"Cleared {data_type} cache")
            else:
                # Clear all cache
                for dt in CACHE_EXPIRY_HOURS.keys():
                    self.clear_cache(dt)
                logger.info("Cleared all cache")
    
    def get_cache_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics and information.
        
        Returns:
            Dictionary with cache information for each data type
        """
        info = {}
        
        for data_type in CACHE_EXPIRY_HOURS.keys():
            cache_dir = self.cache_dir / data_type
            if cache_dir.exists():
                files = list(cache_dir.glob("*.pkl"))
                total_size = sum(f.stat().st_size for f in files)
                valid_files = sum(1 for f in files if self._is_cache_valid(f, data_type))
                
                info[data_type] = {
                    'total_files': len(files),
                    'valid_files': valid_files,
                    'expired_files': len(files) - valid_files,
                    'total_size_mb': total_size / (1024 * 1024),
                    'cache_dir': str(cache_dir)
                }
            else:
                info[data_type] = {
                    'total_files': 0,
                    'valid_files': 0, 
                    'expired_files': 0,
                    'total_size_mb': 0,
                    'cache_dir': str(cache_dir)
                }
        
        return info


# Global cache manager instance
_cache_manager = CacheManager()


def load_historical_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    force_refresh: bool = False
) -> pd.DataFrame:
    """
    Load historical price data for a ticker with caching.
    
    Downloads data from Yahoo Finance and caches it locally. Subsequent calls
    with the same parameters return cached data if it hasn't expired.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        period: Period to fetch data for (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        start: Start date string (YYYY-MM-DD format) - alternative to period
        end: End date string (YYYY-MM-DD format) - alternative to period
        force_refresh: If True, bypass cache and download fresh data
        
    Returns:
        DataFrame with historical price data (Open, High, Low, Close, Volume, etc.)
        
    Raises:
        ValueError: If ticker is invalid or data cannot be retrieved
        ConnectionError: If unable to connect to Yahoo Finance
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    ticker = ticker.upper().strip()
    cache_params = {
        'ticker': ticker,
        'period': period,
        'interval': interval,
        'start': start,
        'end': end
    }
    
    # Check cache first unless force refresh is requested
    if not force_refresh:
        cached_data = _cache_manager.get_cached_data('historical', **cache_params)
        if cached_data is not None:
            return cached_data
    
    # Download fresh data
    logger.info(f"Downloading historical data for {ticker}")
    try:
        ticker_obj = yf.Ticker(ticker)
        
        if start and end:
            data = ticker_obj.history(start=start, end=end, interval=interval)
        else:
            data = ticker_obj.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Cache the data
        _cache_manager.save_cached_data(data, 'historical', **cache_params)
        
        logger.info(f"Successfully loaded {len(data)} records for {ticker}")
        return data
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {e}")
        if "No data found" in str(e):
            raise ValueError(f"Ticker '{ticker}' not found or no data available")
        else:
            raise ConnectionError(f"Failed to download data for {ticker}: {e}")


def load_option_chain(
    ticker: str,
    expiration: Optional[str] = None,
    force_refresh: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Load option chain data for a ticker with caching.
    
    Downloads option chain data from Yahoo Finance and caches it locally.
    Returns both call and put option data.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        expiration: Expiration date in YYYY-MM-DD format. If None, gets nearest expiration
        force_refresh: If True, bypass cache and download fresh data
        
    Returns:
        Dictionary with 'calls' and 'puts' DataFrames containing option data
        
    Raises:
        ValueError: If ticker is invalid or option data cannot be retrieved
        ConnectionError: If unable to connect to Yahoo Finance
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    ticker = ticker.upper().strip()
    cache_params = {
        'ticker': ticker,
        'expiration': expiration
    }
    
    # Check cache first unless force refresh is requested
    if not force_refresh:
        cached_data = _cache_manager.get_cached_data('options', **cache_params)
        if cached_data is not None:
            return cached_data
    
    # Download fresh data
    logger.info(f"Downloading option chain for {ticker}")
    try:
        ticker_obj = yf.Ticker(ticker)
        
        # Get available expiration dates
        options_expirations = ticker_obj.options
        if not options_expirations:
            raise ValueError(f"No options available for ticker {ticker}")
        
        # Use specified expiration or nearest one
        if expiration:
            if expiration not in options_expirations:
                raise ValueError(f"Expiration {expiration} not available for {ticker}. "
                               f"Available: {list(options_expirations)}")
            target_expiration = expiration
        else:
            target_expiration = options_expirations[0]  # Nearest expiration
        
        # Get option chain for the expiration
        option_chain = ticker_obj.option_chain(target_expiration)
        
        result = {
            'calls': option_chain.calls,
            'puts': option_chain.puts,
            'expiration': target_expiration,
            'available_expirations': list(options_expirations)
        }
        
        # Cache the data
        _cache_manager.save_cached_data(result, 'options', **cache_params)
        
        logger.info(f"Successfully loaded option chain for {ticker}, expiration {target_expiration}")
        return result
        
    except Exception as e:
        logger.error(f"Error downloading option chain for {ticker}: {e}")
        if "No options available" in str(e):
            raise ValueError(f"No options available for ticker '{ticker}'")
        else:
            raise ConnectionError(f"Failed to download option chain for {ticker}: {e}")


def get_risk_free_rate(
    duration: str = "3m",
    force_refresh: bool = False
) -> float:
    """
    Get current risk-free rate (Treasury rate) with caching.
    
    Fetches current Treasury rates and caches them locally.
    
    Args:
        duration: Duration for Treasury rate ('1m', '3m', '6m', '1y', '2y', '5y', '10y', '30y')
        force_refresh: If True, bypass cache and download fresh data
        
    Returns:
        Risk-free rate as a decimal (e.g., 0.05 for 5%)
        
    Raises:
        ValueError: If duration is invalid
        ConnectionError: If unable to fetch rate data
    """
    valid_durations = ['1m', '3m', '6m', '1y', '2y', '5y', '10y', '30y']
    if duration not in valid_durations:
        raise ValueError(f"Invalid duration {duration}. Valid options: {valid_durations}")
    
    cache_params = {'duration': duration}
    
    # Check cache first unless force refresh is requested
    if not force_refresh:
        cached_data = _cache_manager.get_cached_data('risk_free', **cache_params)
        if cached_data is not None:
            return cached_data
    
    # Download fresh data - using Treasury rate symbols
    logger.info(f"Fetching {duration} Treasury rate")
    try:
        # Map duration to Yahoo Finance symbols for Treasury rates
        symbol_map = {
            '1m': '^IRX',    # 13 Week Treasury Bill
            '3m': '^IRX',    # 13 Week Treasury Bill  
            '6m': '^IRX',    # 13 Week Treasury Bill (using as proxy)
            '1y': '^TNX',    # 10 Year Treasury Note (using as proxy)
            '2y': '^TNX',    # 10 Year Treasury Note (using as proxy)
            '5y': '^TNX',    # 10 Year Treasury Note (using as proxy) 
            '10y': '^TNX',   # 10 Year Treasury Note
            '30y': '^TYX'    # 30 Year Treasury Bond
        }
        
        symbol = symbol_map[duration]
        ticker_obj = yf.Ticker(symbol)
        
        # Get recent data
        data = ticker_obj.history(period="5d", interval="1d")
        if data.empty:
            raise ValueError(f"No Treasury rate data available")
        
        # Get most recent closing value and convert from percentage to decimal
        rate = float(data['Close'].iloc[-1]) / 100.0
        
        # Cache the rate
        _cache_manager.save_cached_data(rate, 'risk_free', **cache_params)
        
        logger.info(f"Risk-free rate ({duration}): {rate:.4f} ({rate*100:.2f}%)")
        return rate
        
    except Exception as e:
        logger.error(f"Error fetching risk-free rate: {e}")
        # Fallback to a reasonable default rate
        default_rates = {
            '1m': 0.04, '3m': 0.04, '6m': 0.042, '1y': 0.045,
            '2y': 0.048, '5y': 0.05, '10y': 0.05, '30y': 0.052
        }
        rate = default_rates[duration]
        logger.warning(f"Using default risk-free rate ({duration}): {rate:.4f}")
        return rate


def get_dividend_yield(
    ticker: str,
    force_refresh: bool = False
) -> float:
    """
    Get dividend yield for a ticker with caching.
    
    Fetches current dividend yield information and caches it locally.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        force_refresh: If True, bypass cache and download fresh data
        
    Returns:
        Dividend yield as a decimal (e.g., 0.02 for 2%)
        
    Raises:
        ValueError: If ticker is invalid
        ConnectionError: If unable to fetch dividend data
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    
    ticker = ticker.upper().strip()
    cache_params = {'ticker': ticker}
    
    # Check cache first unless force refresh is requested
    if not force_refresh:
        cached_data = _cache_manager.get_cached_data('dividends', **cache_params)
        if cached_data is not None:
            return cached_data
    
    # Download fresh data
    logger.info(f"Fetching dividend yield for {ticker}")
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        # Get dividend yield from info
        dividend_yield = info.get('dividendYield', 0.0)
        if dividend_yield is None:
            dividend_yield = 0.0
        
        # Cache the yield
        _cache_manager.save_cached_data(dividend_yield, 'dividends', **cache_params)
        
        logger.info(f"Dividend yield for {ticker}: {dividend_yield:.4f} ({dividend_yield*100:.2f}%)")
        return dividend_yield
        
    except Exception as e:
        logger.error(f"Error fetching dividend yield for {ticker}: {e}")
        # Return 0 yield if unable to fetch
        logger.warning(f"Assuming zero dividend yield for {ticker}")
        return 0.0


def clear_cache(data_type: Optional[str] = None) -> None:
    """
    Clear cached data.
    
    Args:
        data_type: Specific data type to clear ('historical', 'options', 'risk_free', 'dividends'),
                  or None to clear all cache
    """
    _cache_manager.clear_cache(data_type)


def get_cache_info() -> Dict[str, Dict[str, Any]]:
    """
    Get cache statistics and information.
    
    Returns:
        Dictionary with cache information for each data type including:
        - total_files: Number of cached files
        - valid_files: Number of non-expired cached files
        - expired_files: Number of expired cached files  
        - total_size_mb: Total cache size in megabytes
        - cache_dir: Cache directory path
    """
    return _cache_manager.get_cache_info()


def configure_cache(cache_dir: Optional[str] = None, expiry_hours: Optional[Dict[str, int]] = None) -> None:
    """
    Configure cache settings.
    
    Args:
        cache_dir: New cache directory path
        expiry_hours: Dictionary of expiry hours for each data type
    """
    global _cache_manager, CACHE_EXPIRY_HOURS
    
    if expiry_hours:
        CACHE_EXPIRY_HOURS.update(expiry_hours)
    
    if cache_dir:
        _cache_manager = CacheManager(cache_dir)


# Convenience functions for backward compatibility
def get_stock_data(ticker: str, **kwargs) -> pd.DataFrame:
    """Alias for load_historical_data for backward compatibility."""
    return load_historical_data(ticker, **kwargs)


def get_options_data(ticker: str, **kwargs) -> Dict[str, pd.DataFrame]:
    """Alias for load_option_chain for backward compatibility."""
    return load_option_chain(ticker, **kwargs)