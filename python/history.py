# history.py
"""
Historical data analysis module using the unified data loader.

This module demonstrates how to access historical price data through the
unified loader instead of directly calling Yahoo Finance APIs. All data
access goes through the centralized caching system.
"""

import pandas as pd
from typing import List, Optional, Dict, Any
from optlib.data import load_historical_data, get_cache_info
import numpy as np


def get_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Get historical price data for a ticker.
    
    This function uses the unified loader instead of directly accessing
    Yahoo Finance, ensuring all data goes through the caching system.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for data
        interval: Data interval
        
    Returns:
        DataFrame with historical price data
    """
    # All data access goes through unified loader
    return load_historical_data(ticker=ticker, period=period, interval=interval)


def calculate_returns(ticker: str, period: str = "1y") -> pd.Series:
    """
    Calculate daily returns for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for data
        
    Returns:
        Series with daily returns
    """
    # Use unified loader for data access
    data = load_historical_data(ticker=ticker, period=period)
    return data['Close'].pct_change().dropna()


def calculate_volatility(ticker: str, period: str = "1y", annualized: bool = True) -> float:
    """
    Calculate historical volatility for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for data
        annualized: Whether to annualize the volatility
        
    Returns:
        Historical volatility
    """
    returns = calculate_returns(ticker, period)
    vol = returns.std()
    
    if annualized:
        # Annualize assuming 252 trading days
        vol = vol * np.sqrt(252)
    
    return vol


def get_price_statistics(ticker: str, period: str = "1y") -> Dict[str, float]:
    """
    Get comprehensive price statistics for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Time period for data
        
    Returns:
        Dictionary with various price statistics
    """
    # Use unified loader for data access
    data = load_historical_data(ticker=ticker, period=period)
    
    if data.empty:
        return {}
    
    close_prices = data['Close']
    returns = calculate_returns(ticker, period)
    
    stats = {
        'current_price': float(close_prices.iloc[-1]),
        'min_price': float(close_prices.min()),
        'max_price': float(close_prices.max()),
        'avg_price': float(close_prices.mean()),
        'price_std': float(close_prices.std()),
        'total_return': float((close_prices.iloc[-1] / close_prices.iloc[0]) - 1),
        'volatility': calculate_volatility(ticker, period),
        'avg_volume': float(data['Volume'].mean()),
        'max_volume': float(data['Volume'].max()),
        'trading_days': len(data)
    }
    
    return stats


def compare_tickers(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Compare multiple tickers' performance.
    
    Args:
        tickers: List of ticker symbols
        period: Time period for comparison
        
    Returns:
        DataFrame with comparison statistics
    """
    comparison_data = []
    
    for ticker in tickers:
        try:
            stats = get_price_statistics(ticker, period)
            stats['ticker'] = ticker
            comparison_data.append(stats)
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            continue
    
    if not comparison_data:
        return pd.DataFrame()
    
    return pd.DataFrame(comparison_data).set_index('ticker')


# Example usage and testing
if __name__ == "__main__":
    # Example: Get historical data for AAPL
    print("=== Historical Data Example ===")
    
    try:
        # This will use the unified loader with caching
        aapl_data = get_price_history("AAPL", period="3mo")
        print(f"Loaded {len(aapl_data)} records for AAPL")
        print(f"Latest close price: ${aapl_data['Close'].iloc[-1]:.2f}")
        
        # Calculate volatility
        vol = calculate_volatility("AAPL", period="3mo")
        print(f"3-month volatility: {vol:.2%}")
        
        # Get statistics  
        stats = get_price_statistics("AAPL", period="3mo")
        print(f"Price statistics: {stats}")
        
        print(f"\nCache info: {get_cache_info()}")
        
    except Exception as e:
        print(f"Error: {e}")