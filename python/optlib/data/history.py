# optlib/data/history.py
"""
Historical stock price data analysis module - REFACTORED to use centralized data_store

This module now focuses on data analysis and processing functions that accept
data as parameters, rather than fetching data directly. All data I/O is handled
by the centralized data_store module.
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from .data_store import default_data_store


def analyze_price_history(price_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze historical price data that has been provided as input.
    
    Args:
        price_data: DataFrame with columns: Date, Open, High, Low, Close, Volume
        
    Returns:
        Dictionary with analysis results
    """
    if price_data.empty or 'Close' not in price_data.columns:
        return {}
    
    # Ensure Date column is datetime
    if 'Date' in price_data.columns:
        price_data = price_data.copy()
        price_data['Date'] = pd.to_datetime(price_data['Date'])
        price_data = price_data.sort_values('Date')
    
    close_prices = price_data['Close']
    
    # Calculate returns
    returns = close_prices.pct_change().dropna()
    
    analysis = {
        'total_periods': len(price_data),
        'price_start': float(close_prices.iloc[0]),
        'price_end': float(close_prices.iloc[-1]),
        'price_min': float(close_prices.min()),
        'price_max': float(close_prices.max()),
        'total_return': float((close_prices.iloc[-1] / close_prices.iloc[0]) - 1),
        'average_return': float(returns.mean()),
        'volatility': float(returns.std()),
        'annualized_volatility': float(returns.std() * np.sqrt(252)),
        'sharpe_ratio': float(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0,
        'max_drawdown': calculate_max_drawdown(close_prices),
        'average_volume': float(price_data['Volume'].mean()) if 'Volume' in price_data.columns else None
    }
    
    return analysis


def calculate_max_drawdown(prices: pd.Series) -> float:
    """
    Calculate maximum drawdown from price series.
    
    Args:
        prices: Series of price values
        
    Returns:
        Maximum drawdown as decimal
    """
    peak = prices.cummax()
    drawdown = (prices - peak) / peak
    return float(drawdown.min())


def calculate_returns_statistics(price_data: pd.DataFrame, period: str = 'daily') -> Dict[str, float]:
    """
    Calculate detailed return statistics from price data.
    
    Args:
        price_data: DataFrame with price data
        period: 'daily', 'weekly', or 'monthly'
        
    Returns:
        Dictionary with return statistics
    """
    if price_data.empty or 'Close' not in price_data.columns:
        return {}
    
    price_data = price_data.copy()
    price_data['Date'] = pd.to_datetime(price_data['Date'])
    price_data = price_data.sort_values('Date').set_index('Date')
    
    # Resample based on period
    if period == 'weekly':
        resampled = price_data.resample('W')['Close'].last()
    elif period == 'monthly':
        resampled = price_data.resample('M')['Close'].last()
    else:  # daily
        resampled = price_data['Close']
    
    returns = resampled.pct_change().dropna()
    
    return {
        'mean_return': float(returns.mean()),
        'std_return': float(returns.std()),
        'skewness': float(returns.skew()),
        'kurtosis': float(returns.kurtosis()),
        'min_return': float(returns.min()),
        'max_return': float(returns.max()),
        'positive_days': int((returns > 0).sum()),
        'negative_days': int((returns < 0).sum()),
        'total_days': len(returns)
    }


def calculate_moving_averages(price_data: pd.DataFrame, windows: list = [20, 50, 200]) -> pd.DataFrame:
    """
    Calculate moving averages for given windows.
    
    Args:
        price_data: DataFrame with price data
        windows: List of window sizes for moving averages
        
    Returns:
        DataFrame with original data plus moving averages
    """
    if price_data.empty or 'Close' not in price_data.columns:
        return price_data
    
    result = price_data.copy()
    
    for window in windows:
        result[f'MA_{window}'] = result['Close'].rolling(window=window).mean()
    
    return result


def detect_support_resistance_levels(price_data: pd.DataFrame, window: int = 20) -> Dict[str, list]:
    """
    Detect potential support and resistance levels from price data.
    
    Args:
        price_data: DataFrame with OHLC data
        window: Window size for level detection
        
    Returns:
        Dictionary with support and resistance levels
    """
    if price_data.empty:
        return {'support': [], 'resistance': []}
    
    highs = price_data['High'].rolling(window=window, center=True).max()
    lows = price_data['Low'].rolling(window=window, center=True).min()
    
    # Find local maxima (resistance)
    resistance_levels = []
    for i in range(window, len(price_data) - window):
        if price_data['High'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(float(price_data['High'].iloc[i]))
    
    # Find local minima (support)
    support_levels = []
    for i in range(window, len(price_data) - window):
        if price_data['Low'].iloc[i] == lows.iloc[i]:
            support_levels.append(float(price_data['Low'].iloc[i]))
    
    return {
        'support': sorted(list(set(support_levels))),
        'resistance': sorted(list(set(resistance_levels)), reverse=True)
    }


# === CONVENIENCE FUNCTIONS THAT USE DATA_STORE ===
# These functions provide easy access to data_store functionality for backward compatibility

def get_stock_price_history(symbol: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
    """
    Get historical stock price data using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_stock_price_history(symbol, start_date, end_date, **kwargs)


def get_current_stock_price(symbol: str, **kwargs) -> float:
    """
    Get current stock price using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_current_stock_price(symbol, **kwargs)


def calculate_historical_volatility(symbol: str, period_days: int = 252, **kwargs) -> float:
    """
    Calculate historical volatility using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.calculate_historical_volatility(symbol, period_days, **kwargs)