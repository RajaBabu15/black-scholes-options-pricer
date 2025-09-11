# optlib/data/history.py
"""
Historical stock price data fetching and caching module
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any


def get_stock_price_history(symbol: str, start_date: str, end_date: str, cache_dir: str = "cache") -> pd.DataFrame:
    """
    Fetch historical stock price data for a given symbol and date range.
    
    In a real implementation, this would fetch from Yahoo Finance, Alpha Vantage, or similar.
    For this example, we'll simulate the data fetching with caching.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cache_dir: Directory to store cached data
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{symbol}_{start_date}_{end_date}_history.json")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    # Simulate API call to fetch data (in real implementation would use requests/yfinance)
    print(f"Fetching historical data for {symbol} from {start_date} to {end_date}...")
    
    # Generate mock historical data
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    dates = pd.date_range(start=start, end=end, freq='D')
    
    # Filter out weekends (simple approach)
    business_dates = [d for d in dates if d.weekday() < 5]
    
    # Generate realistic price data
    base_price = 100.0
    np.random.seed(42)  # For reproducible data
    
    data = []
    current_price = base_price
    
    for date in business_dates:
        # Random walk with some volatility
        daily_return = np.random.normal(0.0005, 0.02)  # Small positive drift, 2% daily vol
        current_price *= (1 + daily_return)
        
        # Generate OHLC data
        high = current_price * (1 + abs(np.random.normal(0, 0.01)))
        low = current_price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = current_price * (1 + np.random.normal(0, 0.005))
        volume = int(np.random.normal(1000000, 200000))
        
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(current_price, 2),
            'Volume': max(volume, 100000)
        })
    
    # Cache the data
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    return pd.DataFrame(data)


def get_current_stock_price(symbol: str, cache_dir: str = "cache") -> float:
    """
    Get current stock price for a symbol.
    
    Args:
        symbol: Stock symbol
        cache_dir: Directory to check for cached data
        
    Returns:
        Current stock price
    """
    # In real implementation, would fetch real-time price
    # For now, try to get latest from cached historical data
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        cache_files = [f for f in os.listdir(cache_dir) if f.startswith(symbol) and f.endswith('_history.json')]
    except FileNotFoundError:
        cache_files = []
    
    if cache_files:
        # Get the most recent cache file
        latest_file = sorted(cache_files)[-1]
        cache_path = os.path.join(cache_dir, latest_file)
        
        with open(cache_path, 'r') as f:
            data = json.load(f)
        
        if data:
            return data[-1]['Close']
    
    # Fallback: return simulated current price
    print(f"Fetching current price for {symbol}...")
    np.random.seed(hash(symbol) % 2**32)  # Deterministic but symbol-dependent
    return round(100.0 * (1 + np.random.normal(0, 0.1)), 2)


def calculate_historical_volatility(symbol: str, period_days: int = 252, cache_dir: str = "cache") -> float:
    """
    Calculate historical volatility from stock price data.
    
    Args:
        symbol: Stock symbol
        period_days: Number of days to look back for volatility calculation
        cache_dir: Cache directory
        
    Returns:
        Annualized historical volatility
    """
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=period_days + 30)).strftime('%Y-%m-%d')
    
    df = get_stock_price_history(symbol, start_date, end_date, cache_dir)
    
    if len(df) < 2:
        # Fallback volatility
        return 0.20
    
    # Calculate daily returns
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df['Returns'] = df['Close'].pct_change().dropna()
    
    # Calculate annualized volatility
    daily_volatility = df['Returns'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)  # Assuming 252 trading days per year
    
    return float(annualized_volatility)