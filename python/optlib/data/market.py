# optlib/data/market.py
"""
Market data fetching module for risk-free rates, market indicators, etc.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


def get_risk_free_rate(duration: str = "3M", cache_dir: str = "cache") -> float:
    """
    Fetch risk-free interest rate (typically Treasury rates).
    
    In a real implementation, this would fetch from FRED, Treasury.gov, or financial APIs.
    
    Args:
        duration: Duration for the rate ("1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y")
        cache_dir: Directory to store cached data
        
    Returns:
        Risk-free rate as a decimal (e.g., 0.05 for 5%)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"risk_free_rates_{datetime.now().strftime('%Y-%m-%d')}.json")
    
    # Check if we have today's rates cached
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            rates_data = json.load(f)
        if duration in rates_data:
            return rates_data[duration]
    else:
        rates_data = {}
    
    print(f"Fetching risk-free rate for {duration} duration...")
    
    # Simulate fetching from Treasury or FRED API
    # In reality, would use requests to fetch from:
    # - FRED API: https://fred.stlouisfed.org/series/DGS3MO (3-month)
    # - Treasury: https://www.treasury.gov/resource-center/data-chart-center/interest-rates/
    
    # Generate realistic rates based on typical yield curve
    base_rates = {
        "1M": 0.045,
        "3M": 0.048,
        "6M": 0.050,
        "1Y": 0.052,
        "2Y": 0.055,
        "5Y": 0.058,
        "10Y": 0.062,
        "30Y": 0.065
    }
    
    # Add some random variation to simulate market changes
    np.random.seed(int(datetime.now().strftime('%Y%m%d')))  # Consistent for the day
    variation = np.random.normal(0, 0.005)  # Small daily variation
    
    if duration in base_rates:
        rate = base_rates[duration] + variation
        rate = max(0.001, rate)  # Ensure positive rate
    else:
        # Default to 3-month rate
        rate = base_rates["3M"] + variation
        rate = max(0.001, rate)
    
    # Cache the rate
    rates_data[duration] = round(rate, 6)
    with open(cache_file, 'w') as f:
        json.dump(rates_data, f, indent=2)
    
    return round(rate, 6)


def get_market_indicators(cache_dir: str = "cache") -> Dict[str, Any]:
    """
    Fetch key market indicators (VIX, market indices, etc.).
    
    Args:
        cache_dir: Directory to store cached data
        
    Returns:
        Dictionary with market indicators
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"market_indicators_{datetime.now().strftime('%Y-%m-%d')}.json")
    
    # Check if we have today's indicators cached
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    print("Fetching market indicators...")
    
    # Simulate fetching market indicators
    # In reality, would fetch from financial APIs like Alpha Vantage, Yahoo Finance, etc.
    
    np.random.seed(int(datetime.now().strftime('%Y%m%d')))
    
    # Generate realistic market data
    indicators = {
        "VIX": round(20 + np.random.normal(0, 5), 2),  # Volatility index
        "SPX": round(4500 + np.random.normal(0, 100), 2),  # S&P 500
        "NDX": round(15000 + np.random.normal(0, 500), 2),  # NASDAQ 100
        "DJI": round(35000 + np.random.normal(0, 800), 2),  # Dow Jones
        "USD_INDEX": round(103 + np.random.normal(0, 2), 2),  # Dollar Index
        "GOLD": round(2000 + np.random.normal(0, 50), 2),  # Gold price
        "OIL_WTI": round(80 + np.random.normal(0, 10), 2),  # Oil price
        "BONDS_10Y": get_risk_free_rate("10Y"),  # 10-year Treasury
        "last_updated": datetime.now().isoformat()
    }
    
    # Ensure realistic bounds
    indicators["VIX"] = max(10, min(80, indicators["VIX"]))
    indicators["SPX"] = max(3000, indicators["SPX"])
    indicators["NDX"] = max(10000, indicators["NDX"])
    indicators["DJI"] = max(25000, indicators["DJI"])
    indicators["USD_INDEX"] = max(90, min(120, indicators["USD_INDEX"]))
    indicators["GOLD"] = max(1500, indicators["GOLD"])
    indicators["OIL_WTI"] = max(40, min(150, indicators["OIL_WTI"]))
    
    # Cache the indicators
    with open(cache_file, 'w') as f:
        json.dump(indicators, f, indent=2)
    
    return indicators


def get_dividend_yield(symbol: str, cache_dir: str = "cache") -> float:
    """
    Get dividend yield for a stock symbol.
    
    Args:
        symbol: Stock symbol
        cache_dir: Cache directory
        
    Returns:
        Annual dividend yield as decimal
    """
    cache_file = os.path.join(cache_dir, f"{symbol}_dividend.json")
    
    # Check cache
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            div_data = json.load(f)
        
        # Check if data is recent (within 30 days)
        last_update = datetime.fromisoformat(div_data.get('last_updated', '2000-01-01'))
        if (datetime.now() - last_update).days < 30:
            return div_data.get('dividend_yield', 0.0)
    
    print(f"Fetching dividend information for {symbol}...")
    
    # Simulate dividend data fetching
    # In reality, would fetch from financial APIs
    
    # Generate realistic dividend yields based on symbol hash for consistency
    np.random.seed(hash(symbol) % 2**32)
    
    # Most stocks have dividend yields between 0-6%
    dividend_yield = np.random.exponential(0.02)  # Exponential distribution
    dividend_yield = min(dividend_yield, 0.08)  # Cap at 8%
    
    # Some stocks don't pay dividends
    if np.random.random() < 0.3:  # 30% chance of no dividend
        dividend_yield = 0.0
    
    div_data = {
        'symbol': symbol,
        'dividend_yield': round(dividend_yield, 4),
        'last_updated': datetime.now().isoformat()
    }
    
    # Cache the data
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(div_data, f, indent=2)
    
    return div_data['dividend_yield']


def get_earnings_calendar(symbol: str, cache_dir: str = "cache") -> Optional[str]:
    """
    Get next earnings announcement date for a symbol.
    
    Args:
        symbol: Stock symbol
        cache_dir: Cache directory
        
    Returns:
        Next earnings date in 'YYYY-MM-DD' format, or None if unknown
    """
    # Simulate earnings calendar data
    # In reality, would fetch from financial APIs
    
    np.random.seed(hash(symbol) % 2**32)
    
    # Generate a random earnings date in the next 90 days
    days_until_earnings = np.random.randint(1, 91)
    earnings_date = datetime.now() + timedelta(days=days_until_earnings)
    
    return earnings_date.strftime('%Y-%m-%d')