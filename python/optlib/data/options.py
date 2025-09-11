# optlib/data/options.py
"""
Options chain data fetching and caching module
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional


def get_options_chain(symbol: str, expiration_date: str, cache_dir: str = "cache") -> pd.DataFrame:
    """
    Fetch options chain data for a given symbol and expiration date.
    
    In a real implementation, this would fetch from CBOE, TD Ameritrade API, or similar.
    For this example, we'll simulate the options chain data.
    
    Args:
        symbol: Underlying stock symbol
        expiration_date: Options expiration date in 'YYYY-MM-DD' format
        cache_dir: Directory to store cached data
        
    Returns:
        DataFrame with options chain data
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = os.path.join(cache_dir, f"{symbol}_{expiration_date}_options.json")
    
    # Check if cached data exists
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    
    print(f"Fetching options chain for {symbol} expiring {expiration_date}...")
    
    # Get current stock price for realistic option pricing
    from .history import get_current_stock_price
    current_price = get_current_stock_price(symbol, cache_dir)
    
    # Generate realistic options chain
    options_data = []
    
    # Generate range of strikes around current price
    strike_range = np.arange(
        current_price * 0.8,  # 20% below
        current_price * 1.3,  # 30% above
        max(1.0, current_price * 0.05)  # Strike spacing
    )
    
    # Calculate time to expiration
    exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
    time_to_exp = (exp_date - datetime.now()).days / 365.0
    
    if time_to_exp <= 0:
        time_to_exp = 1/365  # Minimum 1 day
    
    # Risk-free rate and volatility assumptions
    risk_free_rate = 0.05
    volatility = 0.25
    
    np.random.seed(42)  # For consistent data
    
    for strike in strike_range:
        strike = round(strike, 2)
        
        # Calculate theoretical option prices using simplified Black-Scholes
        # Import our existing Black-Scholes functions
        try:
            from src.black_scholes import black_scholes_call_price, black_scholes_put_price
            call_price = black_scholes_call_price(current_price, strike, time_to_exp, risk_free_rate, volatility)
            put_price = black_scholes_put_price(current_price, strike, time_to_exp, risk_free_rate, volatility)
        except ImportError:
            # Fallback: simplified pricing
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = max(0.01, time_to_exp * volatility * current_price * 0.1)
            call_price = intrinsic_call + time_value
            put_price = intrinsic_put + time_value
        
        # Add some randomness to bid-ask spreads
        call_bid_ask_spread = call_price * 0.02 * np.random.uniform(0.5, 1.5)
        put_bid_ask_spread = put_price * 0.02 * np.random.uniform(0.5, 1.5)
        
        # Simulate volume and open interest
        call_volume = int(np.random.exponential(100))
        put_volume = int(np.random.exponential(80))
        call_open_interest = int(np.random.exponential(500))
        put_open_interest = int(np.random.exponential(400))
        
        # Call option data
        options_data.append({
            'Symbol': symbol,
            'Expiration': expiration_date,
            'Strike': strike,
            'Type': 'CALL',
            'Bid': round(max(0.01, call_price - call_bid_ask_spread/2), 2),
            'Ask': round(call_price + call_bid_ask_spread/2, 2),
            'Last': round(call_price + np.random.normal(0, call_bid_ask_spread/4), 2),
            'Volume': call_volume,
            'OpenInterest': call_open_interest,
            'ImpliedVolatility': round(volatility + np.random.normal(0, 0.05), 4)
        })
        
        # Put option data
        options_data.append({
            'Symbol': symbol,
            'Expiration': expiration_date,
            'Strike': strike,
            'Type': 'PUT',
            'Bid': round(max(0.01, put_price - put_bid_ask_spread/2), 2),
            'Ask': round(put_price + put_bid_ask_spread/2, 2),
            'Last': round(put_price + np.random.normal(0, put_bid_ask_spread/4), 2),
            'Volume': put_volume,
            'OpenInterest': put_open_interest,
            'ImpliedVolatility': round(volatility + np.random.normal(0, 0.05), 4)
        })
    
    # Cache the data
    with open(cache_file, 'w') as f:
        json.dump(options_data, f, indent=2)
    
    return pd.DataFrame(options_data)


def get_option_quotes(symbol: str, strike: float, expiration_date: str, option_type: str, cache_dir: str = "cache") -> Dict[str, Any]:
    """
    Get specific option quote data.
    
    Args:
        symbol: Underlying symbol
        strike: Strike price
        expiration_date: Expiration date
        option_type: 'CALL' or 'PUT'
        cache_dir: Cache directory
        
    Returns:
        Dictionary with option quote data
    """
    options_chain = get_options_chain(symbol, expiration_date, cache_dir)
    
    # Filter for specific option
    option_data = options_chain[
        (options_chain['Strike'] == strike) & 
        (options_chain['Type'] == option_type.upper())
    ]
    
    if option_data.empty:
        return {}
    
    return option_data.iloc[0].to_dict()


def get_available_expirations(symbol: str, cache_dir: str = "cache") -> List[str]:
    """
    Get list of available expiration dates for a symbol.
    
    Args:
        symbol: Stock symbol
        cache_dir: Cache directory
        
    Returns:
        List of expiration dates in 'YYYY-MM-DD' format
    """
    # Generate typical expiration dates (3rd Friday of each month)
    current_date = datetime.now()
    expirations = []
    
    for month_offset in range(1, 13):  # Next 12 months
        target_month = current_date.month + month_offset
        target_year = current_date.year
        
        if target_month > 12:
            target_month -= 12
            target_year += 1
        
        # Find 3rd Friday of the month
        first_day = datetime(target_year, target_month, 1)
        first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
        third_friday = first_friday + timedelta(days=14)
        
        expirations.append(third_friday.strftime('%Y-%m-%d'))
    
    return expirations