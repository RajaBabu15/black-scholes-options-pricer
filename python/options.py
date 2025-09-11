# options.py
"""
Options data analysis module using the unified data loader.

This module demonstrates how to access option chain data through the
unified loader instead of directly calling Yahoo Finance APIs. All data
access goes through the centralized caching system.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from optlib.data import load_option_chain, get_cache_info
import numpy as np
from datetime import datetime, timedelta


def get_option_chain(
    ticker: str,
    expiration: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Get option chain data for a ticker.
    
    This function uses the unified loader instead of directly accessing
    Yahoo Finance, ensuring all data goes through the caching system.
    
    Args:
        ticker: Stock ticker symbol
        expiration: Optional expiration date (YYYY-MM-DD format)
        
    Returns:
        Dictionary with 'calls' and 'puts' DataFrames
    """
    # All data access goes through unified loader
    return load_option_chain(ticker=ticker, expiration=expiration)


def find_options_by_moneyness(
    ticker: str,
    current_price: float,
    moneyness_range: Tuple[float, float] = (0.9, 1.1),
    expiration: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Find options within a specific moneyness range.
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price
        moneyness_range: Tuple of (min_moneyness, max_moneyness)
        expiration: Optional expiration date
        
    Returns:
        Dictionary with filtered calls and puts DataFrames
    """
    # Use unified loader for option data
    option_data = load_option_chain(ticker=ticker, expiration=expiration)
    
    min_strike = current_price * moneyness_range[0]
    max_strike = current_price * moneyness_range[1]
    
    # Filter by strike price range
    calls = option_data['calls']
    puts = option_data['puts']
    
    filtered_calls = calls[
        (calls['strike'] >= min_strike) & 
        (calls['strike'] <= max_strike)
    ].copy()
    
    filtered_puts = puts[
        (puts['strike'] >= min_strike) & 
        (puts['strike'] <= max_strike)
    ].copy()
    
    # Add moneyness calculation
    if not filtered_calls.empty:
        filtered_calls['moneyness'] = filtered_calls['strike'] / current_price
    if not filtered_puts.empty:
        filtered_puts['moneyness'] = filtered_puts['strike'] / current_price
    
    return {
        'calls': filtered_calls,
        'puts': filtered_puts,
        'expiration': option_data['expiration'],
        'available_expirations': option_data['available_expirations']
    }


def calculate_implied_volatility_stats(
    ticker: str,
    expiration: Optional[str] = None
) -> Dict[str, float]:
    """
    Calculate implied volatility statistics from option chain.
    
    Args:
        ticker: Stock ticker symbol
        expiration: Optional expiration date
        
    Returns:
        Dictionary with IV statistics
    """
    # Use unified loader for option data
    option_data = load_option_chain(ticker=ticker, expiration=expiration)
    
    calls = option_data['calls']
    puts = option_data['puts']
    
    stats = {}
    
    # Calculate IV stats for calls
    if not calls.empty and 'impliedVolatility' in calls.columns:
        call_iv = calls['impliedVolatility'].dropna()
        if not call_iv.empty:
            stats.update({
                'call_iv_mean': float(call_iv.mean()),
                'call_iv_median': float(call_iv.median()),
                'call_iv_std': float(call_iv.std()),
                'call_iv_min': float(call_iv.min()),
                'call_iv_max': float(call_iv.max())
            })
    
    # Calculate IV stats for puts
    if not puts.empty and 'impliedVolatility' in puts.columns:
        put_iv = puts['impliedVolatility'].dropna()
        if not put_iv.empty:
            stats.update({
                'put_iv_mean': float(put_iv.mean()),
                'put_iv_median': float(put_iv.median()),
                'put_iv_std': float(put_iv.std()),
                'put_iv_min': float(put_iv.min()),
                'put_iv_max': float(put_iv.max())
            })
    
    return stats


def find_liquid_options(
    ticker: str,
    min_volume: int = 10,
    min_open_interest: int = 50,
    expiration: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Find liquid options based on volume and open interest criteria.
    
    Args:
        ticker: Stock ticker symbol
        min_volume: Minimum daily volume
        min_open_interest: Minimum open interest
        expiration: Optional expiration date
        
    Returns:
        Dictionary with filtered calls and puts DataFrames
    """
    # Use unified loader for option data
    option_data = load_option_chain(ticker=ticker, expiration=expiration)
    
    calls = option_data['calls']
    puts = option_data['puts']
    
    # Filter by liquidity criteria
    liquid_calls = calls[
        (calls.get('volume', 0) >= min_volume) &
        (calls.get('openInterest', 0) >= min_open_interest)
    ].copy()
    
    liquid_puts = puts[
        (puts.get('volume', 0) >= min_volume) &
        (puts.get('openInterest', 0) >= min_open_interest)
    ].copy()
    
    return {
        'calls': liquid_calls,
        'puts': liquid_puts,
        'expiration': option_data['expiration'],
        'available_expirations': option_data['available_expirations'],
        'filter_criteria': {
            'min_volume': min_volume,
            'min_open_interest': min_open_interest
        }
    }


def analyze_option_chain(
    ticker: str,
    current_price: Optional[float] = None,
    expiration: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of option chain data.
    
    Args:
        ticker: Stock ticker symbol
        current_price: Current stock price (will fetch if not provided)
        expiration: Optional expiration date
        
    Returns:
        Dictionary with comprehensive option analysis
    """
    # Use unified loader for option data
    option_data = load_option_chain(ticker=ticker, expiration=expiration)
    
    calls = option_data['calls']
    puts = option_data['puts']
    
    analysis = {
        'ticker': ticker,
        'expiration': option_data['expiration'],
        'available_expirations': option_data['available_expirations']
    }
    
    # Basic statistics
    analysis['call_count'] = len(calls)
    analysis['put_count'] = len(puts)
    
    if not calls.empty:
        analysis['call_volume_total'] = int(calls.get('volume', 0).sum())
        analysis['call_open_interest_total'] = int(calls.get('openInterest', 0).sum())
        analysis['call_strike_range'] = {
            'min': float(calls['strike'].min()),
            'max': float(calls['strike'].max())
        }
    
    if not puts.empty:
        analysis['put_volume_total'] = int(puts.get('volume', 0).sum())
        analysis['put_open_interest_total'] = int(puts.get('openInterest', 0).sum())
        analysis['put_strike_range'] = {
            'min': float(puts['strike'].min()),
            'max': float(puts['strike'].max())
        }
    
    # Calculate put/call ratio
    if analysis.get('call_volume_total', 0) > 0:
        put_call_ratio = analysis.get('put_volume_total', 0) / analysis['call_volume_total']
        analysis['put_call_ratio'] = round(put_call_ratio, 3)
    
    # Implied volatility analysis
    iv_stats = calculate_implied_volatility_stats(ticker, expiration)
    if iv_stats:
        analysis['implied_volatility'] = iv_stats
    
    return analysis


def get_options_by_expiration(ticker: str) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Get option chains for all available expirations.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary mapping expiration dates to option chain data
    """
    # First get available expirations
    initial_data = load_option_chain(ticker=ticker)
    available_expirations = initial_data['available_expirations']
    
    all_chains = {}
    for expiration in available_expirations:
        try:
            chain_data = load_option_chain(ticker=ticker, expiration=expiration)
            all_chains[expiration] = {
                'calls': chain_data['calls'],
                'puts': chain_data['puts']
            }
        except Exception as e:
            print(f"Error loading option chain for {expiration}: {e}")
            continue
    
    return all_chains


# Example usage and testing  
if __name__ == "__main__":
    print("=== Options Data Example ===")
    
    try:
        # This will use the unified loader with caching
        ticker = "AAPL"
        option_data = get_option_chain(ticker)
        
        print(f"Option chain for {ticker}")
        print(f"Expiration: {option_data['expiration']}")
        print(f"Calls: {len(option_data['calls'])} contracts")
        print(f"Puts: {len(option_data['puts'])} contracts")
        
        # Analyze the option chain
        analysis = analyze_option_chain(ticker)
        print(f"\nOption analysis: {analysis}")
        
        # Find liquid options
        liquid_options = find_liquid_options(ticker, min_volume=5, min_open_interest=10)
        print(f"\nLiquid calls: {len(liquid_options['calls'])}")
        print(f"Liquid puts: {len(liquid_options['puts'])}")
        
        print(f"\nCache info: {get_cache_info()}")
        
    except Exception as e:
        print(f"Error: {e}")