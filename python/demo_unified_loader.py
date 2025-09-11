#!/usr/bin/env python3
"""
Demo script showing how to use the unified data loader for option pricing.

This script demonstrates the complete workflow:
1. Loading market data through the unified loader
2. Pricing options with live data
3. Calculating Greeks
4. Using cached data for efficiency
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def demo_with_mock_data():
    """Demo using mock data when network is not available."""
    print("="*70)
    print("UNIFIED DATA LOADER DEMO WITH MOCK DATA")
    print("="*70)
    
    # Import our modules
    from src import black_scholes as bs
    from optlib.data.unified_loader import CacheManager
    
    # Create mock market data
    print("\\n1. Creating mock market data...")
    
    # Mock historical data
    dates = pd.date_range(start='2024-01-01', periods=252, freq='D')
    prices = [100]
    for i in range(251):
        # Simple random walk
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        prices.append(prices[-1] * (1 + change))
    
    mock_data = pd.DataFrame({
        'Open': prices[:-1],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'Close': prices[:-1],
        'Volume': [np.random.randint(1000000, 5000000) for _ in range(251)]
    }, index=dates[:-1])
    
    current_price = prices[-2]
    print(f"✓ Mock data created: {len(mock_data)} days, current price: ${current_price:.2f}")
    
    # Calculate historical volatility
    returns = mock_data['Close'].pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)  # Annualized
    print(f"✓ Historical volatility: {volatility:.2%}")
    
    print("\\n2. Testing cache functionality...")
    
    # Test caching
    cache_manager = CacheManager("/tmp/demo_cache")
    cache_manager.save_cached_data(mock_data, 'historical', ticker='DEMO', period='1y')
    
    # Retrieve from cache
    cached_data = cache_manager.get_cached_data('historical', ticker='DEMO', period='1y')
    if cached_data is not None:
        print("✓ Data successfully cached and retrieved")
    else:
        print("✗ Cache test failed")
        return
    
    print("\\n3. Option pricing with mock data...")
    
    # Option parameters
    ticker = "DEMO"
    strike = current_price * 1.05  # 5% OTM
    time_to_expiry = 0.25  # 3 months
    risk_free_rate = 0.05  # 5% annual
    
    print(f"Pricing options for {ticker}:")
    print(f"  Current price: ${current_price:.2f}")
    print(f"  Strike price: ${strike:.2f}")
    print(f"  Time to expiry: {time_to_expiry:.3f} years")
    print(f"  Risk-free rate: {risk_free_rate:.2%}")
    print(f"  Volatility: {volatility:.2%}")
    
    # Price call option
    call_price = bs.black_scholes_call_price(
        S=current_price,
        K=strike,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=volatility
    )
    
    # Price put option
    put_price = bs.black_scholes_put_price(
        S=current_price,
        K=strike,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=volatility
    )
    
    print(f"\\nOption prices:")
    print(f"  Call price: ${call_price:.4f}")
    print(f"  Put price: ${put_price:.4f}")
    
    # Calculate Greeks for call
    call_greeks = bs.calculate_all_greeks(
        S=current_price,
        K=strike,
        T=time_to_expiry,
        r=risk_free_rate,
        sigma=volatility,
        option_type='call'
    )
    
    print(f"\\nCall option Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek}: {value:.6f}")
    
    print("\\n4. Performance comparison...")
    
    import time
    
    # Simulate "first call" (no cache)
    cache_manager.clear_cache('historical')
    start_time = time.time()
    
    # Simulate data loading and processing
    time.sleep(0.01)  # Simulate network delay
    processed_data = mock_data.copy()
    processed_volatility = processed_data['Close'].pct_change().std() * np.sqrt(252)
    
    first_call_time = time.time() - start_time
    
    # Cache the result
    cache_manager.save_cached_data(processed_data, 'historical', ticker='DEMO', period='1y')
    
    # Simulate "cached call"
    start_time = time.time()
    cached_result = cache_manager.get_cached_data('historical', ticker='DEMO', period='1y')
    if cached_result is not None:
        cached_volatility = cached_result['Close'].pct_change().std() * np.sqrt(252)
    cached_call_time = time.time() - start_time
    
    speedup = first_call_time / cached_call_time if cached_call_time > 0 else float('inf')
    
    print(f"Performance comparison:")
    print(f"  First call (with processing): {first_call_time:.4f} seconds")
    print(f"  Cached call: {cached_call_time:.4f} seconds")
    print(f"  Speedup: {speedup:.1f}x faster")
    
    print("\\n5. Cache statistics...")
    cache_info = cache_manager.get_cache_info()
    for data_type, info in cache_info.items():
        if info['total_files'] > 0:
            print(f"{data_type.capitalize()}:")
            print(f"  Files: {info['valid_files']}/{info['total_files']} valid")
            print(f"  Size: {info['total_size_mb']:.3f} MB")
    
    print("\\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\\nKey benefits of the unified loader:")
    print("✓ Centralized data access - no duplicate download logic")
    print("✓ Automatic caching - faster subsequent access")
    print("✓ Thread-safe operations - safe for parallel execution")
    print("✓ Error handling - graceful fallbacks and validation")
    print("✓ Configurable cache expiry - fresh data when needed")
    print("✓ Easy integration - works with existing Black-Scholes code")

def show_usage_examples():
    """Show code examples of how to use the unified loader."""
    print("\\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    
    examples = '''
# Example 1: Load historical data
from optlib.data import load_historical_data

data = load_historical_data('AAPL', period='1y')
current_price = data['Close'].iloc[-1]

# Example 2: Get option chain
from optlib.data import load_option_chain

options = load_option_chain('AAPL')
calls = options['calls']
puts = options['puts']

# Example 3: Get risk-free rate
from optlib.data import get_risk_free_rate

rf_rate = get_risk_free_rate('3m')  # 3-month Treasury rate

# Example 4: Complete option pricing workflow
from optlib.data import (
    load_historical_data, 
    get_risk_free_rate, 
    get_dividend_yield
)
from src import black_scholes as bs

# Get market data
ticker = 'AAPL'
data = load_historical_data(ticker, period='1y')
current_price = data['Close'].iloc[-1]

# Calculate volatility
returns = data['Close'].pct_change().dropna()
volatility = returns.std() * np.sqrt(252)

# Get risk-free rate and dividend yield
risk_free_rate = get_risk_free_rate('3m')
dividend_yield = get_dividend_yield(ticker)

# Adjust price for dividends
time_to_expiry = 0.25
dividend_pv = current_price * dividend_yield * time_to_expiry
adjusted_price = current_price - dividend_pv

# Price option
strike = 150.0
call_price = bs.black_scholes_call_price(
    S=adjusted_price,
    K=strike,
    T=time_to_expiry,
    r=risk_free_rate,
    sigma=volatility
)

# Example 5: Cache management
from optlib.data import clear_cache, get_cache_info

# Clear specific cache
clear_cache('historical')

# Clear all cache
clear_cache()

# Get cache statistics
cache_info = get_cache_info()
    '''
    
    print(examples)

def main():
    """Main demo function."""
    print("Welcome to the Unified Data Loader Demo!")
    print("This demo shows the key features without requiring network access.")
    
    try:
        demo_with_mock_data()
        show_usage_examples()
        
        print("\\n🎉 Demo completed successfully!")
        print("\\nThe unified loader is ready for use with live market data.")
        print("Simply run the modules with network access to fetch real data from Yahoo Finance.")
        
    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)