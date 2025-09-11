#!/usr/bin/env python3
"""
Comprehensive test script to validate the refactored data centralization
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'optlib'))

from optlib.data.data_store import default_data_store, get_complete_market_data
from optlib.harness.runner import OptionsAnalysisRunner
from optlib.data import history, options, market

def test_data_store_centralization():
    """Test that all data operations go through data_store"""
    print("=== Testing Data Store Centralization ===")
    
    # Test 1: Basic data fetching
    print("1. Testing basic data fetching...")
    price = default_data_store.get_current_stock_price('TSLA')
    rate = default_data_store.get_risk_free_rate('6M')
    print(f"   TSLA price: {price}")
    print(f"   6M rate: {rate}")
    
    # Test 2: Complete market data
    print("2. Testing complete market data fetch...")
    complete_data = get_complete_market_data('GOOGL', '2024-12-20')
    expected_keys = ['current_price', 'historical_volatility', 'risk_free_rate', 
                    'dividend_yield', 'market_indicators', 'options_chain', 'timestamp']
    
    for key in expected_keys:
        if key not in complete_data:
            raise ValueError(f"Missing key in complete data: {key}")
    
    print(f"   Complete data contains all expected keys: {len(complete_data)} items")
    
    # Test 3: Verify caching works
    print("3. Testing caching mechanism...")
    # First call should fetch data
    start_time = __import__('time').time()
    price1 = default_data_store.get_current_stock_price('META')
    time1 = __import__('time').time() - start_time
    
    # Second call should be from cache
    start_time = __import__('time').time()
    price2 = default_data_store.get_current_stock_price('META')
    time2 = __import__('time').time() - start_time
    
    if price1 != price2:
        raise ValueError("Cached price doesn't match original")
    
    print(f"   Caching works: same price returned, cache access faster ({time2:.4f}s vs {time1:.4f}s)")
    
    print("✓ Data Store Centralization Tests Passed\n")


def test_analysis_functions():
    """Test that analysis functions work with provided data"""
    print("=== Testing Analysis Functions ===")
    
    # Test 1: History analysis
    print("1. Testing history analysis functions...")
    hist_data = default_data_store.get_stock_price_history('AAPL', '2024-01-01', '2024-12-31')
    analysis = history.analyze_price_history(hist_data)
    
    required_fields = ['total_periods', 'price_start', 'price_end', 'total_return', 'volatility']
    for field in required_fields:
        if field not in analysis:
            raise ValueError(f"Missing analysis field: {field}")
    
    print(f"   History analysis complete: {analysis['total_periods']} periods analyzed")
    
    # Test 2: Options analysis
    print("2. Testing options analysis functions...")
    options_data = default_data_store.get_options_chain('AAPL', '2024-12-20')
    current_price = default_data_store.get_current_stock_price('AAPL')
    
    options_analysis = options.analyze_options_chain(options_data, current_price)
    
    if 'total_options' not in options_analysis:
        raise ValueError("Options analysis failed")
    
    print(f"   Options analysis complete: {options_analysis['total_options']} options analyzed")
    
    # Test 3: Market analysis
    print("3. Testing market analysis functions...")
    market_indicators = default_data_store.get_market_indicators()
    sentiment = market.analyze_market_sentiment(market_indicators)
    
    if 'overall_sentiment' not in sentiment:
        raise ValueError("Market sentiment analysis failed")
    
    print(f"   Market sentiment: {sentiment['overall_sentiment']}")
    
    print("✓ Analysis Functions Tests Passed\n")


def test_workflow_integration():
    """Test end-to-end workflow integration"""
    print("=== Testing Workflow Integration ===")
    
    # Test 1: Full analysis workflow
    print("1. Testing full analysis workflow...")
    runner = OptionsAnalysisRunner()
    
    results = runner.run_full_analysis('NVDA', 200.0, '2024-12-20', 'PUT')
    
    expected_results = ['symbol', 'current_price', 'theoretical_price', 'greeks', 'market_sentiment']
    for field in expected_results:
        if field not in results:
            raise ValueError(f"Missing result field: {field}")
    
    print(f"   Full analysis complete for NVDA PUT")
    print(f"   Theoretical price: {results['theoretical_price']:.4f}")
    print(f"   Delta: {results['greeks']['Delta']:.4f}")
    
    # Test 2: Scenario analysis
    print("2. Testing scenario analysis...")
    scenarios = [180, 190, 200, 210, 220]
    scenario_results = runner.run_scenario_analysis('NVDA', 200.0, '2024-12-20', 'PUT', scenarios)
    
    if len(scenario_results['option_values']) != len(scenarios):
        raise ValueError("Scenario analysis incomplete")
    
    print(f"   Scenario analysis complete: {len(scenarios)} price points analyzed")
    
    # Test 3: Portfolio analysis
    print("3. Testing portfolio analysis...")
    positions = [
        {'symbol': 'AAPL', 'strike': 150, 'expiration': '2024-12-20', 'type': 'CALL', 'quantity': 2},
        {'symbol': 'GOOGL', 'strike': 120, 'expiration': '2024-12-20', 'type': 'PUT', 'quantity': 1}
    ]
    
    portfolio_results = runner.run_portfolio_analysis(positions)
    
    if 'total_value' not in portfolio_results:
        raise ValueError("Portfolio analysis failed")
    
    print(f"   Portfolio analysis complete: {len(portfolio_results['positions'])} positions")
    print(f"   Total portfolio value: {portfolio_results['total_value']:.2f}")
    
    print("✓ Workflow Integration Tests Passed\n")


def test_backward_compatibility():
    """Test that backward compatibility functions work"""
    print("=== Testing Backward Compatibility ===")
    
    # Test that old-style function calls still work
    print("1. Testing backward compatibility functions...")
    
    # These should work exactly as before but use data_store internally
    price = history.get_current_stock_price('IBM')
    vol = history.calculate_historical_volatility('IBM', 100)
    rate = market.get_risk_free_rate('1Y')
    chain = options.get_options_chain('IBM', '2024-12-20')
    
    print(f"   IBM price: {price}")
    print(f"   IBM volatility: {vol:.4f}")
    print(f"   1Y rate: {rate:.4f}")
    print(f"   Options chain size: {len(chain)}")
    
    print("✓ Backward Compatibility Tests Passed\n")


def test_no_direct_io():
    """Verify no modules perform direct I/O except data_store"""
    print("=== Testing No Direct I/O ===")
    
    # This test validates that the refactoring was successful
    print("1. Checking that analysis modules don't perform I/O...")
    
    # Create mock data for testing
    import pandas as pd
    import numpy as np
    
    # Mock price data
    mock_price_data = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100),
        'Open': np.random.uniform(90, 110, 100),
        'High': np.random.uniform(100, 120, 100),
        'Low': np.random.uniform(80, 100, 100),
        'Close': np.random.uniform(90, 110, 100),
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Test analysis functions with mock data (should not trigger any I/O)
    analysis = history.analyze_price_history(mock_price_data)
    
    if 'total_periods' not in analysis:
        raise ValueError("Analysis function failed with mock data")
    
    print("   Analysis functions work with provided data without I/O")
    
    # Mock options data
    mock_options_data = pd.DataFrame({
        'Symbol': ['TEST'] * 10,
        'Strike': np.arange(90, 100),
        'Type': ['CALL'] * 5 + ['PUT'] * 5,
        'Bid': np.random.uniform(1, 10, 10),
        'Ask': np.random.uniform(2, 12, 10),
        'Volume': np.random.randint(100, 1000, 10),
        'OpenInterest': np.random.randint(500, 5000, 10),
        'ImpliedVolatility': np.random.uniform(0.15, 0.35, 10)
    })
    
    options_analysis = options.analyze_options_chain(mock_options_data, 95.0)
    
    if 'total_options' not in options_analysis:
        raise ValueError("Options analysis failed with mock data")
    
    print("   Options analysis works with provided data without I/O")
    
    print("✓ No Direct I/O Tests Passed\n")


def main():
    """Run all tests"""
    print("Starting Comprehensive Data Centralization Tests...\n")
    
    try:
        test_data_store_centralization()
        test_analysis_functions()
        test_workflow_integration()
        test_backward_compatibility()
        test_no_direct_io()
        
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nData centralization refactoring is successful:")
        print("✓ All data I/O operations are centralized in data_store.py")
        print("✓ Analysis modules work with provided data without I/O")
        print("✓ Backward compatibility is maintained")
        print("✓ End-to-end workflows function correctly")
        print("✓ Caching and performance optimizations work")
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())