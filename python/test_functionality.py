#!/usr/bin/env python3
"""Test script for unified loader functionality without network access."""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_cache_functionality():
    """Test cache functionality with mock data."""
    print("Testing cache functionality...")
    
    from optlib.data.unified_loader import CacheManager
    
    # Create a test cache manager
    cache_manager = CacheManager("/tmp/test_cache")
    
    # Test data
    test_data = pd.DataFrame({
        'Close': [100 + i + np.random.randn() for i in range(30)],
        'Volume': [1000000 + i*10000 for i in range(30)]
    })
    
    # Test caching
    cache_manager.save_cached_data(test_data, 'historical', ticker='TEST', period='1mo')
    print("✓ Data saved to cache")
    
    # Test retrieval
    cached_data = cache_manager.get_cached_data('historical', ticker='TEST', period='1mo')
    if cached_data is not None and len(cached_data) == len(test_data):
        print("✓ Data retrieved from cache successfully")
    else:
        print("✗ Cache retrieval failed")
        return False
    
    # Test cache info
    cache_info = cache_manager.get_cache_info()
    print(f"✓ Cache info: {cache_info['historical']['total_files']} files")
    
    # Test cache clearing
    cache_manager.clear_cache('historical')
    cleared_data = cache_manager.get_cached_data('historical', ticker='TEST', period='1mo')
    if cleared_data is None:
        print("✓ Cache cleared successfully")
    else:
        print("✗ Cache clearing failed")
        return False
    
    return True

def test_threading_safety():
    """Test thread safety of cache operations."""
    print("Testing thread safety...")
    
    import threading
    import time
    from optlib.data.unified_loader import CacheManager
    
    cache_manager = CacheManager("/tmp/test_cache_threading")
    results = []
    
    def worker(worker_id):
        """Worker function for threading test."""
        try:
            # Each worker saves and retrieves data
            test_data = pd.DataFrame({
                'Close': [100 + worker_id + i for i in range(10)],
                'Worker': [worker_id] * 10
            })
            
            # Save data
            cache_manager.save_cached_data(
                test_data, 'historical', 
                ticker=f'TEST{worker_id}', worker_id=worker_id
            )
            
            # Brief delay to simulate real usage
            time.sleep(0.1)
            
            # Retrieve data
            retrieved = cache_manager.get_cached_data(
                'historical', ticker=f'TEST{worker_id}', worker_id=worker_id
            )
            
            if retrieved is not None and len(retrieved) == 10:
                results.append(True)
            else:
                results.append(False)
                
        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            results.append(False)
    
    # Create multiple threads
    threads = []
    num_threads = 5
    
    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Check results
    if all(results) and len(results) == num_threads:
        print(f"✓ Thread safety test passed ({num_threads} threads)")
        return True
    else:
        print(f"✗ Thread safety test failed ({sum(results)}/{num_threads} threads successful)")
        return False

def test_black_scholes_integration():
    """Test integration with Black-Scholes calculations."""
    print("Testing Black-Scholes integration...")
    
    try:
        from src import black_scholes as bs
        
        # Test parameters
        S = 100.0    # Current price
        K = 105.0    # Strike price
        T = 0.25     # Time to expiry (3 months)
        r = 0.05     # Risk-free rate (5%)
        sigma = 0.2  # Volatility (20%)
        
        # Calculate call price
        call_price = bs.black_scholes_call_price(S, K, T, r, sigma)
        print(f"✓ Call price: ${call_price:.4f}")
        
        # Calculate put price
        put_price = bs.black_scholes_put_price(S, K, T, r, sigma)
        print(f"✓ Put price: ${put_price:.4f}")
        
        # Calculate Greeks
        greeks = bs.calculate_all_greeks(S, K, T, r, sigma, 'call')
        print(f"✓ Greeks calculated: {list(greeks.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Black-Scholes integration error: {e}")
        return False

def test_module_imports():
    """Test that all modules import correctly."""
    print("Testing module imports...")
    
    try:
        # Test unified loader imports
        from optlib.data import unified_loader
        print("✓ Unified loader imported")
        
        # Test example modules
        import history
        print("✓ History module imported")
        
        import options
        print("✓ Options module imported")
        
        import market
        print("✓ Market module imported")
        
        from harness import runner
        print("✓ Runner module imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("UNIFIED LOADER FUNCTIONALITY TESTS")
    print("="*60)
    
    tests = [
        ("Module Imports", test_module_imports),
        ("Cache Functionality", test_cache_functionality),
        ("Thread Safety", test_threading_safety),
        ("Black-Scholes Integration", test_black_scholes_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results)
    print(f"\nOverall: {total_passed}/{len(tests)} tests passed")
    
    if total_passed == len(tests):
        print("🎉 All tests passed! Unified loader is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return total_passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)