# harness/runner.py
"""
Test harness and runner for comprehensive option pricing with live market data.

This module demonstrates how to use the unified data loader for complete
option pricing workflows. It shows how all market data access should go
through the centralized caching system rather than direct API calls.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Black-Scholes calculations
from src import black_scholes as bs

# Import unified data loader and other modules
from optlib.data import (
    load_historical_data, 
    load_option_chain, 
    get_risk_free_rate, 
    get_dividend_yield,
    clear_cache,
    get_cache_info
)
import history
import options
import market


class OptionPricingRunner:
    """
    Comprehensive option pricing runner using unified data loader.
    
    This class demonstrates the complete workflow for option pricing using
    live market data, ensuring all data access goes through the unified
    caching system for efficiency and consistency.
    """
    
    def __init__(self, cache_warmup: bool = True):
        """
        Initialize the option pricing runner.
        
        Args:
            cache_warmup: Whether to perform cache warmup on initialization
        """
        self.cache_warmup = cache_warmup
        if cache_warmup:
            print("Initializing option pricing runner with cache warmup...")
            self._warmup_cache()
    
    def _warmup_cache(self):
        """Warm up cache with common market data."""
        try:
            # Pre-load common risk-free rates
            get_risk_free_rate("3m")
            get_risk_free_rate("1y")
            print("Cache warmup completed successfully")
        except Exception as e:
            print(f"Cache warmup warning: {e}")
    
    def get_market_data_for_option(
        self,
        ticker: str,
        time_to_expiry: float,
        risk_free_duration: str = "3m"
    ) -> Dict[str, float]:
        """
        Get all market data needed for option pricing.
        
        Args:
            ticker: Stock ticker symbol
            time_to_expiry: Time to expiry in years
            risk_free_duration: Duration for risk-free rate
            
        Returns:
            Dictionary with all market data for option pricing
        """
        print(f"Fetching market data for {ticker}...")
        
        # Use market module which uses unified loader
        market_data = market.get_market_data_for_pricing(
            ticker=ticker,
            time_to_expiry=time_to_expiry,
            risk_free_duration=risk_free_duration
        )
        
        return market_data
    
    def price_option_with_live_data(
        self,
        ticker: str,
        strike: float,
        time_to_expiry: float,
        option_type: str = "call",
        risk_free_duration: str = "3m",
        use_implied_vol: bool = False
    ) -> Dict[str, Any]:
        """
        Price an option using live market data.
        
        Args:
            ticker: Stock ticker symbol
            strike: Option strike price
            time_to_expiry: Time to expiry in years
            option_type: 'call' or 'put'
            risk_free_duration: Duration for risk-free rate
            use_implied_vol: Whether to use implied volatility if available
            
        Returns:
            Dictionary with pricing results and market data
        """
        print(f"\\nPricing {option_type} option for {ticker}")
        print(f"Strike: ${strike}, Time to expiry: {time_to_expiry:.3f} years")
        
        # Get market data using unified loader
        market_data = self.get_market_data_for_option(
            ticker=ticker,
            time_to_expiry=time_to_expiry,
            risk_free_duration=risk_free_duration
        )
        
        # Use historical volatility by default
        volatility = market_data['volatility']
        
        # Try to get implied volatility if requested
        if use_implied_vol:
            try:
                option_data = options.get_option_chain(ticker)
                calls = option_data['calls']
                puts = option_data['puts']
                
                # Find closest strike for implied volatility
                if option_type == "call" and not calls.empty:
                    closest_option = calls.iloc[(calls['strike'] - strike).abs().argsort()[:1]]
                    if not closest_option.empty and 'impliedVolatility' in closest_option.columns:
                        iv = closest_option['impliedVolatility'].iloc[0]
                        if pd.notna(iv) and iv > 0:
                            volatility = float(iv)
                            print(f"Using implied volatility: {volatility:.4f}")
                
                elif option_type == "put" and not puts.empty:
                    closest_option = puts.iloc[(puts['strike'] - strike).abs().argsort()[:1]]
                    if not closest_option.empty and 'impliedVolatility' in closest_option.columns:
                        iv = closest_option['impliedVolatility'].iloc[0]
                        if pd.notna(iv) and iv > 0:
                            volatility = float(iv)
                            print(f"Using implied volatility: {volatility:.4f}")
                            
            except Exception as e:
                print(f"Could not get implied volatility, using historical: {e}")
        
        # Calculate option price using Black-Scholes
        S = market_data['adjusted_price']  # Dividend-adjusted price
        K = strike
        T = time_to_expiry
        r = market_data['risk_free_rate']
        sigma = volatility
        
        if option_type == "call":
            price = bs.black_scholes_call_price(S, K, T, r, sigma)
        else:
            price = bs.black_scholes_put_price(S, K, T, r, sigma)
        
        # Calculate Greeks
        greeks = bs.calculate_all_greeks(S, K, T, r, sigma, option_type)
        
        # Prepare results
        results = {
            'ticker': ticker,
            'option_type': option_type,
            'strike': strike,
            'time_to_expiry': time_to_expiry,
            'theoretical_price': price,
            'greeks': greeks,
            'market_data': market_data,
            'volatility_used': volatility,
            'volatility_type': 'implied' if use_implied_vol and volatility != market_data['volatility'] else 'historical',
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def run_option_strategy_analysis(
        self,
        ticker: str,
        strategies: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze multiple option strategies.
        
        Args:
            ticker: Stock ticker symbol
            strategies: List of strategy definitions
            
        Returns:
            Dictionary with strategy analysis results
        """
        print(f"\\nAnalyzing option strategies for {ticker}")
        
        strategy_results = []
        
        for i, strategy in enumerate(strategies):
            print(f"\\nStrategy {i+1}: {strategy.get('name', 'Unnamed')}")
            
            legs = strategy.get('legs', [])
            total_cost = 0
            strategy_greeks = {
                'Delta': 0, 'Gamma': 0, 'Vega': 0, 'Theta': 0, 'Rho': 0
            }
            
            leg_results = []
            
            for leg in legs:
                option_result = self.price_option_with_live_data(
                    ticker=ticker,
                    strike=leg['strike'],
                    time_to_expiry=leg['time_to_expiry'],
                    option_type=leg['option_type'],
                    use_implied_vol=leg.get('use_implied_vol', False)
                )
                
                position_size = leg.get('position', 1)  # +1 for long, -1 for short
                leg_cost = option_result['theoretical_price'] * position_size
                total_cost += leg_cost
                
                # Add up Greeks
                for greek_name, greek_value in option_result['greeks'].items():
                    strategy_greeks[greek_name] += greek_value * position_size
                
                leg_results.append({
                    'leg': leg,
                    'option_result': option_result,
                    'leg_cost': leg_cost
                })
            
            strategy_result = {
                'strategy': strategy,
                'legs': leg_results,
                'total_cost': total_cost,
                'net_greeks': strategy_greeks,
                'max_profit': strategy.get('max_profit', 'Unlimited'),
                'max_loss': strategy.get('max_loss', total_cost),
                'breakeven': strategy.get('breakeven', 'Calculate')
            }
            
            strategy_results.append(strategy_result)
        
        return {
            'ticker': ticker,
            'strategies': strategy_results,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def run_comprehensive_analysis(
        self,
        tickers: List[str],
        strikes_range: Tuple[float, float] = (0.9, 1.1),
        time_to_expiry: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run comprehensive analysis across multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            strikes_range: Range for strike prices (as ratio of current price)
            time_to_expiry: Time to expiry in years
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        print(f"\\nRunning comprehensive analysis for {len(tickers)} tickers")
        
        analysis_results = {}
        
        for ticker in tickers:
            print(f"\\n{'='*50}")
            print(f"Analyzing {ticker}")
            print(f"{'='*50}")
            
            try:
                # Get current market data
                market_data = market.get_market_data_for_pricing(ticker, time_to_expiry)
                current_price = market_data['current_price']
                
                # Calculate strike prices around current price
                min_strike = current_price * strikes_range[0]
                max_strike = current_price * strikes_range[1]
                strikes = np.linspace(min_strike, max_strike, 5)
                
                # Price options at different strikes
                call_prices = []
                put_prices = []
                
                for strike in strikes:
                    # Price call
                    call_result = self.price_option_with_live_data(
                        ticker=ticker,
                        strike=strike,
                        time_to_expiry=time_to_expiry,
                        option_type="call",
                        use_implied_vol=True
                    )
                    call_prices.append(call_result)
                    
                    # Price put
                    put_result = self.price_option_with_live_data(
                        ticker=ticker,
                        strike=strike,
                        time_to_expiry=time_to_expiry,
                        option_type="put",
                        use_implied_vol=True
                    )
                    put_prices.append(put_result)
                
                # Get additional analysis
                volatility_analysis = market.analyze_volatility_term_structure(ticker)
                option_analysis = options.analyze_option_chain(ticker)
                
                analysis_results[ticker] = {
                    'market_data': market_data,
                    'call_prices': call_prices,
                    'put_prices': put_prices,
                    'volatility_analysis': volatility_analysis,
                    'option_analysis': option_analysis,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                analysis_results[ticker] = {'error': str(e)}
        
        return analysis_results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display formatted results."""
        if 'theoretical_price' in results:
            # Single option result
            print(f"\\n{'='*60}")
            print(f"OPTION PRICING RESULTS")
            print(f"{'='*60}")
            print(f"Ticker: {results['ticker']}")
            print(f"Option Type: {results['option_type'].capitalize()}")
            print(f"Strike Price: ${results['strike']:.2f}")
            print(f"Time to Expiry: {results['time_to_expiry']:.3f} years")
            print(f"Theoretical Price: ${results['theoretical_price']:.4f}")
            print(f"Volatility Used: {results['volatility_used']:.4f} ({results['volatility_type']})")
            
            print(f"\\nGreeks:")
            for name, value in results['greeks'].items():
                print(f"  {name}: {value:.6f}")
            
            print(f"\\nMarket Data:")
            market_data = results['market_data']
            print(f"  Current Price: ${market_data['current_price']:.2f}")
            print(f"  Adjusted Price: ${market_data['adjusted_price']:.2f}")
            print(f"  Risk-Free Rate: {market_data['risk_free_rate']:.4f} ({market_data['risk_free_rate']*100:.2f}%)")
            print(f"  Dividend Yield: {market_data['dividend_yield']:.4f} ({market_data['dividend_yield']*100:.2f}%)")
            print(f"  Historical Volatility: {market_data['volatility']:.4f} ({market_data['volatility']*100:.2f}%)")
            
        elif 'strategies' in results:
            # Strategy analysis result
            print(f"\\n{'='*60}")
            print(f"STRATEGY ANALYSIS RESULTS")
            print(f"{'='*60}")
            
            for i, strategy_result in enumerate(results['strategies']):
                strategy = strategy_result['strategy']
                print(f"\\nStrategy {i+1}: {strategy.get('name', 'Unnamed')}")
                print(f"Total Cost: ${strategy_result['total_cost']:.4f}")
                print(f"Net Greeks: {strategy_result['net_greeks']}")
    
    def run_performance_test(self, iterations: int = 10) -> Dict[str, float]:
        """
        Run performance test of the unified loader caching system.
        
        Args:
            iterations: Number of iterations to test
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"\\nRunning performance test with {iterations} iterations...")
        
        test_ticker = "AAPL"
        timings = {
            'first_call': [],
            'cached_calls': [],
            'cache_hit_ratio': 0
        }
        
        # Clear cache to start fresh
        clear_cache()
        
        # Time first call (no cache)
        start_time = time.time()
        market_data1 = market.get_market_data_for_pricing(test_ticker, 0.25)
        first_call_time = time.time() - start_time
        timings['first_call'].append(first_call_time)
        
        # Time subsequent calls (should hit cache)
        for i in range(iterations - 1):
            start_time = time.time()
            market_data2 = market.get_market_data_for_pricing(test_ticker, 0.25)
            cached_call_time = time.time() - start_time
            timings['cached_calls'].append(cached_call_time)
        
        # Calculate metrics
        avg_first = np.mean(timings['first_call'])
        avg_cached = np.mean(timings['cached_calls'])
        speedup = avg_first / avg_cached if avg_cached > 0 else 0
        
        cache_info = get_cache_info()
        
        performance_results = {
            'avg_first_call_time': avg_first,
            'avg_cached_call_time': avg_cached,
            'speedup_factor': speedup,
            'cache_info': cache_info
        }
        
        print(f"Performance Results:")
        print(f"  First call time: {avg_first:.4f} seconds")
        print(f"  Cached call time: {avg_cached:.4f} seconds")
        print(f"  Speedup factor: {speedup:.2f}x")
        
        return performance_results


def main():
    """Main execution function demonstrating all functionality."""
    print("="*80)
    print("BLACK-SCHOLES OPTION PRICER WITH UNIFIED DATA LOADER")
    print("="*80)
    
    # Initialize runner
    runner = OptionPricingRunner(cache_warmup=True)
    
    try:
        # Example 1: Single option pricing
        print("\\n" + "="*60)
        print("EXAMPLE 1: Single Option Pricing")
        print("="*60)
        
        option_result = runner.price_option_with_live_data(
            ticker="AAPL",
            strike=150.0,
            time_to_expiry=0.25,  # 3 months
            option_type="call",
            use_implied_vol=True
        )
        runner.display_results(option_result)
        
        # Example 2: Strategy analysis
        print("\\n" + "="*60)
        print("EXAMPLE 2: Option Strategy Analysis")
        print("="*60)
        
        strategies = [
            {
                'name': 'Long Straddle',
                'legs': [
                    {'strike': 150, 'time_to_expiry': 0.25, 'option_type': 'call', 'position': 1},
                    {'strike': 150, 'time_to_expiry': 0.25, 'option_type': 'put', 'position': 1}
                ]
            },
            {
                'name': 'Iron Condor',
                'legs': [
                    {'strike': 140, 'time_to_expiry': 0.25, 'option_type': 'put', 'position': 1},
                    {'strike': 145, 'time_to_expiry': 0.25, 'option_type': 'put', 'position': -1},
                    {'strike': 155, 'time_to_expiry': 0.25, 'option_type': 'call', 'position': -1},
                    {'strike': 160, 'time_to_expiry': 0.25, 'option_type': 'call', 'position': 1}
                ]
            }
        ]
        
        strategy_results = runner.run_option_strategy_analysis("AAPL", strategies)
        runner.display_results(strategy_results)
        
        # Example 3: Performance test
        print("\\n" + "="*60)
        print("EXAMPLE 3: Cache Performance Test")
        print("="*60)
        
        performance_results = runner.run_performance_test(iterations=5)
        
        # Example 4: Multi-ticker analysis (limited for demo)
        print("\\n" + "="*60)
        print("EXAMPLE 4: Multi-Ticker Analysis")
        print("="*60)
        
        tickers = ["AAPL", "MSFT"]  # Limited for demo
        comprehensive_results = runner.run_comprehensive_analysis(
            tickers=tickers,
            time_to_expiry=0.25
        )
        
        for ticker, result in comprehensive_results.items():
            if 'error' not in result:
                print(f"\\n{ticker} Analysis Summary:")
                print(f"  Current Price: ${result['market_data']['current_price']:.2f}")
                print(f"  Volatility: {result['market_data']['volatility']:.2%}")
                print(f"  Call Options Analyzed: {len(result['call_prices'])}")
                print(f"  Put Options Analyzed: {len(result['put_prices'])}")
            else:
                print(f"\\n{ticker}: {result['error']}")
        
        # Show final cache info
        print("\\n" + "="*60)
        print("FINAL CACHE STATISTICS")
        print("="*60)
        cache_info = get_cache_info()
        for data_type, info in cache_info.items():
            print(f"{data_type.capitalize()}:")
            print(f"  Files: {info['valid_files']}/{info['total_files']} valid")
            print(f"  Size: {info['total_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()