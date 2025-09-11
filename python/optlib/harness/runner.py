# optlib/harness/runner.py
"""
Options analysis workflow runner that coordinates data fetching and analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Import data modules (will be refactored to use data_store)
from ..data.history import get_stock_price_history, get_current_stock_price, calculate_historical_volatility
from ..data.options import get_options_chain, get_option_quotes, get_available_expirations
from ..data.market import get_risk_free_rate, get_market_indicators, get_dividend_yield


class OptionsAnalysisRunner:
    """
    Coordinates the fetching of market data and running of options analysis.
    This class demonstrates the pattern of directly calling data modules
    that will be refactored to use centralized data_store.
    """
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.analysis_results = {}
    
    def run_full_analysis(self, symbol: str, strike: float, expiration_date: str, 
                         option_type: str = "CALL") -> Dict[str, Any]:
        """
        Run complete options analysis workflow.
        
        This method demonstrates the current pattern where runner directly
        calls various data modules. Will be refactored to use data_store.
        
        Args:
            symbol: Stock symbol
            strike: Option strike price
            expiration_date: Option expiration date
            option_type: 'CALL' or 'PUT'
            
        Returns:
            Complete analysis results
        """
        print(f"Running full options analysis for {symbol} {strike} {option_type} expiring {expiration_date}")
        
        # Fetch all required data (direct calls to data modules)
        results = {
            'symbol': symbol,
            'strike': strike,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'timestamp': datetime.now().isoformat()
        }
        
        # 1. Get current stock data
        print("1. Fetching current stock price...")
        current_price = get_current_stock_price(symbol, self.cache_dir)
        results['current_price'] = current_price
        
        # 2. Get historical data for volatility calculation
        print("2. Fetching historical data...")
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        historical_data = get_stock_price_history(symbol, start_date, end_date, self.cache_dir)
        results['historical_data_points'] = len(historical_data)
        
        # 3. Calculate historical volatility
        print("3. Calculating historical volatility...")
        hist_volatility = calculate_historical_volatility(symbol, 252, self.cache_dir)
        results['historical_volatility'] = hist_volatility
        
        # 4. Get market data
        print("4. Fetching market data...")
        risk_free_rate = get_risk_free_rate("3M", self.cache_dir)
        market_indicators = get_market_indicators(self.cache_dir)
        dividend_yield = get_dividend_yield(symbol, self.cache_dir)
        
        results['risk_free_rate'] = risk_free_rate
        results['market_vix'] = market_indicators.get('VIX')
        results['dividend_yield'] = dividend_yield
        
        # 5. Get options chain data
        print("5. Fetching options chain...")
        options_chain = get_options_chain(symbol, expiration_date, self.cache_dir)
        specific_option = get_option_quotes(symbol, strike, expiration_date, option_type, self.cache_dir)
        
        results['options_chain_size'] = len(options_chain)
        results['option_bid'] = specific_option.get('Bid')
        results['option_ask'] = specific_option.get('Ask')
        results['implied_volatility'] = specific_option.get('ImpliedVolatility')
        
        # 6. Calculate theoretical price and greeks
        print("6. Calculating theoretical values...")
        time_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365.0
        
        try:
            # Import Black-Scholes functions
            from src.black_scholes import (
                black_scholes_call_price, black_scholes_put_price,
                calculate_all_greeks
            )
            
            if option_type.upper() == 'CALL':
                theoretical_price = black_scholes_call_price(
                    current_price, strike, time_to_exp, risk_free_rate, hist_volatility
                )
            else:
                theoretical_price = black_scholes_put_price(
                    current_price, strike, time_to_exp, risk_free_rate, hist_volatility
                )
            
            greeks = calculate_all_greeks(
                current_price, strike, time_to_exp, risk_free_rate, hist_volatility, option_type.lower()
            )
            
            results['theoretical_price'] = theoretical_price
            results['greeks'] = greeks
            
            # Compare with market price
            if results['option_bid'] and results['option_ask']:
                market_mid = (results['option_bid'] + results['option_ask']) / 2
                results['market_mid_price'] = market_mid
                results['price_difference'] = theoretical_price - market_mid
                results['price_difference_pct'] = (theoretical_price - market_mid) / market_mid * 100
            
        except ImportError as e:
            print(f"Warning: Could not import Black-Scholes functions: {e}")
            results['theoretical_price'] = None
            results['greeks'] = None
        
        # Store results
        self.analysis_results[f"{symbol}_{strike}_{option_type}_{expiration_date}"] = results
        
        print("Analysis complete!")
        return results
    
    def generate_volatility_surface(self, symbol: str, expiration_dates: List[str]) -> pd.DataFrame:
        """
        Generate implied volatility surface data.
        
        This method shows another pattern of data access that will be refactored.
        """
        print(f"Generating volatility surface for {symbol}")
        
        surface_data = []
        
        for exp_date in expiration_dates:
            print(f"Processing expiration: {exp_date}")
            
            # Get options chain for this expiration
            options_chain = get_options_chain(symbol, exp_date, self.cache_dir)
            
            for _, option in options_chain.iterrows():
                if option['Type'] == 'CALL':  # Focus on calls for surface
                    time_to_exp = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days / 365.0
                    
                    surface_data.append({
                        'strike': option['Strike'],
                        'expiration': exp_date,
                        'time_to_exp': time_to_exp,
                        'implied_vol': option['ImpliedVolatility'],
                        'bid': option['Bid'],
                        'ask': option['Ask']
                    })
        
        return pd.DataFrame(surface_data)
    
    def run_scenario_analysis(self, symbol: str, strike: float, expiration_date: str,
                            option_type: str, price_scenarios: List[float]) -> Dict[str, List[float]]:
        """
        Run scenario analysis across different stock price levels.
        
        Another example of workflow that coordinates data and calculation.
        """
        print(f"Running scenario analysis for {symbol}")
        
        # Get required market data
        risk_free_rate = get_risk_free_rate("3M", self.cache_dir)
        hist_volatility = calculate_historical_volatility(symbol, 252, self.cache_dir)
        
        time_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365.0
        
        scenario_results = {
            'prices': price_scenarios,
            'option_values': [],
            'deltas': [],
            'gammas': []
        }
        
        try:
            from src.black_scholes import (
                black_scholes_call_price, black_scholes_put_price,
                delta, gamma
            )
            
            for price in price_scenarios:
                if option_type.upper() == 'CALL':
                    option_value = black_scholes_call_price(
                        price, strike, time_to_exp, risk_free_rate, hist_volatility
                    )
                else:
                    option_value = black_scholes_put_price(
                        price, strike, time_to_exp, risk_free_rate, hist_volatility
                    )
                
                delta_val = delta(price, strike, time_to_exp, risk_free_rate, hist_volatility, option_type.lower())
                gamma_val = gamma(price, strike, time_to_exp, risk_free_rate, hist_volatility)
                
                scenario_results['option_values'].append(option_value)
                scenario_results['deltas'].append(delta_val)
                scenario_results['gammas'].append(gamma_val)
                
        except ImportError:
            print("Warning: Could not import Black-Scholes functions for scenario analysis")
        
        return scenario_results