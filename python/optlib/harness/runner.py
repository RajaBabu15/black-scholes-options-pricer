# optlib/harness/runner.py
"""
Options analysis workflow runner - REFACTORED to use centralized data_store

This module coordinates options analysis workflows by using the centralized data_store
for all data fetching operations, ensuring no direct data I/O operations are performed
outside of the data_store module.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# Import centralized data store (only module that should do data I/O)
from ..data.data_store import default_data_store

# Import analysis functions that work with data (no I/O)
from ..data.history import analyze_price_history, calculate_returns_statistics
from ..data.options import analyze_options_chain, calculate_put_call_ratio, create_volatility_smile
from ..data.market import analyze_market_sentiment, assess_macro_environment


class OptionsAnalysisRunner:
    """
    Coordinates the fetching of market data and running of options analysis.
    Now uses centralized data_store for all data operations.
    """
    
    def __init__(self, data_store=None):
        """
        Initialize runner with optional custom data store.
        
        Args:
            data_store: Custom DataStore instance, uses default if None
        """
        self.data_store = data_store or default_data_store
        self.analysis_results = {}
    
    def run_full_analysis(self, symbol: str, strike: float, expiration_date: str, 
                         option_type: str = "CALL", force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run complete options analysis workflow using centralized data store.
        
        Args:
            symbol: Stock symbol
            strike: Option strike price
            expiration_date: Option expiration date
            option_type: 'CALL' or 'PUT'
            force_refresh: If True, refresh all data
            
        Returns:
            Complete analysis results
        """
        print(f"Running full options analysis for {symbol} {strike} {option_type} expiring {expiration_date}")
        
        results = {
            'symbol': symbol,
            'strike': strike,
            'expiration_date': expiration_date,
            'option_type': option_type,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Use centralized data store to get all required data
            print("Fetching all market data from centralized data store...")
            complete_data = self.data_store.get_complete_market_data(
                symbol, expiration_date, force_refresh=force_refresh
            )
            
            # Extract data components
            current_price = complete_data['current_price']
            historical_volatility = complete_data['historical_volatility']
            risk_free_rate = complete_data['risk_free_rate']
            dividend_yield = complete_data['dividend_yield']
            market_indicators = complete_data['market_indicators']
            options_chain = complete_data['options_chain']
            
            # Store basic market data
            results.update({
                'current_price': current_price,
                'historical_volatility': historical_volatility,
                'risk_free_rate': risk_free_rate,
                'dividend_yield': dividend_yield,
                'market_vix': market_indicators.get('VIX'),
                'options_chain_size': len(options_chain)
            })
            
            # Get specific option quotes
            specific_option = self.data_store.get_option_quotes(
                symbol, strike, expiration_date, option_type, force_refresh=force_refresh
            )
            
            if specific_option:
                results.update({
                    'option_bid': specific_option.get('Bid'),
                    'option_ask': specific_option.get('Ask'),
                    'implied_volatility': specific_option.get('ImpliedVolatility')
                })
            
            # Perform analysis using data (no I/O operations)
            print("Performing options chain analysis...")
            options_analysis = analyze_options_chain(options_chain, current_price)
            results['options_analysis'] = options_analysis
            
            print("Analyzing market sentiment...")
            sentiment_analysis = analyze_market_sentiment(market_indicators)
            results['market_sentiment'] = sentiment_analysis
            
            # Calculate theoretical values
            print("Calculating theoretical values...")
            time_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365.0
            
            theoretical_results = self._calculate_theoretical_values(
                current_price, strike, time_to_exp, risk_free_rate, 
                historical_volatility, option_type, dividend_yield
            )
            results.update(theoretical_results)
            
            # Compare with market price
            if results.get('option_bid') and results.get('option_ask'):
                market_mid = (results['option_bid'] + results['option_ask']) / 2
                results['market_mid_price'] = market_mid
                
                if results.get('theoretical_price'):
                    results['price_difference'] = results['theoretical_price'] - market_mid
                    results['price_difference_pct'] = (results['theoretical_price'] - market_mid) / market_mid * 100
            
            # Store results
            self.analysis_results[f"{symbol}_{strike}_{option_type}_{expiration_date}"] = results
            
            print("Analysis complete!")
            return results
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            results['error'] = str(e)
            return results
    
    def generate_volatility_surface(self, symbol: str, expiration_dates: List[str],
                                  force_refresh: bool = False) -> pd.DataFrame:
        """
        Generate implied volatility surface data using centralized data store.
        
        Args:
            symbol: Stock symbol
            expiration_dates: List of expiration dates
            force_refresh: If True, refresh all data
            
        Returns:
            DataFrame with volatility surface data
        """
        print(f"Generating volatility surface for {symbol}")
        
        surface_data = []
        
        for exp_date in expiration_dates:
            print(f"Processing expiration: {exp_date}")
            
            try:
                # Get options chain using centralized data store
                options_chain = self.data_store.get_options_chain(symbol, exp_date, force_refresh=force_refresh)
                
                # Create volatility smile for this expiration
                vol_smile = create_volatility_smile(options_chain, 'CALL')
                
                if not vol_smile.empty:
                    vol_smile['expiration'] = exp_date
                    time_to_exp = (datetime.strptime(exp_date, '%Y-%m-%d') - datetime.now()).days / 365.0
                    vol_smile['time_to_exp'] = time_to_exp
                    surface_data.append(vol_smile)
                    
            except Exception as e:
                print(f"Error processing {exp_date}: {e}")
                continue
        
        if surface_data:
            return pd.concat(surface_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def run_scenario_analysis(self, symbol: str, strike: float, expiration_date: str,
                            option_type: str, price_scenarios: List[float],
                            force_refresh: bool = False) -> Dict[str, List[float]]:
        """
        Run scenario analysis using centralized data store for market data.
        
        Args:
            symbol: Stock symbol
            strike: Option strike
            expiration_date: Expiration date
            option_type: Option type
            price_scenarios: List of price scenarios to test
            force_refresh: If True, refresh market data
            
        Returns:
            Dictionary with scenario results
        """
        print(f"Running scenario analysis for {symbol}")
        
        try:
            # Get market data from centralized data store
            risk_free_rate = self.data_store.get_risk_free_rate("3M", force_refresh=force_refresh)
            hist_volatility = self.data_store.calculate_historical_volatility(symbol, 252, force_refresh=force_refresh)
            
            time_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days / 365.0
            
            scenario_results = {
                'prices': price_scenarios,
                'option_values': [],
                'deltas': [],
                'gammas': []
            }
            
            # Calculate option values for each scenario
            for price in price_scenarios:
                values = self._calculate_theoretical_values(
                    price, strike, time_to_exp, risk_free_rate, hist_volatility, option_type
                )
                
                scenario_results['option_values'].append(values.get('theoretical_price', 0))
                scenario_results['deltas'].append(values.get('greeks', {}).get('Delta', 0))
                scenario_results['gammas'].append(values.get('greeks', {}).get('Gamma', 0))
            
            return scenario_results
            
        except Exception as e:
            print(f"Error in scenario analysis: {e}")
            return {'error': str(e)}
    
    def run_portfolio_analysis(self, positions: List[Dict[str, Any]],
                             force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run portfolio-level analysis for multiple options positions.
        
        Args:
            positions: List of position dictionaries with symbol, strike, expiration, type, quantity
            force_refresh: If True, refresh all data
            
        Returns:
            Portfolio analysis results
        """
        print("Running portfolio analysis...")
        
        portfolio_results = {
            'positions': [],
            'portfolio_greeks': {'Delta': 0, 'Gamma': 0, 'Vega': 0, 'Theta': 0, 'Rho': 0},
            'total_value': 0,
            'risk_metrics': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for position in positions:
            try:
                symbol = position['symbol']
                strike = position['strike']
                expiration = position['expiration']
                option_type = position['type']
                quantity = position.get('quantity', 1)
                
                # Get position analysis
                pos_analysis = self.run_full_analysis(symbol, strike, expiration, option_type, force_refresh)
                
                if 'error' not in pos_analysis:
                    pos_value = pos_analysis.get('theoretical_price', 0) * quantity
                    portfolio_results['total_value'] += pos_value
                    
                    # Aggregate greeks
                    greeks = pos_analysis.get('greeks', {})
                    for greek_name in portfolio_results['portfolio_greeks']:
                        greek_value = greeks.get(greek_name, 0) * quantity
                        portfolio_results['portfolio_greeks'][greek_name] += greek_value
                    
                    position_result = {
                        'symbol': symbol,
                        'strike': strike,
                        'expiration': expiration,
                        'type': option_type,
                        'quantity': quantity,
                        'unit_value': pos_analysis.get('theoretical_price', 0),
                        'total_value': pos_value,
                        'greeks': {k: v * quantity for k, v in greeks.items()}
                    }
                    portfolio_results['positions'].append(position_result)
                
            except Exception as e:
                print(f"Error analyzing position {position}: {e}")
                continue
        
        # Calculate risk metrics
        if portfolio_results['positions']:
            portfolio_results['risk_metrics'] = self._calculate_portfolio_risk_metrics(
                portfolio_results['positions']
            )
        
        return portfolio_results
    
    def _calculate_theoretical_values(self, current_price: float, strike: float, time_to_exp: float,
                                    risk_free_rate: float, volatility: float, option_type: str,
                                    dividend_yield: float = 0.0) -> Dict[str, Any]:
        """
        Calculate theoretical option values and Greeks.
        
        This is a helper method that doesn't perform any I/O operations.
        """
        try:
            # Import Black-Scholes functions (these don't perform I/O)
            from src.black_scholes import (
                black_scholes_call_price, black_scholes_put_price,
                calculate_all_greeks
            )
            
            # Adjust for dividends if needed (simplified)
            adjusted_price = current_price * np.exp(-dividend_yield * time_to_exp)
            
            if option_type.upper() == 'CALL':
                theoretical_price = black_scholes_call_price(
                    adjusted_price, strike, time_to_exp, risk_free_rate, volatility
                )
            else:
                theoretical_price = black_scholes_put_price(
                    adjusted_price, strike, time_to_exp, risk_free_rate, volatility
                )
            
            greeks = calculate_all_greeks(
                adjusted_price, strike, time_to_exp, risk_free_rate, volatility, option_type.lower()
            )
            
            return {
                'theoretical_price': theoretical_price,
                'greeks': greeks
            }
            
        except ImportError as e:
            print(f"Warning: Could not import Black-Scholes functions: {e}")
            return {
                'theoretical_price': None,
                'greeks': None
            }
    
    def _calculate_portfolio_risk_metrics(self, positions: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            positions: List of position analysis results
            
        Returns:
            Dictionary with risk metrics
        """
        if not positions:
            return {}
        
        # Extract position values
        position_values = [pos['total_value'] for pos in positions]
        total_value = sum(position_values)
        
        # Calculate concentration risk
        max_position = max(abs(val) for val in position_values)
        concentration_ratio = max_position / total_value if total_value > 0 else 0
        
        # Calculate net exposure by underlying
        exposures_by_symbol = {}
        for pos in positions:
            symbol = pos['symbol']
            exposure = pos['total_value']
            exposures_by_symbol[symbol] = exposures_by_symbol.get(symbol, 0) + exposure
        
        net_exposures = list(exposures_by_symbol.values())
        max_net_exposure = max(abs(exp) for exp in net_exposures) if net_exposures else 0
        net_concentration = max_net_exposure / total_value if total_value > 0 else 0
        
        return {
            'concentration_ratio': concentration_ratio,
            'net_concentration': net_concentration,
            'num_positions': len(positions),
            'num_underlyings': len(exposures_by_symbol),
            'largest_position': max_position,
            'total_portfolio_value': total_value
        }