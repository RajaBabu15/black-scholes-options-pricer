# optlib/data/options.py
"""
Options analysis module - REFACTORED to use centralized data_store

This module now focuses on options data analysis and processing functions that accept
options data as parameters, rather than fetching data directly. All data I/O is handled
by the centralized data_store module.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from .data_store import default_data_store


def analyze_options_chain(options_chain: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Analyze options chain data that has been provided as input.
    
    Args:
        options_chain: DataFrame with options chain data
        current_price: Current underlying price
        
    Returns:
        Dictionary with options chain analysis
    """
    if options_chain.empty:
        return {}
    
    calls = options_chain[options_chain['Type'] == 'CALL'].copy()
    puts = options_chain[options_chain['Type'] == 'PUT'].copy()
    
    analysis = {
        'total_options': len(options_chain),
        'total_calls': len(calls),
        'total_puts': len(puts),
        'current_price': current_price
    }
    
    if not calls.empty:
        # Call analysis
        calls['Moneyness'] = calls['Strike'] / current_price
        calls['MidPrice'] = (calls['Bid'] + calls['Ask']) / 2
        
        analysis['calls'] = {
            'strike_range': {'min': float(calls['Strike'].min()), 'max': float(calls['Strike'].max())},
            'avg_implied_vol': float(calls['ImpliedVolatility'].mean()),
            'total_volume': int(calls['Volume'].sum()),
            'total_open_interest': int(calls['OpenInterest'].sum()),
            'atm_strike': find_atm_strike(calls, current_price),
            'max_volume_strike': float(calls.loc[calls['Volume'].idxmax(), 'Strike']) if len(calls) > 0 else None,
            'max_oi_strike': float(calls.loc[calls['OpenInterest'].idxmax(), 'Strike']) if len(calls) > 0 else None
        }
    
    if not puts.empty:
        # Put analysis
        puts['Moneyness'] = puts['Strike'] / current_price
        puts['MidPrice'] = (puts['Bid'] + puts['Ask']) / 2
        
        analysis['puts'] = {
            'strike_range': {'min': float(puts['Strike'].min()), 'max': float(puts['Strike'].max())},
            'avg_implied_vol': float(puts['ImpliedVolatility'].mean()),
            'total_volume': int(puts['Volume'].sum()),
            'total_open_interest': int(puts['OpenInterest'].sum()),
            'atm_strike': find_atm_strike(puts, current_price),
            'max_volume_strike': float(puts.loc[puts['Volume'].idxmax(), 'Strike']) if len(puts) > 0 else None,
            'max_oi_strike': float(puts.loc[puts['OpenInterest'].idxmax(), 'Strike']) if len(puts) > 0 else None
        }
    
    return analysis


def find_atm_strike(options_data: pd.DataFrame, current_price: float) -> float:
    """
    Find the at-the-money (ATM) strike closest to current price.
    
    Args:
        options_data: DataFrame with options data
        current_price: Current underlying price
        
    Returns:
        ATM strike price
    """
    if options_data.empty:
        return current_price
    
    strike_diff = abs(options_data['Strike'] - current_price)
    atm_idx = strike_diff.idxmin()
    return float(options_data.loc[atm_idx, 'Strike'])


def calculate_put_call_ratio(options_chain: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate put-call ratios from options chain data.
    
    Args:
        options_chain: DataFrame with options chain data
        
    Returns:
        Dictionary with put-call ratios
    """
    if options_chain.empty:
        return {}
    
    calls = options_chain[options_chain['Type'] == 'CALL']
    puts = options_chain[options_chain['Type'] == 'PUT']
    
    call_volume = calls['Volume'].sum()
    put_volume = puts['Volume'].sum()
    call_oi = calls['OpenInterest'].sum()
    put_oi = puts['OpenInterest'].sum()
    
    ratios = {}
    
    if call_volume > 0:
        ratios['volume_ratio'] = float(put_volume / call_volume)
    
    if call_oi > 0:
        ratios['oi_ratio'] = float(put_oi / call_oi)
    
    return ratios


def create_volatility_smile(options_chain: pd.DataFrame, option_type: str = 'CALL') -> pd.DataFrame:
    """
    Create volatility smile data from options chain.
    
    Args:
        options_chain: DataFrame with options chain data
        option_type: 'CALL' or 'PUT'
        
    Returns:
        DataFrame with strike vs implied volatility
    """
    filtered_options = options_chain[options_chain['Type'] == option_type.upper()].copy()
    
    if filtered_options.empty:
        return pd.DataFrame()
    
    smile_data = filtered_options[['Strike', 'ImpliedVolatility']].copy()
    smile_data = smile_data.sort_values('Strike')
    smile_data['Moneyness'] = smile_data['Strike'] / smile_data['Strike'].median()  # Approximate moneyness
    
    return smile_data


def identify_unusual_options_activity(options_chain: pd.DataFrame, volume_threshold: float = 2.0) -> pd.DataFrame:
    """
    Identify unusual options activity based on volume.
    
    Args:
        options_chain: DataFrame with options chain data
        volume_threshold: Multiplier for average volume to identify unusual activity
        
    Returns:
        DataFrame with unusual activity options
    """
    if options_chain.empty or 'Volume' not in options_chain.columns:
        return pd.DataFrame()
    
    avg_volume = options_chain['Volume'].mean()
    threshold = avg_volume * volume_threshold
    
    unusual_activity = options_chain[options_chain['Volume'] > threshold].copy()
    unusual_activity = unusual_activity.sort_values('Volume', ascending=False)
    
    return unusual_activity


def calculate_option_spreads(options_chain: pd.DataFrame, spread_type: str = 'vertical') -> List[Dict[str, Any]]:
    """
    Calculate potential option spreads from chain data.
    
    Args:
        options_chain: DataFrame with options chain data
        spread_type: Type of spread ('vertical', 'calendar', etc.)
        
    Returns:
        List of spread opportunities
    """
    if options_chain.empty:
        return []
    
    spreads = []
    
    if spread_type == 'vertical':
        # Bull call spreads
        calls = options_chain[options_chain['Type'] == 'CALL'].sort_values('Strike')
        
        for i in range(len(calls) - 1):
            long_leg = calls.iloc[i]
            short_leg = calls.iloc[i + 1]
            
            # Calculate spread cost and max profit
            spread_cost = long_leg['Ask'] - short_leg['Bid']
            max_profit = (short_leg['Strike'] - long_leg['Strike']) - spread_cost
            
            if spread_cost > 0 and max_profit > 0:
                spreads.append({
                    'type': 'bull_call_spread',
                    'long_strike': long_leg['Strike'],
                    'short_strike': short_leg['Strike'],
                    'cost': spread_cost,
                    'max_profit': max_profit,
                    'breakeven': long_leg['Strike'] + spread_cost,
                    'profit_ratio': max_profit / spread_cost if spread_cost > 0 else 0
                })
    
    return sorted(spreads, key=lambda x: x.get('profit_ratio', 0), reverse=True)[:10]  # Top 10


def calculate_implied_forward_price(options_chain: pd.DataFrame, risk_free_rate: float, 
                                  time_to_expiration: float) -> Optional[float]:
    """
    Calculate implied forward price using put-call parity.
    
    Args:
        options_chain: DataFrame with options chain data
        risk_free_rate: Risk-free interest rate
        time_to_expiration: Time to expiration in years
        
    Returns:
        Implied forward price or None if cannot calculate
    """
    if options_chain.empty or time_to_expiration <= 0:
        return None
    
    # Group by strike to find matching calls and puts
    strikes = options_chain['Strike'].unique()
    forward_prices = []
    
    for strike in strikes:
        call_data = options_chain[(options_chain['Strike'] == strike) & (options_chain['Type'] == 'CALL')]
        put_data = options_chain[(options_chain['Strike'] == strike) & (options_chain['Type'] == 'PUT')]
        
        if not call_data.empty and not put_data.empty:
            call_mid = (call_data['Bid'].iloc[0] + call_data['Ask'].iloc[0]) / 2
            put_mid = (put_data['Bid'].iloc[0] + put_data['Ask'].iloc[0]) / 2
            
            # Put-call parity: C - P = S - K * e^(-r*T)
            # Therefore: S = C - P + K * e^(-r*T)
            implied_spot = call_mid - put_mid + strike * np.exp(-risk_free_rate * time_to_expiration)
            forward_price = implied_spot * np.exp(risk_free_rate * time_to_expiration)
            
            forward_prices.append(forward_price)
    
    return float(np.median(forward_prices)) if forward_prices else None


# === CONVENIENCE FUNCTIONS THAT USE DATA_STORE ===
# These functions provide easy access to data_store functionality for backward compatibility

def get_options_chain(symbol: str, expiration_date: str, **kwargs) -> pd.DataFrame:
    """
    Get options chain using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_options_chain(symbol, expiration_date, **kwargs)


def get_option_quotes(symbol: str, strike: float, expiration_date: str, option_type: str, **kwargs) -> Dict[str, Any]:
    """
    Get option quotes using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_option_quotes(symbol, strike, expiration_date, option_type, **kwargs)


def get_available_expirations(symbol: str, **kwargs) -> List[str]:
    """
    Get list of available expiration dates for a symbol.
    
    This generates typical expiration dates since we're simulating data.
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