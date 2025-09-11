# optlib/data/market.py
"""
Market data analysis module - REFACTORED to use centralized data_store

This module now focuses on market data analysis and processing functions that accept
market data as parameters, rather than fetching data directly. All data I/O is handled
by the centralized data_store module.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from .data_store import default_data_store


def analyze_yield_curve(rates_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Analyze yield curve data that has been provided as input.
    
    Args:
        rates_data: Dictionary with duration keys and rate values
        
    Returns:
        Dictionary with yield curve analysis
    """
    if not rates_data:
        return {}
    
    # Standard durations in order
    duration_order = ["1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"]
    duration_months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "2Y": 24, "5Y": 60, "10Y": 120, "30Y": 360}
    
    # Filter available rates
    available_rates = []
    available_durations = []
    
    for duration in duration_order:
        if duration in rates_data:
            available_rates.append(rates_data[duration])
            available_durations.append(duration)
    
    if len(available_rates) < 2:
        return {'error': 'Insufficient rate data for analysis'}
    
    # Calculate curve metrics
    rates_array = np.array(available_rates)
    
    analysis = {
        'curve_shape': determine_curve_shape(available_rates),
        'short_end': available_rates[0],
        'long_end': available_rates[-1],
        'spread_short_long': available_rates[-1] - available_rates[0],
        'max_rate': float(rates_array.max()),
        'min_rate': float(rates_array.min()),
        'average_rate': float(rates_array.mean()),
        'rate_volatility': float(rates_array.std()),
        'available_tenors': available_durations
    }
    
    # Calculate specific spreads if available
    if "3M" in rates_data and "10Y" in rates_data:
        analysis['3m_10y_spread'] = rates_data["10Y"] - rates_data["3M"]
    
    if "2Y" in rates_data and "10Y" in rates_data:
        analysis['2y_10y_spread'] = rates_data["10Y"] - rates_data["2Y"]
    
    return analysis


def determine_curve_shape(rates: List[float]) -> str:
    """
    Determine the shape of the yield curve.
    
    Args:
        rates: List of rates in ascending duration order
        
    Returns:
        String describing curve shape
    """
    if len(rates) < 3:
        return "insufficient_data"
    
    # Simple heuristic for curve shape
    short_end = rates[0]
    long_end = rates[-1]
    middle = rates[len(rates)//2]
    
    if long_end > short_end:
        if middle > short_end and middle < long_end:
            return "normal"  # Upward sloping
        else:
            return "steep"
    elif long_end < short_end:
        return "inverted"  # Downward sloping
    else:
        return "flat"


def analyze_market_sentiment(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze market sentiment from market indicators.
    
    Args:
        indicators: Dictionary with market indicators
        
    Returns:
        Dictionary with sentiment analysis
    """
    if not indicators:
        return {}
    
    sentiment_score = 0
    sentiment_factors = []
    
    # VIX analysis
    if 'VIX' in indicators:
        vix = indicators['VIX']
        if vix < 15:
            sentiment_score += 2
            sentiment_factors.append('Low VIX indicates low fear')
        elif vix < 25:
            sentiment_score += 1
            sentiment_factors.append('Moderate VIX indicates normal volatility')
        elif vix < 35:
            sentiment_score -= 1
            sentiment_factors.append('Elevated VIX indicates increased uncertainty')
        else:
            sentiment_score -= 2
            sentiment_factors.append('High VIX indicates high fear')
    
    # Market indices momentum (simplified)
    indices = ['SPX', 'NDX', 'DJI']
    for index in indices:
        if index in indicators:
            # This is a simplified sentiment - in reality would compare to historical levels
            value = indicators[index]
            if index == 'SPX' and value > 4400:
                sentiment_score += 1
                sentiment_factors.append(f'{index} at strong levels')
            elif index == 'NDX' and value > 14500:
                sentiment_score += 1
                sentiment_factors.append(f'{index} at strong levels')
            elif index == 'DJI' and value > 34000:
                sentiment_score += 1
                sentiment_factors.append(f'{index} at strong levels')
    
    # USD strength
    if 'USD_INDEX' in indicators:
        usd = indicators['USD_INDEX']
        if usd > 105:
            sentiment_score -= 0.5  # Strong USD can be headwind for equities
            sentiment_factors.append('Strong USD may pressure risk assets')
        elif usd < 95:
            sentiment_score += 0.5
            sentiment_factors.append('Weak USD supportive for risk assets')
    
    # Determine overall sentiment
    if sentiment_score >= 2:
        overall_sentiment = "bullish"
    elif sentiment_score >= 0:
        overall_sentiment = "neutral_positive"
    elif sentiment_score >= -2:
        overall_sentiment = "neutral_negative"
    else:
        overall_sentiment = "bearish"
    
    return {
        'sentiment_score': sentiment_score,
        'overall_sentiment': overall_sentiment,
        'sentiment_factors': sentiment_factors,
        'analysis_timestamp': datetime.now().isoformat()
    }


def calculate_correlation_with_market(price_data: pd.DataFrame, market_data: pd.DataFrame, 
                                   period_days: int = 252) -> Dict[str, float]:
    """
    Calculate correlation between a stock and market indices.
    
    Args:
        price_data: DataFrame with stock price data
        market_data: DataFrame with market index data
        period_days: Period for correlation calculation
        
    Returns:
        Dictionary with correlation metrics
    """
    if price_data.empty or market_data.empty:
        return {}
    
    # Ensure both datasets have Date columns
    if 'Date' not in price_data.columns or 'Close' not in price_data.columns:
        return {}
    
    if 'Date' not in market_data.columns or 'Close' not in market_data.columns:
        return {}
    
    # Merge on dates
    stock_data = price_data[['Date', 'Close']].copy()
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data = stock_data.rename(columns={'Close': 'Stock_Close'})
    
    market_df = market_data[['Date', 'Close']].copy()
    market_df['Date'] = pd.to_datetime(market_df['Date'])
    market_df = market_df.rename(columns={'Close': 'Market_Close'})
    
    merged = pd.merge(stock_data, market_df, on='Date', how='inner')
    merged = merged.sort_values('Date')
    
    # Limit to specified period
    if len(merged) > period_days:
        merged = merged.tail(period_days)
    
    if len(merged) < 30:  # Need minimum data points
        return {}
    
    # Calculate returns
    merged['Stock_Returns'] = merged['Stock_Close'].pct_change()
    merged['Market_Returns'] = merged['Market_Close'].pct_change()
    
    # Remove NaN values
    returns_data = merged[['Stock_Returns', 'Market_Returns']].dropna()
    
    if len(returns_data) < 20:
        return {}
    
    # Calculate correlations and beta
    correlation = returns_data['Stock_Returns'].corr(returns_data['Market_Returns'])
    
    # Beta calculation: Cov(stock, market) / Var(market)
    covariance = returns_data['Stock_Returns'].cov(returns_data['Market_Returns'])
    market_variance = returns_data['Market_Returns'].var()
    
    beta = covariance / market_variance if market_variance > 0 else 0
    
    # Additional metrics
    stock_vol = returns_data['Stock_Returns'].std() * np.sqrt(252)
    market_vol = returns_data['Market_Returns'].std() * np.sqrt(252)
    
    return {
        'correlation': float(correlation),
        'beta': float(beta),
        'stock_volatility': float(stock_vol),
        'market_volatility': float(market_vol),
        'data_points': len(returns_data),
        'period_days': period_days
    }


def assess_macro_environment(market_indicators: Dict[str, Any], rates_data: Dict[str, float]) -> Dict[str, Any]:
    """
    Assess the overall macroeconomic environment.
    
    Args:
        market_indicators: Dictionary with market indicators
        rates_data: Dictionary with interest rate data
        
    Returns:
        Dictionary with macro environment assessment
    """
    assessment = {
        'timestamp': datetime.now().isoformat(),
        'factors': []
    }
    
    risk_score = 0  # Higher score = more risk
    
    # Interest rate environment
    if rates_data:
        yield_analysis = analyze_yield_curve(rates_data)
        
        if 'curve_shape' in yield_analysis:
            if yield_analysis['curve_shape'] == 'inverted':
                risk_score += 3
                assessment['factors'].append('Inverted yield curve signals potential recession risk')
            elif yield_analysis['curve_shape'] == 'flat':
                risk_score += 1
                assessment['factors'].append('Flat yield curve indicates economic uncertainty')
            else:
                assessment['factors'].append('Normal yield curve supports economic growth')
        
        if '3m_10y_spread' in yield_analysis:
            spread = yield_analysis['3m_10y_spread']
            if spread < 0:
                risk_score += 2
                assessment['factors'].append('Negative 3M-10Y spread indicates inversion')
            elif spread < 0.5:
                risk_score += 1
                assessment['factors'].append('Narrow 3M-10Y spread suggests flattening')
    
    # Volatility environment
    if 'VIX' in market_indicators:
        vix = market_indicators['VIX']
        if vix > 30:
            risk_score += 2
            assessment['factors'].append('Elevated VIX indicates high market stress')
        elif vix > 20:
            risk_score += 1
            assessment['factors'].append('Moderate VIX indicates some market uncertainty')
        else:
            assessment['factors'].append('Low VIX indicates calm market conditions')
    
    # Dollar strength
    if 'USD_INDEX' in market_indicators:
        usd = market_indicators['USD_INDEX']
        if usd > 110:
            risk_score += 1
            assessment['factors'].append('Very strong USD may impact global growth')
        elif usd < 90:
            assessment['factors'].append('Weak USD supportive for global risk assets')
    
    # Commodity signals
    if 'GOLD' in market_indicators and 'OIL_WTI' in market_indicators:
        gold = market_indicators['GOLD']
        oil = market_indicators['OIL_WTI']
        
        if gold > 2100:  # Flight to safety
            risk_score += 1
            assessment['factors'].append('High gold prices suggest safe-haven demand')
        
        if oil > 100:  # Inflation concerns
            risk_score += 1
            assessment['factors'].append('High oil prices may signal inflation pressures')
        elif oil < 60:  # Growth concerns
            risk_score += 1
            assessment['factors'].append('Low oil prices may signal growth concerns')
    
    # Overall risk assessment
    if risk_score >= 6:
        risk_level = "high"
    elif risk_score >= 3:
        risk_level = "moderate"
    else:
        risk_level = "low"
    
    assessment['risk_score'] = risk_score
    assessment['risk_level'] = risk_level
    
    return assessment


# === CONVENIENCE FUNCTIONS THAT USE DATA_STORE ===
# These functions provide easy access to data_store functionality for backward compatibility

def get_risk_free_rate(duration: str = "3M", **kwargs) -> float:
    """
    Get risk-free rate using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_risk_free_rate(duration, **kwargs)


def get_market_indicators(**kwargs) -> Dict[str, Any]:
    """
    Get market indicators using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_market_indicators(**kwargs)


def get_dividend_yield(symbol: str, **kwargs) -> float:
    """
    Get dividend yield using centralized data store.
    
    This function is maintained for backward compatibility but now uses data_store.
    """
    return default_data_store.get_dividend_yield(symbol, **kwargs)


def get_earnings_calendar(symbol: str, **kwargs) -> Optional[str]:
    """
    Get next earnings announcement date for a symbol.
    
    This generates a simulated earnings date since we're using simulated data.
    """
    # Simulate earnings calendar data
    np.random.seed(hash(symbol) % 2**32)
    
    # Generate a random earnings date in the next 90 days
    days_until_earnings = np.random.randint(1, 91)
    earnings_date = datetime.now() + timedelta(days=days_until_earnings)
    
    return earnings_date.strftime('%Y-%m-%d')