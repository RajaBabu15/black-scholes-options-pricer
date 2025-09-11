# market.py
"""
Market data and risk-free rate analysis module using the unified data loader.

This module demonstrates how to access market data (risk-free rates, dividend yields)
through the unified loader instead of directly calling external APIs. All data
access goes through the centralized caching system.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from optlib.data import get_risk_free_rate, get_dividend_yield, load_historical_data, get_cache_info
from datetime import datetime, timedelta
import numpy as np


def get_current_risk_free_rate(duration: str = "3m") -> float:
    """
    Get current risk-free rate for a specific duration.
    
    This function uses the unified loader instead of directly accessing
    external APIs, ensuring all data goes through the caching system.
    
    Args:
        duration: Duration for risk-free rate ('1m', '3m', '6m', '1y', '2y', '5y', '10y', '30y')
        
    Returns:
        Risk-free rate as a decimal
    """
    # All data access goes through unified loader
    return get_risk_free_rate(duration=duration)


def get_yield_curve() -> Dict[str, float]:
    """
    Get current yield curve across multiple durations.
    
    Returns:
        Dictionary mapping durations to their risk-free rates
    """
    durations = ['3m', '6m', '1y', '2y', '5y', '10y', '30y']
    yield_curve = {}
    
    for duration in durations:
        try:
            rate = get_risk_free_rate(duration=duration)
            yield_curve[duration] = rate
        except Exception as e:
            print(f"Error fetching rate for {duration}: {e}")
            continue
    
    return yield_curve


def get_ticker_dividend_yield(ticker: str) -> float:
    """
    Get dividend yield for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dividend yield as a decimal
    """
    # All data access goes through unified loader
    return get_dividend_yield(ticker=ticker)


def calculate_dividend_adjusted_price(
    ticker: str,
    time_to_expiry: float,
    period: str = "1y"
) -> Dict[str, float]:
    """
    Calculate dividend-adjusted stock price for option pricing.
    
    Args:
        ticker: Stock ticker symbol
        time_to_expiry: Time to option expiry in years
        period: Period for historical data
        
    Returns:
        Dictionary with current price, dividend yield, and adjusted price
    """
    # Use unified loader for both price and dividend data
    price_data = load_historical_data(ticker=ticker, period=period)
    current_price = float(price_data['Close'].iloc[-1])
    
    dividend_yield = get_dividend_yield(ticker=ticker)
    
    # Calculate present value of expected dividends
    dividend_pv = current_price * dividend_yield * time_to_expiry
    adjusted_price = current_price - dividend_pv
    
    return {
        'current_price': current_price,
        'dividend_yield': dividend_yield,
        'dividend_pv': dividend_pv,
        'adjusted_price': adjusted_price,
        'time_to_expiry': time_to_expiry
    }


def get_market_data_for_pricing(
    ticker: str,
    time_to_expiry: float,
    risk_free_duration: str = "3m"
) -> Dict[str, float]:
    """
    Get all market data needed for option pricing.
    
    Args:
        ticker: Stock ticker symbol
        time_to_expiry: Time to option expiry in years
        risk_free_duration: Duration for risk-free rate
        
    Returns:
        Dictionary with all pricing inputs
    """
    # Get stock price data
    price_data = load_historical_data(ticker=ticker, period="1y")
    current_price = float(price_data['Close'].iloc[-1])
    
    # Calculate historical volatility
    returns = price_data['Close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252))  # Annualized
    
    # Get risk-free rate and dividend yield
    risk_free_rate = get_risk_free_rate(duration=risk_free_duration)
    dividend_yield = get_dividend_yield(ticker=ticker)
    
    # Calculate dividend-adjusted price
    dividend_adjustment = calculate_dividend_adjusted_price(ticker, time_to_expiry)
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'adjusted_price': dividend_adjustment['adjusted_price'],
        'volatility': volatility,
        'risk_free_rate': risk_free_rate,
        'dividend_yield': dividend_yield,
        'time_to_expiry': time_to_expiry,
        'data_timestamp': datetime.now().isoformat()
    }


def compare_market_conditions(
    tickers: List[str],
    risk_free_duration: str = "3m"
) -> pd.DataFrame:
    """
    Compare market conditions across multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        risk_free_duration: Duration for risk-free rate
        
    Returns:
        DataFrame with market condition comparison
    """
    market_data = []
    
    # Get risk-free rate once (same for all tickers)
    risk_free_rate = get_risk_free_rate(duration=risk_free_duration)
    
    for ticker in tickers:
        try:
            # Get basic market data
            price_data = load_historical_data(ticker=ticker, period="1y")
            current_price = float(price_data['Close'].iloc[-1])
            
            # Calculate volatility
            returns = price_data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))
            
            # Get dividend yield
            dividend_yield = get_dividend_yield(ticker=ticker)
            
            market_data.append({
                'ticker': ticker,
                'price': current_price,
                'volatility': volatility,
                'dividend_yield': dividend_yield,
                'risk_free_rate': risk_free_rate
            })
            
        except Exception as e:
            print(f"Error loading market data for {ticker}: {e}")
            continue
    
    if not market_data:
        return pd.DataFrame()
    
    return pd.DataFrame(market_data).set_index('ticker')


def analyze_volatility_term_structure(
    ticker: str,
    periods: List[str] = ["1m", "3m", "6m", "1y"]
) -> Dict[str, float]:
    """
    Analyze volatility across different time periods.
    
    Args:
        ticker: Stock ticker symbol
        periods: List of periods to analyze
        
    Returns:
        Dictionary mapping periods to volatilities
    """
    volatilities = {}
    
    for period in periods:
        try:
            # Use unified loader for historical data
            price_data = load_historical_data(ticker=ticker, period=period)
            
            if len(price_data) < 10:  # Need minimum data points
                continue
                
            returns = price_data['Close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(252))  # Annualized
            volatilities[period] = volatility
            
        except Exception as e:
            print(f"Error calculating volatility for {period}: {e}")
            continue
    
    return volatilities


def get_beta_coefficient(
    ticker: str,
    market_ticker: str = "SPY",
    period: str = "1y"
) -> float:
    """
    Calculate beta coefficient relative to market.
    
    Args:
        ticker: Stock ticker symbol
        market_ticker: Market benchmark ticker (default: SPY)
        period: Period for calculation
        
    Returns:
        Beta coefficient
    """
    # Use unified loader for both stock and market data
    stock_data = load_historical_data(ticker=ticker, period=period)
    market_data = load_historical_data(ticker=market_ticker, period=period)
    
    # Align data by date
    combined = pd.merge(
        stock_data[['Close']].rename(columns={'Close': 'stock'}),
        market_data[['Close']].rename(columns={'Close': 'market'}),
        left_index=True,
        right_index=True,
        how='inner'
    )
    
    if len(combined) < 10:
        return 1.0  # Default beta
    
    # Calculate returns
    combined['stock_return'] = combined['stock'].pct_change()
    combined['market_return'] = combined['market'].pct_change()
    combined = combined.dropna()
    
    if len(combined) < 10:
        return 1.0
    
    # Calculate beta as covariance / market variance
    covariance = combined['stock_return'].cov(combined['market_return'])
    market_variance = combined['market_return'].var()
    
    if market_variance == 0:
        return 1.0
    
    beta = covariance / market_variance
    return float(beta)


def get_comprehensive_market_snapshot(ticker: str) -> Dict[str, Any]:
    """
    Get comprehensive market snapshot for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with comprehensive market data
    """
    snapshot = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Basic price data
        price_data = load_historical_data(ticker=ticker, period="1y")
        snapshot['current_price'] = float(price_data['Close'].iloc[-1])
        snapshot['volume'] = int(price_data['Volume'].iloc[-1])
        
        # Volatility analysis
        volatilities = analyze_volatility_term_structure(ticker)
        snapshot['volatility_term_structure'] = volatilities
        
        # Market data
        snapshot['dividend_yield'] = get_dividend_yield(ticker=ticker)
        snapshot['beta'] = get_beta_coefficient(ticker)
        
        # Risk-free rates
        snapshot['yield_curve'] = get_yield_curve()
        
        # Price statistics
        returns = price_data['Close'].pct_change().dropna()
        snapshot['statistics'] = {
            'daily_return_mean': float(returns.mean()),
            'daily_return_std': float(returns.std()),
            'max_drawdown': float((price_data['Close'] / price_data['Close'].cummax() - 1).min()),
            'ytd_return': float((price_data['Close'].iloc[-1] / price_data['Close'].iloc[0]) - 1)
        }
        
    except Exception as e:
        snapshot['error'] = str(e)
    
    return snapshot


# Example usage and testing
if __name__ == "__main__":
    print("=== Market Data Example ===")
    
    try:
        ticker = "AAPL"
        
        # Get risk-free rate
        rf_rate = get_current_risk_free_rate("3m")
        print(f"3-month risk-free rate: {rf_rate:.4f} ({rf_rate*100:.2f}%)")
        
        # Get dividend yield
        div_yield = get_ticker_dividend_yield(ticker)
        print(f"{ticker} dividend yield: {div_yield:.4f} ({div_yield*100:.2f}%)")
        
        # Get yield curve
        yield_curve = get_yield_curve()
        print(f"Yield curve: {yield_curve}")
        
        # Get comprehensive market data for option pricing
        market_data = get_market_data_for_pricing(ticker, 0.25)  # 3 months
        print(f"\nMarket data for {ticker}:")
        for key, value in market_data.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Compare multiple tickers
        comparison = compare_market_conditions(["AAPL", "MSFT", "GOOGL"])
        print(f"\nMarket comparison:\n{comparison}")
        
        print(f"\nCache info: {get_cache_info()}")
        
    except Exception as e:
        print(f"Error: {e}")