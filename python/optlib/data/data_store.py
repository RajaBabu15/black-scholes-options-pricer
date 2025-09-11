# optlib/data/data_store.py
"""
Centralized data store for all financial data fetching, caching, and loading.

This module serves as the single source for acquiring all external data including:
- Stock prices and historical data
- Options chains and quotes
- Risk-free rates and market indicators
- Dividend information and earnings data

All other modules should use this centralized data store instead of directly
accessing external data sources or cache files.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataStore:
    """
    Centralized data store that consolidates all data fetching and caching logic.
    """
    
    def __init__(self, cache_dir: str = "cache", cache_expiry_hours: int = 24):
        """
        Initialize the data store.
        
        Args:
            cache_dir: Directory for storing cached data
            cache_expiry_hours: Hours after which cached data expires
        """
        self.cache_dir = cache_dir
        self.cache_expiry_hours = cache_expiry_hours
        os.makedirs(cache_dir, exist_ok=True)
        
        # Internal cache for session-level data
        self._session_cache = {}
        
        logger.info(f"DataStore initialized with cache_dir: {cache_dir}")
    
    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cached data is still valid based on expiry time."""
        if not os.path.exists(cache_file):
            return False
        
        file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        expiry_time = datetime.now() - timedelta(hours=self.cache_expiry_hours)
        
        return file_mod_time > expiry_time
    
    def _save_to_cache(self, data: Any, cache_file: str) -> None:
        """Save data to cache file."""
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_file}: {e}")
    
    def _load_from_cache(self, cache_file: str) -> Optional[Any]:
        """Load data from cache file."""
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            logger.debug(f"Loaded data from cache: {cache_file}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_file}: {e}")
            return None
    
    # === STOCK PRICE DATA ===
    
    def get_stock_price_history(self, symbol: str, start_date: str, end_date: str, 
                               force_refresh: bool = False) -> pd.DataFrame:
        """
        Get historical stock price data.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            force_refresh: If True, bypass cache and fetch fresh data
            
        Returns:
            DataFrame with historical price data
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_history"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Check session cache first
        if cache_key in self._session_cache and not force_refresh:
            logger.debug(f"Returning {symbol} history from session cache")
            return pd.DataFrame(self._session_cache[cache_key])
        
        # Check file cache
        if self._is_cache_valid(cache_file) and not force_refresh:
            data = self._load_from_cache(cache_file)
            if data:
                self._session_cache[cache_key] = data
                return pd.DataFrame(data)
        
        # Fetch fresh data
        logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
        data = self._fetch_stock_history(symbol, start_date, end_date)
        
        # Cache the data
        self._save_to_cache(data, cache_file)
        self._session_cache[cache_key] = data
        
        return pd.DataFrame(data)
    
    def get_current_stock_price(self, symbol: str, force_refresh: bool = False) -> float:
        """
        Get current stock price.
        
        Args:
            symbol: Stock symbol
            force_refresh: If True, bypass cache
            
        Returns:
            Current stock price
        """
        cache_key = f"{symbol}_current_price"
        
        # Check session cache (short-lived for current prices)
        if cache_key in self._session_cache and not force_refresh:
            cache_time = self._session_cache.get(f"{cache_key}_time", datetime.min)
            if (datetime.now() - cache_time).seconds < 300:  # 5 minutes
                return self._session_cache[cache_key]
        
        logger.info(f"Fetching current price for {symbol}")
        
        # Try to get from recent historical data first
        try:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
            recent_data = self.get_stock_price_history(symbol, start_date, end_date)
            
            if not recent_data.empty:
                recent_data['Date'] = pd.to_datetime(recent_data['Date'])
                latest_price = recent_data.loc[recent_data['Date'].idxmax(), 'Close']
                
                # Cache for short term
                self._session_cache[cache_key] = latest_price
                self._session_cache[f"{cache_key}_time"] = datetime.now()
                
                return float(latest_price)
        except Exception as e:
            logger.warning(f"Could not get recent historical data for {symbol}: {e}")
        
        # Fallback: simulate current price
        price = self._simulate_current_price(symbol)
        
        # Cache for short term
        self._session_cache[cache_key] = price
        self._session_cache[f"{cache_key}_time"] = datetime.now()
        
        return price
    
    def calculate_historical_volatility(self, symbol: str, period_days: int = 252,
                                      force_refresh: bool = False) -> float:
        """
        Calculate historical volatility from stock price data.
        
        Args:
            symbol: Stock symbol
            period_days: Number of days for volatility calculation
            force_refresh: If True, bypass cache
            
        Returns:
            Annualized historical volatility
        """
        cache_key = f"{symbol}_volatility_{period_days}"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Check cache
        if self._is_cache_valid(cache_file) and not force_refresh:
            data = self._load_from_cache(cache_file)
            if data and 'volatility' in data:
                return data['volatility']
        
        logger.info(f"Calculating historical volatility for {symbol} ({period_days} days)")
        
        # Get historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=period_days + 30)).strftime('%Y-%m-%d')
        
        df = self.get_stock_price_history(symbol, start_date, end_date, force_refresh)
        
        if len(df) < 2:
            return 0.20  # Default volatility
        
        # Calculate volatility
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df['Returns'] = df['Close'].pct_change().dropna()
        
        daily_volatility = df['Returns'].std()
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        # Cache the result
        vol_data = {
            'symbol': symbol,
            'volatility': float(annualized_volatility),
            'period_days': period_days,
            'data_points': len(df['Returns'].dropna()),
            'calculated_at': datetime.now().isoformat()
        }
        self._save_to_cache(vol_data, cache_file)
        
        return float(annualized_volatility)
    
    # === OPTIONS DATA ===
    
    def get_options_chain(self, symbol: str, expiration_date: str, 
                         force_refresh: bool = False) -> pd.DataFrame:
        """
        Get options chain data for a symbol and expiration.
        
        Args:
            symbol: Underlying symbol
            expiration_date: Expiration date in 'YYYY-MM-DD' format
            force_refresh: If True, bypass cache
            
        Returns:
            DataFrame with options chain data
        """
        cache_key = f"{symbol}_{expiration_date}_options"
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Check cache
        if self._is_cache_valid(cache_file) and not force_refresh:
            data = self._load_from_cache(cache_file)
            if data:
                return pd.DataFrame(data)
        
        logger.info(f"Fetching options chain for {symbol} expiring {expiration_date}")
        
        # Fetch fresh options data
        options_data = self._fetch_options_chain(symbol, expiration_date)
        
        # Cache the data
        self._save_to_cache(options_data, cache_file)
        
        return pd.DataFrame(options_data)
    
    def get_option_quotes(self, symbol: str, strike: float, expiration_date: str, 
                         option_type: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get specific option quote data.
        
        Args:
            symbol: Underlying symbol
            strike: Strike price
            expiration_date: Expiration date
            option_type: 'CALL' or 'PUT'
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with option quote data
        """
        options_chain = self.get_options_chain(symbol, expiration_date, force_refresh)
        
        # Filter for specific option
        option_data = options_chain[
            (options_chain['Strike'] == strike) & 
            (options_chain['Type'] == option_type.upper())
        ]
        
        if option_data.empty:
            return {}
        
        return option_data.iloc[0].to_dict()
    
    # === MARKET DATA ===
    
    def get_risk_free_rate(self, duration: str = "3M", force_refresh: bool = False) -> float:
        """
        Get risk-free interest rate.
        
        Args:
            duration: Duration ("1M", "3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y")
            force_refresh: If True, bypass cache
            
        Returns:
            Risk-free rate as decimal
        """
        cache_file = os.path.join(self.cache_dir, f"risk_free_rates_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        # Check cache
        if self._is_cache_valid(cache_file) and not force_refresh:
            data = self._load_from_cache(cache_file)
            if data and duration in data:
                return data[duration]
        
        logger.info(f"Fetching risk-free rate for {duration}")
        
        # Load existing rates or create new
        rates_data = self._load_from_cache(cache_file) or {}
        
        # Fetch new rate
        rate = self._fetch_risk_free_rate(duration)
        rates_data[duration] = rate
        
        # Cache the updated rates
        self._save_to_cache(rates_data, cache_file)
        
        return rate
    
    def get_market_indicators(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get market indicators (VIX, indices, etc.).
        
        Args:
            force_refresh: If True, bypass cache
            
        Returns:
            Dictionary with market indicators
        """
        cache_file = os.path.join(self.cache_dir, f"market_indicators_{datetime.now().strftime('%Y-%m-%d')}.json")
        
        # Check cache
        if self._is_cache_valid(cache_file) and not force_refresh:
            data = self._load_from_cache(cache_file)
            if data:
                return data
        
        logger.info("Fetching market indicators")
        
        # Fetch fresh indicators
        indicators = self._fetch_market_indicators()
        
        # Cache the data
        self._save_to_cache(indicators, cache_file)
        
        return indicators
    
    def get_dividend_yield(self, symbol: str, force_refresh: bool = False) -> float:
        """
        Get dividend yield for a symbol.
        
        Args:
            symbol: Stock symbol
            force_refresh: If True, bypass cache
            
        Returns:
            Annual dividend yield as decimal
        """
        cache_file = os.path.join(self.cache_dir, f"{symbol}_dividend.json")
        
        # Check cache (longer expiry for dividend data)
        if not force_refresh and os.path.exists(cache_file):
            data = self._load_from_cache(cache_file)
            if data:
                last_update = datetime.fromisoformat(data.get('last_updated', '2000-01-01'))
                if (datetime.now() - last_update).days < 30:  # 30 days for dividend data
                    return data.get('dividend_yield', 0.0)
        
        logger.info(f"Fetching dividend information for {symbol}")
        
        # Fetch fresh dividend data
        dividend_yield = self._fetch_dividend_yield(symbol)
        
        # Cache the data
        div_data = {
            'symbol': symbol,
            'dividend_yield': dividend_yield,
            'last_updated': datetime.now().isoformat()
        }
        self._save_to_cache(div_data, cache_file)
        
        return dividend_yield
    
    # === BULK DATA OPERATIONS ===
    
    def get_complete_market_data(self, symbol: str, expiration_date: str, 
                               force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get all market data needed for options analysis in one call.
        
        Args:
            symbol: Stock symbol
            expiration_date: Option expiration date
            force_refresh: If True, refresh all data
            
        Returns:
            Dictionary with all market data
        """
        logger.info(f"Fetching complete market data for {symbol}")
        
        return {
            'current_price': self.get_current_stock_price(symbol, force_refresh),
            'historical_volatility': self.calculate_historical_volatility(symbol, 252, force_refresh),
            'risk_free_rate': self.get_risk_free_rate("3M", force_refresh),
            'dividend_yield': self.get_dividend_yield(symbol, force_refresh),
            'market_indicators': self.get_market_indicators(force_refresh),
            'options_chain': self.get_options_chain(symbol, expiration_date, force_refresh),
            'timestamp': datetime.now().isoformat()
        }
    
    def clear_cache(self, pattern: str = None) -> None:
        """
        Clear cached data.
        
        Args:
            pattern: If provided, only clear files matching pattern
        """
        if pattern:
            files_to_remove = [f for f in os.listdir(self.cache_dir) if pattern in f]
        else:
            files_to_remove = os.listdir(self.cache_dir)
        
        for file_name in files_to_remove:
            file_path = os.path.join(self.cache_dir, file_name)
            try:
                os.remove(file_path)
                logger.info(f"Removed cache file: {file_name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_name}: {e}")
        
        # Clear session cache
        if pattern:
            keys_to_remove = [k for k in self._session_cache.keys() if pattern in k]
            for key in keys_to_remove:
                del self._session_cache[key]
        else:
            self._session_cache.clear()
    
    # === PRIVATE METHODS (Data Fetching Implementation) ===
    
    def _fetch_stock_history(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch historical stock data (simulated implementation)."""
        # In real implementation, this would call external APIs
        # For now, simulate realistic data
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        dates = pd.date_range(start=start, end=end, freq='D')
        
        # Filter business days
        business_dates = [d for d in dates if d.weekday() < 5]
        
        # Generate realistic price data
        base_price = 100.0
        np.random.seed(hash(symbol) % 2**32)  # Consistent per symbol
        
        data = []
        current_price = base_price
        
        for date in business_dates:
            # Random walk
            daily_return = np.random.normal(0.0005, 0.02)
            current_price *= (1 + daily_return)
            
            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.01)))
            low = current_price * (1 - abs(np.random.normal(0, 0.01)))
            open_price = current_price * (1 + np.random.normal(0, 0.005))
            volume = int(np.random.normal(1000000, 200000))
            
            data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(current_price, 2),
                'Volume': max(volume, 100000)
            })
        
        return data
    
    def _simulate_current_price(self, symbol: str) -> float:
        """Simulate current price for a symbol."""
        np.random.seed(hash(symbol) % 2**32)
        return round(100.0 * (1 + np.random.normal(0, 0.1)), 2)
    
    def _fetch_options_chain(self, symbol: str, expiration_date: str) -> List[Dict]:
        """Fetch options chain data (simulated implementation)."""
        current_price = self.get_current_stock_price(symbol)
        
        options_data = []
        strike_range = np.arange(
            current_price * 0.8,
            current_price * 1.3,
            max(1.0, current_price * 0.05)
        )
        
        exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
        time_to_exp = max(1/365, (exp_date - datetime.now()).days / 365.0)
        
        risk_free_rate = 0.05
        volatility = 0.25
        
        np.random.seed(42)
        
        for strike in strike_range:
            strike = round(strike, 2)
            
            # Simplified option pricing
            intrinsic_call = max(0, current_price - strike)
            intrinsic_put = max(0, strike - current_price)
            time_value = max(0.01, time_to_exp * volatility * current_price * 0.1)
            
            call_price = intrinsic_call + time_value
            put_price = intrinsic_put + time_value
            
            call_spread = call_price * 0.02 * np.random.uniform(0.5, 1.5)
            put_spread = put_price * 0.02 * np.random.uniform(0.5, 1.5)
            
            # Add call and put options
            for opt_type, price, spread in [('CALL', call_price, call_spread), ('PUT', put_price, put_spread)]:
                options_data.append({
                    'Symbol': symbol,
                    'Expiration': expiration_date,
                    'Strike': strike,
                    'Type': opt_type,
                    'Bid': round(max(0.01, price - spread/2), 2),
                    'Ask': round(price + spread/2, 2),
                    'Last': round(price + np.random.normal(0, spread/4), 2),
                    'Volume': int(np.random.exponential(100)),
                    'OpenInterest': int(np.random.exponential(500)),
                    'ImpliedVolatility': round(volatility + np.random.normal(0, 0.05), 4)
                })
        
        return options_data
    
    def _fetch_risk_free_rate(self, duration: str) -> float:
        """Fetch risk-free rate (simulated implementation)."""
        base_rates = {
            "1M": 0.045, "3M": 0.048, "6M": 0.050, "1Y": 0.052,
            "2Y": 0.055, "5Y": 0.058, "10Y": 0.062, "30Y": 0.065
        }
        
        np.random.seed(int(datetime.now().strftime('%Y%m%d')))
        variation = np.random.normal(0, 0.005)
        
        rate = base_rates.get(duration, 0.048) + variation
        return max(0.001, round(rate, 6))
    
    def _fetch_market_indicators(self) -> Dict[str, Any]:
        """Fetch market indicators (simulated implementation)."""
        np.random.seed(int(datetime.now().strftime('%Y%m%d')))
        
        indicators = {
            "VIX": max(10, min(80, 20 + np.random.normal(0, 5))),
            "SPX": max(3000, 4500 + np.random.normal(0, 100)),
            "NDX": max(10000, 15000 + np.random.normal(0, 500)),
            "DJI": max(25000, 35000 + np.random.normal(0, 800)),
            "USD_INDEX": max(90, min(120, 103 + np.random.normal(0, 2))),
            "GOLD": max(1500, 2000 + np.random.normal(0, 50)),
            "OIL_WTI": max(40, min(150, 80 + np.random.normal(0, 10))),
            "BONDS_10Y": self.get_risk_free_rate("10Y"),
            "last_updated": datetime.now().isoformat()
        }
        
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in indicators.items()}
    
    def _fetch_dividend_yield(self, symbol: str) -> float:
        """Fetch dividend yield (simulated implementation)."""
        np.random.seed(hash(symbol) % 2**32)
        
        dividend_yield = np.random.exponential(0.02)
        dividend_yield = min(dividend_yield, 0.08)
        
        # 30% chance of no dividend
        if np.random.random() < 0.3:
            dividend_yield = 0.0
        
        return round(dividend_yield, 4)


# Create a global instance for easy access
default_data_store = DataStore()


# === CONVENIENCE FUNCTIONS ===
# These functions provide a simple interface that other modules can use

def get_stock_price_history(symbol: str, start_date: str, end_date: str, **kwargs) -> pd.DataFrame:
    """Get historical stock price data using the default data store."""
    return default_data_store.get_stock_price_history(symbol, start_date, end_date, **kwargs)

def get_current_stock_price(symbol: str, **kwargs) -> float:
    """Get current stock price using the default data store."""
    return default_data_store.get_current_stock_price(symbol, **kwargs)

def calculate_historical_volatility(symbol: str, period_days: int = 252, **kwargs) -> float:
    """Calculate historical volatility using the default data store."""
    return default_data_store.calculate_historical_volatility(symbol, period_days, **kwargs)

def get_options_chain(symbol: str, expiration_date: str, **kwargs) -> pd.DataFrame:
    """Get options chain using the default data store."""
    return default_data_store.get_options_chain(symbol, expiration_date, **kwargs)

def get_option_quotes(symbol: str, strike: float, expiration_date: str, option_type: str, **kwargs) -> Dict[str, Any]:
    """Get option quotes using the default data store."""
    return default_data_store.get_option_quotes(symbol, strike, expiration_date, option_type, **kwargs)

def get_risk_free_rate(duration: str = "3M", **kwargs) -> float:
    """Get risk-free rate using the default data store."""
    return default_data_store.get_risk_free_rate(duration, **kwargs)

def get_market_indicators(**kwargs) -> Dict[str, Any]:
    """Get market indicators using the default data store."""
    return default_data_store.get_market_indicators(**kwargs)

def get_dividend_yield(symbol: str, **kwargs) -> float:
    """Get dividend yield using the default data store."""
    return default_data_store.get_dividend_yield(symbol, **kwargs)

def get_complete_market_data(symbol: str, expiration_date: str, **kwargs) -> Dict[str, Any]:
    """Get complete market data using the default data store."""
    return default_data_store.get_complete_market_data(symbol, expiration_date, **kwargs)