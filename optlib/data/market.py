import yfinance as yf
from optlib.utils.cache import get_session_cache, get_ticker_registry
# Market data functions (rates, dividends)

def fetch_risk_free_rate(fallback_rate: float = 0.04) -> float:
    """Fetch risk-free rate with caching to avoid repeated network calls."""
    cache = get_session_cache()
    cache_key = "risk_free_rate"
    
    # Check cache first
    cached_rate = cache.get(cache_key)
    if cached_rate is not None:
        print(f"Using cached risk-free rate: {cached_rate:.3f} ({cached_rate*100:.1f}%)")
        return cached_rate
    
    try:
        ticker_registry = get_ticker_registry()
        treasury = ticker_registry.get_ticker("^TNX")
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100.0
            if 0.001 <= rate <= 0.15:
                print(f"Fetched risk-free rate from Treasury: {rate:.3f} ({rate*100:.1f}%)")
                # Cache for 1 hour
                cache.set(cache_key, rate, ttl=3600)
                return rate
    except Exception as e:
        print(f"Warning: Could not fetch Treasury rate ({e}), using fallback")
    
    print(f"Using fallback risk-free rate: {fallback_rate:.3f} ({fallback_rate*100:.1f}%)")
    # Cache fallback rate for shorter time (15 minutes)
    cache.set(cache_key, fallback_rate, ttl=900)
    return fallback_rate


def fetch_dividend_yield(ticker: str, fallback_yield: float = 0.0) -> float:
    """Fetch dividend yield with caching to avoid repeated network calls."""
    cache = get_session_cache()
    cache_key = f"dividend_yield_{ticker}"
    
    # Check cache first
    cached_yield = cache.get(cache_key)
    if cached_yield is not None:
        print(f"Using cached dividend yield for {ticker}: {cached_yield:.4f} ({cached_yield*100:.2f}%)")
        return cached_yield
    
    try:
        ticker_registry = get_ticker_registry()
        stock = ticker_registry.get_ticker(ticker)
        info = stock.info
        yield_fields = ['dividendYield', 'trailingAnnualDividendYield', 'forwardAnnualDividendYield']
        for field in yield_fields:
            if field in info and info[field] is not None:
                div_yield = float(info[field])
                if 0 <= div_yield <= 0.1:
                    print(f"Fetched dividend yield for {ticker}: {div_yield:.4f} ({div_yield*100:.2f}%)")
                    # Cache for 1 hour
                    cache.set(cache_key, div_yield, ttl=3600)
                    return div_yield
        if 'dividendRate' in info and 'currentPrice' in info:
            div_rate = info.get('dividendRate', 0)
            price = info.get('currentPrice', 1)
            if div_rate > 0 and price > 0:
                div_yield = div_rate / price
                print(f"Calculated dividend yield for {ticker}: {div_yield:.4f} ({div_yield*100:.2f}%)")
                # Cache for 1 hour
                cache.set(cache_key, div_yield, ttl=3600)
                return div_yield
    except Exception as e:
        print(f"Warning: Could not fetch dividend yield for {ticker} ({e}), using fallback")
    
    print(f"Using fallback dividend yield: {fallback_yield:.4f} ({fallback_yield*100:.2f}%)")
    # Cache fallback for shorter time (15 minutes)
    cache.set(cache_key, fallback_yield, ttl=900)
    return fallback_yield

