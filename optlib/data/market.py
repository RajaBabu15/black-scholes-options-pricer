# Market data functions (rates, dividends) and historical cache
try:
    from options_engine_torch import fetch_risk_free_rate, fetch_dividend_yield
except Exception:
    # Fallback no-op implementations
    def fetch_risk_free_rate(fallback_rate: float = 0.04) -> float:
        return fallback_rate
    def fetch_dividend_yield(ticker: str, fallback_yield: float = 0.0) -> float:
        return fallback_yield

