import yfinance as yf
# Market data functions (rates, dividends)

def fetch_risk_free_rate(fallback_rate: float = 0.04) -> float:
    try:
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100.0
            if 0.001 <= rate <= 0.15:
                print(f"Fetched risk-free rate from Treasury: {rate:.3f} ({rate*100:.1f}%)")
                return rate
    except Exception as e:
        print(f"Warning: Could not fetch Treasury rate ({e}), using fallback")
    print(f"Using fallback risk-free rate: {fallback_rate:.3f} ({fallback_rate*100:.1f}%)")
    return fallback_rate


def fetch_dividend_yield(ticker: str, fallback_yield: float = 0.0) -> float:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        yield_fields = ['dividendYield', 'trailingAnnualDividendYield', 'forwardAnnualDividendYield']
        for field in yield_fields:
            if field in info and info[field] is not None:
                div_yield = float(info[field])
                if 0 <= div_yield <= 0.1:
                    print(f"Fetched dividend yield for {ticker}: {div_yield:.4f} ({div_yield*100:.2f}%)")
                    return div_yield
        if 'dividendRate' in info and 'currentPrice' in info:
            div_rate = info.get('dividendRate', 0)
            price = info.get('currentPrice', 1)
            if div_rate > 0 and price > 0:
                div_yield = div_rate / price
                print(f"Calculated dividend yield for {ticker}: {div_yield:.4f} ({div_yield*100:.2f}%)")
                return div_yield
    except Exception as e:
        print(f"Warning: Could not fetch dividend yield for {ticker} ({e}), using fallback")
    print(f"Using fallback dividend yield: {fallback_yield:.4f} ({fallback_yield*100:.2f}%)")
    return fallback_yield

