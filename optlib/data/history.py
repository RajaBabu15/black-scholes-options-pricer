import os
from typing import List, Dict
import pandas as pd
from datetime import datetime


def load_or_download_hist(ticker: str, years: int, data_dir: str) -> pd.DataFrame:
    path = os.path.join(data_dir, f"{ticker}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        return df
    start = (datetime.utcnow().date().replace(day=1)).replace(year=datetime.utcnow().year - (years-1))
    import yfinance as yf
    df = yf.download(ticker, start=str(start))
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(path)
    return df


def load_or_download_many(tickers: List[str], years: int, data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load or download historical data for a list of tickers.
    For each ticker:
      - Try to load data_dir/<TICKER>.csv
      - If missing or unreadable, download with yfinance and save to the same path
      - Append the loaded DataFrame to the results dict under the ticker key
    If no ticker could be loaded, try AAPL as a fallback (load or download) and return that.
    """
    results: Dict[str, pd.DataFrame] = {}
    os.makedirs(data_dir, exist_ok=True)

    # Normalize input list
    norm = [t.strip().upper() for t in (tickers or []) if t and t.strip()]
    if not norm:
        norm = ['AAPL']

    for tk in norm:
        path = os.path.join(data_dir, f"{tk}.csv")
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
                results[tk] = df
            else:
                # Download and persist
                start = (datetime.utcnow().date().replace(day=1)).replace(year=datetime.utcnow().year - (years-1))
                import yfinance as yf
                df = yf.download(tk, start=str(start))
                df.to_csv(path)
                results[tk] = df
        except Exception as e:
            # Skip ticker on error
            continue

    if not results:
        # Fallback to AAPL
        tk = 'AAPL'
        path = os.path.join(data_dir, f"{tk}.csv")
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
            else:
                start = (datetime.utcnow().date().replace(day=1)).replace(year=datetime.utcnow().year - (years-1))
                import yfinance as yf
                df = yf.download(tk, start=str(start))
                df.to_csv(path)
            results[tk] = df
        except Exception:
            # Return empty dict if even fallback fails
            return {}

    return results

