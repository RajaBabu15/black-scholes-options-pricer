import os
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime
import yfinance as yf
from optlib.utils.cache import get_ticker_registry


def _load_csv_data(path: str) -> pd.DataFrame:
    """Helper function to load and process CSV data with consistent formatting."""
    df = pd.read_csv(path)
    # Handle incorrect CSV format - skip header rows and rename Price column to Date
    if 'Price' in df.columns and df.iloc[0]['Price'] == 'Ticker':
        # Skip the first 2 rows (header and ticker row)
        df = df.iloc[2:].copy()
        # Rename Price column to Date
        df = df.rename(columns={'Price': 'Date'})
    # Parse dates and set index
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    # Convert numeric columns to float
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _download_and_save_data(ticker: str, years: int, path: str) -> pd.DataFrame:
    """Helper function to download and save data from yfinance."""
    start = (datetime.utcnow().date().replace(day=1)).replace(year=datetime.utcnow().year - (years-1))
    df = yf.download(ticker, start=str(start))
    df.to_csv(path)
    return df

def load_data(tickers: List[str], years: int, data_dir: str) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    os.makedirs(data_dir, exist_ok=True)

    norm = [t.strip().upper() for t in (tickers or []) if t and t.strip()]
    if not norm:
        norm = ['AAPL']

    for tk in norm:
        path = os.path.join(data_dir, f"{tk}.csv")
        try:
            if os.path.exists(path):
                df = _load_csv_data(path)
                results[tk] = df
            else:
                df = _download_and_save_data(tk, years, path)
                results[tk] = df
        except Exception as e:
            continue

    if not results:
        tk = 'AAPL'
        path = os.path.join(data_dir, f"{tk}.csv")
        try:
            if os.path.exists(path):
                df = _load_csv_data(path)
            else:
                df = _download_and_save_data(tk, years, path)
            results[tk] = df
        except Exception:
            return {}

    return results


def load_data_with_tickers(tickers: List[str], years: int, data_dir: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, yf.Ticker]]:
    hist_data = load_data(tickers, years, data_dir)
    ticker_objects = {}
    
    # Use ticker registry to reuse ticker objects
    ticker_registry = get_ticker_registry()
    for ticker in hist_data.keys():
        ticker_objects[ticker] = ticker_registry.get_ticker(ticker)
    
    return hist_data, ticker_objects

