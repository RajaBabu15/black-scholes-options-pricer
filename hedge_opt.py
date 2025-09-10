#!/usr/bin/env python3
import os
from typing import List
from datetime import datetime
import pandas as pd

# High-level harness API
from optlib.harness.runner import run_hedging_optimization, run
from optlib.data.tickers import default_100_tickers
from optlib.data.history import load_or_download_hist

# Default IO dirs
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def main(ticker: str | None = None, tickers: List[str] | None = None, limit: int = 100):
    """Local-data-first unified entrypoint.
    - If `tickers` list is None or empty, default to ['AAPL'].
    - Else use the provided list.
    - For each ticker, try to load its CSV from DATA_DIR (<TICKER>.csv).
      Only include tickers whose CSV loads successfully.
    - Run the portfolio across the filtered tickers.
    """
    # Resolve list of tickers
    if tickers is None or len(tickers) == 0:
        resolved = ['AAPL']
    else:
        resolved = [t.strip().upper() for t in tickers if t and t.strip()]

    # Ensure data exists: load or download tickers; use AAPL fallback if needed
    from optlib.data.history import load_or_download_many
    data_map = load_or_download_many(resolved, years=2, data_dir=DATA_DIR)
    valid_tickers = list(data_map.keys())
    if not valid_tickers:
        print("[ERROR] No data available even after download attempts.")
        return []
    # Enforce limit and run portfolio
    valid_tickers = valid_tickers[:max(1, int(limit))]
    return run_many(tickers=valid_tickers, limit=len(valid_tickers), data_dir=DATA_DIR, log_dir=LOG_DIR)

if __name__ == '__main__':
    main(tickers=default_100_tickers())

