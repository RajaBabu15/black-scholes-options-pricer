#!/usr/bin/env python3
import os
from typing import List
from datetime import datetime
import pandas as pd

# High-level harness API
from optlib.harness.runner import run_hedging_optimization, run_many
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
    - For each ticker, try to load its CSV from DATA_DIR (<TICKER>_hist.csv).
      Only include tickers whose CSV loads successfully.
    - Run the portfolio across the filtered tickers.
    """
    # Resolve list of tickers
    if tickers is None or len(tickers) == 0:
        resolved = ['AAPL']
    else:
        resolved = [t.strip().upper() for t in tickers if t and t.strip()]

    # Try to load local CSVs; keep only those that succeed
    valid_tickers: List[str] = []
    for tk in resolved:
        csv_path = os.path.join(DATA_DIR, f"{tk}_hist.csv")
        try:
            if os.path.exists(csv_path):
                # Try loading to validate
                _df = pd.read_csv(csv_path)
                valid_tickers.append(tk)
            else:
                print(f"[WARN] Missing local data file: {csv_path} — skipping {tk}")
        except Exception as e:
            print(f"[WARN] Failed to load {csv_path}: {e} — skipping {tk}")

    if not valid_tickers:
        # Nothing local available; fall back to AAPL if present
        fallback_csv = os.path.join(DATA_DIR, 'AAPL_hist.csv')
        if os.path.exists(fallback_csv):
            valid_tickers = ['AAPL']
            print("[INFO] Falling back to AAPL only.")
        else:
            print("[ERROR] No local CSVs found in data/. Aborting.")
            return []

    # Enforce limit and run portfolio
    valid_tickers = valid_tickers[:max(1, int(limit))]
    return run_many(tickers=valid_tickers, limit=len(valid_tickers), data_dir=DATA_DIR, log_dir=LOG_DIR)

if __name__ == '__main__':
    main()

