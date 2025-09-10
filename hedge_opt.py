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

def main(ticker='AAPL'):
    return run_hedging_optimization(ticker=ticker, data_dir=DATA_DIR, log_dir=LOG_DIR)
def main_many(tickers: List[str], limit: int = 100):
    return run_many(tickers=tickers, limit=limit, data_dir=DATA_DIR, log_dir=LOG_DIR)

if __name__ == '__main__':
    tickers_path = os.path.join(DATA_DIR, 'tickers.txt')
    if os.path.exists(tickers_path):
        with open(tickers_path) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    else:
        tickers = default_100_tickers()
    main_many(tickers, limit=100)

