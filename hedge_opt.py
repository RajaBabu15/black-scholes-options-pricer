#!/usr/bin/env python3
import os
import argparse
from typing import List
from datetime import datetime
import pandas as pd

# High-level harness API
from optlib.harness.runner import run
from optlib.data.tickers import default_100_tickers
from optlib.data.history import load_data_with_tickers

# Default IO dirs
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def main(tickers: List[str] | None = None, limit: int = 100, parallel: bool = True, max_workers: int = None):
    if tickers is None:
        tickers = default_100_tickers()

    data_map, ticker_objects = load_data_with_tickers(tickers, years=2, data_dir=DATA_DIR)
    
    if len(data_map) == 0:
        print("No data loaded. Exiting.")
        return []
    
    return run(data_map=data_map, ticker_objects=ticker_objects, limit=limit, data_dir=DATA_DIR, log_dir=LOG_DIR, 
              parallel=parallel, max_workers=max_workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run hedge optimization for multiple tickers')
    parser.add_argument('--limit', type=int, default=5,
                        help='Maximum number of tickers to process (default: 5 for testing)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--max-workers', type=int,
                        help='Maximum number of worker processes')
    parser.add_argument('--tickers', nargs='*',
                        help='Specific tickers to process (e.g., --tickers AAPL MSFT)')

    args = parser.parse_args()

    tickers = args.tickers if args.tickers else None
    parallel = not args.no_parallel


    results = main(tickers=tickers, limit=args.limit,
                   parallel=parallel, max_workers=args.max_workers)
