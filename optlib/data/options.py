import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf


def choose_expiries(stock: yf.Ticker, min_td=10, max_td=100, targets=(30, 60, 90)) -> List[str]:
    expiries = stock.options
    today = pd.Timestamp.utcnow().tz_localize(None)
    choices = []
    for t in targets:
        best = None
        best_diff = 1e9
        for exp in expiries:
            d = pd.Timestamp(exp)
            td = max((d - today).days, 0)
            td_trading = int(td * 5 / 7)
            if td_trading < min_td or td_trading > max_td:
                continue
            diff = abs(td_trading - t)
            if diff < best_diff:
                best = exp
                best_diff = diff
        if best is not None:
            choices.append(best)
    # Deduplicate while preserving order
    seen = set(); out = []
    for e in choices:
        if e not in seen:
            seen.add(e); out.append(e)
    return out


def fetch_clean_chain(stock: yf.Ticker, expiry: str, S0: float, r: float, q: float) -> Tuple[np.ndarray, np.ndarray, float]:
    oc = stock.option_chain(expiry)
    calls = oc.calls.copy()
    # mid-price
    calls['mid'] = np.where((calls['bid'] > 0) & (calls['ask'] > 0), (calls['bid'] + calls['ask']) / 2, calls['lastPrice'])
    # filters
    T_days = max((pd.Timestamp(expiry) - pd.Timestamp.utcnow().tz_localize(None)).days, 1)
    T = int(T_days * 5 / 7) / 252
    mny = (calls['strike'] / S0).astype(float)
    spread = (calls['ask'] - calls['bid']).abs()
    valid = (
        (mny >= 0.8) & (mny <= 1.2) &
        (calls['mid'] > 0.05) &
        (calls.get('volume', 0).fillna(0) > 10) &
        (np.where(calls['mid'] > 0, spread / calls['mid'], 1.0) < 0.3)
    )
    calls = calls[valid]
    if len(calls) == 0:
        return np.array([]), np.array([]), T
    strikes = calls['strike'].values.astype(float)
    mids = calls['mid'].values.astype(float)
    # remove near-zero time value deep ITM
    fwd = S0 * math.exp((r - q) * T)
    intrinsic = np.maximum(fwd - strikes * math.exp(-r * T), 0.0)
    time_val = mids - intrinsic
    keep = (time_val >= max(0.01, 0.001 * S0))
    strikes, mids = strikes[keep], mids[keep]
    return strikes, mids, T


def load_or_download_chain_clean(ticker: str, expiry: str, S0: float, r: float, q: float, stock: yf.Ticker, data_dir: str) -> Tuple[np.ndarray, np.ndarray, float]:
    import os
    fname = os.path.join(data_dir, f"{ticker}_chain_{expiry}.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        # Recompute T from current date to ensure no negative expiry
        try:
            T_days = max((pd.Timestamp(expiry) - pd.Timestamp.utcnow().tz_localize(None)).days, 1)
        except Exception:
            T_days = int(df.get('T_days', pd.Series([30])).iloc[0])
        T = int(T_days * 5 / 7) / 252
        strikes = df['strike'].values.astype(float)
        mids = df['mid'].values.astype(float)
        return strikes, mids, T
    # otherwise fetch and clean now
    Ks, mids, T = fetch_clean_chain(stock, expiry, S0, r, q)
    if len(Ks) > 0:
        out = pd.DataFrame({'strike': Ks, 'mid': mids})
        out['expiry'] = expiry
        out['T_days'] = int(max((pd.Timestamp(expiry) - pd.Timestamp.utcnow().tz_localize(None)).days, 1))
        out.to_csv(fname, index=False)
    return Ks, mids, T

