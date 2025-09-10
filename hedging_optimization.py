#!/usr/bin/env python3
import os
import math
import json
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime

import yfinance as yf

# Import modularized API
from optlib.data.market import fetch_risk_free_rate, fetch_dividend_yield
from optlib.data.options import choose_expiries, fetch_clean_chain, load_or_download_chain_clean
from optlib.pricing.iv import implied_vol_from_price
from optlib.models.heston import heston_char_func
from optlib.pricing.cos import cos_price_from_cf
from optlib.sim.paths import generate_heston_paths
from optlib.hedge.delta import delta_hedge_sim, compute_per_path_deltas_scaling
from optlib.metrics.performance import calculate_performance_metrics
from optlib.utils.tensor import tensor_dtype
from optlib.optimization.scale import optimize_exposure_scale
from optlib.calibration.heston import calibrate_heston_multi
from optlib.optimization.evaluate import evaluate_configs

import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)






def _log(msg: str, ticker: str):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(os.path.join(LOG_DIR, f"{ticker}.log"), 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass

def load_or_download_hist(ticker: str, years: int = 2) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}_hist.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
        return df
    start = (datetime.utcnow().date().replace(day=1)).replace(year=datetime.utcnow().year - (years-1))
    df = yf.download(ticker, start=str(start))
    df.to_csv(path)
    return df

def load_or_download_chain_clean(ticker: str, expiry: str, S0: float, r: float, q: float, stock: yf.Ticker) -> Tuple[np.ndarray, np.ndarray, float]:
    fname = os.path.join(DATA_DIR, f"{ticker}_chain_{expiry}.csv")
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        # Recompute T from current date to ensure no negative expiry
        try:
            T_days = max((pd.Timestamp(expiry)-pd.Timestamp.utcnow().tz_localize(None)).days,1)
        except Exception:
            T_days = int(df.get('T_days', pd.Series([30])).iloc[0])
        T = int(T_days*5/7)/252
        strikes = df['strike'].values.astype(float)
        mids = df['mid'].values.astype(float)
        return strikes, mids, T
    # otherwise fetch and clean now
    Ks, mids, T = fetch_clean_chain(stock, expiry, S0, r, q)
    if len(Ks)>0:
        out = pd.DataFrame({'strike': Ks, 'mid': mids})
        out['expiry'] = expiry
        out['T_days'] = int(max((pd.Timestamp(expiry)-pd.Timestamp.utcnow().tz_localize(None)).days,1))
        out.to_csv(fname, index=False)
    return Ks, mids, T

def main(ticker='AAPL'):
    _log(f"=== Optimization Harness for {ticker} ===", ticker)
    stock = yf.Ticker(ticker)
    r = fetch_risk_free_rate(fallback_rate=0.041)
    q = fetch_dividend_yield(ticker, fallback_yield=0.004)
    # Load or download historical to get S0
    hist = load_or_download_hist(ticker, years=2)
    S0 = float(hist['Close'].iloc[-1])
    _log(f"S0={S0:.2f}, r={r:.3f}, q={q:.4f}", ticker)

    exps = choose_expiries(stock)
    if len(exps)==0:
        _log("No suitable expiries found.", ticker)
        return None

    strikes_by_exp={}; prices_by_exp={}; T_by_exp={}
    for exp in exps:
        Ks, mids, T = load_or_download_chain_clean(ticker, exp, S0, r, q, stock)
        strikes_by_exp[exp] = Ks
        prices_by_exp[exp] = mids
        T_by_exp[exp] = T
        _log(f"Expiry {exp}: {len(Ks)} options, T={T:.3f}y", ticker)

    # Calibrate Heston across these expiries (FAST)
    t0=time.time()
    params = calibrate_heston_multi(stock, S0, r, q, exps, strikes_by_exp, prices_by_exp, T_by_exp)
    kappa, theta, sigma_v, rho, v0 = params
    _log(f"Calibrated params: kappa={kappa:.3f}, theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}, v0={v0:.4f}", ticker)
    _log(f"Calibration took {time.time()-t0:.2f}s", ticker)

    # Choose target expiry (closest to 60d)
    target_exp = exps[1] if len(exps)>1 else exps[0]
    T = T_by_exp[target_exp]
    Ks = strikes_by_exp[target_exp]
    mids = prices_by_exp[target_exp]
    if len(Ks)==0:
        print("No options in target expiry.", flush=True)
        return
    K = float(np.median(Ks))

    # FAST TRAINING/EVAL: Train exposure_scale for a small set of discrete rebal/tc/impact configs (no redundant BS-only grid)
    steps_per_day_eval = 8
    n_paths_eval = 150

    # Simulate Heston paths (evaluation set)
    _log("Simulating Heston paths...", ticker)
    n_steps = max(int(T*252*steps_per_day_eval), 120)
    n_paths = n_paths_eval
    t1=time.time()
    S_paths, v_paths = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps)
    times = torch.linspace(0., T, n_steps+1, dtype=tensor_dtype)
    _log(f"Simulated {n_paths} paths with {n_steps} steps in {time.time()-t1:.2f}s", ticker)

    # Small discrete set to try (trainable scale will be learned per config)
    rebal_list = [2,5,10]
    tc_list = [0.0005,0.001]
    impact_list = [0.0, 1e-6]

    # Prepare a small training set with different seed
    torch.manual_seed(int(time.time()) % 1_000_000)
    n_steps_train = max(int(T*252*steps_per_day_eval), 200)
    n_paths_train = 120
    S_paths_train, v_paths_train = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths_train, n_steps_train)
    times_train = torch.linspace(0., T, n_steps_train+1, dtype=tensor_dtype)
    periods_per_year = 252 * steps_per_day_eval

    rows = []
    for rebal in rebal_list:
        for tc_v in tc_list:
            for impact_v in impact_list:
                _log(f"Training scale for rebal={rebal}, tc={tc_v}, impact={impact_v}", ticker)
                try:
                    res = optimize_exposure_scale(S_paths_train, v_paths_train, times_train, K, r, q, rebal, 'bs', tc_v, impact_v, target_periods_per_year=periods_per_year, steps=100, lr=0.05)
                    scale_star = res['scale']
                    _log(f"   -> learned scale={scale_star:.3f}", ticker)
                    pnl_e, C0_e, step_ts_e, diag_e = delta_hedge_sim(S_paths, v_paths, times, K, r, q, tc=tc_v, impact_lambda=impact_v, rebal_freq=rebal, deltas_mode='bs', exposure_scale=scale_star, return_timeseries=True, anti_lookahead_checks=True)
                    ret_series = step_ts_e.mean(axis=0)
                    met_e = calculate_performance_metrics(ret_series, risk_free_rate=r, periods_per_year=float(periods_per_year))
                    sh, so, ca = met_e['sharpe_ratio'], met_e['sortino_ratio'], met_e['calmar_ratio']
                    dd, av = met_e['max_drawdown'], met_e['annual_volatility']
                    penalty = 0.0
                    if sh < 1.0: penalty += 10.0 * (1.0 - sh)**2
                    if sh > 4.0: penalty += 5.0 * (sh - 4.0)**2
                    if so < 1.5: penalty += 6.0 * (1.5 - so)**2
                    if ca < 1.0: penalty += 8.0 * (1.0 - ca)**2
                    if dd > 0.25: penalty += 12.0 * (dd - 0.25)**2
                    if av < 0.05: penalty += 6.0 * (0.05 - av)**2
                    if av > 0.25: penalty += 6.0 * (av - 0.25)**2
                    if sh < -0.5: penalty += 20.0 * abs(sh + 0.5)
                    if so < -0.5: penalty += 20.0 * abs(so + 0.5)
                    if ca < 0.0:  penalty += 15.0 * abs(ca)
                    if met_e['annual_return'] < 0.0: penalty += 10.0 * abs(met_e['annual_return'])
                    score = 0.5*sh + 0.3*so + 0.2*ca - penalty
                    meets_gates = (1.0 <= sh <= 3.0) and (1.5 <= so <= 4.0) and (ca >= 1.0) and (dd <= 0.25) and (0.05 <= av <= 0.25)
                    reasons = []
                    if not (1.0 <= sh <= 3.0):
                        reasons.append(f"Sharpe {sh:.3f} not in [1.0, 3.0]")
                    if not (1.5 <= so <= 4.0):
                        reasons.append(f"Sortino {so:.3f} not in [1.5, 4.0]")
                    if ca < 1.0:
                        reasons.append(f"Calmar {ca:.3f} < 1.0")
                    if dd > 0.25:
                        reasons.append(f"MaxDD {dd:.3f} > 0.25")
                    if not (0.05 <= av <= 0.25):
                        reasons.append(f"AnnVol {av:.3f} not in [0.05, 0.25]")
                    rejection_reason = "OK" if meets_gates else "; ".join(reasons) if reasons else "Did not meet gates"
                    rows.append({'rebal': rebal, 'tc': tc_v, 'impact': impact_v, 'scale': scale_star, 'mode': 'bs', 'score': score, 'penalty': penalty, 'meets_gates': meets_gates,
                                 **met_e,
                                 'rejection_reason': rejection_reason,
                                 'num_trades_mean': float(np.mean(diag_e['trades'])),
                                 'avg_spread_cost_mean': float(np.mean(diag_e['avg_spread_cost'])),
                                 'avg_impact_cost_mean': float(np.mean(diag_e['avg_impact_cost']))})
                except Exception as e:
                    _log(f"   -> training/eval failed: {e}", ticker)

    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    _log("Top 10 configs by score:", ticker)
    _log(str(df.head(10)), ticker)

    # Acceptance gates
    if not df.empty:
        best = df.iloc[0]
        ok = (best['sharpe_ratio']>=1.0 and best['sortino_ratio']>=1.0 and best['calmar_ratio']>=1.0 and best['max_drawdown']<=0.25 and 0.05<=best['annual_volatility']<=0.25)
    else:
        ok = False
    _log(f"Best config meets gates: {ok}", ticker)
    out_path = os.path.join(DATA_DIR, f"opt_results_{ticker}.csv")
    df.to_csv(out_path, index=False)
    _log(f"Saved all config results to {out_path}", ticker)

    # Save best config JSON
    try:
        df_sorted = df.sort_values('score', ascending=False)
        best_row = df_sorted[df_sorted['meets_gates']].head(1)
        if best_row.empty:
            best_row = df_sorted.head(1)
        best = best_row.to_dict(orient='records')[0]
        # include calibration
        best['calibration'] = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}
        best['ticker'] = ticker
        best['timestamp'] = datetime.utcnow().isoformat()
        json_path = os.path.join(DATA_DIR, f"{ticker}_best_config.json")
        with open(json_path, 'w') as f:
            json.dump(best, f, indent=2)
        _log(f"Saved best configuration to {json_path}", ticker)
    except Exception as e:
        _log(f"Failed to save best config: {e}", ticker)
    
    return best
def main_many(tickers: List[str], limit: int = 100):
    results = []
    for i, tk in enumerate(tickers[:limit], 1):
        _log(f"\n=== Running ticker {i}/{min(limit, len(tickers))}: {tk} ===", tk)
        try:
            res = main(tk)
            results.append(res)
        except Exception as e:
            _log(f"Error on {tk}: {e}", tk)
    # Write portfolio summary CSV
    rows = [r for r in results if r is not None]
    if rows:
        dfp = pd.DataFrame(rows)
        outp = os.path.join(DATA_DIR, f"portfolio_best_configs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        dfp.to_csv(outp, index=False)
        print(f"Saved portfolio best configs to {outp}")


def _default_100_tickers() -> List[str]:
    # Fall back to a static list of 100 liquid US tickers if no tickers.txt
    base = [
        'AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','BRK-B','LLY','AVGO',
        'JPM','V','JNJ','WMT','PG','XOM','MA','UNH','HD','ABBV',
        'CVX','MRK','PEP','COST','KO','PFE','BAC','TMO','CSCO','ACN',
        'DIS','ABT','DHR','MCD','CRM','AMD','NFLX','INTC','LIN','CMCSA',
        'ADBE','WFC','NKE','TXN','PM','NEE','IBM','HON','ORCL','AMAT',
        'UPS','MS','CAT','QCOM','SBUX','LOW','BLK','RTX','AMGN','CVS',
        'AVB','BKNG','SPGI','GS','PLD','MDT','T','MDLZ','DE','UNP',
        'GE','LMT','PYPL','BA','ADP','NOW','INTU','ISRG','MO','EL',
        'C','DUK','USB','ADI','MMC','PNC','ZTS','SO','CL','MMM',
        'GM','F','FDX','GMAB','ETN','APD','SCHW','REGN','CME','GILD'
    ]
    return base[:100]

if __name__ == '__main__':
    # Load tickers from data/tickers.txt if present, else use defaults
    tickers_path = os.path.join(DATA_DIR, 'tickers.txt')
    if os.path.exists(tickers_path):
        with open(tickers_path) as f:
            tickers = [line.strip().upper() for line in f if line.strip()]
    else:
        tickers = _default_100_tickers()
    main_many(tickers, limit=100)

