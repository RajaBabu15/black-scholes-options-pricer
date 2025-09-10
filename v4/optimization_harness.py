#!/usr/bin/env python3
import os
import math
import json
import time
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

import yfinance as yf

# Import core functions from the main engine
from hedge_heston_torch import (
    fetch_risk_free_rate,
    fetch_dividend_yield,
    implied_vol_from_price,
    heston_char_func,
    cos_price_from_cf,
    generate_heston_paths,
    delta_hedge_sim,
    calculate_performance_metrics,
    compute_per_path_deltas_scaling,
    tensor_dtype,
)

import torch

def choose_expiries(stock: yf.Ticker, min_td=10, max_td=100, targets=(30,60,90)) -> List[str]:
    expiries = stock.options
    today = pd.Timestamp.utcnow().tz_localize(None)
    choices = []
    for t in targets:
        best = None
        best_diff = 1e9
        for exp in expiries:
            d = pd.Timestamp(exp)
            td = max((d - today).days, 0)
            td_trading = int(td * 5/7)
            if td_trading < min_td or td_trading > max_td:
                continue
            diff = abs(td_trading - t)
            if diff < best_diff:
                best = exp
                best_diff = diff
        if best is not None:
            choices.append(best)
    # Deduplicate while preserving order
    seen = set(); out=[]
    for e in choices:
        if e not in seen:
            seen.add(e); out.append(e)
    return out

def fetch_clean_chain(stock: yf.Ticker, expiry: str, S0: float, r: float, q: float) -> Tuple[np.ndarray, np.ndarray, float]:
    oc = stock.option_chain(expiry)
    calls = oc.calls.copy()
    # mid-price
    calls['mid'] = np.where((calls['bid']>0)&(calls['ask']>0),(calls['bid']+calls['ask'])/2,calls['lastPrice'])
    # filters
    T_days = max((pd.Timestamp(expiry)-pd.Timestamp.utcnow().tz_localize(None)).days,1)
    T = int(T_days*5/7)/252
    mny = (calls['strike']/S0).astype(float)
    spread = (calls['ask']-calls['bid']).abs()
    valid = (
        (mny>=0.8) & (mny<=1.2) &
        (calls['mid']>0.05) &
        (calls.get('volume',0).fillna(0) > 10) &
        (np.where(calls['mid']>0, spread/calls['mid'], 1.0) < 0.3)
    )
    calls = calls[valid]
    if len(calls)==0:
        return np.array([]), np.array([]), T
    strikes = calls['strike'].values.astype(float)
    mids = calls['mid'].values.astype(float)
    # remove near-zero time value deep ITM
    fwd = S0*math.exp((r-q)*T)
    intrinsic = np.maximum(fwd - strikes*math.exp(-r*T), 0.0)
    time_val = mids - intrinsic
    keep = (time_val >= max(0.01, 0.001*S0))
    strikes, mids = strikes[keep], mids[keep]
    return strikes, mids, T

def calibrate_heston_multi(stock: yf.Ticker, S0: float, r: float, q: float, expiries: List[str], strikes_by_exp: Dict[str, np.ndarray], prices_by_exp: Dict[str, np.ndarray], T_by_exp: Dict[str, float]) -> List[float]:
    # Build a fixed universe of (T, K, P, iv_mkt, vega_mkt) to ensure constant residual length and speed
    samples = []
    for exp in expiries:
        Ks = strikes_by_exp.get(exp, np.array([]))
        Ps = prices_by_exp.get(exp, np.array([]))
        T = T_by_exp.get(exp, None)
        if T is None or len(Ks)==0:
            continue
        # Subsample up to 15 strikes per expiry for speed and stability
        idx = np.linspace(0, len(Ks)-1, num=min(15, len(Ks)), dtype=int)
        for j in idx:
            K = float(Ks[j]); P = float(Ps[j])
            iv_mkt = implied_vol_from_price(P, S0, K, r, q, T)
            if not np.isfinite(iv_mkt):
                continue
            # Black-Scholes vega at market IV
            if T <= 1e-8:
                continue
            d1 = (math.log(S0/K) + (r-q+0.5*iv_mkt*iv_mkt)*T) / (iv_mkt*math.sqrt(T))
            vega_mkt = S0*math.exp(-q*T)*math.exp(-0.5*d1*d1)/math.sqrt(2*math.pi)*math.sqrt(T)
            vega_mkt = max(vega_mkt, 1e-6)
            samples.append((T, K, P, iv_mkt, vega_mkt))
    if len(samples)==0:
        # Fallback default
        return [3.0, 0.04, 0.4, -0.6, 0.04]

    # initial guess (moderate)
    params0 = np.array([3.0, 0.2**2, 0.4, -0.6, 0.2**2], dtype=float)
    bounds = ([0.1, 0.0001, 0.05, -0.99, 0.0001], [10.0, 1.0, 1.5, 0.0, 1.0])

    def objective(p):
        kappa, theta, sigma_v, rho, v0 = p
        # Feller and theta cap
        if 2*kappa*theta <= sigma_v**2 or theta>1.0 or any(np.array(p)<np.array(bounds[0])) or any(np.array(p)>np.array(bounds[1])):
            return np.ones(len(samples))*1e3
        errs = []
        for T, K, P, iv_mkt, vega_mkt in samples:
            try:
                cf = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
                modelP = cos_price_from_cf(S0, float(K), r, q, T, cf, params=[kappa, theta, sigma_v, rho, v0])
                if not math.isfinite(modelP):
                    errs.append(10.0)
                    continue
                # Approximate IV error via price error / vega
                errs.append((modelP - P) / vega_mkt)
            except Exception:
                errs.append(10.0)
        return np.array(errs)

    from scipy.optimize import least_squares
    res = least_squares(objective, x0=params0, bounds=bounds, method='trf', max_nfev=80, loss='soft_l1', f_scale=0.05)
    if not res.success:
        return params0.tolist()
    return res.x.tolist()

def evaluate_configs(S_paths, v_paths, times, K, r, q, per_path_deltas, configs, verbose=True):
    rows=[]
    total = len(configs)
    for i, cfg in enumerate(configs, 1):
        rebal, tc, impact, scale, mode = cfg
        if verbose:
            print(f"  [{i}/{total}] rebal={rebal}, tc={tc:.4f}, impact={impact:.1e}, scale={scale:.2f}, mode={mode}", flush=True)
        try:
            t0 = time.time()
            with torch.no_grad():
                pnl,_ = delta_hedge_sim(S_paths, v_paths, times, K, r, q, tc=tc, impact_lambda=impact, rebal_freq=rebal, deltas_mode=mode, per_path_deltas=per_path_deltas if mode=='perpath' else None, exposure_scale=scale)
            met = calculate_performance_metrics(pnl, risk_free_rate=r)
            score = 0.5*met['sharpe_ratio'] + 0.3*met['sortino_ratio'] + 0.2*met['calmar_ratio']
            penalty = 0.0
            if met['max_drawdown']>0.2: penalty -= 0.5
            if met['annual_volatility']>0.20: penalty -= 0.5
            if verbose:
                print(f"     -> score={score+penalty:.3f}, sharpe={met['sharpe_ratio']:.3f}, calmar={met['calmar_ratio']:.3f}, vol={met['annual_volatility']:.3f} (took {time.time()-t0:.2f}s)", flush=True)
            rows.append({
                'rebal': rebal,
                'tc': tc,
                'impact': impact,
                'scale': scale,
                'mode': mode,
                'score': score+penalty,
                **met
            })
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.", flush=True)
            break
        except Exception as e:
            print(f"     -> skipped due to error: {e}", flush=True)
            rows.append({'rebal': rebal,'tc': tc,'impact': impact,'scale': scale,'mode': mode,'score': -999,'total_pnl':0.0,'annual_return':0.0,'annual_volatility':0.0,'sharpe_ratio':0.0,'sortino_ratio':0.0,'max_drawdown':0.0,'calmar_ratio':0.0})
    return pd.DataFrame(rows)

def main(ticker='AAPL'):
    print(f"=== Optimization Harness for {ticker} ===", flush=True)
    stock = yf.Ticker(ticker)
    r = fetch_risk_free_rate(fallback_rate=0.041)
    q = fetch_dividend_yield(ticker, fallback_yield=0.004)
    S0 = float(stock.history(period='5d')['Close'].iloc[-1])
    print(f"S0={S0:.2f}, r={r:.3f}, q={q:.4f}", flush=True)

    exps = choose_expiries(stock)
    if len(exps)==0:
        print("No suitable expiries found.", flush=True)
        return

    strikes_by_exp={}; prices_by_exp={}; T_by_exp={}
    for exp in exps:
        Ks, mids, T = fetch_clean_chain(stock, exp, S0, r, q)
        strikes_by_exp[exp] = Ks
        prices_by_exp[exp] = mids
        T_by_exp[exp] = T
        print(f"Expiry {exp}: {len(Ks)} options, T={T:.3f}y", flush=True)

    # Calibrate Heston across these expiries (FAST)
    t0=time.time()
    params = calibrate_heston_multi(stock, S0, r, q, exps, strikes_by_exp, prices_by_exp, T_by_exp)
    kappa, theta, sigma_v, rho, v0 = params
    print(f"Calibrated params: kappa={kappa:.3f}, theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}, v0={v0:.4f}", flush=True)
    print(f"Calibration took {time.time()-t0:.2f}s", flush=True)

    # Choose target expiry (closest to 60d)
    target_exp = exps[1] if len(exps)>1 else exps[0]
    T = T_by_exp[target_exp]
    Ks = strikes_by_exp[target_exp]
    mids = prices_by_exp[target_exp]
    if len(Ks)==0:
        print("No options in target expiry.", flush=True)
        return
    K = float(np.median(Ks))

    # FAST EVAL SETTINGS
    fast_eval = True
    steps_per_day_eval = 8 if fast_eval else 24
    n_paths_eval = 150 if fast_eval else 500

    # Simulate Heston paths
    print("Simulating Heston paths...", flush=True)
    n_steps = max(int(T*252*steps_per_day_eval), 120)
    n_paths = n_paths_eval
    t1=time.time()
    S_paths, v_paths = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps)
    times = torch.linspace(0., T, n_steps+1, dtype=tensor_dtype)
    print(f"Simulated {n_paths} paths with {n_steps} steps in {time.time()-t1:.2f}s", flush=True)

    # Build config grid (BS only for fast evaluation)
    rebal_list = [2,5,10]
    tc_list = [0.0005,0.001]
    impact_list = [0.0, 1e-6]
    scale_list = [0.8, 1.0]
    modes = ['bs']
    configs = [(a,b,c,d,e) for a in rebal_list for b in tc_list for c in impact_list for d in scale_list for e in modes]

    print(f"Evaluating {len(configs)} configs (fast BS-only)...", flush=True)
    t2=time.time()
    df = evaluate_configs(S_paths, v_paths, times, K, r, q, per_path_deltas=None, configs=configs)
    print(f"Evaluation completed in {time.time()-t2:.2f}s", flush=True)

    df_sorted = df.sort_values('score', ascending=False)
    print("Top 10 configs by score:\n", df_sorted.head(10), flush=True)

    # Optional: refine top-3 with per-path deltas and higher fidelity
    do_refine = False  # disable by default to avoid long runtimes
    if do_refine:
        try:
            top = df_sorted.head(3).copy()
            print("Refining top-3 with per-path deltas and higher fidelity...", flush=True)
            steps_per_day_final = 24
            n_paths_final = 400
            n_steps_f = max(int(T*252*steps_per_day_final), 240)
            print(f"Re-simulating {n_paths_final} paths x {n_steps_f} steps...", flush=True)
            S_paths_f, v_paths_f = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths_final, n_steps_f)
            times_f = torch.linspace(0., T, n_steps_f+1, dtype=tensor_dtype)
            print("Computing per-path deltas (this can take a while)...", flush=True)
            ppd = compute_per_path_deltas_scaling(S_paths_f.cpu().numpy(), K, times_f.cpu().numpy(), r, q, relative_eps=0.001)
            refined_rows=[]
            for _, row in top.iterrows():
                cfg_bs = (int(row['rebal']), float(row['tc']), float(row['impact']), float(row['scale']), 'bs')
                cfg_pp = (int(row['rebal']), float(row['tc']), float(row['impact']), float(row['scale']), 'perpath')
                for cfg in [cfg_bs, cfg_pp]:
                    pnl,_ = delta_hedge_sim(S_paths_f, v_paths_f, times_f, K, r, q, tc=cfg[1], impact_lambda=cfg[2], rebal_freq=cfg[0], deltas_mode=cfg[4], per_path_deltas=ppd if cfg[4]=='perpath' else None, exposure_scale=cfg[3])
                    met = calculate_performance_metrics(pnl, risk_free_rate=r)
                    score = 0.5*met['sharpe_ratio'] + 0.3*met['sortino_ratio'] + 0.2*met['calmar_ratio']
                    refined_rows.append({'mode': cfg[4], 'rebal': cfg[0], 'tc': cfg[1], 'impact': cfg[2], 'scale': cfg[3], 'score': score, **met})
            df_refined = pd.DataFrame(refined_rows).sort_values('score', ascending=False)
            print("Refined results:\n", df_refined.head(10), flush=True)
        except KeyboardInterrupt:
            print("Refinement interrupted by user.", flush=True)
        except Exception as e:
            print(f"Refinement skipped due to error: {e}", flush=True)

    # Acceptance gates
    best = df_sorted.iloc[0]
    ok = (best['sharpe_ratio']>=1.0 and best['sortino_ratio']>=1.0 and best['calmar_ratio']>=0.5 and best['max_drawdown']<=0.25 and best['annual_volatility']<=0.20)
    print("\nBest fast-eval config meets gates:", ok, flush=True)
    out_path = f"opt_results_{ticker}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved all config results to {out_path}", flush=True)
if __name__ == '__main__':
    main()

