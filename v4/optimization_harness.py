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

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Differentiable metrics (torch) ===

def sharpe_torch(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = returns.mean()
    sigma = returns.std(unbiased=False) + eps
    return mu / sigma

def sortino_torch(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = returns.mean()
    downside = returns[returns < 0.0]
    if downside.numel() == 0:
        ds = torch.tensor(1.0, device=returns.device)
    else:
        ds = downside.std(unbiased=False) + eps
    return mu / ds

def compound_equity_from_returns(returns: torch.Tensor, init: float = 1.0) -> torch.Tensor:
    # returns is 1D (time) torch tensor; equity compounding
    return torch.cumprod(1.0 + returns, dim=0) * init

def max_drawdown_torch(equity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # running peak via cumulative maximum
    peak = torch.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + eps)
    return dd.max()

def calmar_torch(ann_return: torch.Tensor, max_dd: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ann_return / (max_dd + eps)


def composite_loss(returns: torch.Tensor, periods_per_year: float) -> torch.Tensor:
    # returns: 1D time-series torch tensor
    sh = sharpe_torch(returns)
    so = sortino_torch(returns)
    eq = compound_equity_from_returns(returns)
    mdd = max_drawdown_torch(eq)
    ann_mu = returns.mean() * periods_per_year
    ann_vol = returns.std(unbiased=False) * torch.sqrt(torch.tensor(periods_per_year, device=returns.device))
    ca = calmar_torch(ann_mu, mdd)

    # penalties (target ranges)
    penalty = torch.tensor(0.0, device=returns.device)
    penalty += torch.where(sh < 1.0, 10.0 * (1.0 - sh) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(sh > 4.0, 5.0 * (sh - 4.0) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(so < 1.5, 6.0 * (1.5 - so) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ca < 1.0, 8.0 * (1.0 - ca) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ann_vol < 0.05, 6.0 * (0.05 - ann_vol) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ann_vol > 0.25, 6.0 * (ann_vol - 0.25) ** 2, torch.tensor(0.0, device=returns.device))

    # asymmetric heavy penalties for deeply negative performance
    penalty += torch.where(sh < -0.5, 20.0 * torch.abs(sh + 0.5), torch.tensor(0.0, device=returns.device))
    penalty += torch.where(so < -0.5, 20.0 * torch.abs(so + 0.5), torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ca < 0.0, 15.0 * torch.abs(ca), torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ann_mu < 0.0, 10.0 * torch.abs(ann_mu), torch.tensor(0.0, device=returns.device))

    # hard reject (return huge loss)
    if (sh.item() < -1.0) or (so.item() < -1.0) or (ca.item() < 0.0):
        return torch.tensor(1e6, device=returns.device)

    score = 0.5 * sh + 0.3 * so + 0.2 * ca
    loss = -score + penalty
    return loss


def optimize_exposure_scale(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year: float, steps: int = 200, lr: float = 5e-2) -> Dict:
    # Trainable exposure scale in (0,1) via sigmoid(param)
    scale_param = torch.nn.Parameter(torch.tensor(0.0, dtype=S_paths_t.dtype, device=S_paths_t.device))
    optimizer = torch.optim.Adam([scale_param], lr=lr)

    for step in range(steps):
        optimizer.zero_grad()
        scale = torch.sigmoid(scale_param)
        pnl_t, C0_t, step_ts_t, diag = delta_hedge_sim(
            S_paths_t, v_paths_t, times_t, K, r, q,
            tc=tc, impact_lambda=impact, rebal_freq=rebal_freq,
            deltas_mode=mode, per_path_deltas=None,
            exposure_scale=scale, return_timeseries=True, anti_lookahead_checks=True, return_torch=True)
        # average across paths -> time-series
        ret_ts = step_ts_t.mean(dim=0)
        loss = composite_loss(ret_ts, periods_per_year=target_periods_per_year)
        loss.backward()
        optimizer.step()
    # return learned scale
    return {'scale': torch.sigmoid(scale_param).detach().cpu().item()}

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
    # compute steps_per_year from timeline
    T = float(times[-1]) if hasattr(times, '__len__') else 1.0
    steps_per_day = max(int(round((len(times)-1) / max(T*252, 1e-8))), 1)
    periods_per_year = 252 * steps_per_day
    base_seed = int(time.time()) % 10_000_000
    for i, cfg in enumerate(configs, 1):
        rebal, tc, impact, scale, mode = cfg
        rand_seed = base_seed + i
        if verbose:
            print(f"  [{i}/{total}] rebal={rebal}, tc={tc:.4f}, impact={impact:.1e}, scale={scale:.2f}, mode={mode}, seed={rand_seed}", flush=True)
        try:
            t0 = time.time()
            with torch.no_grad():
                pnl, C0, step_ts, diag = delta_hedge_sim(
                    S_paths, v_paths, times, K, r, q,
                    tc=tc, impact_lambda=impact, rebal_freq=rebal,
                    deltas_mode=mode, per_path_deltas=per_path_deltas if mode=='perpath' else None,
                    exposure_scale=scale, return_timeseries=True, anti_lookahead_checks=True)
            # bootstrap across paths for this config to avoid reusing same seed deterministically
            if step_ts is None:
                raise RuntimeError("step time series not returned")
            np.random.seed(rand_seed)
            idx = np.random.choice(step_ts.shape[0], size=step_ts.shape[0], replace=True)
            step_series = step_ts[idx].mean(axis=0)  # average return across resampled paths
            # compute metrics on time series
            met = calculate_performance_metrics(step_series, risk_free_rate=r, periods_per_year=float(periods_per_year))
            # penalty function
            penalty = 0.0
            sh, so, ca = met['sharpe_ratio'], met['sortino_ratio'], met['calmar_ratio']
            dd, av = met['max_drawdown'], met['annual_volatility']
            # hard rejections
            if not np.isfinite(sh) or not np.isfinite(so) or not np.isfinite(ca) or not np.isfinite(dd) or not np.isfinite(av):
                raise RuntimeError("NaN/Inf metric detected")
            if scale > 1.0:
                raise RuntimeError("Exposure scale > 1.0 not allowed")
            # penalties (symmetric around target ranges)
            if sh < 1.0: penalty += 10.0 * (1.0 - sh)**2
            if sh > 4.0: penalty += 5.0 * (sh - 4.0)**2
            if so < 1.5: penalty += 6.0 * (1.5 - so)**2
            if ca < 1.0: penalty += 8.0 * (1.0 - ca)**2
            if dd > 0.25: penalty += 12.0 * (dd - 0.25)**2
            if av < 0.05: penalty += 6.0 * (0.05 - av)**2
            if av > 0.25: penalty += 6.0 * (av - 0.25)**2
            # asymmetric heavy penalties for deeply negative performance
            if sh < -0.5: penalty += 20.0 * abs(sh + 0.5)
            if so < -0.5: penalty += 20.0 * abs(so + 0.5)
            if ca < 0.0:  penalty += 15.0 * abs(ca)
            if met['annual_return'] < 0.0: penalty += 10.0 * abs(met['annual_return'])
            # hard rejections for hopeless configs
            if (sh < -1.0) or (so < -1.0) or (ca < 0.0):
                raise RuntimeError(f"Hard reject: sh={sh:.3f}, so={so:.3f}, ca={ca:.3f}")
            score = 0.5*sh + 0.3*so + 0.2*ca - penalty
            meets_gates = (1.0 <= sh <= 3.0) and (1.5 <= so <= 4.0) and (ca >= 1.0) and (dd <= 0.25) and (0.05 <= av <= 0.25)
            if verbose:
                print(f"     -> score={score:.3f}, pen={penalty:.3f}, Sharpe={sh:.3f}, Sortino={so:.3f}, Calmar={ca:.3f}, DD={dd:.3f}, Vol={av:.3f} (took {time.time()-t0:.2f}s)", flush=True)
            rows.append({
                'rebal': rebal,
                'tc': tc,
                'impact': impact,
                'scale': scale,
                'mode': mode,
                'random_seed': rand_seed,
                'score': score,
                'penalty': penalty,
                'meets_gates': bool(meets_gates),
                **met,
                'num_trades_mean': float(np.mean(diag['trades'])),
                'avg_spread_cost_mean': float(np.mean(diag['avg_spread_cost'])),
                'avg_impact_cost_mean': float(np.mean(diag['avg_impact_cost'])),
            })
        except KeyboardInterrupt:
            print("Evaluation interrupted by user.", flush=True)
            break
        except Exception as e:
            print(f"     -> REJECTED due to error: {e}", flush=True)
            rows.append({'rebal': rebal,'tc': tc,'impact': impact,'scale': scale,'mode': mode,'random_seed': rand_seed,'score': -999.0,'penalty': 0.0,'meets_gates': False,
                         'total_pnl':0.0,'annual_return':0.0,'annual_volatility':np.nan,'sharpe_ratio':np.nan,'sortino_ratio':np.nan,'max_drawdown':np.nan,'calmar_ratio':np.nan,
                         'num_trades_mean':0.0,'avg_spread_cost_mean':0.0,'avg_impact_cost_mean':0.0})
    return pd.DataFrame(rows)

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

