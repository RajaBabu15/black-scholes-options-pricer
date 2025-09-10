import os
import time
import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Tuple

from optlib.data.market import fetch_risk_free_rate, fetch_dividend_yield
from optlib.data.options import choose_expiries, load_or_download_chain_clean
from optlib.data.history import load_or_download_hist
from optlib.calibration.heston import calibrate_heston_multi
from optlib.sim.paths import generate_heston_paths
from optlib.utils.tensor import tensor_dtype
from optlib.optimization.scale import optimize_exposure_scale
from optlib.hedge.delta import delta_hedge_sim
from optlib.metrics.performance import calculate_performance_metrics


def log_message(msg: str, ticker: str, log_dir: str):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{ticker}.log"), 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass


def run_hedging_optimization(ticker: str, data_dir: str, log_dir: str) -> Dict:
    log = lambda m: log_message(m, ticker, log_dir)
    log(f"=== Optimization Harness for {ticker} ===")

    # Market params and S0
    r = fetch_risk_free_rate(fallback_rate=0.041)
    q = fetch_dividend_yield(ticker, fallback_yield=0.004)
    hist = load_or_download_hist(ticker, years=2, data_dir=data_dir)
    S0 = float(hist['Close'].iloc[-1])
    log(f"S0={S0:.2f}, r={r:.3f}, q={q:.4f}")

    # Option universe
    import yfinance as yf
    stock = yf.Ticker(ticker)
    exps = choose_expiries(stock)
    if len(exps) == 0:
        log("No suitable expiries found.")
        return None

    strikes_by_exp={}; prices_by_exp={}; T_by_exp={}
    for exp in exps:
        Ks, mids, T = load_or_download_chain_clean(ticker, exp, S0, r, q, stock, data_dir)
        strikes_by_exp[exp] = Ks
        prices_by_exp[exp] = mids
        T_by_exp[exp] = T
        log(f"Expiry {exp}: {len(Ks)} options, T={T:.3f}y")

    # Calibrate Heston
    t0=time.time()
    params = calibrate_heston_multi(stock, S0, r, q, exps, strikes_by_exp, prices_by_exp, T_by_exp)
    kappa, theta, sigma_v, rho, v0 = params
    log(f"Calibrated params: kappa={kappa:.3f}, theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}, v0={v0:.4f}")
    log(f"Calibration took {time.time()-t0:.2f}s")

    # Target expiry near 60d
    target_exp = exps[1] if len(exps)>1 else exps[0]
    T = T_by_exp[target_exp]
    Ks = strikes_by_exp[target_exp]
    if len(Ks)==0:
        print("No options in target expiry.", flush=True)
        return None
    K = float(np.median(Ks))

    # Simulation settings
    steps_per_day_eval = 8
    n_paths_eval = 150

    # Simulate eval set
    log("Simulating Heston paths...")
    n_steps = max(int(T*252*steps_per_day_eval), 120)
    n_paths = n_paths_eval
    t1=time.time()
    S_paths, v_paths = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps)
    times = torch.linspace(0., T, n_steps+1, dtype=tensor_dtype)
    log(f"Simulated {n_paths} paths with {n_steps} steps in {time.time()-t1:.2f}s")

    # Config grid
    rebal_list = [2,5,10]
    tc_list = [0.0005,0.001]
    impact_list = [0.0, 1e-6]

    # Training set
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
                log(f"Training scale for rebal={rebal}, tc={tc_v}, impact={impact_v}")
                try:
                    res = optimize_exposure_scale(S_paths_train, v_paths_train, times_train, K, r, q, rebal, 'bs', tc_v, impact_v, target_periods_per_year=periods_per_year, steps=100, lr=0.05)
                    scale_star = res['scale']
                    log(f"   -> learned scale={scale_star:.3f}")
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
                    if not (1.0 <= sh <= 3.0): reasons.append(f"Sharpe {sh:.3f} not in [1.0, 3.0]")
                    if not (1.5 <= so <= 4.0): reasons.append(f"Sortino {so:.3f} not in [1.5, 4.0]")
                    if ca < 1.0: reasons.append(f"Calmar {ca:.3f} < 1.0")
                    if dd > 0.25: reasons.append(f"MaxDD {dd:.3f} > 0.25")
                    if not (0.05 <= av <= 0.25): reasons.append(f"AnnVol {av:.3f} not in [0.05, 0.25]")
                    rejection_reason = "OK" if meets_gates else "; ".join(reasons) if reasons else "Did not meet gates"
                    rows.append({'rebal': rebal, 'tc': tc_v, 'impact': impact_v, 'scale': scale_star, 'mode': 'bs', 'score': score, 'penalty': penalty, 'meets_gates': meets_gates,
                                 **met_e,
                                 'rejection_reason': rejection_reason,
                                 'num_trades_mean': float(np.mean(diag_e['trades'])),
                                 'avg_spread_cost_mean': float(np.mean(diag_e['avg_spread_cost'])),
                                 'avg_impact_cost_mean': float(np.mean(diag_e['avg_impact_cost']))})
                except Exception as e:
                    log(f"   -> training/eval failed: {e}")

    df = pd.DataFrame(rows).sort_values('score', ascending=False)
    log("Top 10 configs by score:")
    log(str(df.head(10)))

    # Acceptance gates
    if not df.empty:
        best = df.iloc[0]
        ok = (best['sharpe_ratio']>=1.0 and best['sortino_ratio']>=1.0 and best['calmar_ratio']>=1.0 and best['max_drawdown']<=0.25 and 0.05<=best['annual_volatility']<=0.25)
    else:
        ok = False
    log(f"Best config meets gates: {ok}")

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"opt_results_{ticker}.csv")
    df.to_csv(out_path, index=False)
    log(f"Saved all config results to {out_path}")

    # Save best config JSON
    try:
        df_sorted = df.sort_values('score', ascending=False)
        best_row = df_sorted[df_sorted['meets_gates']].head(1)
        if best_row.empty:
            best_row = df_sorted.head(1)
        best = best_row.to_dict(orient='records')[0]
        best['calibration'] = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}
        best['ticker'] = ticker
        best['timestamp'] = datetime.utcnow().isoformat()
        json_path = os.path.join(data_dir, f"{ticker}_best_config.json")
        with open(json_path, 'w') as f:
            json.dump(best, f, indent=2)
        log(f"Saved best configuration to {json_path}")
    except Exception as e:
        log(f"Failed to save best config: {e}")

    return best


def run_many(tickers: List[str], limit: int, data_dir: str, log_dir: str):
    results = []
    for i, tk in enumerate(tickers[:limit], 1):
        log_message(f"\n=== Running ticker {i}/{min(limit, len(tickers))}: {tk} ===", tk, log_dir)
        try:
            res = run_hedging_optimization(tk, data_dir=data_dir, log_dir=log_dir)
            results.append(res)
        except Exception as e:
            log_message(f"Error on {tk}: {e}", tk, log_dir)
    rows = [r for r in results if r is not None]
    if rows:
        dfp = pd.DataFrame(rows)
        outp = os.path.join(data_dir, f"portfolio_best_configs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        dfp.to_csv(outp, index=False)
        print(f"Saved portfolio best configs to {outp}")
    return rows

