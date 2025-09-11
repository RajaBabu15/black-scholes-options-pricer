import os
import time
import json
import numpy as np
import pandas as pd
# from pandas.tseries import offsets  # Will import locally
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import cloudpickle  # For serialization
from optlib.data.market import fetch_risk_free_rate, fetch_dividend_yield
from optlib.data.options import choose_expiries, load_or_download_chain_clean
# from optlib.data.tickers import get_ticker_registry  # Not available
from optlib.calibration.heston import calibrate_heston_multi
from optlib.sim.paths import generate_heston_paths
from optlib.utils.tensor import tensor_dtype
from optlib.optimization.scale import optimize_exposure_scale
from optlib.hedge.delta import delta_hedge_sim
from optlib.metrics.performance import calculate_performance_metrics

def log_message(msg: str, ticker: str, log_dir: str):
    ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{ts}] {msg}"
    console_line = f"[{ts}] [{ticker}] {msg}"
    print(console_line, flush=True)
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, f"{ticker}.log"), 'a') as f:
            f.write(line + '\n')
    except Exception:
        pass

def run_hedging_optimization(ticker: str, hist_data: pd.DataFrame, stock_ticker: object, data_dir: str, log_dir: str, risk_free_rate: float) -> Dict:
    log = lambda m: log_message(m, ticker, log_dir)
    log(f"=== Optimization Harness for {ticker} ===")
    # Market params and S0 - use passed risk_free_rate to avoid redundant fetching
    r = risk_free_rate
    q = fetch_dividend_yield(ticker, fallback_yield=0.004)
    hist = hist_data
    S0 = float(hist['Close'].iloc[-1])
    log(f"S0={S0:.2f}, r={r:.3f}, q={q:.4f}")
    # Option universe - use pre-created ticker object
    stock = stock_ticker
    exps = choose_expiries(stock)
    if len(exps) == 0:
        log("No suitable expiries found.")
        return None
    # Build market_data: expiry -> DataFrame(strike, price, T)
    market_data = {}
    T_by_exp = {}
    for exp in exps:
        Ks, mids, T = load_or_download_chain_clean(ticker, exp, S0, r, q, stock, data_dir)
        # Guard against empty/None and filter liquidity/arb
        if Ks is None or len(Ks) == 0 or mids is None or len(mids) == 0:
            log(f"Expiry {exp}: no chain returned (skipping).")
            continue
        # Ensure arrays are numeric and same length
        Ks = np.asarray(Ks, dtype=float)
        mids = np.asarray(mids, dtype=float)
        if Ks.shape[0] != mids.shape[0]:
            log(f"Expiry {exp}: strikes/prices length mismatch (skipping).")
            continue
        # Filter: positive prices, no arb (P >= intrinsic), reasonable moneyness
        F = S0 * np.exp((r - q) * T)
        intrinsic_call = np.maximum(Ks - F * np.exp(-r * T), 0)
        intrinsic_put = np.maximum(F * np.exp(-r * T) - Ks, 0)
        valid = (mids > 0) & (mids >= np.minimum(intrinsic_call, intrinsic_put) * 0.9) & (np.abs(np.log(Ks / F)) < 0.4)
        Ks, mids = Ks[valid], mids[valid]
        if len(Ks) == 0:
            log(f"Expiry {exp}: no valid options after filter (skipping).")
            continue
        df_exp = pd.DataFrame({
            'strike': Ks,
            'price': mids,
            'T': np.full(len(Ks), T)  # Add T column
        })
        market_data[exp] = df_exp
        T_by_exp[exp] = float(T)
        log(f"Expiry {exp}: {len(Ks)} valid options, T={T:.3f}y")
    if len(market_data) == 0:
        log("No market data built for any expiry.")
        return None
    # Calibrate Heston
    t0 = time.time()
    params = calibrate_heston_multi(market_data, S0, r, q, expiries=list(market_data.keys()), T_by_exp=T_by_exp)
    kappa, theta, sigma_v, rho, v0 = params
    log(f"Calibrated params: kappa={kappa:.3f}, theta={theta:.4f}, sigma_v={sigma_v:.3f}, rho={rho:.3f}, v0={v0:.4f}")
    log(f"Calibration took {time.time() - t0:.2f}s")
    # Choose a target expiry (2nd if available) and ATM-forward strike
    exps_list = list(market_data.keys())
    target_exp = exps_list[1] if len(exps_list) > 1 else exps_list[0]
    Ks_arr = market_data[target_exp]['strike'].values
    if len(Ks_arr) == 0:
        log("No options in target expiry.")
        return None
    F = S0 * np.exp((r - q) * T_by_exp[target_exp])
    logK_F = np.log(Ks_arr / F)
    K_idx = np.argmin(np.abs(logK_F))
    K = float(Ks_arr[K_idx])
    T = float(T_by_exp[target_exp])
    log(f"Target: expiry={target_exp}, ATM K={K:.2f} (F={F:.2f})")
    # Original simulation settings - full computational load
    steps_per_day = 1  # Daily for realism
    n_paths_eval = 1000
    n_paths_train = 2000  # Restored original training paths
    # Business day steps - full resolution
    from datetime import date
    start_date = date.today()
    # Simple approximation: use business days
    n_business_days = int(T * 252)
    end_date = start_date + pd.Timedelta(days=int(n_business_days * 1.4))  # Account for weekends
    dates = pd.bdate_range(start_date, periods=n_business_days)
    n_steps_eval = len(dates) - 1
    n_steps_train = max(n_steps_eval, 252)  # At least yearly resolution
    periods_per_year = 252
    torch.manual_seed(42 + hash(ticker))  # Reproducible
    log(f"Simulation setup: {n_paths_train} train paths, {n_steps_train} steps; {n_paths_eval} eval paths, {n_steps_eval} steps")
    
    # Generate paths using vectorized method
    log("Simulating Heston paths...")
    S_paths_eval, v_paths_eval = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths_eval, n_steps_eval)
    times_eval = torch.linspace(0.0, T, n_steps_eval + 1, dtype=tensor_dtype, device=S_paths_eval.device)
    
    S_paths_train, v_paths_train = generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths_train, n_steps_train)
    times_train = torch.linspace(0.0, T, n_steps_train + 1, dtype=tensor_dtype, device=S_paths_train.device)
    # Extreme configuration grid - targeting Sharpe=1.0 and Return=30%
    rebal_list = [0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5, 7, 10]  # Including sub-daily rebalancing
    tc_list = [0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001]  # Very low transaction costs
    impact_list = [0.0, 1e-8, 1e-7, 5e-7, 1e-6]  # Minimal market impact
    rows = []
    
    total_configs = len(rebal_list) * len(tc_list) * len(impact_list)
    log(f"Testing {total_configs} configurations (was 2, now restored to original {total_configs})")
    config_num = 0
    
    for rebal in rebal_list:
        for tc_v in tc_list:
            for impact_v in impact_list:
                config_num += 1
                log(f"[{config_num}/{total_configs}] Training scale for rebal={rebal}, tc={tc_v}, impact={impact_v}")
                try:
                    # Optimize exposure scale
                    res = optimize_exposure_scale(
                        S_paths_train, v_paths_train, times_train, K, r, q,
                        rebal, 'bs', tc_v, impact_v, target_periods_per_year=periods_per_year
                    )
                    scale_star = float(res['scale'])
                    log(f" -> learned scale={scale_star:.3f}")
                    # Evaluate on evaluation set with vectorized simulation
                    pnl_e, C0_e, step_ts_e, diag_e = delta_hedge_sim(
                        S_paths_eval, v_paths_eval, times_eval, K, r, q,
                        tc=tc_v, impact_lambda=impact_v, rebal_freq=rebal,
                        exposure_scale=scale_star, return_timeseries=True, return_torch=False
                    )
                    if step_ts_e is None:
                        raise RuntimeError("step time series not returned")
                    ret_series = step_ts_e.mean(axis=0)
                    met_e = calculate_performance_metrics(ret_series, risk_free_rate=r, periods_per_year=float(periods_per_year))
                    # Target performance score: Sharpe=1.0, Return=30%
                    sh, so, ca = met_e['sharpe_ratio'], met_e['sortino_ratio'], met_e['calmar_ratio']
                    ann_ret = met_e.get('annualized_return', -0.5)
                    max_dd = met_e.get('max_drawdown', 0.2)
                    volatility = met_e.get('annual_volatility', 0.3)
                    
                    # Handle NaN values
                    if np.isnan(sh): sh = -10
                    if np.isnan(so): so = -10
                    if np.isnan(ca): ca = -10
                    if np.isnan(ann_ret): ann_ret = -0.5
                    if np.isnan(max_dd): max_dd = 0.5
                    if np.isnan(volatility): volatility = 0.3
                    
                    # Target-based scoring with penalties
                    target_sharpe = 1.0
                    target_return = 0.30
                    target_max_dd = 0.10
                    
                    # Sharpe component (40% weight)
                    sharpe_diff = sh - target_sharpe
                    if sh >= target_sharpe:
                        sharpe_score = 100 + min(sharpe_diff * 50, 200)
                    else:
                        sharpe_score = sharpe_diff * 200
                    
                    # Return component (30% weight)
                    return_diff = ann_ret - target_return
                    if ann_ret >= target_return:
                        return_score = 100 + min(return_diff * 100, 100)
                    else:
                        return_score = return_diff * 300
                    
                    # Drawdown component (20% weight)
                    dd_diff = max_dd - target_max_dd
                    if max_dd <= target_max_dd:
                        dd_score = 50 - dd_diff * 200
                    else:
                        dd_score = -dd_diff * 500
                    
                    # Consistency component (10% weight)
                    consistency_score = min(so * 10, 50) + min(ca * 10, 50)
                    
                    # Combine with penalties
                    score = (0.4 * sharpe_score + 0.3 * return_score + 0.2 * dd_score + 0.1 * consistency_score)
                    
                    # Additional penalties
                    if volatility > 0.8: score -= 100
                    if ann_ret < -0.3: score -= 200
                    if max_dd > 0.5: score -= 500
                    rows.append({
                        'rebal': rebal,
                        'tc': tc_v,
                        'impact': impact_v,
                        'scale': scale_star,
                        'mode': 'bs',
                        'score': score,
                        **met_e,
                        'num_trades_mean': float(np.mean(diag_e['trades'])),
                        'avg_spread_cost_mean': float(np.mean(diag_e['avg_spread_cost'])),
                        'avg_impact_cost_mean': float(np.mean(diag_e['avg_impact_cost'])),
                    })
                except Exception as e:
                    log(f" -> training/eval failed: {e}")
                    continue
    df = pd.DataFrame(rows).sort_values('score', ascending=False) if rows else pd.DataFrame(columns=['score'])
    log(f"Completed {len(rows)} configurations successfully")
    if len(rows) > 0:
        log(f"Top 5 configs by score:")
        for i, (_, row) in enumerate(df.head(5).iterrows()):
            log(f"  {i+1}. rebal={row['rebal']}, tc={row['tc']:.4f}, scale={row['scale']:.3f}, score={row['score']:.4f}")
    # Save results
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, f"opt_results_{ticker}.csv")
    df.to_csv(out_path, index=False)
    log(f"Saved all config results to {out_path}")
    # Save best config JSON
    if not df.empty:
        try:
            df_sorted = df.sort_values('score', ascending=False)
            best = df_sorted.head(1).to_dict(orient='records')[0]
            best['calibration'] = {'kappa': kappa, 'theta': theta, 'sigma_v': sigma_v, 'rho': rho, 'v0': v0}
            best['ticker'] = ticker
            best['timestamp'] = datetime.utcnow().isoformat()
            json_path = os.path.join(data_dir, f"{ticker}_best_config.json")
            with open(json_path, 'w') as f:
                json.dump(best, f, indent=2)
            log(f"Saved best configuration to {json_path}")
        except Exception as e:
            log(f"Failed to save best config: {e}")
    return df.head(1).to_dict(orient='records')[0] if not df.empty else None

def run_single_ticker(args):
    """Wrapper for parallel processing"""
    ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate = args
    # Deserialize DataFrame and recreate ticker object
    hist_data = cloudpickle.loads(hist_data_bytes)
    import yfinance as yf
    stock_ticker = yf.Ticker(ticker)
    return run_hedging_optimization(ticker, hist_data, stock_ticker, data_dir, log_dir, risk_free_rate)

def run(data_map: Dict[str, pd.DataFrame], ticker_objects: Dict[str, object], limit: int, data_dir: str, log_dir: str, parallel: bool = True, max_workers: int = None):
    tickers = list(data_map.keys())[:limit]
    print(f"Processing {len(tickers)} tickers: {tickers[:5]}{'...' if len(tickers) > 5 else ''}")
   
    # Fetch risk-free rate once
    print("Fetching risk-free rate...")
    risk_free_rate = fetch_risk_free_rate(fallback_rate=0.041)
    print(f"Using risk-free rate: {risk_free_rate:.3f} ({risk_free_rate*100:.1f}%)")
   
    if parallel and len(tickers) > 1:
        # Prepare serializable args
        args_list = []
        for tk in tickers:
            hist_data_bytes = cloudpickle.dumps(data_map[tk])
            args_list.append((tk, hist_data_bytes, data_dir, log_dir, risk_free_rate))
       
        if max_workers is None:
            max_workers = min(cpu_count(), len(tickers))
       
        print(f"Running in parallel with {max_workers} workers...")
        results = []
       
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(run_single_ticker, args): args[0]  # Submit without timeout
                               for args in args_list}
           
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                ticker = future_to_ticker[future]
                print(f"[{i}/{len(tickers)}] Completed: {ticker}")
                try:
                    result = future.result(timeout=300)  # Apply timeout here
                    if result is not None:
                        results.append(result)
                except TimeoutError:
                    print(f"Timeout on {ticker}")
                    log_message(f"Timeout on {ticker}", ticker, log_dir)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    log_message(f"Parallel error on {ticker}: {e}", ticker, log_dir)
    else:
        # Sequential
        print("Running sequentially...")
        results = []
        for i, tk in enumerate(tickers, 1):
            log_message(f"\n=== Running ticker {i}/{len(tickers)}: {tk} ===", tk, log_dir)
            try:
                res = run_hedging_optimization(tk, data_map[tk], ticker_objects[tk], data_dir=data_dir, log_dir=log_dir, risk_free_rate=risk_free_rate)
                if res is not None:
                    results.append(res)
            except Exception as e:
                log_message(f"Error on {tk}: {e}", tk, log_dir)
                print(f"Error processing {tk}: {e}")
   
    # Save results
    print(f"\nCompleted processing. Got {len(results)} successful results.")
    if results:
        dfp = pd.DataFrame(results)
        outp = os.path.join(data_dir, f"portfolio_best_configs_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        dfp.to_csv(outp, index=False)
        print(f"Saved portfolio best configs to {outp}")
    return results
