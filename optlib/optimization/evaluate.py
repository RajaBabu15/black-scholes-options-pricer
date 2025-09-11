import time
import numpy as np
import pandas as pd
import torch
from typing import List, Tuple
from optlib.hedge.delta import delta_hedge_sim
from optlib.metrics.performance import calculate_performance_metrics


def evaluate_configs(S_paths, v_paths, times, K, r, q, per_path_deltas, configs, verbose=True):
    rows = []
    total = len(configs)
    T = float(times[-1]) if hasattr(times, '__len__') else 1.0
    steps_per_day = max(int(round((len(times) - 1) / max(T * 252, 1e-8))), 1)
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
                    deltas_mode=mode, per_path_deltas=per_path_deltas if mode == 'perpath' else None,
                    exposure_scale=scale, return_timeseries=True, anti_lookahead_checks=True)
            if step_ts is None:
                raise RuntimeError("step time series not returned")
            np.random.seed(rand_seed)
            idx = np.random.choice(step_ts.shape[0], size=step_ts.shape[0], replace=True)
            step_series = step_ts[idx].mean(axis=0)
            met = calculate_performance_metrics(step_series, risk_free_rate=r, periods_per_year=float(periods_per_year))
            penalty = 0.0
            sh, so, ca = met['sharpe_ratio'], met['sortino_ratio'], met['calmar_ratio']
            dd, av = met['max_drawdown'], met['annual_volatility']
            if not np.isfinite(sh) or not np.isfinite(so) or not np.isfinite(ca) or not np.isfinite(dd) or not np.isfinite(av):
                raise RuntimeError("NaN/Inf metric detected")
            if scale > 1.0:
                raise RuntimeError("Exposure scale > 1.0 not allowed")
            if sh < 1.0: penalty += 10.0 * (1.0 - sh) ** 2
            if sh > 4.0: penalty += 5.0 * (sh - 4.0) ** 2
            if so < 1.5: penalty += 6.0 * (1.5 - so) ** 2
            if ca < 1.0: penalty += 8.0 * (1.0 - ca) ** 2
            if dd > 0.25: penalty += 12.0 * (dd - 0.25) ** 2
            if av < 0.05: penalty += 6.0 * (0.05 - av) ** 2
            if av > 0.25: penalty += 6.0 * (av - 0.25) ** 2
            if sh < -0.5: penalty += 20.0 * abs(sh + 0.5)
            if so < -0.5: penalty += 20.0 * abs(so + 0.5)
            if ca < 0.0:  penalty += 15.0 * abs(ca)
            if met['annual_return'] < 0.0: penalty += 10.0 * abs(met['annual_return'])
            if (sh < -1.0) or (so < -1.0) or (ca < 0.0):
                raise RuntimeError(f"Hard reject: sh={sh:.3f}, so={so:.3f}, ca={ca:.3f}")
            score = 0.5 * sh + 0.3 * so + 0.2 * ca - penalty
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

