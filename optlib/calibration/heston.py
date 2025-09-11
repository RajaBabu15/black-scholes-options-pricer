import math
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import logging
from optlib.pricing.iv import implied_vol_from_price
from optlib.models.heston import heston_char_func
from optlib.pricing.cos import cos_price_from_cf
from scipy.optimize import least_squares

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calibrate_heston_multi(
    market_data: Dict[str, pd.DataFrame],
    S0: float,
    r: float,
    q: float,
    expiries: Optional[List[str]] = None,
    T_by_exp: Optional[Dict[str, float]] = None,
    strike_col: str = "strike",
    price_col: str = "price",
    T_col: str = "T",
    max_strikes_per_expiry: int = 15,
) -> List[float]:
    """
    Calibrate Heston parameters to market option data.
    market_data: dict mapping expiry keys -> pandas.DataFrame.
                 Each DataFrame must contain columns for strikes and prices (defaults: 'strike','price').
                 Optionally it can be provided via T_by_exp[expiry].
    S0, r, q: spot, domestic rate, dividend yield (used in pricing / implied vol).
    expiries: optional list of expiry keys to use; default = market_data.keys() order.
    Returns: list of calibrated params [kappa, theta, sigma_v, rho, v0] or default if fit fails.
    """
    if expiries is None:
        expiries = list(market_data.keys())
    if T_by_exp is None:
        T_by_exp = {}
    samples = []
    samples_by_expiry = {}  
    for exp in expiries:
        df = market_data.get(exp, None)
        if df is None or df.shape[0] == 0:
            # logger.debug(f"Skipping expiry {exp}: empty or missing data")
            continue
        if strike_col not in df.columns or price_col not in df.columns:
            # logger.debug(f"Skipping expiry {exp}: missing {strike_col} or {price_col}")
            continue
        if T_col in df.columns:
            Ts = df[T_col].astype(float).to_numpy()
        else:
            T_scalar = T_by_exp.get(exp, None)
            if T_scalar is None:
                # logger.debug(f"Skipping expiry {exp}: no T provided")
                continue
            Ts = np.full(len(df), float(T_scalar))
        Ks = df[strike_col].to_numpy(dtype=float)
        Ps = df[price_col].to_numpy(dtype=float)
        if len(Ks) == 0:
            # logger.debug(f"Skipping expiry {exp}: no strikes")
            continue
        idx = np.linspace(0, len(Ks) - 1, num=min(max_strikes_per_expiry, len(Ks)), dtype=int)
        exp_samples = []
        for j in idx:
            K = float(Ks[j])
            P = float(Ps[j])
            T = float(Ts[j])
            if T <= 1e-12:
                # logger.debug(f"Skipping strike {K} expiry {exp}: T={T} too small")
                continue
            if P <= 0:
                # logger.debug(f"Skipping strike {K} expiry {exp}: negative price {P}")
                continue
            try:
                iv_mkt = implied_vol_from_price(P, S0, K, r, q, T)
            except Exception as e:
                # logger.debug(f"Skipping strike {K} expiry {exp}: IV computation failed: {e}")
                continue
            if not np.isfinite(iv_mkt):
                # logger.debug(f"Skipping strike {K} expiry {exp}: non-finite IV {iv_mkt}")
                continue
            d1 = (math.log(S0 / K) + (r - q + 0.5 * iv_mkt * iv_mkt) * T) / (iv_mkt * math.sqrt(T))
            vega_mkt = S0 * math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)
            vega_mkt = max(vega_mkt, 1e-6)
            exp_samples.append((T, K, P, iv_mkt, vega_mkt))
            samples.append((T, K, P, iv_mkt, vega_mkt))
        if exp_samples:
            samples_by_expiry[exp] = exp_samples
    if len(samples) == 0:
        logger.warning("No valid samples, returning default params")
        return [3.0, 0.04, 0.4, -0.6, 0.04]
    params0 = np.array([3.0, 0.2 ** 2, 0.4, -0.6, 0.2 ** 2], dtype=float)
    lb = np.array([0.1, 0.0001, 0.05, -0.99, 0.0001], dtype=float)
    ub = np.array([10.0, 2.0, 1.5, 0.99, 1.0], dtype=float)

    eval_count = [0]  
    def objective(p):
        eval_count[0] += 1
        p = np.asarray(p, dtype=float)
        kappa, theta, sigma_v, rho, v0 = p

        feller_ratio = sigma_v ** 2 / (2 * kappa * theta) if (kappa * theta) > 0 else float('inf')
        errs = np.zeros(len(samples) + 1, dtype=float)
        if np.any(p < lb) or np.any(p > ub):
            errs[:] = 1e3
            return errs

        idx = 0
        for exp, exp_samples in samples_by_expiry.items():
            Ts = np.array([s[0] for s in exp_samples])
            Ks = np.array([s[1] for s in exp_samples])
            Ps = np.array([s[2] for s in exp_samples])
            vega_mkts = np.array([s[4] for s in exp_samples])
            T = Ts[0]  
            try:
                cf = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
                modelPs = cos_price_from_cf(S0, Ks, r, q, T, cf, params=p)
                if not np.all(np.isfinite(modelPs)):
                    errs[idx:idx+len(Ks)] = 10.0
                    idx += len(Ks)
                    continue
                errs[idx:idx+len(Ks)] = (modelPs - Ps) / vega_mkts
                idx += len(Ks)
            except Exception as e:
                # logger.debug(f"Pricing failed for expiry {exp}: {e}")
                errs[idx:idx+len(Ks)] = 10.0
                idx += len(Ks)

        rms_err = np.sqrt(np.mean(errs[:-1] ** 2)) if len(errs[:-1]) > 0 else 1.0
        penalty = 100.0 * rms_err * max(0.0, feller_ratio - 1.0)
        errs[-1] = 10.0 * penalty  
        if eval_count[0] % 100 == 0:
            print(f"Calibration progress: {eval_count[0]}/1000 evaluations, RMS error: {rms_err:.4f}")
        return errs

    res = least_squares(objective, x0=params0, bounds=(lb, ub),
                        method='trf', max_nfev=1000, loss='soft_l1', f_scale=0.05, ftol=1e-8)
    if not res.success:
        logger.warning("Optimization failed, returning initial params")
        return params0.tolist()
    logger.info(f"Calibrated params: {res.x.tolist()}")
    return res.x.tolist()
