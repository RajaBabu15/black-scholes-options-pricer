import math
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import yfinance as yf
from optlib.pricing.iv import implied_vol_from_price
from optlib.models.heston import heston_char_func
from optlib.pricing.cos import cos_price_from_cf


def calibrate_heston_multi(stock: yf.Ticker, S0: float, r: float, q: float, expiries: List[str], strikes_by_exp: Dict[str, np.ndarray], prices_by_exp: Dict[str, np.ndarray], T_by_exp: Dict[str, float]) -> List[float]:
    # Build a fixed universe of (T, K, P, iv_mkt, vega_mkt) to ensure constant residual length and speed
    samples = []
    for exp in expiries:
        Ks = strikes_by_exp.get(exp, np.array([]))
        Ps = prices_by_exp.get(exp, np.array([]))
        T = T_by_exp.get(exp, None)
        if T is None or len(Ks) == 0:
            continue
        # Subsample up to 15 strikes per expiry for speed and stability
        idx = np.linspace(0, len(Ks) - 1, num=min(15, len(Ks)), dtype=int)
        for j in idx:
            K = float(Ks[j]); P = float(Ps[j])
            iv_mkt = implied_vol_from_price(P, S0, K, r, q, T)
            if not np.isfinite(iv_mkt):
                continue
            if T <= 1e-8:
                continue
            d1 = (math.log(S0 / K) + (r - q + 0.5 * iv_mkt * iv_mkt) * T) / (iv_mkt * math.sqrt(T))
            vega_mkt = S0 * math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)
            vega_mkt = max(vega_mkt, 1e-6)
            samples.append((T, K, P, iv_mkt, vega_mkt))
    if len(samples) == 0:
        return [3.0, 0.04, 0.4, -0.6, 0.04]

    params0 = np.array([3.0, 0.2 ** 2, 0.4, -0.6, 0.2 ** 2], dtype=float)
    bounds = ([0.1, 0.0001, 0.05, -0.99, 0.0001], [10.0, 1.0, 1.5, 0.0, 1.0])

    def objective(p):
        kappa, theta, sigma_v, rho, v0 = p
        if 2 * kappa * theta <= sigma_v ** 2 or theta > 1.0 or any(np.array(p) < np.array(bounds[0])) or any(np.array(p) > np.array(bounds[1])):
            return np.ones(len(samples)) * 1e3
        errs = []
        for T, K, P, iv_mkt, vega_mkt in samples:
            try:
                cf = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
                modelP = cos_price_from_cf(S0, float(K), r, q, T, cf, params=[kappa, theta, sigma_v, rho, v0])
                if not math.isfinite(modelP):
                    errs.append(10.0)
                    continue
                errs.append((modelP - P) / vega_mkt)
            except Exception:
                errs.append(10.0)
        return np.array(errs)

    from scipy.optimize import least_squares
    res = least_squares(objective, x0=params0, bounds=bounds, method='trf', max_nfev=80, loss='soft_l1', f_scale=0.05)
    if not res.success:
        return params0.tolist()
    return res.x.tolist()

