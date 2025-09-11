import math
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from optlib.pricing.iv import implied_vol_from_price
from optlib.models.heston import heston_char_func
from optlib.pricing.cos import cos_price_from_cf
from scipy.optimize import least_squares


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
                 Optionally it can include a column named `T` giving time-to-expiry per row; if absent,
                 a scalar T can be provided via T_by_exp[expiry].
    S0, r, q: spot, domestic rate, dividend yield (used in pricing / implied vol).
    expiries: optional list of expiry keys to use; default = market_data.keys() order.
    Returns: list of calibrated params [kappa, theta, sigma_v, rho, v0] or default if fit fails.
    """

    if expiries is None:
        expiries = list(market_data.keys())
    if T_by_exp is None:
        T_by_exp = {}

    samples = []
    for exp in expiries:
        df = market_data.get(exp, None)
        if df is None or df.shape[0] == 0:
            continue
        # check required columns
        if strike_col not in df.columns or price_col not in df.columns:
            # skip expiry if required columns missing
            continue

        # get time-to-expiry per-row or fallback to T_by_exp[exp]
        if T_col in df.columns:
            Ts = df[T_col].astype(float).to_numpy()
        else:
            T_scalar = T_by_exp.get(exp, None)
            if T_scalar is None:
                continue
            Ts = np.full(len(df), float(T_scalar))

        Ks = df[strike_col].to_numpy(dtype=float)
        Ps = df[price_col].to_numpy(dtype=float)

        if len(Ks) == 0:
            continue

        # subsample for stability/speed
        idx = np.linspace(0, len(Ks) - 1, num=min(max_strikes_per_expiry, len(Ks)), dtype=int)
        for j in idx:
            K = float(Ks[j])
            P = float(Ps[j])
            T = float(Ts[j])
            if T <= 1e-12:
                continue
            try:
                iv_mkt = implied_vol_from_price(P, S0, K, r, q, T)
            except Exception:
                continue
            if not np.isfinite(iv_mkt):
                continue
            # Black vega approx (per-strike)
            d1 = (math.log(S0 / K) + (r - q + 0.5 * iv_mkt * iv_mkt) * T) / (iv_mkt * math.sqrt(T))
            vega_mkt = S0 * math.exp(-q * T) * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(T)
            vega_mkt = max(vega_mkt, 1e-6)
            samples.append((T, K, P, iv_mkt, vega_mkt))

    if len(samples) == 0:
        # fallback default params (kappa, theta, sigma_v, rho, v0)
        return [3.0, 0.04, 0.4, -0.6, 0.04]

    params0 = np.array([3.0, 0.2 ** 2, 0.4, -0.6, 0.2 ** 2], dtype=float)
    lb = np.array([0.1, 0.0001, 0.05, -0.99, 0.0001], dtype=float)
    ub = np.array([10.0, 1.0, 1.5, 0.0, 1.0], dtype=float)

    def objective(p):
        p = np.asarray(p, dtype=float)
        kappa, theta, sigma_v, rho, v0 = p
        if (2.0 * kappa * theta <= sigma_v ** 2) or (theta > 1.0) or np.any(p < lb) or np.any(p > ub):
            return np.full(len(samples), 1e3, dtype=float)

        errs = []
        for T, K, P, iv_mkt, vega_mkt in samples:
            try:
                cf = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
                modelP = cos_price_from_cf(S0, float(K), r, q, T, cf)
                if not np.isfinite(modelP):
                    errs.append(10.0)
                    continue
                errs.append((float(modelP) - P) / vega_mkt)
            except Exception:
                errs.append(10.0)
        return np.asarray(errs, dtype=float)

    res = least_squares(objective, x0=params0, bounds=(lb, ub),
                        method='trf', max_nfev=1000, loss='soft_l1', f_scale=0.05)

    if not res.success:
        return params0.tolist()
    return res.x.tolist()
