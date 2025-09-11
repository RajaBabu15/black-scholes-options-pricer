import math
from optlib.pricing.bs import bs_price


def implied_vol_from_price(price, S, K, r, q, T, option_type='call', tol=1e-8, maxiter=200):
    price = float(price)
    if option_type == 'call':
        intrinsic_value = max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
        upper_bound = S * math.exp(-q*T)
    else:
        intrinsic_value = max(0.0, K * math.exp(-r*T) - S * math.exp(-q*T))
        upper_bound = K * math.exp(-r*T)
    if not (intrinsic_value - 1e-6 <= price <= upper_bound + 1e-6):
        return float('nan')
    time_value = price - intrinsic_value
    if time_value <= max(1e-4 * S, 1e-6):
        return 0.05
    if T > 1e-12:
        sigma = max(0.05, min(0.8, math.sqrt(2 * math.pi / T) * (price - intrinsic_value) / max(S, 1e-8)))
    else:
        sigma = 0.2
    converged = False
    for _ in range(maxiter):
        sigma = float(max(1e-4, min(sigma, 1.0)))
        price_est = bs_price(S, K, r, q, sigma, T, option_type=option_type)
        d1 = (math.log(S/K) + (r-q+0.5*sigma*sigma)*T) / (sigma*math.sqrt(T)) if T>1e-12 else 0
        vega = S * math.exp(-q*T) * math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi) * math.sqrt(T) if T>1e-12 else 0
        diff = price_est - price
        if abs(diff) < tol:
            converged = True
            break
        if abs(vega) > 1e-8:
            sigma -= diff / vega
        else:
            sigma *= 1.02 if diff < 0 else 0.98
    if converged:
        return max(0.05, min(sigma, 1.0))
    lo, hi = 1e-4, 1.0
    def price_at(sig):
        return bs_price(S, K, r, q, sig, T, option_type=option_type)
    plo = price_at(lo) - price
    phi = price_at(hi) - price
    if plo * phi > 0:
        return max(0.05, min(sigma, 1.0))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        pm = price_at(mid) - price
        if abs(pm) < tol:
            return mid
        if plo * pm <= 0:
            hi = mid
            phi = pm
        else:
            lo = mid
            plo = pm
    return 0.5 * (lo + hi)

