# python/src/black_scholes.py

import numpy as np
from scipy.stats import norm

# N(x) is the Cumulative Distribution Function (CDF) for standard normal distribution
N = norm.cdf
# n(x) is the Probability Density Function (PDF) for standard normal distribution
n = norm.pdf

def d1(S, K, T, r, sigma):
    """Calculates d1 for the Black-Scholes model."""
    if T <= 0 or sigma <= 0:
        # Handle edge cases: If time or volatility is zero/negative,
        # d1 tends towards +/- infinity depending on S/K ratio,
        # leading to price being max(S-K*exp(-rT), 0) or max(K*exp(-rT)-S, 0).
        # Returning a very large/small number or handling it in the main functions
        # ensures correct limit behavior (e.g., N(inf)=1, N(-inf)=0).
        # A large number is sufficient here as N() handles the limits.
        if S > K * np.exp(-r * T):
            return 10.0 # Effectively infinity for N()
        else:
            return -10.0 # Effectively -infinity for N()
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    """Calculates d2 using the relationship d2 = d1 - sigma * sqrt(T)."""
    if T <= 0 or sigma <= 0:
         # Handle edge case similar to d1
         if S > K * np.exp(-r * T):
             return 10.0
         else:
             return -10.0
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def black_scholes_call_price(S, K, T, r, sigma):
    """Calculates the Black-Scholes price for a European call option."""
    if T <= 0: # Option expired
        return max(0.0, S - K) # Intrinsic value at expiry
    if sigma <= 0: # No volatility, price is deterministic discount
        return max(0.0, S - K * np.exp(-r * T))

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    price = S * N(d1_val) - K * np.exp(-r * T) * N(d2_val)
    return price

def black_scholes_put_price(S, K, T, r, sigma):
    """Calculates the Black-Scholes price for a European put option."""
    if T <= 0: # Option expired
        return max(0.0, K - S) # Intrinsic value at expiry
    if sigma <= 0: # No volatility
        return max(0.0, K * np.exp(-r * T) - S)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)
    price = K * np.exp(-r * T) * N(-d2_val) - S * N(-d1_val)
    return price

# --- Greeks ---

def delta(S, K, T, r, sigma, option_type='call'):
    """Calculates Delta: Sensitivity of option price to underlying stock price change."""
    if T <= 0 or sigma <= 0: # Handle expired or zero volatility
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else: # put
            return -1.0 if S < K else 0.0

    d1_val = d1(S, K, T, r, sigma)
    if option_type == 'call':
        return N(d1_val)
    elif option_type == 'put':
        return N(d1_val) - 1.0
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def gamma(S, K, T, r, sigma):
    """Calculates Gamma: Sensitivity of Delta to underlying stock price change."""
    if T <= 0 or sigma <= 0:
        return 0.0 # Gamma is zero at expiry or zero vol

    d1_val = d1(S, K, T, r, sigma)
    gamma_val = n(d1_val) / (S * sigma * np.sqrt(T))
    return gamma_val

def vega(S, K, T, r, sigma):
    """Calculates Vega: Sensitivity of option price to volatility change.
       Note: Vega is typically presented per 1% change in vol (multiplied by 0.01)."""
    if T <= 0 or sigma <= 0:
        return 0.0 # Vega is zero at expiry or zero vol

    d1_val = d1(S, K, T, r, sigma)
    vega_val = S * n(d1_val) * np.sqrt(T)
    return vega_val * 0.01 # Per 1% change in vol

def theta(S, K, T, r, sigma, option_type='call'):
    """Calculates Theta: Sensitivity of option price to time decay (passage of time).
       Note: Theta is typically presented per day (divided by 365)."""
    if T <= 0 or sigma <= 0:
        return 0.0 # Theta is zero at expiry or zero vol

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    term1 = - (S * n(d1_val) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        term2 = r * K * np.exp(-r * T) * N(d2_val)
        theta_val = term1 - term2
    elif option_type == 'put':
        term2 = r * K * np.exp(-r * T) * N(-d2_val)
        theta_val = term1 + term2
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return theta_val / 365.0 # Per day

def rho(S, K, T, r, sigma, option_type='call'):
    """Calculates Rho: Sensitivity of option price to interest rate change.
       Note: Rho is typically presented per 1% change in rate (multiplied by 0.01)."""
    if T <= 0:
        return 0.0 # Rho is zero at expiry

    d2_val = d2(S, K, T, r, sigma)

    if option_type == 'call':
        rho_val = K * T * np.exp(-r * T) * N(d2_val)
    elif option_type == 'put':
        rho_val = -K * T * np.exp(-r * T) * N(-d2_val)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return rho_val * 0.01 # Per 1% change in rate

def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculates all Greeks for the specified option type."""
    return {
        'Delta': delta(S, K, T, r, sigma, option_type),
        'Gamma': gamma(S, K, T, r, sigma),
        'Vega': vega(S, K, T, r, sigma),
        'Theta': theta(S, K, T, r, sigma, option_type),
        'Rho': rho(S, K, T, r, sigma, option_type)
    }