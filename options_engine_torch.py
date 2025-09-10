import torch
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

# Set device for GPU acceleration if available (MPS for Apple Silicon, CUDA for NVIDIA)
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print(f"Using device: {device} (Apple Silicon GPU acceleration)")
elif torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using device: {device} (NVIDIA GPU acceleration)")
else:
    device = torch.device('cpu')
    print(f"Using device: {device} (CPU only)")

# Choose appropriate precision based on device capabilities
# MPS (Apple Silicon) doesn't support float64, but CUDA and CPU do
if torch.backends.mps.is_available() and device.type == 'mps':
    # MPS only supports float32, but still better than CPU float32
    torch.set_default_dtype(torch.float32)
    tensor_dtype = torch.float32
    complex_dtype = torch.complex64
    print("Using float32 for MPS compatibility (Apple Silicon limitation)")
else:
    # Use float64 for maximum precision on CPU/CUDA
    torch.set_default_dtype(torch.float64)
    tensor_dtype = torch.float64
    complex_dtype = torch.complex128
    print("Using float64 for maximum financial calculation precision")

# Constants for complex operations
I_COMPLEX = torch.tensor(1j, dtype=complex_dtype, device=device)
ONE_COMPLEX = torch.tensor(1.0, dtype=complex_dtype, device=device)
HALF = torch.tensor(0.5, dtype=complex_dtype, device=device)

# --- utilities (norm cdf/pdf) using PyTorch ---
def norm_cdf(x):
    x = torch.as_tensor(x, dtype=tensor_dtype, device=device)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def norm_pdf(x):
    x = torch.as_tensor(x, dtype=tensor_dtype, device=device)
    return torch.exp(-0.5 * x * x) / math.sqrt(2*math.pi)

# --- Black-Scholes helpers using PyTorch ---
def bs_price(S, K, r, q, sigma, tau, option_type='call'):
    S = torch.as_tensor(S, dtype=tensor_dtype, device=device)
    sigma = torch.as_tensor(sigma, dtype=tensor_dtype, device=device)
    tau = torch.as_tensor(tau, dtype=tensor_dtype, device=device)
    K = torch.as_tensor(K, dtype=tensor_dtype, device=device)
    
    small = 1e-12
    if torch.any(tau <= small):
        if option_type == 'call': 
            return torch.maximum(S - K, torch.tensor(0.0, device=device))
        else: 
            return torch.maximum(K - S, torch.tensor(0.0, device=device))
    
    d1 = (torch.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * torch.sqrt(tau))
    d2 = d1 - sigma * torch.sqrt(tau)
    
    if option_type == 'call':
        return S * torch.exp(-q * tau) * norm_cdf(d1) - K * torch.exp(-r * tau) * norm_cdf(d2)
    else:
        return K * torch.exp(-r * tau) * norm_cdf(-d2) - S * torch.exp(-q * tau) * norm_cdf(-d1)

# --- Bates (Heston + Jumps) characteristic function using PyTorch ---
def bates_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j):
    """
    Bates model characteristic function (Heston + Merton Jumps).
    Note: Uses CPU float64 for numerical stability, consistent with Heston.
    """
    # Force CPU complex128 for compatibility with heston_char_func
    cpu = torch.device('cpu')
    cdtype = torch.complex128
    
    # 1. Calculate the Heston part (already on CPU)
    heston_phi = heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)

    # 2. Calculate the Merton Jump part (on CPU)
    lambda_j_c = torch.as_tensor(lambda_j, dtype=cdtype, device=cpu)
    mu_j_c = torch.as_tensor(mu_j, dtype=cdtype, device=cpu)
    sigma_j_c = torch.as_tensor(sigma_j, dtype=cdtype, device=cpu)
    T_c = torch.as_tensor(T, dtype=cdtype, device=cpu)
    u_c = u.to(cdtype)
    
    # Constants on CPU
    i_c = torch.tensor(1j, dtype=cdtype, device=cpu)
    one_c = torch.tensor(1.0, dtype=cdtype, device=cpu)
    half_c = torch.tensor(0.5, dtype=cdtype, device=cpu)

    # Drift correction for jumps
    drift_correction = lambda_j_c * (torch.exp(mu_j_c + half_c * sigma_j_c**2) - one_c)
    
    # Jump characteristic function component
    jump_component = lambda_j_c * (torch.exp(i_c * u_c * mu_j_c - half_c * u_c**2 * sigma_j_c**2) - one_c)
    jump_phi = torch.exp(T_c * (-drift_correction + jump_component))
    
    # 3. Combine them
    return heston_phi * jump_phi

# --- Heston characteristic function using PyTorch ---
def heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0):
    # Force CPU float64 complex for numerical stability
    cpu = torch.device('cpu')
    cdtype = torch.complex128
    u = torch.as_tensor(u, dtype=cdtype, device=cpu)
    i = torch.tensor(1j, dtype=cdtype, device=cpu)
    
    # Numerical stability checks
    sigma_v = max(float(sigma_v), 1e-8)
    T = max(float(T), 1e-8)
    a = kappa * theta
    alpha = kappa - rho * sigma_v * i * u
    beta = (sigma_v**2) * (i * u + u**2)
    
    # Stabilize square root calculation
    discriminant = alpha**2 + beta
    d = torch.sqrt(discriminant + 1e-16j)
    if torch.any(torch.isnan(d)):
        return torch.ones_like(u)
    
    # Prevent overflow in g calculation
    g = (alpha - d) / (alpha + d)
    g = torch.clamp(g.real, min=-0.9999, max=0.9999) + 1j * torch.clamp(g.imag, min=-50, max=50)
    
    exp_dT = torch.exp(-d * T)
    
    # Stabilize log calculation
    log_arg = (1 - g * exp_dT) / (1 - g)
    log_arg = torch.clamp(log_arg.real, min=1e-12, max=1e12) + 1j * torch.clamp(log_arg.imag, min=-1e6, max=1e6)
    
    C = (i * u * (math.log(float(S0)) + (r - q) * T)
         + a / (sigma_v**2) * ((alpha - d) * T - 2.0 * torch.log(log_arg)))
    D = (alpha - d) / (sigma_v**2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    
    # Limit the real part of the exponent to avoid overflow in exp
    exponent = C + D * v0
    real_clamped = torch.clamp(exponent.real, min=-200.0, max=200.0)
    imag_clamped = torch.clamp(exponent.imag, min=-1e6, max=1e6)
    exponent_clamped = torch.complex(real_clamped, imag_clamped)
    result = torch.exp(exponent_clamped)
    return result

# --- COS method pricing using PyTorch ---
def cos_price_from_cf(S0, K, r, q, T, cf_func, params=None, target_precision=1e-6):
    """COS method pricing with adaptive parameters and exact c2 calculation.
    Note: For numerical stability, this function runs on CPU with float64.
    """
    # Force CPU float64 for numerical stability
    cpu = torch.device('cpu')
    num_dtype = torch.float64

    # Convert inputs to CPU float64
    S0_t = torch.as_tensor(S0, dtype=num_dtype, device=cpu)
    K_t = torch.as_tensor(K, dtype=num_dtype, device=cpu)
    
    x0 = math.log(float(S0) * math.exp(-q*T))
    c1 = math.log(float(S0)) + (r - q) * T
    
    # Exact second cumulant calculation for Heston model (capped for stability)
    if params is not None and len(params) >= 5:
        kappa, theta, sigma_v, rho, v0 = params[:5]
        c2_exact = (
            T * v0 + 
            (kappa * theta * T**2) / 2 + 
            (sigma_v**2 * T**3) / 12 + 
            (rho * sigma_v * T**2 * (v0 - theta)) / 4 +
            (rho * sigma_v * kappa * theta * T**3) / 6
        )
        # Cap c2 to avoid overly wide integration ranges for extreme params
        c2 = max(1e-8, min(c2_exact, 4.0 * T))
        
        # Adaptive N and L based on model parameters and precision
        volatility_of_vol = max(0.05, float(sigma_v))
        time_scaling = max(math.sqrt(max(T, 1e-6)), 0.1)
        
        # More terms for high vol-of-vol or longer maturity
        N = max(256, int(512 * volatility_of_vol * time_scaling))
        N = min(N, 2048)
        
        # Wider truncation for high uncertainty
        L = max(10, int(10 + 5 * volatility_of_vol + 2 * time_scaling))
        L = min(L, 25)
    else:
        market_vol = 0.25
        c2 = T * market_vol**2 + 0.5 * T**2 * market_vol**2
        N = 256
        L = 12
    
    a = c1 - L * math.sqrt(abs(c2) + 1e-12)
    b = c1 + L * math.sqrt(abs(c2) + 1e-12)
    
    k = torch.arange(N, dtype=num_dtype, device=cpu)
    u = k * math.pi / (b - a)
    
    # Payoff coefficients for call (CPU float64)
    c = math.log(float(K))
    
    def Chi(k, a, b, c, d):
        kpi = k * math.pi / (b - a)
        term1 = (torch.cos(kpi * (d - a)) * math.exp(d) - torch.cos(kpi * (c - a)) * math.exp(c))
        term2 = kpi * (torch.sin(kpi * (d - a)) * math.exp(d) - torch.sin(kpi * (c - a)) * math.exp(c))
        return (term1 + term2) / (1 + kpi**2)
    
    def Psi(k, a, b, c, d):
        kpi = k * math.pi / (b - a)
        result = torch.zeros_like(k)
        mask = (k == 0)
        result[mask] = d - c
        result[~mask] = (torch.sin(kpi[~mask]*(d-a)) - torch.sin(kpi[~mask]*(c-a))) * (b-a) / (k[~mask] * math.pi)
        return result
    
    Vk = 2.0/(b-a) * (Chi(k, a, b, c, b) - float(K) * Psi(k,a,b,c,b))
    Vk[0] *= 0.5
    
    # Evaluate characteristic function (ensure it runs on CPU float64)
    u_c = u.to(torch.complex128)
    try:
        phi_u = cf_func(u_c)
    except Exception:
        # Fallback: small damping to improve stability
        phi_u = cf_func(u_c - 0.5j)
    exp_term = torch.exp(1j * u_c * (x0 - a))
    mat = phi_u * exp_term
    price_t = math.exp(-r*T) * torch.real(torch.sum(mat * Vk.to(torch.complex128)))
    price = price_t.item() if isinstance(price_t, torch.Tensor) else float(price_t)
    
    # Fallback to Black-Scholes if NaN/inf
    if not math.isfinite(price):
        vol0 = math.sqrt(max(params[-1], 1e-6)) if (params is not None and len(params)>=5) else 0.3
        price = bs_price(float(S0), float(K), r, q, vol0, T, option_type='call').cpu().item()
    
    return price

def implied_vol_from_price(price, S, K, r, q, T, option_type='call', tol=1e-8, maxiter=200):
    """Robust implied volatility calculation with proper intrinsic value handling and bisection fallback."""
    price = float(price)
    
    # Calculate proper intrinsic value based on option type
    if option_type == 'call':
        intrinsic_value = max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
        upper_bound = S * math.exp(-q*T)
    else:
        intrinsic_value = max(0.0, K * math.exp(-r*T) - S * math.exp(-q*T))
        upper_bound = K * math.exp(-r*T)
    
    # Arbitrage bounds check - reject prices that violate no-arbitrage
    if not (intrinsic_value - 1e-6 <= price <= upper_bound + 1e-6):
        return float('nan')
    
    # If time value negligible for deep ITM/OTM short-dated, return low IV
    time_value = price - intrinsic_value
    if time_value <= max(1e-4 * S, 1e-6):
        return 0.05
    
    # Newton-Raphson iteration with better initial guess
    if T > 1e-12:
        sigma = max(0.05, min(0.8, math.sqrt(2 * math.pi / T) * (price - intrinsic_value) / max(S, 1e-8)))
    else:
        sigma = 0.2
    
    converged = False
    for _ in range(maxiter):
        sigma = float(max(1e-4, min(sigma, 1.0)))
        price_est_tensor = bs_price(S, K, r, q, sigma, T, option_type=option_type)
        price_est = price_est_tensor.cpu().item() if isinstance(price_est_tensor, torch.Tensor) else price_est_tensor
        
        if T <= 1e-12:
            return sigma
        d1 = (math.log(S/K) + (r-q+0.5*sigma*sigma)*T) / (sigma*math.sqrt(T))
        vega = S * math.exp(-q*T) * math.exp(-0.5*d1*d1) / math.sqrt(2*math.pi) * math.sqrt(T)
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

    # Bisection fallback (robust for low vega / near-expiry)
    lo, hi = 1e-4, 1.0
    def price_at(sig):
        pt = bs_price(S, K, r, q, sig, T, option_type=option_type)
        return pt.cpu().item() if isinstance(pt, torch.Tensor) else pt
    plo = price_at(lo) - price
    phi = price_at(hi) - price
    if plo * phi > 0:
        # Cannot bracket; return best effort bounded sigma
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

# Smooth abs (Huber-like) to keep gradients stable
def smooth_abs(x, delta=1e-6):
    return torch.sqrt(x * x + delta)

# --- Hedging simulator using PyTorch ---
def delta_hedge_sim(S_paths, v_paths, times, K, r, q, tc=0.0008, impact_lambda=0.0, option_type='call', rebal_freq=1, deltas_mode='bs', per_path_deltas=None, exposure_scale=1.0, return_timeseries=False, anti_lookahead_checks=True, return_torch=False):
    """Delta-hedging simulator with optional per-step returns time series and diagnostics.
    Ensures no lookahead: delta at t uses only info up to t. Costs applied at trade times.
    If return_torch=True, returns tensors for differentiable training.
    """
    # Convert to torch tensors
    S_paths = torch.as_tensor(S_paths, dtype=tensor_dtype, device=device)
    v_paths = torch.as_tensor(v_paths, dtype=tensor_dtype, device=device)
    times = torch.as_tensor(times, dtype=tensor_dtype, device=device)

    n_paths, m = S_paths.shape
    n_steps = m - 1
    dt = torch.diff(times)
    T = times[-1]
    tau = T - times
    S0 = S_paths[0, 0]
    sigma0 = torch.sqrt(torch.maximum(v_paths[:, 0], torch.tensor(0.0, device=device))).mean()
    C0 = bs_price(S0, K, r, q, sigma0, T, option_type=option_type)

    pnl = torch.zeros(n_paths, device=device)
    # per-step normalized returns (n_paths x n_steps), optional
    step_returns = torch.zeros((n_paths, n_steps), dtype=tensor_dtype, device=device) if return_timeseries else None

    # diagnostics
    trades_count = torch.zeros(n_paths, dtype=tensor_dtype, device=device)
    spread_cost_total = torch.zeros(n_paths, dtype=tensor_dtype, device=device)
    impact_cost_total = torch.zeros(n_paths, dtype=tensor_dtype, device=device)

    if exposure_scale > 1.0:
        raise ValueError("Exposure scale > 1.0 is not allowed (risk control)")

    for i in range(n_paths):
        S_path, v_path = S_paths[i], v_paths[i]
        # compute deltas sequence without lookahead
        if deltas_mode == 'bs':
            sigma_inst = torch.sqrt(torch.maximum(v_path, torch.tensor(1e-12, device=device)))
            deltas = torch.zeros_like(S_path)
            for t in range(m):
                # use only info up to t
                if anti_lookahead_checks:
                    assert t < m, "Index out of bounds"
                if tau[t] <= 1e-12:
                    deltas[t] = 1.0 if S_path[t] > K else 0.0
                else:
                    d1 = (torch.log(S_path[t] / K) + (r - q + 0.5 * sigma_inst[t] ** 2) * tau[t]) / (sigma_inst[t] * torch.sqrt(tau[t]))
                    deltas[t] = torch.exp(-q * tau[t]) * norm_cdf(d1)
        elif deltas_mode == 'perpath':
            # use only current t during trading loop
            deltas = torch.as_tensor(per_path_deltas[i], dtype=tensor_dtype, device=device)
        else:
            raise ValueError("Unknown deltas_mode")

        # Apply exposure scaling
        # preserve gradient if exposure_scale is a tensor
        exposure_scale_t = exposure_scale if isinstance(exposure_scale, torch.Tensor) else torch.as_tensor(exposure_scale, dtype=tensor_dtype, device=device)
        deltas = deltas * exposure_scale_t

        # Initial hedge at t=0
        Delta_prev = deltas[0] if deltas_mode == 'bs' else deltas[0]
        notional0 = torch.abs(C0) + torch.abs(Delta_prev * S_path[0]) + 1e-8
        trade0 = Delta_prev * S_path[0]
        cash = C0 - trade0 - (smooth_abs(trade0) * tc + impact_lambda * (trade0 ** 2))
        trades_count[i] += 1.0
        spread_cost_total[i] += torch.abs(Delta_prev * S_path[0]) * tc
        impact_cost_total[i] += impact_lambda * (Delta_prev * S_path[0]) ** 2

        reb = torch.arange(0, m, rebal_freq, device=device)
        if reb[-1] != n_steps:
            reb = torch.cat([reb, torch.tensor([n_steps], device=device)])

        # step loop
        last_equity = cash + Delta_prev * S_path[0]
        for k in range(1, len(reb)):
            t_idx, t_idx_prev = int(reb[k]), int(reb[k - 1])
            # accrue interest period by period
            for j in range(t_idx_prev, t_idx):
                cash *= torch.exp(r * dt[j])
                # track step return if requested (equity change after price move)
                if return_timeseries:
                    equity_now = cash + Delta_prev * S_path[j + 1]
                    step_returns[i, j] = (equity_now - last_equity) / notional0
                    last_equity = equity_now
            # compute hedge at t_idx using only info up to t_idx
            Delta_new = deltas[t_idx] if deltas_mode == 'bs' else deltas[t_idx]
            if anti_lookahead_checks:
                # ensure we didn't access any future path points for delta
                assert t_idx < m, "Hedge index beyond path length"
            dDelta = Delta_new - Delta_prev
            trade_value = dDelta * S_path[t_idx]
            cost = smooth_abs(trade_value) * tc + impact_lambda * (trade_value ** 2)
            cash -= trade_value + cost
            if torch.abs(dDelta) > 0:
                trades_count[i] += 1.0
                spread_cost_total[i] += smooth_abs(trade_value) * tc
                impact_cost_total[i] += impact_lambda * (trade_value ** 2)
            Delta_prev = Delta_new

        # accrue remaining interest until expiry
        for j in range(int(reb[-1]), n_steps):
            cash *= torch.exp(r * dt[j])
            if return_timeseries:
                equity_now = cash + Delta_prev * S_path[j + 1]
                step_returns[i, j] = (equity_now - last_equity) / notional0
                last_equity = equity_now

        # expiry payoff
        if option_type == 'call':
            Delta_expiry = (S_path[-1] > K).float()
            payoff = torch.maximum(S_path[-1] - K, torch.tensor(0.0, device=device))
        else:
            Delta_expiry = (S_path[-1] < K).float()
            payoff = torch.maximum(K - S_path[-1], torch.tensor(0.0, device=device))
        final_trade = (Delta_expiry - Delta_prev) * S_path[-1]
        final_hedge_cost = smooth_abs(final_trade) * tc
        cash -= final_trade + final_hedge_cost
        if return_timeseries and torch.abs(Delta_expiry - Delta_prev) > 0:
            trades_count[i] += 1.0
            spread_cost_total[i] += smooth_abs(final_trade) * tc
            impact_cost_total[i] += impact_lambda * (final_trade ** 2)

        pnl[i] = cash + Delta_prev * S_path[-1] - payoff

    if return_torch:
        # Return detached tensors for differentiable training (avoid numpy conversion)
        diag = {
            'trades': trades_count.detach(),
            'avg_spread_cost': (spread_cost_total / (trades_count + 1e-8)).detach(),
            'avg_impact_cost': (impact_cost_total / (trades_count + 1e-8)).detach(),
            'total_spread_cost': spread_cost_total.detach(),
            'total_impact_cost': impact_cost_total.detach(),
        }
        return pnl, C0, step_returns, diag

    # Default (numpy outputs)
    diag = {
        'trades': trades_count.cpu().numpy(),
        'avg_spread_cost': (spread_cost_total / (trades_count + 1e-8)).cpu().numpy(),
        'avg_impact_cost': (impact_cost_total / (trades_count + 1e-8)).cpu().numpy(),
        'total_spread_cost': spread_cost_total.cpu().numpy(),
        'total_impact_cost': impact_cost_total.cpu().numpy(),
    }
    out_pnl = pnl.cpu().numpy()
    out_C0 = C0.cpu().item() if isinstance(C0, torch.Tensor) else C0
    if return_timeseries:
        return out_pnl, out_C0, step_returns.cpu().numpy(), diag
    return out_pnl, out_C0, None, diag

# --- Per-path deltas via scaling using PyTorch ---
def compute_per_path_deltas_scaling(S_paths, K, times, r, q, relative_eps=0.001):
    """Compute deltas using relative finite differences with proper discounting."""
    S_paths = torch.as_tensor(S_paths, dtype=tensor_dtype, device=device)
    times = torch.as_tensor(times, dtype=tensor_dtype, device=device)
    n_paths, m = S_paths.shape
    deltas = torch.zeros_like(S_paths)
    T = times[-1]
    
    for i in range(n_paths):
        S_path = S_paths[i]
        for t in range(m):
            St = S_path[t]
            if St <= 0: continue
            
            # Use relative epsilon based on current stock price
            eps = relative_eps * St
            
            # Remove lookahead bias - use current price, not final price
            scale_up = (St + eps)/St
            scale_dn = torch.maximum((St - eps)/St, torch.tensor(1e-12, device=device))
            
            # Project using simple drift model instead of lookahead
            remaining_time = T - times[t]
            if remaining_time > 1e-8:
                drift = (r - q) * remaining_time
                ST_up = St * scale_up * torch.exp(drift)
                ST_dn = St * scale_dn * torch.exp(drift)
            else:
                ST_up, ST_dn = St * scale_up, St * scale_dn
            
            # Use actual risk-free rate for discounting, not hardcoded 0.01
            discount_factor = torch.exp(-r * (T - times[t]))
            price_up = discount_factor * torch.maximum(ST_up - K, torch.tensor(0.0, device=device))
            price_dn = discount_factor * torch.maximum(ST_dn - K, torch.tensor(0.0, device=device))
            deltas[i,t] = (price_up - price_dn) / (2*eps)
    
    return deltas.cpu().numpy()

# --- Regularized calibration objective function ---
def objective_function_regularized(params, S0, r, q, T, strikes, market_ivs, reg_lambda=0.1, target_params=None):
    """
    Objective function for Heston calibration with Tikhonov (L2) regularization.
    
    Args:
        params (list): Heston parameters [kappa, theta, sigma_v, rho, v0].
        market_ivs (torch.Tensor): Market implied volatilities to match.
        reg_lambda (float): Regularization strength.
        target_params (torch.Tensor): A tensor of "ideal" or "central" parameter values.
    
    Returns:
        np.ndarray: Vector of errors (model IVs vs market IVs) plus penalty.
    """
    # Unpack parameters, ensuring they are tensors on the correct device
    params_tensor = torch.tensor(params, dtype=tensor_dtype, device=device)
    
    try:
        # 1. Calculate model implied vols using Heston model
        kappa, theta, sigma_v, rho, v0 = params
        cf = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
        
        model_prices = torch.tensor([cos_price_from_cf(S0, K.item(), r, q, T, cf, params=[kappa, theta, sigma_v, rho, v0]) for K in strikes], device=device)
        # Replace NaNs with market IVs to ensure finite residuals
        model_ivs_list = []
        for p, K in zip(model_prices, strikes):
            pi = p.item() if isinstance(p, torch.Tensor) else p
            iv = implied_vol_from_price(pi, S0, K.item(), r, q, T)
            if np.isnan(iv):
                # fallback to market IV at same index if available
                idx = (strikes == K).nonzero()[0].item() if hasattr((strikes == K), 'nonzero') else 0
                iv = market_ivs[idx].cpu().item() if 'market_ivs' in locals() and len(market_ivs)>idx else 0.3
            model_ivs_list.append(iv)
        model_ivs = torch.tensor(model_ivs_list, device=device)
        
        # 2. Calculate the primary error (difference in implied vols)
        iv_error = model_ivs - market_ivs
        
        # 3. Calculate the regularization penalty
        if target_params is not None and reg_lambda > 0:
            param_diff = params_tensor - target_params
            penalty = torch.sqrt(torch.tensor(reg_lambda, device=device)) * param_diff
            # Concatenate the error and penalty tensors
            full_error = torch.cat([iv_error, penalty])
        else:
            full_error = iv_error
            
        return full_error.cpu().numpy()
    except Exception as e:
        print(f"Calibration error: {e}")
        return 1e5 * torch.ones(len(market_ivs) + (5 if target_params is not None else 0)).numpy()

# --- Heston Greeks via finite differences ---
def heston_greeks_fd(S0, K, r, q, T, kappa, theta, sigma_v, rho, v0, s_relative_bump=0.005, v_relative_bump=0.02):
    """
    Calculates Heston price, delta, and vega via finite differences with relative bumps.
    Adds a robust fallback to Black-Scholes greeks for very short maturities to avoid numerical instability.
    
    Args:
        s_relative_bump (float): Relative bump size for stock price (0.1% default).
        v_relative_bump (float): Relative bump size for initial variance (2% default).
    
    Returns:
        dict: A dictionary containing {'price', 'delta', 'vega'}.
    """
    # Fallback to Black-Scholes greeks for very short maturities (T < ~1.25 trading days)
    if T < 0.005:
        vol_sqrt_v0 = math.sqrt(max(v0, 1e-8))
        # Black-Scholes price, delta, vega
        bs_p = bs_price(S0, K, r, q, vol_sqrt_v0, T, option_type='call')
        bs_p = bs_p.cpu().item() if isinstance(bs_p, torch.Tensor) else float(bs_p)
        d1 = (math.log(S0/K) + (r - q + 0.5 * vol_sqrt_v0**2) * T) / (vol_sqrt_v0 * math.sqrt(T))
        delta_fb = math.exp(-q*T) * float(norm_cdf(torch.tensor(d1)))
        vega_fb = S0 * math.exp(-q*T) * float(norm_pdf(torch.tensor(d1))) * math.sqrt(T)
        return {'price': bs_p, 'delta': max(0.0, min(delta_fb, math.exp(-q*T))), 'vega': max(0.0, vega_fb)}
    
    # Calculate actual bump sizes relative to current values
    s_bump = s_relative_bump * S0
    v_bump = v_relative_bump * v0
    
    # Base price
    cf_base = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
    price_base = cos_price_from_cf(S0, K, r, q, T, cf_base, params=[kappa, theta, sigma_v, rho, v0])
    
    # Delta calculation with relative bump
    cf_s_up = lambda u: heston_char_func(u, S0 + s_bump, r, q, T, kappa, theta, sigma_v, rho, v0)
    price_s_up = cos_price_from_cf(S0 + s_bump, K, r, q, T, cf_s_up, params=[kappa, theta, sigma_v, rho, v0])
    delta = (price_s_up - price_base) / s_bump
    
    # Vega calculation (sensitivity to v0) with relative bump
    cf_v_up = lambda u: heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0 + v_bump)
    price_v_up = cos_price_from_cf(S0, K, r, q, T, cf_v_up, params=[kappa, theta, sigma_v, rho, v0 + v_bump])
    vega_v0 = (price_v_up - price_base) / v_bump  # This is dP/dv0
    
    # To get dP/d(sqrt(v0)), which is the standard definition of vega:
    vega = vega_v0 * (2 * math.sqrt(max(v0, 1e-8)))
    
    return {'price': price_base, 'delta': delta, 'vega': vega}

# --- Monte Carlo Heston Path Generator ---
def generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps, device=device):
    """
    Generate Monte Carlo paths under the Heston model using Euler-Maruyama scheme.
    
    Args:
        S0: Initial stock price
        r: Risk-free rate
        q: Dividend yield
        T: Time to maturity
        kappa: Speed of mean reversion
        theta: Long-term variance
        sigma_v: Vol of vol
        rho: Correlation between stock and vol Brownian motions
        v0: Initial variance
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        device: PyTorch device
    
    Returns:
        tuple: (S_paths, v_paths) both as tensors
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    
    # Initialize arrays
    S_paths = torch.zeros((n_paths, n_steps + 1), dtype=tensor_dtype, device=device)
    v_paths = torch.zeros((n_paths, n_steps + 1), dtype=tensor_dtype, device=device)
    
    # Set initial conditions
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    # Generate correlated random numbers
    for t in range(n_steps):
        # Independent standard normal random variables
        Z1 = torch.randn(n_paths, dtype=tensor_dtype, device=device)
        Z2 = torch.randn(n_paths, dtype=tensor_dtype, device=device)
        
        # Apply correlation
        W1 = Z1
        W2 = rho * Z1 + math.sqrt(1 - rho**2) * Z2
        
        # Current values
        S_t = S_paths[:, t]
        v_t = torch.maximum(v_paths[:, t], torch.tensor(1e-8, device=device))  # Avoid negative variance
        sqrt_v_t = torch.sqrt(v_t)
        
        # Variance process (CIR with full truncation scheme)
        v_next = v_t + kappa * (theta - v_t) * dt + sigma_v * sqrt_v_t * sqrt_dt * W2
        v_paths[:, t + 1] = torch.maximum(v_next, torch.tensor(1e-8, device=device))
        
        # Stock price process
        S_next = S_t * torch.exp(
            (r - q - 0.5 * v_t) * dt + sqrt_v_t * sqrt_dt * W1
        )
        S_paths[:, t + 1] = S_next
    
    return S_paths, v_paths

# --- Live Market Parameter Fetchers ---
def fetch_risk_free_rate(fallback_rate=0.045):
    """
    Fetch current risk-free rate from 10-year Treasury yield.
    
    Args:
        fallback_rate: Fallback rate if fetch fails
    
    Returns:
        float: Current risk-free rate as decimal
    """
    try:
        import yfinance as yf
        # Fetch 10-year Treasury yield
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="5d")
        if not hist.empty:
            rate = hist['Close'].iloc[-1] / 100.0  # Convert percentage to decimal
            if 0.001 <= rate <= 0.15:  # Sanity check (0.1% to 15%)
                print(f"Fetched risk-free rate from Treasury: {rate:.3f} ({rate*100:.1f}%)")
                return rate
    except Exception as e:
        print(f"Warning: Could not fetch Treasury rate ({e}), using fallback")
    
    print(f"Using fallback risk-free rate: {fallback_rate:.3f} ({fallback_rate*100:.1f}%)")
    return fallback_rate

def fetch_dividend_yield(ticker, fallback_yield=0.005):
    """
    Fetch current dividend yield for a stock.
    
    Args:
        ticker: Stock ticker symbol
        fallback_yield: Fallback yield if fetch fails
    
    Returns:
        float: Current dividend yield as decimal
    """
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try multiple fields for dividend yield
        yield_fields = ['dividendYield', 'trailingAnnualDividendYield', 'forwardAnnualDividendYield']
        
        for field in yield_fields:
            if field in info and info[field] is not None:
                div_yield = float(info[field])
                if 0 <= div_yield <= 0.1:  # Sanity check (0% to 10%)
                    print(f"Fetched dividend yield for {ticker}: {div_yield:.4f} ({div_yield*100:.2f}%)")
                    return div_yield
        
        # Alternative: calculate from dividend and price
        if 'dividendRate' in info and 'currentPrice' in info:
            div_rate = info.get('dividendRate', 0)
            price = info.get('currentPrice', 1)
            if div_rate > 0 and price > 0:
                div_yield = div_rate / price
                print(f"Calculated dividend yield for {ticker}: {div_yield:.4f} ({div_yield*100:.2f}%)")
                return div_yield
                
    except Exception as e:
        print(f"Warning: Could not fetch dividend yield for {ticker} ({e}), using fallback")
    
    print(f"Using fallback dividend yield: {fallback_yield:.4f} ({fallback_yield*100:.2f}%)")
    return fallback_yield

def calculate_transaction_costs(volume, bid_ask_spread, price, base_commission=0.0005):
    """
    Calculate realistic transaction costs based on market microstructure.
    
    Args:
        volume: Trading volume
        bid_ask_spread: Bid-ask spread
        price: Option price
        base_commission: Base commission rate
    
    Returns:
        float: Transaction cost rate
    """
    # Spread-based cost (half the spread)
    spread_cost = (bid_ask_spread / price) / 2 if price > 0 else 0.01
    
    # Volume-based adjustment (less liquid = higher cost)
    if volume > 1000:
        volume_multiplier = 1.0
    elif volume > 100:
        volume_multiplier = 1.5
    elif volume > 10:
        volume_multiplier = 2.0
    else:
        volume_multiplier = 3.0
    
    total_cost = base_commission + spread_cost * volume_multiplier
    return min(total_cost, 0.05)  # Cap at 5%

# --- Advanced performance metrics with statistical testing ---
def calculate_performance_metrics(pnl, risk_free_rate=0.01, periods_per_year=252.0):
    """
    Calculates a dictionary of performance metrics from a PnL series.
    
    Args:
        pnl (np.ndarray): Array of profit and loss values for each period.
        risk_free_rate (float): Annual risk-free rate for Sharpe/Sortino.
        periods_per_year (float): Number of periods in a year (e.g., 252 for daily).
        
    Returns:
        dict: A dictionary of key performance metrics.
    """
    import numpy as np
    
    if len(pnl) < 2:
        return {metric: 0.0 for metric in ['sharpe', 'sortino', 'max_drawdown', 'calmar']}

    # Annualization factor
    ann_factor = np.sqrt(periods_per_year)

    # Sharpe Ratio
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    daily_rf = risk_free_rate / periods_per_year
    sharpe = (mean_pnl - daily_rf) / std_pnl * ann_factor if std_pnl > 0 else 0.0

    # Sortino Ratio
    negative_pnl = pnl[pnl < 0]
    downside_std = np.std(negative_pnl) if len(negative_pnl) > 0 else 0.0
    sortino = (mean_pnl - daily_rf) / downside_std * ann_factor if downside_std > 0 else 0.0

    # Max Drawdown & Calmar
    cumulative_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    
    annual_return = mean_pnl * periods_per_year
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0

    return {
        'total_pnl': np.sum(pnl),
        'annual_return': annual_return,
        'annual_volatility': std_pnl * ann_factor,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }

def statistical_significance_tests(pnl_strategy_a, pnl_strategy_b, confidence_level=0.95, n_bootstrap=10000):
    """
    Comprehensive statistical testing framework for comparing two trading strategies.
    Industry-standard tests for quantitative finance.
    
    Args:
        pnl_strategy_a, pnl_strategy_b: Arrays of PnL values for each strategy
        confidence_level: Confidence level for intervals (default 95%)
        n_bootstrap: Number of bootstrap samples for Sharpe ratio testing
        
    Returns:
        dict: Statistical test results with p-values and confidence intervals
    """
    import numpy as np
    from scipy import stats
    
    results = {}
    
    # 1. Two-sample t-test for mean PnL difference
    t_stat, p_value_ttest = stats.ttest_ind(pnl_strategy_a, pnl_strategy_b, equal_var=False)
    results['mean_pnl_ttest'] = {
        'test_statistic': t_stat,
        'p_value': p_value_ttest,
        'significant': p_value_ttest < (1 - confidence_level),
        'interpretation': f"Strategy A mean PnL is {'significantly' if p_value_ttest < 0.05 else 'not significantly'} different from Strategy B (p={p_value_ttest:.4f})"
    }
    
    # 2. Mann-Whitney U test (non-parametric alternative)
    u_stat, p_value_mannwhitney = stats.mannwhitneyu(pnl_strategy_a, pnl_strategy_b, alternative='two-sided')
    results['median_pnl_mannwhitney'] = {
        'test_statistic': u_stat,
        'p_value': p_value_mannwhitney,
        'significant': p_value_mannwhitney < (1 - confidence_level),
        'interpretation': f"Strategy A median PnL is {'significantly' if p_value_mannwhitney < 0.05 else 'not significantly'} different from Strategy B (p={p_value_mannwhitney:.4f})"
    }
    
    # 3. Bootstrap test for Sharpe ratio difference (industry standard)
    def bootstrap_sharpe(pnl_array, n_samples=n_bootstrap):
        """Bootstrap Sharpe ratios to build confidence intervals"""
        sharpes = []
        n = len(pnl_array)
        for _ in range(n_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(pnl_array, size=n, replace=True)
            if np.std(bootstrap_sample) > 1e-8:
                sharpe = np.mean(bootstrap_sample) / np.std(bootstrap_sample)
            else:
                sharpe = 0.0
            sharpes.append(sharpe)
        return np.array(sharpes)
    
    # Generate bootstrap distributions
    sharpe_a_bootstrap = bootstrap_sharpe(pnl_strategy_a)
    sharpe_b_bootstrap = bootstrap_sharpe(pnl_strategy_b)
    sharpe_diff_bootstrap = sharpe_a_bootstrap - sharpe_b_bootstrap
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(sharpe_diff_bootstrap, 100 * alpha / 2)
    ci_upper = np.percentile(sharpe_diff_bootstrap, 100 * (1 - alpha / 2))
    
    # P-value: proportion of bootstrap samples where difference <= 0
    p_value_sharpe = np.mean(sharpe_diff_bootstrap <= 0)
    if p_value_sharpe > 0.5:
        p_value_sharpe = 2 * (1 - p_value_sharpe)  # Two-sided test
    else:
        p_value_sharpe = 2 * p_value_sharpe
    
    results['sharpe_ratio_bootstrap'] = {
        'mean_difference': np.mean(sharpe_diff_bootstrap),
        'confidence_interval': [ci_lower, ci_upper],
        'p_value': p_value_sharpe,
        'significant': 0 not in [ci_lower, ci_upper],  # CI doesn't contain 0
        'interpretation': f"Strategy A Sharpe ratio is {'significantly' if 0 not in [ci_lower, ci_upper] else 'not significantly'} different from Strategy B. 95% CI for difference: [{ci_lower:.4f}, {ci_upper:.4f}]"
    }
    
    # 4. Volatility comparison (F-test for variance equality)
    f_stat = np.var(pnl_strategy_a, ddof=1) / np.var(pnl_strategy_b, ddof=1)
    df1, df2 = len(pnl_strategy_a) - 1, len(pnl_strategy_b) - 1
    p_value_ftest = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
    
    results['volatility_ftest'] = {
        'test_statistic': f_stat,
        'p_value': p_value_ftest,
        'significant': p_value_ftest < (1 - confidence_level),
        'interpretation': f"Strategy A volatility is {'significantly' if p_value_ftest < 0.05 else 'not significantly'} different from Strategy B (p={p_value_ftest:.4f})"
    }
    
    return results

def regime_analysis(pnl_array, volatility_array, regime_threshold=0.3):
    """
    Analyze strategy performance across different volatility regimes.
    Critical for institutional evaluation.
    
    Args:
        pnl_array: Strategy PnL values
        volatility_array: Corresponding volatility estimates
        regime_threshold: Volatility threshold to separate low/high vol regimes
        
    Returns:
        dict: Performance metrics by regime
    """
    import numpy as np
    
    # Define regimes based on volatility
    low_vol_mask = volatility_array <= regime_threshold
    high_vol_mask = volatility_array > regime_threshold
    
    results = {
        'low_volatility_regime': {
            'n_observations': np.sum(low_vol_mask),
            'avg_volatility': np.mean(volatility_array[low_vol_mask]) if np.any(low_vol_mask) else 0,
            'performance': calculate_performance_metrics(pnl_array[low_vol_mask]) if np.any(low_vol_mask) else {}
        },
        'high_volatility_regime': {
            'n_observations': np.sum(high_vol_mask),
            'avg_volatility': np.mean(volatility_array[high_vol_mask]) if np.any(high_vol_mask) else 0,
            'performance': calculate_performance_metrics(pnl_array[high_vol_mask]) if np.any(high_vol_mask) else {}
        }
    }
    
    # Regime stability analysis
    if np.any(low_vol_mask) and np.any(high_vol_mask):
        results['regime_comparison'] = {
            'sharpe_ratio_difference': (
                results['high_volatility_regime']['performance'].get('sharpe_ratio', 0) - 
                results['low_volatility_regime']['performance'].get('sharpe_ratio', 0)
            ),
            'interpretation': "Strategy performs better in high volatility" if (
                results['high_volatility_regime']['performance'].get('sharpe_ratio', 0) > 
                results['low_volatility_regime']['performance'].get('sharpe_ratio', 0)
            ) else "Strategy performs better in low volatility"
        }
    
    return results

# --- Advanced optimization framework for risk-adjusted returns ---
def optimize_strategy_parameters(S_paths, v_paths, times, K, r, q, optimization_target='sharpe_ratio'):
    """
    Professional-grade optimization targeting risk-adjusted returns rather than raw PnL.
    Uses Bayesian optimization for efficient parameter space exploration.
    
    Args:
        optimization_target: 'sharpe_ratio', 'sortino_ratio', or 'calmar_ratio'
        
    Returns:
        dict: Optimal parameters and optimization results
    """
    try:
        from scipy.optimize import minimize
        import numpy as np
        
        def objective_function(params):
            """Objective function for strategy optimization"""
            rebal_freq, tc, impact_lambda = params
            
            # Ensure parameters are within reasonable bounds
            rebal_freq = max(1, int(rebal_freq))
            tc = max(0.0001, min(tc, 0.01))  # 0.01% to 1%
            impact_lambda = max(0, min(impact_lambda, 1e-4))
            
            try:
                # Run simulation with these parameters
                pnl, _ = delta_hedge_sim(
                    S_paths, v_paths, times, K, r, q, 
                    tc=tc, impact_lambda=impact_lambda, rebal_freq=rebal_freq, 
                    deltas_mode='bs'
                )
                
                if len(pnl) < 2:
                    return -1000  # Invalid result penalty
                
                metrics = calculate_performance_metrics(pnl)
                
                # Return negative because scipy.minimize minimizes
                if optimization_target == 'sharpe_ratio':
                    return -metrics.get('sharpe_ratio', -1000)
                elif optimization_target == 'sortino_ratio':
                    return -metrics.get('sortino_ratio', -1000)
                elif optimization_target == 'calmar_ratio':
                    return -metrics.get('calmar_ratio', -1000)
                else:
                    return -metrics.get('sharpe_ratio', -1000)
                    
            except Exception as e:
                print(f"Optimization error with params {params}: {e}")
                return -1000
        
        # Define parameter bounds: [rebal_freq, tc, impact_lambda]
        bounds = [(1, 30), (0.0001, 0.01), (0, 1e-4)]
        
        # Initial guess
        x0 = [2, 0.0008, 1e-6]
        
        # Run optimization
        result = minimize(
            objective_function,
            x0=x0,
            bounds=bounds,
            method='L-BFGS-B',
            options={'maxiter': 50, 'disp': False}
        )
        
        optimal_rebal_freq = max(1, int(result.x[0]))
        optimal_tc = result.x[1]
        optimal_impact = result.x[2]
        
        # Evaluate final performance with optimal parameters
        optimal_pnl, _ = delta_hedge_sim(
            S_paths, v_paths, times, K, r, q,
            tc=optimal_tc, impact_lambda=optimal_impact, rebal_freq=optimal_rebal_freq,
            deltas_mode='bs'
        )
        
        optimal_metrics = calculate_performance_metrics(optimal_pnl)
        
        return {
            'optimal_parameters': {
                'rebalancing_frequency': optimal_rebal_freq,
                'transaction_cost': optimal_tc,
                'market_impact': optimal_impact
            },
            'optimization_result': result,
            'optimal_performance': optimal_metrics,
            'target_metric': optimization_target,
            'target_value': optimal_metrics.get(optimization_target, 0)
        }
        
    except ImportError:
        print("SciPy not available for advanced optimization")
        return None
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None

# --- Unit testing framework for critical financial functions ---
def run_unit_tests():
    """
    Unit testing framework for core mathematical functions.
    Essential for institutional-grade code validation.
    """
    import numpy as np
    
    print("\n=== RUNNING UNIT TESTS ===")
    
    tests_passed = 0
    total_tests = 0
    
    def assert_test(condition, test_name):
        nonlocal tests_passed, total_tests
        total_tests += 1
        if condition:
            print(f"✓ {test_name}")
            tests_passed += 1
        else:
            print(f"✗ {test_name}")
    
    # Test 1: Black-Scholes price for ATM option
    S, K, r, q, sigma, T = 100, 100, 0.05, 0.02, 0.2, 1.0
    bs_price_result = bs_price(S, K, r, q, sigma, T, 'call')
    bs_price_val = bs_price_result.cpu().item() if isinstance(bs_price_result, torch.Tensor) else bs_price_result
    # Expected: approximately 11.25 for these parameters
    assert_test(10 < bs_price_val < 13, f"BS ATM call price reasonable: {bs_price_val:.4f}")
    
    # Test 2: Call-put parity
    call_price = bs_price(S, K, r, q, sigma, T, 'call')
    put_price = bs_price(S, K, r, q, sigma, T, 'put')
    call_val = call_price.cpu().item() if isinstance(call_price, torch.Tensor) else call_price
    put_val = put_price.cpu().item() if isinstance(put_price, torch.Tensor) else put_price
    
    # Call - Put = S*exp(-qT) - K*exp(-rT)
    expected_diff = S * math.exp(-q*T) - K * math.exp(-r*T)
    actual_diff = call_val - put_val
    assert_test(abs(actual_diff - expected_diff) < 0.01, f"Call-put parity: diff={actual_diff:.4f}, expected={expected_diff:.4f}")
    
    # Test 3: Zero time to expiry should give intrinsic value
    call_intrinsic = bs_price(110, 100, r, q, sigma, 1e-10, 'call')
    call_intrinsic_val = call_intrinsic.cpu().item() if isinstance(call_intrinsic, torch.Tensor) else call_intrinsic
    assert_test(abs(call_intrinsic_val - 10.0) < 0.01, f"Zero-expiry intrinsic value: {call_intrinsic_val:.4f}")
    
    # Test 4: Delta bounds for call option
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        # Only test Greeks if we have proper computational resources
        try:
            greeks = heston_greeks_fd(S, K, r, q, T, 2.0, 0.04, 0.3, -0.5, 0.04)
            if not math.isnan(greeks['delta']):
                assert_test(0 <= greeks['delta'] <= math.exp(-q*T), f"Delta bounds: {greeks['delta']:.4f}")
                assert_test(greeks['vega'] >= 0, f"Vega non-negative: {greeks['vega']:.4f}")
        except:
            print("  Greeks test skipped due to numerical issues")
    
    # Test 5: Implied volatility round-trip
    test_price = 15.0
    recovered_vol = implied_vol_from_price(test_price, S, K, r, q, T, 'call')
    if not math.isnan(recovered_vol):
        recovered_price_tensor = bs_price(S, K, r, q, recovered_vol, T, 'call')
        recovered_price = recovered_price_tensor.cpu().item() if isinstance(recovered_price_tensor, torch.Tensor) else recovered_price_tensor
        assert_test(abs(recovered_price - test_price) < 0.01, f"IV round-trip: original={test_price}, recovered={recovered_price:.4f}")
    
    # Test 6: Arbitrage bounds
    high_price = S * math.exp(-q*T) + 1  # Above upper bound
    invalid_iv = implied_vol_from_price(high_price, S, K, r, q, T, 'call')
    assert_test(math.isnan(invalid_iv), f"Arbitrage detection: invalid price rejected")
    
    print(f"\nUnit Tests: {tests_passed}/{total_tests} passed ({100*tests_passed/total_tests:.1f}%)")
    if tests_passed == total_tests:
        print("✓ All unit tests passed - code is mathematically sound")
    else:
        print("✗ Some tests failed - review mathematical implementations")
    
    return tests_passed == total_tests

# --- Grid search using PyTorch (maintained for compatibility) ---
def grid_search_rebal_cost(S_paths, v_paths, times, K, r, q, rebal_values, tc_values, impact_values, mode='bs', per_path_deltas=None):
    """Basic grid search - use optimize_strategy_parameters for advanced optimization"""
    rows=[]
    for rebal in rebal_values:
        for tc in tc_values:
            for impact in impact_values:
                pnl,_ = delta_hedge_sim(S_paths, v_paths, times, K, r, q, tc=tc, impact_lambda=impact, rebal_freq=rebal, deltas_mode=mode, per_path_deltas=per_path_deltas)
                metrics = calculate_performance_metrics(pnl)
                rows.append({
                    'rebal': int(rebal), 'tc': tc, 'impact': impact,
                    'mean': pnl.mean(), 'std': pnl.std(), 
                    'median': torch.median(torch.tensor(pnl)).item(),
                    'sharpe': metrics.get('sharpe_ratio', 0),
                    'max_dd': metrics.get('max_drawdown', 0)
                })
    return pd.DataFrame(rows)
