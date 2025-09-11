"""
Fully vectorized delta hedge simulation for maximum performance
All operations are batched across paths and time steps
"""
import torch
from optlib.utils.tensor import tensor_dtype, device
from optlib.pricing.bs import bs_price, norm_cdf
def smooth_abs_vectorized(x, delta=1e-6):
    """Vectorized smooth absolute value"""
    return torch.sqrt(x * x + delta)

def compute_bs_deltas_vectorized(S_paths, v_paths, times, K_t, r_t, q_t, option_type='call'):
    """Compute Black-Scholes deltas for all paths and timesteps simultaneously"""
    n_paths, n_timesteps = S_paths.shape
    T = times[-1]
    
    # Broadcast tau for all paths and times: (1, n_timesteps)
    tau = (T - times).unsqueeze(0)  # Shape: (1, n_timesteps)
    
    # Clamp volatilities and compute vol_sqrt_t
    sigma_inst = torch.sqrt(torch.clamp(v_paths, min=1e-12))  # Shape: (n_paths, n_timesteps)
    
    # Create masks for expiry vs normal cases
    tau_safe = torch.clamp(tau, min=1e-12)
    is_expiry = tau <= 1e-12
    
    # Compute ratios and d1 for all paths/times
    s_ratio = torch.clamp(S_paths / K_t, min=1e-8, max=1e8)
    log_s_k = torch.log(s_ratio)
    vol_sqrt_t = sigma_inst * torch.sqrt(tau_safe)
    vol_sqrt_t = torch.clamp(vol_sqrt_t, min=1e-8)
    
    d1 = (log_s_k + (r_t - q_t + 0.5 * sigma_inst ** 2) * tau_safe) / vol_sqrt_t
    d1 = torch.clamp(d1, min=-10, max=10)
    
    # Compute deltas
    discount_factor = torch.exp(-q_t * tau_safe)
    normal_deltas = discount_factor * norm_cdf(d1)
    
    if option_type == 'put':
        normal_deltas = normal_deltas - discount_factor
    
    # Handle expiry cases: delta = 1 if ITM, 0 if OTM
    if option_type == 'call':
        expiry_deltas = (S_paths > K_t).float()
    else:
        expiry_deltas = -(S_paths < K_t).float()
    
    # Select based on mask
    deltas = torch.where(is_expiry, expiry_deltas, normal_deltas)
    
    return deltas

def simulate_cash_evolution_vectorized(S_paths, deltas, times, r_t, rebal_freq, tc_t, impact_t,
                                     return_timeseries=False):
    """Vectorized cash evolution simulation"""
    n_paths, n_timesteps = S_paths.shape
    n_steps = n_timesteps - 1
    dt = torch.diff(times)
    
    # Initialize arrays
    cash = torch.zeros(n_paths, device=S_paths.device, dtype=tensor_dtype)
    trades_count = torch.zeros(n_paths, device=S_paths.device, dtype=tensor_dtype)
    spread_cost_total = torch.zeros(n_paths, device=S_paths.device, dtype=tensor_dtype)
    impact_cost_total = torch.zeros(n_paths, device=S_paths.device, dtype=tensor_dtype)
    
    if return_timeseries:
        step_returns = torch.zeros((n_paths, n_steps), dtype=tensor_dtype, device=S_paths.device)
    else:
        step_returns = None
    
    # Create rebalancing schedule
    reb_indices = torch.arange(0, n_timesteps, rebal_freq, device=S_paths.device)
    if reb_indices[-1] != n_steps:
        reb_indices = torch.cat([reb_indices, torch.tensor([n_steps], device=S_paths.device)])
    
    # Initial trade (t=0)
    Delta_prev = deltas[:, 0]  # Shape: (n_paths,)
    trade0 = Delta_prev * S_paths[:, 0]
    cost0 = smooth_abs_vectorized(trade0) * tc_t + impact_t * (trade0 ** 2)
    cash = -trade0 - cost0  # Start with negative trade value + costs
    
    # Update counters
    trades_count += (torch.abs(Delta_prev) > 0).float()
    spread_cost_total += torch.abs(trade0) * tc_t
    impact_cost_total += impact_t * (trade0 ** 2)
    
    # Track notional and last equity for returns
    notional0 = torch.abs(trade0) + 1e-8
    if return_timeseries:
        last_equity = cash + Delta_prev * S_paths[:, 0]
    
    # Process each rebalancing period
    for k in range(1, len(reb_indices)):
        t_idx_prev = int(reb_indices[k-1])
        t_idx = int(reb_indices[k])
        
        # Accrue interest and track returns for steps between rebalancing
        for j in range(t_idx_prev, t_idx):
            cash *= torch.exp(r_t * dt[j])
            
            if return_timeseries and j < n_steps:
                equity_now = cash + Delta_prev * S_paths[:, j + 1]
                step_returns[:, j] = (equity_now - last_equity) / notional0
                last_equity = equity_now
        
        # Rebalance at t_idx (if not at end)
        if t_idx < n_timesteps:
            Delta_new = deltas[:, t_idx]
            dDelta = Delta_new - Delta_prev
            trade_value = dDelta * S_paths[:, t_idx]
            cost = smooth_abs_vectorized(trade_value) * tc_t + impact_t * (trade_value ** 2)
            
            cash -= trade_value + cost
            
            # Update counters
            has_trade = torch.abs(dDelta) > 0
            trades_count += has_trade.float()
            spread_cost_total += torch.abs(trade_value) * tc_t
            impact_cost_total += impact_t * (trade_value ** 2)
            
            Delta_prev = Delta_new
    
    return cash, Delta_prev, trades_count, spread_cost_total, impact_cost_total, step_returns

def delta_hedge_sim_vectorized(S_paths, v_paths, times, K, r, q, tc=0.0008, impact_lambda=0.0,
                              option_type='call', rebal_freq=1, exposure_scale=1.0, 
                              return_timeseries=False, return_torch=False):
    """
    Fully vectorized delta hedge simulation
    
    Args:
        S_paths: (n_paths, n_timesteps) stock price paths
        v_paths: (n_paths, n_timesteps) variance paths  
        times: (n_timesteps,) time grid
        K: strike price
        r: risk-free rate
        q: dividend yield
        tc: transaction cost rate
        impact_lambda: market impact parameter
        option_type: 'call' or 'put'
        rebal_freq: rebalancing frequency (in steps)
        exposure_scale: scaling factor for delta exposure
        return_timeseries: whether to return step-by-step returns
        return_torch: whether to return torch tensors or numpy arrays
    """
    # Convert inputs to tensors on correct device
    S_paths = torch.as_tensor(S_paths, dtype=tensor_dtype, device=device)
    v_paths = torch.as_tensor(v_paths, dtype=tensor_dtype, device=device)
    times = torch.as_tensor(times, dtype=tensor_dtype, device=device)
    
    # Convert parameters to tensors on same device as input data
    device_to_use = S_paths.device
    K_t = torch.tensor(K, dtype=tensor_dtype, device=device_to_use)
    r_t = torch.tensor(r, dtype=tensor_dtype, device=device_to_use)
    q_t = torch.tensor(q, dtype=tensor_dtype, device=device_to_use)
    tc_t = torch.tensor(tc, dtype=tensor_dtype, device=device_to_use)
    impact_t = torch.tensor(impact_lambda, dtype=tensor_dtype, device=device_to_use)
    exposure_scale_t = torch.tensor(exposure_scale, dtype=tensor_dtype, device=device_to_use)
    
    n_paths, n_timesteps = S_paths.shape
    T = times[-1]
    
    # Compute initial option price
    S0 = S_paths[0, 0]
    sigma0 = torch.sqrt(torch.clamp(v_paths[:, 0].mean(), min=1e-12))
    C0 = bs_price(S0, K_t, r_t, q_t, sigma0, T, option_type=option_type)
    
    # Compute all deltas at once
    deltas = compute_bs_deltas_vectorized(S_paths, v_paths, times, K_t, r_t, q_t, option_type)
    deltas = deltas * exposure_scale_t  # Apply scaling
    
    # Simulate cash evolution
    cash, final_delta, trades_count, spread_cost_total, impact_cost_total, step_returns = \
        simulate_cash_evolution_vectorized(S_paths, deltas, times, r_t, rebal_freq, tc_t, impact_t, return_timeseries)
    
    # Handle final expiry
    if option_type == 'call':
        final_intrinsic_delta = (S_paths[:, -1] > K_t).float()
        payoff = torch.clamp(S_paths[:, -1] - K_t, min=0.0)
    else:
        final_intrinsic_delta = -(S_paths[:, -1] < K_t).float()
        payoff = torch.clamp(K_t - S_paths[:, -1], min=0.0)
    
    # Final closing trade
    final_trade = (final_intrinsic_delta - final_delta) * S_paths[:, -1]
    final_cost = smooth_abs_vectorized(final_trade) * tc_t + impact_t * (final_trade ** 2)
    cash -= final_trade + final_cost
    
    # Update final trade statistics
    has_final_trade = torch.abs(final_intrinsic_delta - final_delta) > 0
    trades_count += has_final_trade.float()
    spread_cost_total += torch.abs(final_trade) * tc_t
    impact_cost_total += impact_t * (final_trade ** 2)
    
    # Final P&L: add initial option premium, subtract final payoff
    pnl = C0 + cash - payoff
    
    # Prepare diagnostics
    diag = {
        'trades': trades_count,
        'avg_spread_cost': spread_cost_total / torch.clamp(trades_count, min=1e-8),
        'avg_impact_cost': impact_cost_total / torch.clamp(trades_count, min=1e-8),
        'total_spread_cost': spread_cost_total,
        'total_impact_cost': impact_cost_total,
    }
    
    # Return results
    if return_torch:
        return pnl, C0, step_returns, diag
    else:
        # Convert to numpy
        diag_np = {k: v.detach().cpu().numpy() for k, v in diag.items()}
        pnl_np = pnl.detach().cpu().numpy()
        C0_np = C0.detach().cpu().item()
        step_returns_np = step_returns.detach().cpu().numpy() if step_returns is not None else None
        return pnl_np, C0_np, step_returns_np, diag_np

# Compatibility aliases
delta_hedge_sim = delta_hedge_sim_vectorized

def compute_per_path_deltas_scaling(S_paths, K, times, r, q, relative_eps=0.001):
    """Compatibility function - returns simple deltas"""
    import torch
    S_paths = torch.as_tensor(S_paths, dtype=tensor_dtype, device=device)
    times = torch.as_tensor(times, dtype=tensor_dtype, device=device)
    K_t = torch.tensor(K, dtype=tensor_dtype, device=device)
    r_t = torch.tensor(r, dtype=tensor_dtype, device=device)
    q_t = torch.tensor(q, dtype=tensor_dtype, device=device)
    
    # Simple Black-Scholes deltas for compatibility
    n_paths, m = S_paths.shape
    T = times[-1]
    tau = T - times
    deltas = torch.zeros_like(S_paths)
    
    # Compute deltas using vectorized method
    sigma = 0.2  # Default volatility for compatibility
    v_paths = torch.full_like(S_paths, sigma**2)
    deltas = compute_bs_deltas_vectorized(S_paths, v_paths, times, K_t, r_t, q_t)
    
    return deltas.cpu().numpy()
