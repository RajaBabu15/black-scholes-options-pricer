"""
Vectorized scale optimization that tests multiple scales in parallel
"""
import torch
from typing import Dict, List
from optlib.hedge.delta import delta_hedge_sim_vectorized
from optlib.utils.tensor import tensor_dtype, device

def compute_batch_sharpe_ratio(step_returns_batch, eps=1e-8):
    """
    Compute Sharpe ratios for batch of return series
    
    Args:
        step_returns_batch: (n_scales, n_paths, n_steps) return series for different scales
        eps: small number to avoid division by zero
    
    Returns:
        sharpe_ratios: (n_scales,) Sharpe ratios for each scale
    """
    # Compute mean returns across paths and time: (n_scales,)
    mean_returns = step_returns_batch.mean(dim=(1, 2))  
    
    # Compute standard deviation across paths and time: (n_scales,)
    std_returns = step_returns_batch.view(step_returns_batch.shape[0], -1).std(dim=1, unbiased=False)
    
    # Compute Sharpe ratios
    sharpe_ratios = mean_returns / (std_returns + eps)
    
    return sharpe_ratios

def batch_delta_hedge_sim_vectorized(S_paths, v_paths, times, K, r, q, scales_to_test,
                                   rebal_freq, tc, impact):
    """
    Run delta hedge simulation for multiple scales in parallel
    
    Args:
        S_paths, v_paths, times: Path data
        K, r, q: Option parameters
        scales_to_test: List of exposure scales to test
        rebal_freq, tc, impact: Trading parameters
        
    Returns:
        results: List of (pnl, step_returns, diag) for each scale
    """
    n_scales = len(scales_to_test)
    results = []
    
    # Since we need to modify exposure scaling, we run each scale separately
    # but use the vectorized simulation for each
    for scale in scales_to_test:
        pnl, C0, step_returns, diag = delta_hedge_sim_vectorized(
            S_paths, v_paths, times, K, r, q,
            tc=tc, impact_lambda=impact, rebal_freq=rebal_freq,
            exposure_scale=scale, return_timeseries=True, return_torch=True
        )
        results.append((pnl, step_returns, diag))
    
    return results

def optimize_exposure_scale_vectorized(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq,
                                     mode, tc, impact, target_periods_per_year: float) -> Dict:
    """
    Vectorized exposure scale optimization using batch processing
    """
    # Original scale testing with full training data
    max_test_paths = 200  # Restored from 100 to 200
    n_paths = S_paths_t.shape[0]
    if n_paths > max_test_paths:
        S_test = S_paths_t[:max_test_paths]
        v_test = v_paths_t[:max_test_paths]
    else:
        S_test = S_paths_t
        v_test = v_paths_t
    
    # Use more time steps for better accuracy
    n_timesteps = times_t.shape[0]
    if n_timesteps > 50:  # Increased threshold from 25 to 50
        step_indices = torch.arange(0, n_timesteps, 2, device=times_t.device)  # Every 2nd step instead of 3rd
        if step_indices[-1] != n_timesteps - 1:
            step_indices = torch.cat([step_indices, torch.tensor([n_timesteps - 1], device=times_t.device)])
        times_test = times_t[step_indices]
        S_test = S_test[:, step_indices]
        v_test = v_test[:, step_indices]
    else:
        times_test = times_t
    
    # Extreme scale testing range for target performance (Sharpe=1.0, Return=30%)
    scales_to_test = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]  # Very wide exposure range
    
    # Run batch simulation
    results = batch_delta_hedge_sim_vectorized(
        S_test, v_test, times_test, K, r, q, scales_to_test, 
        rebal_freq, tc, impact
    )
    
    # Evaluate each scale
    best_scale = scales_to_test[0]
    best_score = float('-inf')
    
    for i, (scale, (pnl, step_returns, diag)) in enumerate(zip(scales_to_test, results)):
        if step_returns is not None:
            # Compute Sharpe ratio across all paths and steps
            ret_flat = step_returns.view(-1)  # Flatten to 1D
            sharpe = ret_flat.mean() / (ret_flat.std(unbiased=False) + 1e-8)
            sharpe_val = float(sharpe)
            
            if sharpe_val > best_score:
                best_score = sharpe_val
                best_scale = scale
    return {'scale': best_scale}

# Convenience function that maintains the same interface as the original
def optimize_exposure_scale(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year: float) -> Dict:
    """Wrapper function that uses vectorized optimization"""
    return optimize_exposure_scale_vectorized(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year)
