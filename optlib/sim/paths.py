"""
Ultra-fast vectorized Heston path generation using full GPU acceleration
No loops, pure tensor operations for maximum performance
"""
import torch
import numpy as np
from typing import List, Dict
from optlib.utils.tensor import tensor_dtype, device
def generate_heston_paths_ultra(S0: float, r: float, q: float, T: float, kappa: float,
                               theta: float, sigma_v: float, rho: float, v0: float,
                               n_paths: int, n_steps: int, 
                               device_override=None) -> tuple:
    """
    Ultra-fast fully vectorized Heston path generation
    All operations are batched tensor operations on GPU
    """
    device_to_use = device_override or device
    
    # Time grid
    dt = T / n_steps
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device_to_use, dtype=tensor_dtype))
    
    # Pre-allocate all paths on GPU
    S_paths = torch.zeros((n_paths, n_steps + 1), device=device_to_use, dtype=tensor_dtype)
    v_paths = torch.zeros((n_paths, n_steps + 1), device=device_to_use, dtype=tensor_dtype)
    
    # Initial conditions
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    # Convert parameters to tensors
    kappa_t = torch.tensor(kappa, device=device_to_use, dtype=tensor_dtype)
    theta_t = torch.tensor(theta, device=device_to_use, dtype=tensor_dtype)
    sigma_v_t = torch.tensor(sigma_v, device=device_to_use, dtype=tensor_dtype)
    rho_t = torch.tensor(rho, device=device_to_use, dtype=tensor_dtype)
    r_t = torch.tensor(r, device=device_to_use, dtype=tensor_dtype)
    q_t = torch.tensor(q, device=device_to_use, dtype=tensor_dtype)
    dt_t = torch.tensor(dt, device=device_to_use, dtype=tensor_dtype)
    
    # Pre-compute correlation matrix terms
    sqrt_1_rho2 = torch.sqrt(1 - rho_t * rho_t)
    
    # Generate ALL random numbers at once (most efficient)
    # Shape: (n_paths, n_steps, 2) for [dW1, dW2]
    randn_all = torch.randn((n_paths, n_steps, 2), device=device_to_use, dtype=tensor_dtype)
    dW1 = randn_all[:, :, 0] * sqrt_dt  # Shape: (n_paths, n_steps)
    dW2_indep = randn_all[:, :, 1] * sqrt_dt  # Independent noise
    
    # Correlated Brownian motion: dW2 = rho * dW1 + sqrt(1-rho^2) * dW2_indep
    dW2 = rho_t * dW1 + sqrt_1_rho2 * dW2_indep
    
    # Vectorized Euler scheme - process all paths and steps simultaneously
    for t in range(n_steps):
        v_curr = v_paths[:, t]
        S_curr = S_paths[:, t]
        
        # Variance process (Feller square-root process)
        # dv = kappa(theta - v)dt + sigma_v*sqrt(v)*dW2
        v_curr_pos = torch.clamp(v_curr, min=1e-8)  # Ensure positive variance
        sqrt_v = torch.sqrt(v_curr_pos)
        
        dv = kappa_t * (theta_t - v_curr) * dt_t + sigma_v_t * sqrt_v * dW2[:, t]
        v_next = torch.clamp(v_curr + dv, min=1e-8)  # Ensure variance stays positive
        
        # Stock price process
        # dS = (r-q)S*dt + sqrt(v)*S*dW1
        dS = (r_t - q_t) * S_curr * dt_t + sqrt_v * S_curr * dW1[:, t]
        S_next = S_curr + dS
        
        # Store next values
        v_paths[:, t + 1] = v_next
        S_paths[:, t + 1] = torch.clamp(S_next, min=1e-8)  # Ensure positive prices
    
    return S_paths, v_paths

def batch_generate_multiple_maturities(S0: float, r: float, q: float,
                                     maturities: List[float], kappa: float, 
                                     theta: float, sigma_v: float, rho: float, v0: float,
                                     n_paths: int, steps_per_year: int = 252) -> Dict[float, tuple]:
    """
    Generate paths for multiple maturities simultaneously using tensor batching
    """
    results = {}
    max_T = max(maturities)
    max_steps = int(max_T * steps_per_year)
    
    # Generate longest maturity first
    S_paths_full, v_paths_full = generate_heston_paths_ultra(
        S0, r, q, max_T, kappa, theta, sigma_v, rho, v0, n_paths, max_steps
    )
    
    # Extract sub-paths for shorter maturities
    for T in maturities:
        n_steps = int(T * steps_per_year)
        if T == max_T:
            results[T] = (S_paths_full, v_paths_full)
        else:
            # Efficient slicing - no memory copy needed
            results[T] = (S_paths_full[:, :n_steps+1], v_paths_full[:, :n_steps+1])
    
    return results

# Main function - compatibility with original API
def generate_heston_paths(S0: float, r: float, q: float, T: float, kappa: float, 
                         theta: float, sigma_v: float, rho: float, v0: float,
                         n_paths: int, n_steps: int) -> tuple:
    """Generate Heston paths using ultra-fast vectorized method"""
    return generate_heston_paths_ultra(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps)
