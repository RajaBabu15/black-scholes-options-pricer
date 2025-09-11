import math
import torch
from optlib.utils.tensor import tensor_dtype, device

def generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps, device=device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)
    S_paths = torch.zeros((n_paths, n_steps + 1), dtype=tensor_dtype, device=device)
    v_paths = torch.zeros((n_paths, n_steps + 1), dtype=tensor_dtype, device=device)
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    for t in range(n_steps):
        Z1 = torch.randn(n_paths, dtype=tensor_dtype, device=device)
        Z2 = torch.randn(n_paths, dtype=tensor_dtype, device=device)
        W1 = Z1
        W2 = rho * Z1 + math.sqrt(1 - rho**2) * Z2
        S_t = S_paths[:, t]
        v_t = torch.maximum(v_paths[:, t], torch.tensor(1e-8, device=device))
        sqrt_v_t = torch.sqrt(v_t)
        # Milstein for v: Euler + 0.5 sigma_v^2 v (dW^2 - dt)
        v_euler = v_t + kappa * (theta - v_t) * dt + sigma_v * sqrt_v_t * sqrt_dt * W2
        v_milstein = v_euler + 0.5 * sigma_v**2 * v_t * (W2**2 - 1) * dt
        v_next = torch.maximum(v_milstein, torch.tensor(1e-8, device=device))
        v_paths[:, t + 1] = v_next
        S_next = S_t * torch.exp((r - q - 0.5 * v_t) * dt + sqrt_v_t * sqrt_dt * W1)
        S_paths[:, t + 1] = S_next
    return S_paths, v_paths
