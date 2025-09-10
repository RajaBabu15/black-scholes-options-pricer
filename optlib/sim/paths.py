import math
import torch
from optlib.utils.tensor import tensor_dtype, device


def generate_heston_paths(S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps, device=device):
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
        v_next = v_t + kappa * (theta - v_t) * dt + sigma_v * sqrt_v_t * sqrt_dt * W2
        v_paths[:, t + 1] = torch.maximum(v_next, torch.tensor(1e-8, device=device))
        S_next = S_t * torch.exp((r - q - 0.5 * v_t) * dt + sqrt_v_t * sqrt_dt * W1)
        S_paths[:, t + 1] = S_next
    return S_paths, v_paths

