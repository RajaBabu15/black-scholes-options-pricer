import math
import torch
from optlib.utils.tensor import tensor_dtype, device


def norm_cdf(x):
    x = torch.as_tensor(x, dtype=tensor_dtype, device=device)
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    x = torch.as_tensor(x, dtype=tensor_dtype, device=device)
    return torch.exp(-0.5 * x * x) / math.sqrt(2*math.pi)


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

