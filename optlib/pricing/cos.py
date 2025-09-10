import math
import torch
from optlib.utils.tensor import tensor_dtype


def cos_price_from_cf(S0, K, r, q, T, cf_func, params=None, target_precision=1e-6):
    """COS method pricing with adaptive parameters and exact c2 calculation.
    Runs on CPU float64 for numerical stability.
    """
    cpu = torch.device('cpu')
    num_dtype = torch.float64

    S0_t = torch.as_tensor(S0, dtype=num_dtype, device=cpu)
    K_t = torch.as_tensor(K, dtype=num_dtype, device=cpu)

    x0 = math.log(float(S0) * math.exp(-q*T))
    c1 = math.log(float(S0)) + (r - q) * T

    if params is not None and len(params) >= 5:
        kappa, theta, sigma_v, rho, v0 = params[:5]
        c2_exact = (
            T * v0 +
            (kappa * theta * T**2) / 2 +
            (sigma_v**2 * T**3) / 12 +
            (rho * sigma_v * T**2 * (v0 - theta)) / 4 +
            (rho * sigma_v * kappa * theta * T**3) / 6
        )
        c2 = max(1e-8, min(c2_exact, 4.0 * T))
        volatility_of_vol = max(0.05, float(sigma_v))
        time_scaling = max(math.sqrt(max(T, 1e-6)), 0.1)
        N = max(256, int(512 * volatility_of_vol * time_scaling))
        N = min(N, 2048)
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

    u_c = u.to(torch.complex128)
    try:
        phi_u = cf_func(u_c)
    except Exception:
        phi_u = cf_func(u_c - 0.5j)
    exp_term = torch.exp(1j * u_c * (x0 - a))
    mat = phi_u * exp_term
    price_t = math.exp(-r*T) * torch.real(torch.sum(mat * Vk.to(torch.complex128)))
    price = price_t.item() if isinstance(price_t, torch.Tensor) else float(price_t)
    return price

