import math
import torch
from optlib.utils.tensor import tensor_dtype
import logging

# Reduce verbosity for COS pricing
logging.getLogger(__name__).setLevel(logging.INFO)
logger = logging.getLogger(__name__)
def cos_price_from_cf(S0, K, r, q, T, cf_func, params=None):
    """COS method pricing with adaptive parameters and numerical c2 calculation.
    Runs on CPU float64 for numerical stability. Supports vectorized K.
    """
    cpu = torch.device('cpu')
    num_dtype = torch.float64
    K_t = torch.as_tensor(K, dtype=num_dtype, device=cpu)
    c1 = math.log(float(S0)) + (r - q) * T
    if params is not None and len(params) >= 5:
        kappa, theta, sigma_v, rho, v0 = params[:5]
        h = 1e-5
        try:
            phi_pos = cf_func(torch.tensor([h], dtype=torch.complex128, device=cpu))
            phi_neg = cf_func(torch.tensor([-h], dtype=torch.complex128, device=cpu))
            phi_0 = cf_func(torch.tensor([0.0], dtype=torch.complex128, device=cpu))
            if torch.any(torch.abs(phi_pos - 1.0) < 1e-10) or torch.any(torch.abs(phi_neg - 1.0) < 1e-10):
                logger.warning("CF too close to 1 for c2, reducing h")
                h *= 0.1
                phi_pos = cf_func(torch.tensor([h], dtype=torch.complex128, device=cpu))
                phi_neg = cf_func(torch.tensor([-h], dtype=torch.complex128, device=cpu))
            log_phi_pos = torch.log(phi_pos)
            log_phi_neg = torch.log(phi_neg)
            log_phi_0 = torch.log(phi_0)
            c2_num = - (log_phi_pos.real - 2 * log_phi_0.real + log_phi_neg.real) / (h ** 2)
            c2 = max(1e-8, float(c2_num))
    # Reduced verbosity - only log occasionally
    # logger.debug(f"Numerical c2: {c2:.6f} for params {params}")
        except Exception as e:
            logger.warning(f"Failed numerical c2: {e}, falling back to approx")
            c2 = (
                T * v0 +
                (kappa * theta * T**2) / 2 +
                (sigma_v**2 * T**3) / 12 +
                (rho * sigma_v * T**2 * (v0 - theta)) / 4 +
                (rho * sigma_v * kappa * theta * T**3) / 6
            )
            c2 = max(1e-8, min(float(c2), 4.0 * T))
        volatility_of_vol = max(0.05, float(sigma_v))
        time_scaling = max(math.sqrt(max(T, 1e-6)), 0.1)
        N = max(256, int(512 * volatility_of_vol * time_scaling))
        N = min(N, 2048)
        L = max(10, int(10 + 5 * volatility_of_vol + 2 * time_scaling))
        L = min(L, 25)
    # Reduced verbosity - removed iteration logging
    else:
        market_vol = 0.25
        c2 = T * market_vol**2 + 0.5 * T**2 * market_vol**2
        N = 256
        L = 12
        logger.debug(f"Default N={N}, L={L}, c2={c2:.6f}")
    a = c1 - L * math.sqrt(abs(c2) + 1e-12)
    b = c1 + L * math.sqrt(abs(c2) + 1e-12)

    if K_t.dim() == 0:
        c = torch.log(K_t).reshape(1)
    else:
        c = torch.log(K_t).reshape(-1)  

    k = torch.arange(N, dtype=num_dtype, device=cpu)               
    u = k * math.pi / (b - a)                                      
    def ChiPsi(k, a, b, c_vec, d):

        kpi = k[:, None] * math.pi / (b - a)                       
        c_t = c_vec[None, :]                                       
        d_t = torch.tensor(d, dtype=num_dtype, device=cpu)         

        exp_c = torch.exp(c_t)
        exp_d = torch.exp(d_t)

        term1 = torch.cos(kpi * (d_t - a)) * exp_d - torch.cos(kpi * (c_t - a)) * exp_c
        term2 = kpi * (torch.sin(kpi * (d_t - a)) * exp_d - torch.sin(kpi * (c_t - a)) * exp_c)
        Chi = (term1 + term2) / (1.0 + kpi**2)

        Psi = torch.zeros_like(Chi)
        mask0 = (k == 0)
        Psi[mask0, :] = (d_t - c_t)
        if (~mask0).any():
            kpi_nz = kpi[~mask0, :]
            Psi[~mask0, :] = (torch.sin(kpi_nz * (d_t - a)) - torch.sin(kpi_nz * (c_t - a))) * (b - a) / (k[~mask0, None] * math.pi)
        return Chi, Psi
    Chi, Psi = ChiPsi(k, a, b, c, b)                               
    K_col = K_t.reshape(-1).to(num_dtype)                           
    Vk = 2.0 / (b - a) * (Chi - K_col[None, :] * Psi)               
    Vk[0, :] *= 0.5
    u_c = u.to(torch.complex128)                                    
    try:
        phi_u = cf_func(u_c)                                        
    except Exception as e:
        logger.error(f"Characteristic function failed: {e}")
        raise

    exp_term = torch.exp(1j * u_c * (0.0 - a))                      
    mat = (phi_u * exp_term)[:, None]                               
    price_t = math.exp(-r * T) * (mat * Vk.to(torch.complex128)).sum(dim=0).real  
    price = price_t.numpy() if K_t.dim() > 0 else price_t.item()
    return price
