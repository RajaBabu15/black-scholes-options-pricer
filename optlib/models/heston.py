import math
import torch

def heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0):
    cpu = torch.device('cpu')
    cdtype = torch.complex128
    u = torch.as_tensor(u, dtype=cdtype, device=cpu)
    i = torch.tensor(1j, dtype=cdtype, device=cpu)
    sigma_v = max(float(sigma_v), 1e-8)
    T = max(float(T), 1e-8)
    a = kappa * theta
    alpha = kappa - rho * sigma_v * i * u
    beta = (sigma_v**2) * (i * u + u**2)
    discriminant = alpha**2 + beta
    d = torch.sqrt(discriminant + 1e-16j)
    if torch.any(torch.isnan(d)):
        return torch.ones_like(u)
    g = (alpha - d) / (alpha + d)
    g = torch.clamp(g.real, min=-0.9999, max=0.9999) + 1j * torch.clamp(g.imag, min=-50, max=50)
    exp_dT = torch.exp(-d * T)
    log_arg = (1 - g * exp_dT) / (1 - g)
    log_arg = torch.clamp(log_arg.real, min=1e-12, max=1e12) + 1j * torch.clamp(log_arg.imag, min=-1e6, max=1e6)
    C = (i * u * (math.log(float(S0)) + (r - q) * T)
         + a / (sigma_v**2) * ((alpha - d) * T - 2.0 * torch.log(log_arg)))
    D = (alpha - d) / (sigma_v**2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)
    exponent = C + D * v0
    real_clamped = torch.clamp(exponent.real, min=-200.0, max=200.0)
    imag_clamped = torch.clamp(exponent.imag, min=-1e6, max=1e6)
    exponent_clamped = torch.complex(real_clamped, imag_clamped)
    result = torch.exp(exponent_clamped)
    return result

def bates_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0, lambda_j, mu_j, sigma_j):
    cpu = torch.device('cpu')
    cdtype = torch.complex128
    heston_phi = heston_char_func(u, S0, r, q, T, kappa, theta, sigma_v, rho, v0)
    lambda_j_c = torch.as_tensor(lambda_j, dtype=cdtype, device=cpu)
    mu_j_c = torch.as_tensor(mu_j, dtype=cdtype, device=cpu)
    sigma_j_c = torch.as_tensor(sigma_j, dtype=cdtype, device=cpu)
    T_c = torch.as_tensor(T, dtype=cdtype, device=cpu)
    u_c = torch.as_tensor(u, dtype=cdtype, device=cpu)
    i_c = torch.tensor(1j, dtype=cdtype, device=cpu)
    one_c = torch.tensor(1.0, dtype=cdtype, device=cpu)
    half_c = torch.tensor(0.5, dtype=cdtype, device=cpu)

    m = torch.exp(mu_j_c + half_c * sigma_j_c**2)  
    drift_correction = lambda_j_c * (m - one_c)  
    r_eff = r - float(drift_correction.real)  

    heston_phi_eff = heston_char_func(u_c, S0, r_eff, q, T, kappa, theta, sigma_v, rho, v0)

    psi_u = torch.exp(i_c * u_c * mu_j_c - half_c * u_c**2 * sigma_j_c**2)
    jump_component = lambda_j_c * T_c * (psi_u - one_c)
    jump_phi = torch.exp(jump_component)
    return heston_phi_eff * jump_phi
