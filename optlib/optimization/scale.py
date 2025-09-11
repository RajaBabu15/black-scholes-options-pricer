import torch
from typing import Dict


def sharpe_torch(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = returns.mean()
    sigma = returns.std(unbiased=False) + eps
    return mu / sigma


def sortino_torch(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mu = returns.mean()
    downside = returns[returns < 0.0]
    if downside.numel() == 0:
        ds = torch.ones(1, device=returns.device, dtype=returns.dtype)
    else:
        ds = downside.std(unbiased=False) + eps
    return mu / ds


def compound_equity_from_returns(returns: torch.Tensor, init: float = 1.0) -> torch.Tensor:
    return torch.cumprod(1.0 + returns, dim=0) * init


def max_drawdown_torch(equity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = torch.cummax(equity, dim=0).values
    dd = (peak - equity) / (peak + eps)
    return dd.max()


def calmar_torch(ann_return: torch.Tensor, max_dd: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ann_return / (max_dd + eps)


def composite_loss(returns: torch.Tensor, periods_per_year: float) -> torch.Tensor:
    sh = sharpe_torch(returns)
    so = sortino_torch(returns)
    eq = compound_equity_from_returns(returns)
    mdd = max_drawdown_torch(eq)
    ann_mu = returns.mean() * periods_per_year
    periods_tensor = torch.tensor(periods_per_year, device=returns.device, dtype=returns.dtype)
    ann_vol = returns.std(unbiased=False) * torch.sqrt(periods_tensor)
    ca = calmar_torch(ann_mu, mdd)

    penalty = torch.zeros(1, device=returns.device, dtype=returns.dtype)
    zero_tensor = torch.zeros(1, device=returns.device, dtype=returns.dtype)
    penalty += torch.where(sh < 1.0, 10.0 * (1.0 - sh) ** 2, zero_tensor)
    penalty += torch.where(sh > 4.0, 5.0 * (sh - 4.0) ** 2, zero_tensor)
    penalty += torch.where(so < 1.5, 6.0 * (1.5 - so) ** 2, zero_tensor)
    penalty += torch.where(ca < 1.0, 8.0 * (1.0 - ca) ** 2, zero_tensor)
    penalty += torch.where(ann_vol < 0.05, 6.0 * (0.05 - ann_vol) ** 2, zero_tensor)
    penalty += torch.where(ann_vol > 0.25, 6.0 * (ann_vol - 0.25) ** 2, zero_tensor)

    penalty += torch.where(sh < -0.5, 20.0 * torch.abs(sh + 0.5), zero_tensor)
    penalty += torch.where(so < -0.5, 20.0 * torch.abs(so + 0.5), zero_tensor)
    penalty += torch.where(ca < 0.0, 15.0 * torch.abs(ca), zero_tensor)
    penalty += torch.where(ann_mu < 0.0, 10.0 * torch.abs(ann_mu), zero_tensor)

    # Use differentiable operations to avoid breaking gradient chain
    hard_reject_penalty = torch.zeros(1, device=returns.device, dtype=returns.dtype)
    large_penalty = torch.tensor(1e6, device=returns.device, dtype=returns.dtype)
    hard_reject_penalty += torch.where(sh < -1.0, large_penalty, zero_tensor)
    hard_reject_penalty += torch.where(so < -1.0, large_penalty, zero_tensor) 
    hard_reject_penalty += torch.where(ca < 0.0, large_penalty, zero_tensor)

    score = 0.5 * sh + 0.3 * so + 0.2 * ca
    loss = -score + penalty + hard_reject_penalty
    return loss


def optimize_exposure_scale_simple_fallback(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year: float) -> Dict:
    """Simple fallback that tests a few fixed scales without gradient optimization"""
    from optlib.hedge.delta import delta_hedge_sim
    
    print("Using simple fallback optimization...")
    scales_to_test = [0.3, 0.5, 0.7, 0.9, 1.0]
    best_scale = 0.5
    best_score = float('-inf')
    
    for scale in scales_to_test:
        try:
            pnl, C0, step_ts, diag = delta_hedge_sim(
                S_paths_t.detach().cpu().numpy(), v_paths_t.detach().cpu().numpy(), times_t.detach().cpu().numpy(), 
                K, r, q, tc=tc, impact_lambda=impact, rebal_freq=rebal_freq,
                deltas_mode=mode, exposure_scale=scale, return_timeseries=True, return_torch=False)
            
            if step_ts is not None:
                ret_series = step_ts.mean(axis=0)
                sharpe = ret_series.mean() / (ret_series.std() + 1e-8)
                if sharpe > best_score:
                    best_score = sharpe
                    best_scale = scale
                    
        except Exception as e:
            print(f"Error testing scale {scale}: {e}")
            continue
    
    print(f"Simple optimization completed. Best scale: {best_scale:.3f}")
    return {'scale': best_scale}


def optimize_exposure_scale(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year: float, steps: int = 200, lr: float = 5e-2) -> Dict:
    """Primary optimization using simple grid search - gradient optimization is too slow/unstable"""
    return optimize_exposure_scale_simple_fallback(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year)

