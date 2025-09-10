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
        ds = torch.tensor(1.0, device=returns.device)
    else:
        ds = downside.std(unbiased=False) + eps
    return mu / ds


def compound_equity_from_returns(returns: torch.Tensor, init: float = 1.0) -> torch.Tensor:
    return torch.cumprod(1.0 + returns, dim=0) * init


def max_drawdown_torch(equity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    peak = torch.maximum.accumulate(equity)
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
    ann_vol = returns.std(unbiased=False) * torch.sqrt(torch.tensor(periods_per_year, device=returns.device))
    ca = calmar_torch(ann_mu, mdd)

    penalty = torch.tensor(0.0, device=returns.device)
    penalty += torch.where(sh < 1.0, 10.0 * (1.0 - sh) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(sh > 4.0, 5.0 * (sh - 4.0) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(so < 1.5, 6.0 * (1.5 - so) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ca < 1.0, 8.0 * (1.0 - ca) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ann_vol < 0.05, 6.0 * (0.05 - ann_vol) ** 2, torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ann_vol > 0.25, 6.0 * (ann_vol - 0.25) ** 2, torch.tensor(0.0, device=returns.device))

    penalty += torch.where(sh < -0.5, 20.0 * torch.abs(sh + 0.5), torch.tensor(0.0, device=returns.device))
    penalty += torch.where(so < -0.5, 20.0 * torch.abs(so + 0.5), torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ca < 0.0, 15.0 * torch.abs(ca), torch.tensor(0.0, device=returns.device))
    penalty += torch.where(ann_mu < 0.0, 10.0 * torch.abs(ann_mu), torch.tensor(0.0, device=returns.device))

    if (sh.item() < -1.0) or (so.item() < -1.0) or (ca.item() < 0.0):
        return torch.tensor(1e6, device=returns.device)

    score = 0.5 * sh + 0.3 * so + 0.2 * ca
    loss = -score + penalty
    return loss


def optimize_exposure_scale(S_paths_t, v_paths_t, times_t, K, r, q, rebal_freq, mode, tc, impact, target_periods_per_year: float, steps: int = 200, lr: float = 5e-2) -> Dict:
    from optlib.hedge.delta import delta_hedge_sim
    scale_param = torch.nn.Parameter(torch.tensor(0.0, dtype=S_paths_t.dtype, device=S_paths_t.device))
    optimizer = torch.optim.Adam([scale_param], lr=lr)
    for step in range(steps):
        optimizer.zero_grad()
        scale = torch.sigmoid(scale_param)
        pnl_t, C0_t, step_ts_t, diag = delta_hedge_sim(
            S_paths_t, v_paths_t, times_t, K, r, q,
            tc=tc, impact_lambda=impact, rebal_freq=rebal_freq,
            deltas_mode=mode, per_path_deltas=None,
            exposure_scale=scale, return_timeseries=True, anti_lookahead_checks=True, return_torch=True)
        ret_ts = step_ts_t.mean(dim=0)
        loss = composite_loss(ret_ts, periods_per_year=target_periods_per_year)
        loss.backward()
        optimizer.step()
    return {'scale': torch.sigmoid(scale_param).detach().cpu().item()}

