import torch
from optlib.utils.tensor import tensor_dtype, device
from optlib.pricing.bs import bs_price, norm_cdf


def smooth_abs(x, delta=1e-6):
    return torch.sqrt(x * x + delta)


def delta_hedge_sim(S_paths, v_paths, times, K, r, q, tc=0.0008, impact_lambda=0.0, option_type='call', rebal_freq=1, deltas_mode='bs', per_path_deltas=None, exposure_scale=1.0, return_timeseries=False, anti_lookahead_checks=True, return_torch=False):
    S_paths = torch.as_tensor(S_paths, dtype=tensor_dtype, device=device)
    v_paths = torch.as_tensor(v_paths, dtype=tensor_dtype, device=device)
    times = torch.as_tensor(times, dtype=tensor_dtype, device=device)
    n_paths, m = S_paths.shape
    n_steps = m - 1
    dt = torch.diff(times)
    T = times[-1]
    tau = T - times
    S0 = S_paths[0, 0]
    sigma0 = torch.sqrt(torch.maximum(v_paths[:, 0], torch.tensor(0.0, device=device))).mean()
    C0 = bs_price(S0, K, r, q, sigma0, T, option_type=option_type)
    pnl = torch.zeros(n_paths, device=device)
    step_returns = torch.zeros((n_paths, n_steps), dtype=tensor_dtype, device=device) if return_timeseries else None
    trades_count = torch.zeros(n_paths, dtype=tensor_dtype, device=device)
    spread_cost_total = torch.zeros(n_paths, dtype=tensor_dtype, device=device)
    impact_cost_total = torch.zeros(n_paths, dtype=tensor_dtype, device=device)
    if exposure_scale > 1.0:
        raise ValueError("Exposure scale > 1.0 is not allowed (risk control)")
    for i in range(n_paths):
        S_path, v_path = S_paths[i], v_paths[i]
        if deltas_mode == 'bs':
            sigma_inst = torch.sqrt(torch.maximum(v_path, torch.tensor(1e-12, device=device)))
            deltas = torch.zeros_like(S_path)
            for t in range(m):
                if anti_lookahead_checks:
                    assert t < m, "Index out of bounds"
                if tau[t] <= 1e-12:
                    deltas[t] = 1.0 if S_path[t] > K else 0.0
                else:
                    d1 = (torch.log(S_path[t] / K) + (r - q + 0.5 * sigma_inst[t] ** 2) * tau[t]) / (sigma_inst[t] * torch.sqrt(tau[t]))
                    deltas[t] = torch.exp(-q * tau[t]) * norm_cdf(d1)
        elif deltas_mode == 'perpath':
            deltas = torch.as_tensor(per_path_deltas[i], dtype=tensor_dtype, device=device)
        else:
            raise ValueError("Unknown deltas_mode")
        exposure_scale_t = exposure_scale if isinstance(exposure_scale, torch.Tensor) else torch.as_tensor(exposure_scale, dtype=tensor_dtype, device=device)
        deltas = deltas * exposure_scale_t
        Delta_prev = deltas[0]
        notional0 = torch.abs(C0) + torch.abs(Delta_prev * S_path[0]) + 1e-8
        trade0 = Delta_prev * S_path[0]
        cash = C0 - trade0 - (smooth_abs(trade0) * tc + impact_lambda * (trade0 ** 2))
        trades_count[i] += 1.0
        spread_cost_total[i] += torch.abs(Delta_prev * S_path[0]) * tc
        impact_cost_total[i] += impact_lambda * (Delta_prev * S_path[0]) ** 2
        reb = torch.arange(0, m, rebal_freq, device=device)
        if reb[-1] != n_steps:
            reb = torch.cat([reb, torch.tensor([n_steps], device=device)])
        last_equity = cash + Delta_prev * S_path[0]
        for k in range(1, len(reb)):
            t_idx, t_idx_prev = int(reb[k]), int(reb[k - 1])
            for j in range(t_idx_prev, t_idx):
                cash *= torch.exp(r * dt[j])
                if return_timeseries:
                    equity_now = cash + Delta_prev * S_path[j + 1]
                    step_returns[i, j] = (equity_now - last_equity) / notional0
                    last_equity = equity_now
            Delta_new = deltas[t_idx]
            if anti_lookahead_checks:
                assert t_idx < m, "Hedge index beyond path length"
            dDelta = Delta_new - Delta_prev
            trade_value = dDelta * S_path[t_idx]
            cost = smooth_abs(trade_value) * tc + impact_lambda * (trade_value ** 2)
            cash -= trade_value + cost
            if torch.abs(dDelta) > 0:
                trades_count[i] += 1.0
                spread_cost_total[i] += smooth_abs(trade_value) * tc
                impact_cost_total[i] += impact_lambda * (trade_value ** 2)
            Delta_prev = Delta_new
        for j in range(int(reb[-1]), n_steps):
            cash *= torch.exp(r * dt[j])
            if return_timeseries:
                equity_now = cash + Delta_prev * S_path[j + 1]
                step_returns[i, j] = (equity_now - last_equity) / notional0
                last_equity = equity_now
        if option_type == 'call':
            Delta_expiry = (S_path[-1] > K).float()
            payoff = torch.maximum(S_path[-1] - K, torch.tensor(0.0, device=device))
        else:
            Delta_expiry = (S_path[-1] < K).float()
            payoff = torch.maximum(K - S_path[-1], torch.tensor(0.0, device=device))
        final_trade = (Delta_expiry - Delta_prev) * S_path[-1]
        final_hedge_cost = smooth_abs(final_trade) * tc
        cash -= final_trade + final_hedge_cost
        if return_timeseries and torch.abs(Delta_expiry - Delta_prev) > 0:
            trades_count[i] += 1.0
            spread_cost_total[i] += smooth_abs(final_trade) * tc
            impact_cost_total[i] += impact_lambda * (final_trade ** 2)
        pnl[i] = cash + Delta_prev * S_path[-1] - payoff
    if return_torch:
        diag = {
            'trades': trades_count.detach(),
            'avg_spread_cost': (spread_cost_total / (trades_count + 1e-8)).detach(),
            'avg_impact_cost': (impact_cost_total / (trades_count + 1e-8)).detach(),
            'total_spread_cost': spread_cost_total.detach(),
            'total_impact_cost': impact_cost_total.detach(),
        }
        return pnl, C0, step_returns, diag
    diag = {
        'trades': trades_count.cpu().numpy(),
        'avg_spread_cost': (spread_cost_total / (trades_count + 1e-8)).cpu().numpy(),
        'avg_impact_cost': (impact_cost_total / (trades_count + 1e-8)).cpu().numpy(),
        'total_spread_cost': spread_cost_total.cpu().numpy(),
        'total_impact_cost': impact_cost_total.cpu().numpy(),
    }
    out_pnl = pnl.cpu().numpy()
    out_C0 = C0.cpu().item() if isinstance(C0, torch.Tensor) else C0
    if return_timeseries:
        return out_pnl, out_C0, step_returns.cpu().numpy(), diag
    return out_pnl, out_C0, None, diag


def compute_per_path_deltas_scaling(S_paths, K, times, r, q, relative_eps=0.001):
    S_paths = torch.as_tensor(S_paths, dtype=tensor_dtype, device=device)
    times = torch.as_tensor(times, dtype=tensor_dtype, device=device)
    n_paths, m = S_paths.shape
    deltas = torch.zeros_like(S_paths)
    T = times[-1]
    for i in range(n_paths):
        S_path = S_paths[i]
        for t in range(m):
            St = S_path[t]
            if St <= 0:
                continue
            eps = relative_eps * St
            scale_up = (St + eps)/St
            scale_dn = torch.maximum((St - eps)/St, torch.tensor(1e-12, device=device))
            remaining_time = T - times[t]
            if remaining_time > 1e-8:
                drift = (r - q) * remaining_time
                ST_up = St * scale_up * torch.exp(drift)
                ST_dn = St * scale_dn * torch.exp(drift)
            else:
                ST_up, ST_dn = St * scale_up, St * scale_dn
            discount_factor = torch.exp(-r * (T - times[t]))
            price_up = discount_factor * torch.maximum(ST_up - K, torch.tensor(0.0, device=device))
            price_dn = discount_factor * torch.maximum(ST_dn - K, torch.tensor(0.0, device=device))
            deltas[i,t] = (price_up - price_dn) / (2*eps)
    return deltas.cpu().numpy()

