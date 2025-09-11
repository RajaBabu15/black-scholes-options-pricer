import numpy as np

def calculate_performance_metrics(pnl, risk_free_rate=0.01, periods_per_year=252.0):
    if len(pnl) < 2:
        return {metric: 0.0 for metric in ['sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'calmar_ratio']}
    ann_factor = np.sqrt(periods_per_year)
    mean_pnl = np.mean(pnl)
    std_pnl = np.std(pnl)
    daily_rf = risk_free_rate / periods_per_year
    sharpe = (mean_pnl - daily_rf) / std_pnl * ann_factor if std_pnl > 0 else 0.0
    negative_pnl = pnl[pnl < 0]
    downside_std = np.std(negative_pnl) if len(negative_pnl) > 0 else 0.0
    sortino = (mean_pnl - daily_rf) / downside_std * ann_factor if downside_std > 0 else 0.0
    cumulative_pnl = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    annual_return = mean_pnl * periods_per_year
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    return {
        'total_pnl': np.sum(pnl),
        'annual_return': annual_return,
        'annual_volatility': std_pnl * ann_factor,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'calmar_ratio': calmar
    }

