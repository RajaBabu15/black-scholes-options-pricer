"""
Hedging Strategy Comparison Module
=================================

Framework for comparing P&L and risk metrics between different hedging strategies:
- Dynamic delta hedging vs static hedging
- Performance analysis and visualization
- Risk-adjusted metrics computation
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from optlib.utils.tensor import tensor_dtype, device
from optlib.hedge.delta import delta_hedge_sim_vectorized
from optlib.hedge.static import (
    static_delta_hedge, buy_and_hold_hedge, static_gamma_hedge,
    protective_put_strategy, covered_call_strategy, no_hedge_strategy
)
from optlib.metrics.performance import calculate_performance_metrics


def compare_hedging_strategies(S_paths: torch.Tensor,
                             v_paths: torch.Tensor,
                             times: torch.Tensor,
                             K: Union[float, torch.Tensor],
                             r: Union[float, torch.Tensor],
                             q: Union[float, torch.Tensor],
                             option_type: str = 'call',
                             dynamic_hedge_params: Optional[Dict] = None,
                             static_strategies: Optional[List[str]] = None,
                             transaction_cost: float = 0.0008,
                             impact_lambda: float = 0.0) -> Dict[str, Dict]:
    """
    Compare multiple hedging strategies on the same set of paths.
    
    Args:
        S_paths, v_paths, times: Monte Carlo paths
        K, r, q: Option parameters
        option_type: 'call' or 'put'
        dynamic_hedge_params: Parameters for dynamic hedging
        static_strategies: List of static strategies to compare
        transaction_cost: Transaction cost rate
        impact_lambda: Market impact parameter
        
    Returns:
        Dictionary with results for each strategy
    """
    results = {}
    
    # Default parameters
    if dynamic_hedge_params is None:
        dynamic_hedge_params = {
            'rebal_freq': 1.0,
            'exposure_scale': 1.0
        }
    
    if static_strategies is None:
        static_strategies = ['static_delta', 'buy_and_hold', 'no_hedge']
    
    # Dynamic delta hedging
    try:
        dynamic_pnl, _, dynamic_returns, dynamic_diag = delta_hedge_sim_vectorized(
            S_paths, v_paths, times, K, r, q,
            tc=transaction_cost,
            impact_lambda=impact_lambda, 
            option_type=option_type,
            rebal_freq=dynamic_hedge_params['rebal_freq'],
            exposure_scale=dynamic_hedge_params['exposure_scale'],
            return_timeseries=True,
            return_torch=True
        )
        
        results['dynamic_delta'] = {
            'pnl': dynamic_pnl,
            'returns': dynamic_returns,
            'trades_count': dynamic_diag['trades'],
            'total_transaction_costs': dynamic_diag['total_spread_cost'] + dynamic_diag['total_impact_cost'],
            'avg_transaction_cost': dynamic_diag['avg_spread_cost'] + dynamic_diag['avg_impact_cost'],
            'strategy_type': 'dynamic'
        }
    except Exception as e:
        print(f"Error in dynamic hedging: {e}")
        results['dynamic_delta'] = {'error': str(e)}
    
    # Static strategies
    for strategy in static_strategies:
        try:
            if strategy == 'static_delta':
                static_result = static_delta_hedge(S_paths, v_paths, times, K, r, q, option_type)
            elif strategy == 'buy_and_hold':
                static_result = buy_and_hold_hedge(S_paths, v_paths, times, K, r, q, option_type)
            elif strategy == 'no_hedge':
                static_result = no_hedge_strategy(S_paths, v_paths, times, K, r, q, option_type)
            elif strategy == 'protective_put':
                if option_type == 'call':
                    # For call options, protective put doesn't apply directly
                    continue
                static_result = protective_put_strategy(S_paths, v_paths, times, K, r, q)
            elif strategy == 'covered_call':
                if option_type == 'put':
                    # For put options, covered call doesn't apply directly  
                    continue
                static_result = covered_call_strategy(S_paths, v_paths, times, K, r, q)
            else:
                print(f"Unknown static strategy: {strategy}")
                continue
                
            results[strategy] = {
                'pnl': static_result['pnl'],
                'returns': None,  # Static strategies don't have return timeseries
                'trades_count': static_result['trades_count'],
                'total_transaction_costs': static_result.get('total_transaction_costs', torch.zeros_like(static_result['pnl'])),
                'strategy_type': 'static',
                **static_result  # Include all strategy-specific results
            }
        except Exception as e:
            print(f"Error in {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    return results


def analyze_strategy_performance(results: Dict[str, Dict],
                               risk_free_rate: float = 0.02,
                               target_periods_per_year: float = 252.0) -> pd.DataFrame:
    """
    Analyze and compare performance metrics across strategies.
    
    Args:
        results: Results from compare_hedging_strategies
        risk_free_rate: Risk-free rate for Sharpe ratio calculation
        target_periods_per_year: Periods per year for annualization
        
    Returns:
        DataFrame with performance metrics for each strategy
    """
    performance_data = []
    
    for strategy_name, strategy_results in results.items():
        if 'error' in strategy_results:
            continue
            
        pnl = strategy_results['pnl']
        if isinstance(pnl, torch.Tensor):
            pnl_np = pnl.cpu().numpy()
        else:
            pnl_np = np.array(pnl)
        
        # Basic statistics
        mean_pnl = np.mean(pnl_np)
        std_pnl = np.std(pnl_np)
        min_pnl = np.min(pnl_np)
        max_pnl = np.max(pnl_np)
        median_pnl = np.median(pnl_np)
        
        # Risk metrics
        profit_prob = np.mean(pnl_np > 0)
        loss_prob = np.mean(pnl_np < 0)  
        
        # VaR and CVaR (5% level)
        var_5 = np.percentile(pnl_np, 5)
        cvar_5 = np.mean(pnl_np[pnl_np <= var_5])
        
        # Sharpe-like ratio (mean/std)
        if std_pnl > 1e-8:
            sharpe_like = mean_pnl / std_pnl
        else:
            sharpe_like = 0.0
        
        # Trading activity
        trades_count = strategy_results.get('trades_count', torch.zeros(len(pnl_np)))
        if isinstance(trades_count, torch.Tensor):
            avg_trades = float(torch.mean(trades_count))
        else:
            avg_trades = np.mean(trades_count)
        
        # Transaction costs
        total_costs = strategy_results.get('total_transaction_costs', torch.zeros(len(pnl_np)))
        if isinstance(total_costs, torch.Tensor):
            avg_cost = float(torch.mean(total_costs))
        else:
            avg_cost = np.mean(total_costs)
        
        # Compile metrics
        performance_data.append({
            'Strategy': strategy_name,
            'Mean_PnL': mean_pnl,
            'Std_PnL': std_pnl,
            'Min_PnL': min_pnl,
            'Max_PnL': max_pnl,
            'Median_PnL': median_pnl,
            'Sharpe_Like': sharpe_like,
            'Profit_Prob': profit_prob,
            'Loss_Prob': loss_prob,
            'VaR_5pct': var_5,
            'CVaR_5pct': cvar_5,
            'Avg_Trades': avg_trades,
            'Avg_Transaction_Cost': avg_cost,
            'Strategy_Type': strategy_results.get('strategy_type', 'unknown')
        })
    
    return pd.DataFrame(performance_data)


def hedging_efficiency_analysis(dynamic_results: Dict,
                              static_results: Dict,
                              benchmark_strategy: str = 'no_hedge') -> Dict[str, float]:
    """
    Analyze hedging efficiency by comparing variance reduction.
    
    Args:
        dynamic_results: Results from dynamic hedging
        static_results: Results from static hedging strategies  
        benchmark_strategy: Strategy to use as unhedged benchmark
        
    Returns:
        Dictionary with efficiency metrics
    """
    # Get benchmark (unhedged) variance
    if benchmark_strategy in static_results:
        benchmark_pnl = static_results[benchmark_strategy]['pnl']
        if isinstance(benchmark_pnl, torch.Tensor):
            benchmark_var = float(torch.var(benchmark_pnl))
        else:
            benchmark_var = np.var(benchmark_pnl)
    else:
        benchmark_var = None
    
    efficiency_metrics = {}
    
    # Dynamic hedging efficiency
    if 'dynamic_delta' in dynamic_results:
        dynamic_pnl = dynamic_results['dynamic_delta']['pnl']
        if isinstance(dynamic_pnl, torch.Tensor):
            dynamic_var = float(torch.var(dynamic_pnl))
        else:
            dynamic_var = np.var(dynamic_pnl)
        
        if benchmark_var is not None and benchmark_var > 1e-8:
            efficiency_metrics['dynamic_variance_reduction'] = 1 - dynamic_var / benchmark_var
        else:
            efficiency_metrics['dynamic_variance_reduction'] = None
        
        efficiency_metrics['dynamic_variance'] = dynamic_var
    
    # Static strategies efficiency
    for strategy_name, strategy_data in static_results.items():
        if strategy_name == benchmark_strategy:
            continue
            
        strategy_pnl = strategy_data['pnl']
        if isinstance(strategy_pnl, torch.Tensor):
            strategy_var = float(torch.var(strategy_pnl))
        else:
            strategy_var = np.var(strategy_pnl)
        
        if benchmark_var is not None and benchmark_var > 1e-8:
            efficiency_metrics[f'{strategy_name}_variance_reduction'] = 1 - strategy_var / benchmark_var
        else:
            efficiency_metrics[f'{strategy_name}_variance_reduction'] = None
            
        efficiency_metrics[f'{strategy_name}_variance'] = strategy_var
    
    if benchmark_var is not None:
        efficiency_metrics['benchmark_variance'] = benchmark_var
    
    return efficiency_metrics


def cost_benefit_analysis(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Analyze cost vs benefit tradeoff for different hedging strategies.
    
    Args:
        results: Results from compare_hedging_strategies
        
    Returns:
        DataFrame with cost-benefit analysis
    """
    analysis_data = []
    
    for strategy_name, strategy_results in results.items():
        if 'error' in strategy_results:
            continue
            
        pnl = strategy_results['pnl']
        if isinstance(pnl, torch.Tensor):
            pnl_np = pnl.cpu().numpy()
        else:
            pnl_np = np.array(pnl)
        
        # Benefits (risk reduction)
        pnl_std = np.std(pnl_np)
        pnl_mean = np.mean(pnl_np)
        
        # Costs (transaction costs + operational complexity)
        total_costs = strategy_results.get('total_transaction_costs', torch.zeros(len(pnl_np)))
        if isinstance(total_costs, torch.Tensor):
            avg_cost = float(torch.mean(total_costs))
        else:
            avg_cost = np.mean(total_costs)
        
        trades_count = strategy_results.get('trades_count', torch.zeros(len(pnl_np)))
        if isinstance(trades_count, torch.Tensor):
            avg_trades = float(torch.mean(trades_count))
        else:
            avg_trades = np.mean(trades_count)
        
        # Operational complexity score (based on number of trades)
        if avg_trades <= 1:
            complexity_score = 1  # Simple
        elif avg_trades <= 10:
            complexity_score = 2  # Moderate
        else:
            complexity_score = 3  # Complex
        
        # Cost-benefit ratio
        if pnl_std > 1e-8:
            cost_benefit_ratio = avg_cost / pnl_std
        else:
            cost_benefit_ratio = float('inf') if avg_cost > 0 else 0.0
        
        analysis_data.append({
            'Strategy': strategy_name,
            'Risk_Reduction_Benefit': 1.0 / (pnl_std + 1e-8),  # Higher is better
            'Mean_PnL': pnl_mean,
            'PnL_Volatility': pnl_std,
            'Avg_Transaction_Cost': avg_cost,
            'Avg_Trades_Per_Path': avg_trades,
            'Operational_Complexity': complexity_score,
            'Cost_Benefit_Ratio': cost_benefit_ratio,  # Lower is better
            'Net_Benefit_Score': pnl_mean - avg_cost - complexity_score * 0.1  # Simple scoring
        })
    
    return pd.DataFrame(analysis_data)


def generate_comparison_report(S_paths: torch.Tensor,
                             v_paths: torch.Tensor, 
                             times: torch.Tensor,
                             K: Union[float, torch.Tensor],
                             r: Union[float, torch.Tensor],
                             q: Union[float, torch.Tensor],
                             option_type: str = 'call',
                             transaction_cost: float = 0.0008,
                             impact_lambda: float = 0.0,
                             output_file: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Generate comprehensive comparison report for hedging strategies.
    
    Args:
        S_paths, v_paths, times: Monte Carlo paths
        K, r, q: Option parameters
        option_type: 'call' or 'put'
        transaction_cost: Transaction cost rate
        impact_lambda: Market impact parameter
        output_file: Optional file to save results
        
    Returns:
        Dictionary containing various analysis DataFrames
    """
    print("Running hedging strategy comparison...")
    
    # Compare strategies
    results = compare_hedging_strategies(
        S_paths, v_paths, times, K, r, q, option_type,
        transaction_cost=transaction_cost,
        impact_lambda=impact_lambda
    )
    
    # Performance analysis
    print("Analyzing performance metrics...")
    performance_df = analyze_strategy_performance(results, risk_free_rate=float(r))
    
    # Separate dynamic and static results
    dynamic_results = {k: v for k, v in results.items() if v.get('strategy_type') == 'dynamic'}
    static_results = {k: v for k, v in results.items() if v.get('strategy_type') == 'static'}
    
    # Efficiency analysis
    print("Computing hedging efficiency...")
    efficiency_metrics = hedging_efficiency_analysis(dynamic_results, static_results)
    
    # Cost-benefit analysis
    print("Performing cost-benefit analysis...")
    cost_benefit_df = cost_benefit_analysis(results)
    
    # Compile report
    report = {
        'performance_metrics': performance_df,
        'cost_benefit_analysis': cost_benefit_df,
        'efficiency_metrics': efficiency_metrics,
        'raw_results': results
    }
    
    # Save to file if requested
    if output_file:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            performance_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            cost_benefit_df.to_excel(writer, sheet_name='Cost_Benefit', index=False)
            
            # Add efficiency metrics as a separate sheet
            efficiency_df = pd.DataFrame([efficiency_metrics]).T
            efficiency_df.columns = ['Value']
            efficiency_df.to_excel(writer, sheet_name='Efficiency_Metrics')
        
        print(f"Report saved to {output_file}")
    
    return report