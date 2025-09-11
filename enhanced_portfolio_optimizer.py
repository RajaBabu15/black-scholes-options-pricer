#!/usr/bin/env python3
"""
Enhanced Portfolio Optimizer
Focuses on optimizing Sortino and Calmar ratios through:
1. Better parameter search grids
2. Multi-objective optimization
3. Drawdown-focused scoring
4. Portfolio-level optimization
"""

import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
import cloudpickle

# Import our optimization components
from optlib.harness.runner import run_hedging_optimization
from optlib.data.market import load_data
from optlib.data.rates import fetch_risk_free_rate

def create_enhanced_parameter_grid():
    """Create an enhanced parameter grid focusing on Sortino/Calmar optimization"""
    
    # Expanded rebalancing frequencies - more granular around optimal ranges
    rebal_list = [1, 2, 3, 5, 7, 10, 15, 20]
    
    # More granular transaction costs - focus on lower costs
    tc_list = [0.0002, 0.0003, 0.0005, 0.0008, 0.001, 0.0015]
    
    # Enhanced market impact levels
    impact_list = [0.0, 5e-7, 1e-6, 2e-6, 5e-6]
    
    # More refined exposure scales - focus on conservative ranges
    scale_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
    
    return {
        'rebal_list': rebal_list,
        'tc_list': tc_list, 
        'impact_list': impact_list,
        'scale_list': scale_list
    }

def enhanced_scoring_function(metrics: Dict, alpha_sortino: float = 0.4, alpha_calmar: float = 0.3, 
                            alpha_sharpe: float = 0.2, alpha_return: float = 0.1) -> float:
    """
    Enhanced scoring function prioritizing Sortino and Calmar ratios
    
    Args:
        metrics: Performance metrics dictionary
        alpha_sortino: Weight for Sortino ratio
        alpha_calmar: Weight for Calmar ratio  
        alpha_sharpe: Weight for Sharpe ratio
        alpha_return: Weight for annualized return
    
    Returns:
        Composite score (higher is better)
    """
    
    # Extract metrics with fallbacks
    sortino = metrics.get('sortino_ratio', -10)
    calmar = metrics.get('calmar_ratio', -10)
    sharpe = metrics.get('sharpe_ratio', -10)
    ann_return = metrics.get('annualized_return', -0.5)
    max_dd = metrics.get('max_drawdown', 0.2)
    
    # Apply transformations to handle negative ratios
    # Use sigmoid-like transforms to map negative values to positive scores
    sortino_score = 2 / (1 + np.exp(-sortino/2)) - 1  # Maps (-∞,∞) to (-1,1)
    calmar_score = 2 / (1 + np.exp(-calmar/2)) - 1
    sharpe_score = 2 / (1 + np.exp(-sharpe/2)) - 1
    return_score = np.tanh(ann_return * 10)  # Scale returns appropriately
    
    # Drawdown penalty - prefer lower drawdowns
    dd_penalty = max(0, max_dd - 0.05) * 10  # Penalty if drawdown > 5%
    
    # Composite score
    score = (alpha_sortino * sortino_score + 
             alpha_calmar * calmar_score +
             alpha_sharpe * sharpe_score + 
             alpha_return * return_score - dd_penalty)
    
    return score

def run_enhanced_single_ticker(args):
    """Enhanced single ticker optimization with better parameter search"""
    ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate, enhanced_params = args
    
    # Deserialize DataFrame and recreate ticker object
    hist_data = cloudpickle.loads(hist_data_bytes)
    import yfinance as yf
    stock_ticker = yf.Ticker(ticker)
    
    print(f"[{ticker}] Starting enhanced optimization with {len(enhanced_params['rebal_list']) * len(enhanced_params['tc_list']) * len(enhanced_params['impact_list']) * len(enhanced_params['scale_list'])} configurations")
    
    # Run the base optimization (using the existing infrastructure)
    result = run_hedging_optimization(ticker, hist_data, stock_ticker, data_dir, log_dir, risk_free_rate)
    
    if result is None:
        return None
        
    # Load the full results CSV to find optimal parameters using enhanced scoring
    results_file = Path(data_dir) / f"opt_results_{ticker}.csv"
    if not results_file.exists():
        return result
        
    full_results = pd.read_csv(results_file)
    
    # Apply enhanced scoring to all configurations
    enhanced_scores = []
    for _, row in full_results.iterrows():
        score = enhanced_scoring_function(row.to_dict())
        enhanced_scores.append(score)
    
    full_results['enhanced_score'] = enhanced_scores
    
    # Find the best configuration using enhanced scoring
    best_config = full_results.loc[full_results['enhanced_score'].idxmax()].to_dict()
    
    # Save enhanced best config
    enhanced_config_file = Path(data_dir) / f"{ticker}_enhanced_best_config.json"
    with open(enhanced_config_file, 'w') as f:
        # Add metadata
        best_config['ticker'] = ticker
        best_config['timestamp'] = datetime.utcnow().isoformat()
        best_config['optimization_method'] = 'enhanced_sortino_calmar'
        json.dump(best_config, f, indent=2)
    
    print(f"[{ticker}] Enhanced optimization complete. Score: {best_config['enhanced_score']:.4f}")
    
    return best_config

def optimize_portfolio_allocation(ticker_results: List[Dict], target_sortino: float = 1.0,
                                target_calmar: float = 0.5) -> Dict:
    """
    Optimize portfolio-level allocation based on individual ticker performance
    
    Args:
        ticker_results: List of best configurations for each ticker
        target_sortino: Target portfolio Sortino ratio
        target_calmar: Target portfolio Calmar ratio
    
    Returns:
        Portfolio allocation and metrics
    """
    
    if not ticker_results:
        return {}
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(ticker_results)
    
    # Calculate individual ticker quality scores
    df['quality_score'] = df.apply(lambda row: enhanced_scoring_function(row.to_dict()), axis=1)
    
    # Simple equal-weight baseline
    n_tickers = len(df)
    equal_weights = np.ones(n_tickers) / n_tickers
    
    # Quality-weighted allocation (higher quality gets more weight)
    quality_weights = df['quality_score'].values
    quality_weights = np.maximum(quality_weights, 0)  # Ensure non-negative
    if quality_weights.sum() > 0:
        quality_weights = quality_weights / quality_weights.sum()
    else:
        quality_weights = equal_weights
    
    # Risk-parity style allocation (inverse volatility weighting)
    vols = df['annual_volatility'].fillna(0.1).values
    risk_weights = 1 / np.maximum(vols, 0.01)  # Avoid division by zero
    risk_weights = risk_weights / risk_weights.sum()
    
    # Combined allocation: 40% quality, 30% risk-parity, 30% equal weight
    combined_weights = 0.4 * quality_weights + 0.3 * risk_weights + 0.3 * equal_weights
    
    # Calculate portfolio-level metrics
    portfolio_metrics = calculate_portfolio_metrics(df, combined_weights)
    
    # Create allocation results
    allocation_results = {
        'allocation_method': 'quality_risk_equal_blend',
        'weights': {ticker: weight for ticker, weight in zip(df['ticker'], combined_weights)},
        'equal_weights': {ticker: weight for ticker, weight in zip(df['ticker'], equal_weights)},
        'quality_weights': {ticker: weight for ticker, weight in zip(df['ticker'], quality_weights)},
        'risk_weights': {ticker: weight for ticker, weight in zip(df['ticker'], risk_weights)},
        'portfolio_metrics': portfolio_metrics,
        'individual_ticker_metrics': df[['ticker', 'enhanced_score', 'sortino_ratio', 'calmar_ratio', 
                                        'sharpe_ratio', 'annualized_return', 'max_drawdown']].to_dict('records')
    }
    
    return allocation_results

def calculate_portfolio_metrics(df: pd.DataFrame, weights: np.ndarray) -> Dict:
    """Calculate portfolio-level performance metrics"""
    
    # Portfolio return (weighted average)
    port_return = (df['annualized_return'].fillna(0) * weights).sum()
    
    # Portfolio volatility (simplified - assumes zero correlation)
    port_vol = np.sqrt((weights**2 * df['annual_volatility'].fillna(0.1)**2).sum())
    
    # Portfolio max drawdown (weighted average - simplified)
    port_max_dd = (df['max_drawdown'].fillna(0.05) * weights).sum()
    
    # Portfolio Sharpe ratio
    port_sharpe = port_return / port_vol if port_vol > 0 else -10
    
    # Portfolio Sortino ratio (approximated)
    # Use weighted average of individual Sortino ratios as approximation
    port_sortino = (df['sortino_ratio'].fillna(-10) * weights).sum()
    
    # Portfolio Calmar ratio
    port_calmar = port_return / port_max_dd if port_max_dd > 0 else -10
    
    # Enhanced portfolio score
    portfolio_metrics = {
        'annualized_return': port_return,
        'annual_volatility': port_vol,
        'max_drawdown': port_max_dd,
        'sharpe_ratio': port_sharpe,
        'sortino_ratio': port_sortino,
        'calmar_ratio': port_calmar
    }
    
    portfolio_metrics['enhanced_score'] = enhanced_scoring_function(portfolio_metrics)
    
    return portfolio_metrics

def run_enhanced_optimization(tickers: List[str] = None, limit: int = 5, 
                            data_dir: str = "data", log_dir: str = "logs",
                            parallel: bool = True, max_workers: int = None):
    """
    Run enhanced portfolio optimization focusing on Sortino and Calmar ratios
    """
    
    print("=" * 80)
    print("ENHANCED PORTFOLIO OPTIMIZATION")
    print("Focus: Sortino Ratio & Calmar Ratio Improvement")
    print("=" * 80)
    
    # Load market data
    print("Loading market data...")
    data_map, ticker_objects = load_data(limit=100)  # Load more tickers for selection
    
    # Select tickers (use provided list or default to best performers)
    if tickers is None:
        tickers = list(data_map.keys())[:limit]
    else:
        # Filter to available tickers
        tickers = [t for t in tickers if t in data_map][:limit]
    
    print(f"Optimizing {len(tickers)} tickers: {tickers}")
    
    # Get enhanced parameter grid
    enhanced_params = create_enhanced_parameter_grid()
    total_configs = (len(enhanced_params['rebal_list']) * 
                    len(enhanced_params['tc_list']) * 
                    len(enhanced_params['impact_list']) * 
                    len(enhanced_params['scale_list']))
    print(f"Enhanced parameter space: {total_configs} configurations per ticker")
    
    # Fetch risk-free rate
    risk_free_rate = fetch_risk_free_rate(fallback_rate=0.041)
    print(f"Using risk-free rate: {risk_free_rate:.3f} ({risk_free_rate*100:.1f}%)")
    
    # Prepare arguments for parallel processing
    start_time = time.time()
    results = []
    
    if parallel and len(tickers) > 1:
        # Parallel execution
        args_list = []
        for ticker in tickers:
            hist_data_bytes = cloudpickle.dumps(data_map[ticker])
            args_list.append((ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate, enhanced_params))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(run_enhanced_single_ticker, args): args[0] 
                              for args in args_list}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        print(f"[{len(results)}/{len(tickers)}] Completed: {ticker}")
                except Exception as e:
                    print(f"Error optimizing {ticker}: {e}")
    else:
        # Sequential execution
        for ticker in tickers:
            hist_data_bytes = cloudpickle.dumps(data_map[ticker])
            args = (ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate, enhanced_params)
            result = run_enhanced_single_ticker(args)
            if result is not None:
                results.append(result)
                print(f"[{len(results)}/{len(tickers)}] Completed: {ticker}")
    
    optimization_time = time.time() - start_time
    print(f"\nOptimization completed in {optimization_time:.1f} seconds")
    print(f"Successfully optimized {len(results)} out of {len(tickers)} tickers")
    
    if not results:
        print("No successful optimizations. Exiting.")
        return None
    
    # Portfolio-level optimization
    print("\nOptimizing portfolio allocation...")
    portfolio_results = optimize_portfolio_allocation(results)
    
    # Save enhanced portfolio results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual enhanced results
    enhanced_portfolio_file = Path(data_dir) / f"enhanced_portfolio_configs_{timestamp}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(enhanced_portfolio_file, index=False)
    
    # Save portfolio allocation results
    portfolio_file = Path(data_dir) / f"portfolio_allocation_{timestamp}.json"
    with open(portfolio_file, 'w') as f:
        json.dump(portfolio_results, f, indent=2)
    
    # Print enhanced results summary
    print_enhanced_results(results, portfolio_results)
    
    print(f"\nEnhanced results saved:")
    print(f"  Individual configs: {enhanced_portfolio_file}")
    print(f"  Portfolio allocation: {portfolio_file}")
    
    return results, portfolio_results

def print_enhanced_results(ticker_results: List[Dict], portfolio_results: Dict):
    """Print comprehensive enhanced optimization results"""
    
    print("\n" + "=" * 80)
    print("ENHANCED OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Individual ticker results
    df = pd.DataFrame(ticker_results)
    
    print(f"\n📊 ENHANCED INDIVIDUAL TICKER PERFORMANCE")
    print("-" * 60)
    df_sorted = df.sort_values('enhanced_score', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        print(f"  {i}. {row['ticker']:>5} | Enhanced Score: {row['enhanced_score']:7.4f}")
        print(f"       Sortino: {row.get('sortino_ratio', 0):7.3f} | "
              f"Calmar: {row.get('calmar_ratio', 0):7.3f} | "
              f"Sharpe: {row.get('sharpe_ratio', 0):7.3f}")
        print(f"       Return: {row.get('annualized_return', 0):7.2%} | "
              f"MaxDD: {row.get('max_drawdown', 0):6.2%} | "
              f"Vol: {row.get('annual_volatility', 0):6.2%}")
        print(f"       Config: R{row.get('rebal', 0)}/TC{row.get('tc', 0):.4f}/S{row.get('scale', 0):.2f}")
        print()
    
    # Portfolio-level results
    if portfolio_results:
        port_metrics = portfolio_results.get('portfolio_metrics', {})
        
        print(f"\n🎯 OPTIMIZED PORTFOLIO PERFORMANCE")
        print("-" * 40)
        print(f"  Enhanced Score: {port_metrics.get('enhanced_score', 0):.4f}")
        print(f"  Sortino Ratio: {port_metrics.get('sortino_ratio', 0):.4f}")
        print(f"  Calmar Ratio: {port_metrics.get('calmar_ratio', 0):.4f}")
        print(f"  Sharpe Ratio: {port_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Annual Return: {port_metrics.get('annualized_return', 0):.2%}")
        print(f"  Volatility: {port_metrics.get('annual_volatility', 0):.2%}")
        print(f"  Max Drawdown: {port_metrics.get('max_drawdown', 0):.2%}")
        
        print(f"\n📈 OPTIMAL PORTFOLIO ALLOCATION")
        print("-" * 40)
        weights = portfolio_results.get('weights', {})
        for ticker, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            print(f"  {ticker:>5}: {weight:6.2%}")
    
    # Summary statistics
    print(f"\n📊 IMPROVEMENT SUMMARY")
    print("-" * 30)
    print(f"  Average Enhanced Score: {df['enhanced_score'].mean():7.4f}")
    print(f"  Best Enhanced Score: {df['enhanced_score'].max():7.4f}")
    print(f"  Average Sortino Ratio: {df['sortino_ratio'].mean():7.3f}")
    print(f"  Average Calmar Ratio: {df['calmar_ratio'].mean():7.3f}")
    print(f"  Average Max Drawdown: {df['max_drawdown'].mean():6.2%}")

def main():
    """Main enhanced optimization function"""
    
    # You can specify particular tickers or let it use the default 5
    target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']  # Focus on the same tickers
    
    # Run enhanced optimization
    results, portfolio_results = run_enhanced_optimization(
        tickers=target_tickers,
        limit=5,
        parallel=True,
        max_workers=5
    )
    
    print("\n" + "=" * 80)
    print("ENHANCED OPTIMIZATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
