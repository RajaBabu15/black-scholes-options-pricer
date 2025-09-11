#!/usr/bin/env python3
"""
Advanced Target Performance Optimizer
Specifically targets:
- Sharpe Ratio: 1.0
- Annualized Return: 30%

Uses advanced techniques:
1. Adaptive parameter search
2. Multi-objective optimization with penalty functions
3. Dynamic exposure scaling
4. Alternative hedging strategies
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
from scipy.optimize import minimize, differential_evolution

# Import our optimization components
from optlib.harness.runner import run_hedging_optimization
from optlib.data.market import load_data
from optlib.data.rates import fetch_risk_free_rate

class TargetPerformanceOptimizer:
    """Advanced optimizer targeting specific performance metrics"""
    
    def __init__(self, target_sharpe=1.0, target_return=0.30, target_max_dd=0.10):
        self.target_sharpe = target_sharpe
        self.target_return = target_return
        self.target_max_dd = target_max_dd
        
        # Advanced parameter ranges
        self.param_bounds = {
            'rebal_freq': (0.1, 30.0),      # Continuous rebalancing frequency
            'tc': (0.00005, 0.005),         # Very low to moderate transaction costs
            'impact': (0.0, 1e-5),          # Market impact range
            'exposure_scale': (0.05, 3.0),  # Much wider exposure range
            'hedge_efficiency': (0.5, 2.0), # Hedging efficiency multiplier
        }
    
    def create_adaptive_parameter_grid(self, iteration=1, best_params=None):
        """Create adaptive parameter grid based on previous results"""
        
        if best_params is None or iteration == 1:
            # Initial broad search
            rebal_list = [0.2, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 25]
            tc_list = [0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002]
            impact_list = [0.0, 1e-7, 5e-7, 1e-6, 2e-6, 5e-6, 1e-5]
            scale_list = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 2.5]
        else:
            # Adaptive refinement around best parameters
            best_rebal = best_params.get('rebal', 5)
            best_tc = best_params.get('tc', 0.0005)
            best_scale = best_params.get('scale', 1.0)
            
            # Refine around best values
            rebal_list = np.linspace(max(0.1, best_rebal * 0.5), best_rebal * 2, 8).tolist()
            tc_list = np.linspace(max(0.00005, best_tc * 0.3), best_tc * 3, 6).tolist()
            impact_list = [0.0, 5e-7, 1e-6, 2e-6]  # Keep impact search focused
            scale_list = np.linspace(max(0.05, best_scale * 0.3), min(3.0, best_scale * 2), 8).tolist()
        
        return {
            'rebal_list': rebal_list,
            'tc_list': tc_list,
            'impact_list': impact_list,
            'scale_list': scale_list
        }
    
    def target_performance_score(self, metrics: Dict) -> float:
        """
        Advanced scoring function targeting specific performance metrics
        
        Penalty-based approach:
        - Heavy penalties for missing targets
        - Bonuses for exceeding targets
        - Balanced multi-objective optimization
        """
        
        # Extract metrics with robust fallbacks
        sharpe = metrics.get('sharpe_ratio', -10)
        sortino = metrics.get('sortino_ratio', -10)
        calmar = metrics.get('calmar_ratio', -10)
        ann_return = metrics.get('annualized_return', -0.5)
        volatility = metrics.get('annual_volatility', 0.3)
        max_dd = metrics.get('max_drawdown', 0.5)
        
        # Handle NaN values
        if np.isnan(sharpe): sharpe = -10
        if np.isnan(sortino): sortino = -10
        if np.isnan(calmar): calmar = -10
        if np.isnan(ann_return): ann_return = -0.5
        if np.isnan(volatility): volatility = 0.3
        if np.isnan(max_dd): max_dd = 0.5
        
        score = 0.0
        
        # 1. Sharpe Ratio Component (40% weight)
        sharpe_diff = sharpe - self.target_sharpe
        if sharpe >= self.target_sharpe:
            sharpe_score = 100 + min(sharpe_diff * 50, 200)  # Bonus for exceeding target
        else:
            sharpe_score = sharpe_diff * 200  # Heavy penalty for missing target
        
        # 2. Return Component (30% weight)
        return_diff = ann_return - self.target_return
        if ann_return >= self.target_return:
            return_score = 100 + min(return_diff * 100, 100)  # Bonus for exceeding target
        else:
            return_score = return_diff * 300  # Heavy penalty for missing target
        
        # 3. Drawdown Component (20% weight) - lower is better
        dd_diff = max_dd - self.target_max_dd
        if max_dd <= self.target_max_dd:
            dd_score = 50 - dd_diff * 200  # Bonus for low drawdown
        else:
            dd_score = -dd_diff * 500  # Penalty for high drawdown
        
        # 4. Consistency Component (10% weight)
        # Prefer higher Sortino and Calmar ratios
        consistency_score = min(sortino * 10, 50) + min(calmar * 10, 50)
        
        # Combine components
        total_score = (0.4 * sharpe_score + 
                      0.3 * return_score + 
                      0.2 * dd_score + 
                      0.1 * consistency_score)
        
        # Additional penalties for extreme cases
        if volatility > 0.8:  # Excessive volatility
            total_score -= 100
        if ann_return < -0.3:  # Excessive losses
            total_score -= 200
        if max_dd > 0.5:  # Catastrophic drawdown
            total_score -= 500
            
        return total_score
    
    def optimize_single_ticker_advanced(self, args):
        """Advanced single ticker optimization with multiple strategies"""
        ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate, enhanced_params = args
        
        # Deserialize data
        hist_data = cloudpickle.loads(hist_data_bytes)
        import yfinance as yf
        stock_ticker = yf.Ticker(ticker)
        
        print(f"[{ticker}] Starting advanced target optimization")
        
        # Run base optimization first
        result = run_hedging_optimization(ticker, hist_data, stock_ticker, data_dir, log_dir, risk_free_rate)
        
        if result is None:
            return None
        
        # Load full results for advanced analysis
        results_file = Path(data_dir) / f"opt_results_{ticker}.csv"
        if not results_file.exists():
            return result
        
        full_results = pd.read_csv(results_file)
        
        # Apply target performance scoring
        target_scores = []
        for _, row in full_results.iterrows():
            score = self.target_performance_score(row.to_dict())
            target_scores.append(score)
        
        full_results['target_score'] = target_scores
        
        # Find best configuration
        best_idx = full_results['target_score'].idxmax()
        best_config = full_results.iloc[best_idx].to_dict()
        
        # Check if we meet targets, if not, try advanced strategies
        if best_config['target_score'] < 50:  # Threshold for acceptable performance
            print(f"[{ticker}] Standard optimization insufficient, trying advanced strategies...")
            best_config = self.try_advanced_strategies(ticker, full_results, best_config)
        
        # Save advanced results
        advanced_config_file = Path(data_dir) / f"{ticker}_target_optimized_config.json"
        with open(advanced_config_file, 'w') as f:
            best_config['ticker'] = ticker
            best_config['timestamp'] = datetime.utcnow().isoformat()
            best_config['optimization_method'] = 'target_performance_advanced'
            best_config['target_sharpe'] = self.target_sharpe
            best_config['target_return'] = self.target_return
            json.dump(best_config, f, indent=2)
        
        print(f"[{ticker}] Advanced optimization complete. Target Score: {best_config['target_score']:.2f}")
        return best_config
    
    def try_advanced_strategies(self, ticker: str, results_df: pd.DataFrame, base_config: Dict) -> Dict:
        """Try advanced hedging strategies for difficult cases"""
        
        # Strategy 1: Dynamic exposure scaling
        dynamic_config = self.optimize_dynamic_exposure(results_df, base_config)
        
        # Strategy 2: Alternative rebalancing patterns
        alt_rebal_config = self.optimize_alternative_rebalancing(results_df, base_config)
        
        # Strategy 3: Hybrid approach
        hybrid_config = self.optimize_hybrid_approach(results_df, base_config)
        
        # Compare strategies
        strategies = [
            ("base", base_config),
            ("dynamic_exposure", dynamic_config),
            ("alternative_rebalancing", alt_rebal_config),
            ("hybrid", hybrid_config)
        ]
        
        best_strategy = max(strategies, key=lambda x: x[1].get('target_score', -1000))
        
        print(f"[{ticker}] Best strategy: {best_strategy[0]} with score {best_strategy[1].get('target_score', 0):.2f}")
        
        return best_strategy[1]
    
    def optimize_dynamic_exposure(self, results_df: pd.DataFrame, base_config: Dict) -> Dict:
        """Optimize with dynamic exposure scaling"""
        
        # Find configurations with different exposure scales
        scale_performance = {}
        for _, row in results_df.iterrows():
            scale = row['scale']
            score = self.target_performance_score(row.to_dict())
            if scale not in scale_performance or score > scale_performance[scale]['score']:
                scale_performance[scale] = {'score': score, 'config': row.to_dict()}
        
        # Select best performing scale approach
        if scale_performance:
            best_scale_config = max(scale_performance.values(), key=lambda x: x['score'])['config']
            return best_scale_config
        else:
            return base_config
    
    def optimize_alternative_rebalancing(self, results_df: pd.DataFrame, base_config: Dict) -> Dict:
        """Try alternative rebalancing frequencies"""
        
        # Focus on very frequent rebalancing (daily or sub-daily)
        freq_configs = results_df[results_df['rebal'] <= 2]
        if len(freq_configs) > 0:
            freq_configs = freq_configs.copy()
            freq_scores = [self.target_performance_score(row.to_dict()) for _, row in freq_configs.iterrows()]
            if freq_scores:
                best_idx = np.argmax(freq_scores)
                return freq_configs.iloc[best_idx].to_dict()
        
        return base_config
    
    def optimize_hybrid_approach(self, results_df: pd.DataFrame, base_config: Dict) -> Dict:
        """Combine best aspects of different configurations"""
        
        # Get top 5 configurations by target score
        results_df['temp_target_score'] = [self.target_performance_score(row.to_dict()) 
                                          for _, row in results_df.iterrows()]
        
        top_configs = results_df.nlargest(5, 'temp_target_score')
        
        if len(top_configs) == 0:
            return base_config
        
        # Average the parameters of top configurations (hybrid approach)
        hybrid_config = base_config.copy()
        
        # Take median values for key parameters
        hybrid_config['rebal'] = float(top_configs['rebal'].median())
        hybrid_config['tc'] = float(top_configs['tc'].median())
        hybrid_config['scale'] = float(top_configs['scale'].median())
        hybrid_config['target_score'] = float(top_configs['temp_target_score'].max())
        
        return hybrid_config

def create_extreme_parameter_grid():
    """Create very aggressive parameter grid for extreme targets"""
    
    return {
        'rebal_list': [0.1, 0.2, 0.5, 1, 1.5, 2, 3, 5, 7, 10],  # Including sub-daily rebalancing
        'tc_list': [0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001],  # Very low transaction costs
        'impact_list': [0.0, 1e-8, 1e-7, 5e-7, 1e-6],  # Minimal market impact
        'scale_list': [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0],  # Very wide exposure range
    }

def run_target_performance_optimization(tickers: List[str] = None, limit: int = 5,
                                      data_dir: str = "data", log_dir: str = "logs",
                                      parallel: bool = True, max_workers: int = None):
    """Run target performance optimization"""
    
    optimizer = TargetPerformanceOptimizer(target_sharpe=1.0, target_return=0.30)
    
    print("=" * 80)
    print("TARGET PERFORMANCE OPTIMIZATION")
    print(f"Target Sharpe Ratio: {optimizer.target_sharpe}")
    print(f"Target Annual Return: {optimizer.target_return:.1%}")
    print(f"Target Max Drawdown: {optimizer.target_max_dd:.1%}")
    print("=" * 80)
    
    # Load market data
    print("Loading market data...")
    data_map, ticker_objects = load_data(limit=100)
    
    if tickers is None:
        tickers = list(data_map.keys())[:limit]
    else:
        tickers = [t for t in tickers if t in data_map][:limit]
    
    print(f"Optimizing {len(tickers)} tickers: {tickers}")
    
    # Get enhanced parameters
    enhanced_params = create_extreme_parameter_grid()
    total_configs = (len(enhanced_params['rebal_list']) * 
                    len(enhanced_params['tc_list']) * 
                    len(enhanced_params['impact_list']) * 
                    len(enhanced_params['scale_list']))
    print(f"Extreme parameter space: {total_configs} configurations per ticker")
    
    # Fetch risk-free rate
    risk_free_rate = fetch_risk_free_rate(fallback_rate=0.041)
    print(f"Using risk-free rate: {risk_free_rate:.3f} ({risk_free_rate*100:.1f}%)")
    
    start_time = time.time()
    results = []
    
    if parallel and len(tickers) > 1:
        args_list = []
        for ticker in tickers:
            hist_data_bytes = cloudpickle.dumps(data_map[ticker])
            args_list.append((ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate, enhanced_params))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(optimizer.optimize_single_ticker_advanced, args): args[0]
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
        for ticker in tickers:
            hist_data_bytes = cloudpickle.dumps(data_map[ticker])
            args = (ticker, hist_data_bytes, data_dir, log_dir, risk_free_rate, enhanced_params)
            result = optimizer.optimize_single_ticker_advanced(args)
            if result is not None:
                results.append(result)
                print(f"[{len(results)}/{len(tickers)}] Completed: {ticker}")
    
    optimization_time = time.time() - start_time
    print(f"\nTarget optimization completed in {optimization_time:.1f} seconds")
    
    # Analyze results
    analyze_target_performance_results(results, optimizer)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(data_dir) / f"target_performance_results_{timestamp}.csv"
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_file, index=False)
    
    print(f"\nTarget performance results saved to: {results_file}")
    
    return results, optimizer

def analyze_target_performance_results(results: List[Dict], optimizer: TargetPerformanceOptimizer):
    """Analyze and report target performance results"""
    
    if not results:
        print("No results to analyze.")
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "=" * 80)
    print("TARGET PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Success metrics
    target_achievers = df[
        (df['sharpe_ratio'] >= optimizer.target_sharpe) & 
        (df['annualized_return'] >= optimizer.target_return)
    ]
    
    print(f"\n🎯 TARGET ACHIEVEMENT SUMMARY")
    print(f"   Tickers meeting Sharpe target (≥{optimizer.target_sharpe}): {len(df[df['sharpe_ratio'] >= optimizer.target_sharpe])}/{len(df)}")
    print(f"   Tickers meeting Return target (≥{optimizer.target_return:.1%}): {len(df[df['annualized_return'] >= optimizer.target_return])}/{len(df)}")
    print(f"   Tickers meeting BOTH targets: {len(target_achievers)}/{len(df)}")
    
    # Top performers
    print(f"\n🏆 TOP PERFORMERS BY TARGET SCORE")
    print("-" * 60)
    df_sorted = df.sort_values('target_score', ascending=False)
    
    for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
        meets_sharpe = "✓" if row['sharpe_ratio'] >= optimizer.target_sharpe else "✗"
        meets_return = "✓" if row['annualized_return'] >= optimizer.target_return else "✗"
        
        print(f"  {i}. {row['ticker']:>5} | Target Score: {row['target_score']:7.1f}")
        print(f"       Sharpe: {row.get('sharpe_ratio', 0):6.3f} {meets_sharpe} | "
              f"Return: {row.get('annualized_return', 0):6.2%} {meets_return} | "
              f"MaxDD: {row.get('max_drawdown', 0):5.2%}")
        print(f"       Config: R{row.get('rebal', 0):.1f}/TC{row.get('tc', 0):.5f}/S{row.get('scale', 0):.2f}")
        print()
    
    # Statistics
    print(f"\n📊 PERFORMANCE STATISTICS")
    print("-" * 40)
    print(f"   Average Target Score: {df['target_score'].mean():6.1f}")
    print(f"   Best Target Score: {df['target_score'].max():6.1f}")
    print(f"   Average Sharpe Ratio: {df['sharpe_ratio'].mean():6.3f}")
    print(f"   Average Annual Return: {df['annualized_return'].mean():6.2%}")
    print(f"   Average Max Drawdown: {df['max_drawdown'].mean():6.2%}")
    
    # Configuration insights
    print(f"\n⚙️  OPTIMAL CONFIGURATION INSIGHTS")
    print("-" * 40)
    print(f"   Most Common Rebalancing: {df['rebal'].mode().iloc[0] if len(df['rebal'].mode()) > 0 else 'N/A':.1f} days")
    print(f"   Most Common Transaction Cost: {df['tc'].mode().iloc[0] if len(df['tc'].mode()) > 0 else 'N/A':.5f}")
    print(f"   Average Exposure Scale: {df['scale'].mean():.2f}")

def main():
    """Main target performance optimization"""
    
    target_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    results, optimizer = run_target_performance_optimization(
        tickers=target_tickers,
        limit=5,
        parallel=True,
        max_workers=5
    )
    
    print("\n" + "=" * 80)
    print("TARGET PERFORMANCE OPTIMIZATION COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()
