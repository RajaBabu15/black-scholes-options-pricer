#!/usr/bin/env python3
"""
Portfolio Analysis Script
Plots portfolio performance and calculates comprehensive financial ratios
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_portfolio_data(data_dir: str = "data"):
    """Load portfolio results and individual ticker data"""
    data_path = Path(data_dir)
    
    # Find all best config JSON files to get tickers
    config_files = list(data_path.glob("*_best_config.json"))
    if not config_files:
        raise FileNotFoundError("No best config files found")
    
    # Extract ticker names from config files
    tickers = [f.stem.replace('_best_config', '') for f in config_files]
    print(f"Found tickers: {tickers}")
    
    # Load individual ticker results
    ticker_data = {}
    ticker_configs = {}
    portfolio_data = []
    
    for ticker in tickers:
        # Load full results CSV
        results_file = data_path / f"opt_results_{ticker}.csv"
        if results_file.exists():
            ticker_data[ticker] = pd.read_csv(results_file)
        
        # Load best config JSON
        config_file = data_path / f"{ticker}_best_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                ticker_configs[ticker] = config_data
                # Add ticker name to the config data
                config_data['ticker'] = ticker
                portfolio_data.append(config_data)
    
    # Create portfolio DataFrame from best configs
    portfolio_df = pd.DataFrame(portfolio_data)
    
    return portfolio_df, ticker_data, ticker_configs

def calculate_portfolio_metrics(portfolio_df):
    """Calculate comprehensive portfolio-level metrics"""
    
    metrics = {}
    
    # Basic statistics
    metrics['num_tickers'] = len(portfolio_df)
    metrics['avg_score'] = portfolio_df['score'].mean()
    metrics['score_std'] = portfolio_df['score'].std()
    metrics['min_score'] = portfolio_df['score'].min()
    metrics['max_score'] = portfolio_df['score'].max()
    
    # Performance ratios (portfolio averages)
    ratio_cols = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
    for col in ratio_cols:
        if col in portfolio_df.columns:
            metrics[f'avg_{col}'] = portfolio_df[col].mean()
            metrics[f'std_{col}'] = portfolio_df[col].std()
    
    # Return metrics
    return_cols = ['annualized_return', 'volatility', 'max_drawdown']
    for col in return_cols:
        if col in portfolio_df.columns:
            metrics[f'avg_{col}'] = portfolio_df[col].mean()
            metrics[f'std_{col}'] = portfolio_df[col].std()
    
    # Configuration analysis
    config_cols = ['rebal', 'tc', 'impact', 'scale']
    for col in config_cols:
        if col in portfolio_df.columns:
            metrics[f'avg_{col}'] = portfolio_df[col].mean()
            metrics[f'mode_{col}'] = portfolio_df[col].mode().iloc[0] if len(portfolio_df[col].mode()) > 0 else None
    
    # Trading metrics
    trade_cols = ['num_trades_mean', 'avg_spread_cost_mean', 'avg_impact_cost_mean']
    for col in trade_cols:
        if col in portfolio_df.columns:
            metrics[f'avg_{col}'] = portfolio_df[col].mean()
    
    return metrics

def print_portfolio_summary(portfolio_df, metrics):
    """Print comprehensive portfolio summary"""
    print("=" * 80)
    print("PORTFOLIO HEDGE OPTIMIZATION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 PORTFOLIO OVERVIEW")
    print(f"   Number of Tickers: {metrics['num_tickers']}")
    print(f"   Average Score: {metrics['avg_score']:.4f} ± {metrics['score_std']:.4f}")
    print(f"   Score Range: [{metrics['min_score']:.4f}, {metrics['max_score']:.4f}]")
    
    print(f"\n📈 PERFORMANCE RATIOS (Portfolio Average)")
    if 'avg_sharpe_ratio' in metrics:
        print(f"   Sharpe Ratio: {metrics['avg_sharpe_ratio']:.4f} ± {metrics['std_sharpe_ratio']:.4f}")
    if 'avg_sortino_ratio' in metrics:
        print(f"   Sortino Ratio: {metrics['avg_sortino_ratio']:.4f} ± {metrics['std_sortino_ratio']:.4f}")
    if 'avg_calmar_ratio' in metrics:
        print(f"   Calmar Ratio: {metrics['avg_calmar_ratio']:.4f} ± {metrics['std_calmar_ratio']:.4f}")
    
    print(f"\n💰 RETURN CHARACTERISTICS")
    if 'avg_annualized_return' in metrics:
        print(f"   Annualized Return: {metrics['avg_annualized_return']:.2%} ± {metrics['std_annualized_return']:.2%}")
    if 'avg_volatility' in metrics:
        print(f"   Volatility: {metrics['avg_volatility']:.2%} ± {metrics['std_volatility']:.2%}")
    if 'avg_max_drawdown' in metrics:
        print(f"   Max Drawdown: {metrics['avg_max_drawdown']:.2%} ± {metrics['std_max_drawdown']:.2%}")
    
    print(f"\n⚙️  OPTIMAL CONFIGURATION SUMMARY")
    if 'mode_rebal' in metrics:
        print(f"   Most Common Rebalancing: {metrics['mode_rebal']} days (avg: {metrics['avg_rebal']:.1f})")
    if 'mode_tc' in metrics:
        print(f"   Most Common Transaction Cost: {metrics['mode_tc']:.4f} (avg: {metrics['avg_tc']:.4f})")
    if 'mode_scale' in metrics:
        print(f"   Most Common Exposure Scale: {metrics['mode_scale']:.2f} (avg: {metrics['avg_scale']:.2f})")
    
    print(f"\n🔄 TRADING ACTIVITY")
    if 'avg_num_trades_mean' in metrics:
        print(f"   Average Trades per Path: {metrics['avg_num_trades_mean']:.1f}")
    if 'avg_avg_spread_cost_mean' in metrics:
        print(f"   Average Spread Cost: {metrics['avg_avg_spread_cost_mean']:.6f}")
    if 'avg_avg_impact_cost_mean' in metrics:
        print(f"   Average Impact Cost: {metrics['avg_avg_impact_cost_mean']:.8f}")
    
    print(f"\n📋 INDIVIDUAL TICKER PERFORMANCE")
    print("-" * 60)
    portfolio_sorted = portfolio_df.sort_values('score', ascending=False)
    for i, (_, row) in enumerate(portfolio_sorted.iterrows(), 1):
        print(f"   {i}. {row['ticker']:>5} | Score: {row['score']:7.4f} | "
              f"Sharpe: {row.get('sharpe_ratio', 0):6.3f} | "
              f"Return: {row.get('annualized_return', 0):6.2%} | "
              f"Config: R{row.get('rebal', 0)}/TC{row.get('tc', 0):.4f}/S{row.get('scale', 0):.2f}")

def create_portfolio_plots(portfolio_df, ticker_data, output_dir="plots"):
    """Create comprehensive portfolio visualization plots"""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # 1. Portfolio Performance Overview
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Portfolio Hedge Optimization Performance Overview', fontsize=16, fontweight='bold')
    
    # Score distribution
    axes[0, 0].hist(portfolio_df['score'], bins=10, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(portfolio_df['score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {portfolio_df["score"].mean():.4f}')
    axes[0, 0].set_xlabel('Optimization Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distribution Across Tickers')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sharpe vs Sortino
    if 'sharpe_ratio' in portfolio_df.columns and 'sortino_ratio' in portfolio_df.columns:
        scatter = axes[0, 1].scatter(portfolio_df['sharpe_ratio'], portfolio_df['sortino_ratio'], 
                                   c=portfolio_df['score'], cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_xlabel('Sharpe Ratio')
        axes[0, 1].set_ylabel('Sortino Ratio')
        axes[0, 1].set_title('Risk-Adjusted Performance Comparison')
        plt.colorbar(scatter, ax=axes[0, 1], label='Score')
        
        # Add ticker labels
        for i, (_, row) in enumerate(portfolio_df.iterrows()):
            axes[0, 1].annotate(row['ticker'], 
                              (row['sharpe_ratio'], row['sortino_ratio']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Return vs Risk
    if 'annualized_return' in portfolio_df.columns and 'volatility' in portfolio_df.columns:
        scatter = axes[1, 0].scatter(portfolio_df['volatility'], portfolio_df['annualized_return'], 
                                   c=portfolio_df['score'], cmap='viridis', s=100, alpha=0.7)
        axes[1, 0].set_xlabel('Volatility (Annual)')
        axes[1, 0].set_ylabel('Annualized Return')
        axes[1, 0].set_title('Return vs Risk Profile')
        plt.colorbar(scatter, ax=axes[1, 0], label='Score')
        
        # Add ticker labels
        for i, (_, row) in enumerate(portfolio_df.iterrows()):
            axes[1, 0].annotate(row['ticker'], 
                              (row['volatility'], row['annualized_return']),
                              xytext=(5, 5), textcoords='offset points', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Configuration heatmap
    if all(col in portfolio_df.columns for col in ['rebal', 'tc', 'scale', 'score']):
        pivot_data = portfolio_df.pivot_table(values='score', 
                                            index='rebal', 
                                            columns='tc', 
                                            aggfunc='mean')
        if not pivot_data.empty:
            im = axes[1, 1].imshow(pivot_data.values, cmap='viridis', aspect='auto')
            axes[1, 1].set_xticks(range(len(pivot_data.columns)))
            axes[1, 1].set_xticklabels([f'{x:.4f}' for x in pivot_data.columns])
            axes[1, 1].set_yticks(range(len(pivot_data.index)))
            axes[1, 1].set_yticklabels(pivot_data.index)
            axes[1, 1].set_xlabel('Transaction Cost')
            axes[1, 1].set_ylabel('Rebalancing Frequency (days)')
            axes[1, 1].set_title('Score Heatmap by Configuration')
            plt.colorbar(im, ax=axes[1, 1], label='Average Score')
            
            # Add values to heatmap
            for i in range(len(pivot_data.index)):
                for j in range(len(pivot_data.columns)):
                    if not np.isnan(pivot_data.iloc[i, j]):
                        axes[1, 1].text(j, i, f'{pivot_data.iloc[i, j]:.2f}', 
                                       ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/portfolio_overview.png', dpi=300, bbox_inches='tight')
    print(f"Saved portfolio overview plot to {output_dir}/portfolio_overview.png")
    
    # 2. Individual Ticker Performance Comparison
    if len(ticker_data) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Individual Ticker Analysis', fontsize=16, fontweight='bold')
        
        tickers = list(ticker_data.keys())
        
        # Performance metrics comparison
        metrics = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'annualized_return', 'volatility', 'max_drawdown']
        metric_names = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Annualized Return', 'Volatility', 'Max Drawdown']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            if i < 6:  # Only plot first 6 metrics
                row, col = i // 3, i % 3
                
                if metric in portfolio_df.columns:
                    values = portfolio_df.set_index('ticker')[metric]
                    bars = axes[row, col].bar(range(len(values)), values.values, 
                                            color=plt.cm.viridis(np.linspace(0, 1, len(values))))
                    axes[row, col].set_xticks(range(len(values)))
                    axes[row, col].set_xticklabels(values.index, rotation=45)
                    axes[row, col].set_ylabel(name)
                    axes[row, col].set_title(f'{name} by Ticker')
                    axes[row, col].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for j, (bar, val) in enumerate(zip(bars, values.values)):
                        height = bar.get_height()
                        if metric in ['annualized_return', 'volatility', 'max_drawdown']:
                            label = f'{val:.2%}'
                        else:
                            label = f'{val:.3f}'
                        axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                          label, ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ticker_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved ticker comparison plot to {output_dir}/ticker_comparison.png")
    
    # 3. Configuration Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Optimal Configuration Analysis', fontsize=16, fontweight='bold')
    
    # Rebalancing frequency distribution
    if 'rebal' in portfolio_df.columns:
        rebal_counts = portfolio_df['rebal'].value_counts()
        axes[0, 0].pie(rebal_counts.values, labels=[f'{int(x)} days' for x in rebal_counts.index], 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Optimal Rebalancing Frequency')
    
    # Transaction cost distribution
    if 'tc' in portfolio_df.columns:
        tc_counts = portfolio_df['tc'].value_counts()
        axes[0, 1].pie(tc_counts.values, labels=[f'{x:.4f}' for x in tc_counts.index], 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Optimal Transaction Cost')
    
    # Exposure scale distribution
    if 'scale' in portfolio_df.columns:
        axes[1, 0].hist(portfolio_df['scale'], bins=10, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(portfolio_df['scale'].mean(), color='red', linestyle='--',
                          label=f'Mean: {portfolio_df["scale"].mean():.3f}')
        axes[1, 0].set_xlabel('Exposure Scale')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Exposure Scale Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Score vs Scale relationship
    if 'scale' in portfolio_df.columns:
        scatter = axes[1, 1].scatter(portfolio_df['scale'], portfolio_df['score'], 
                                   c=portfolio_df['rebal'], cmap='viridis', s=100, alpha=0.7)
        axes[1, 1].set_xlabel('Exposure Scale')
        axes[1, 1].set_ylabel('Optimization Score')
        axes[1, 1].set_title('Score vs Exposure Scale')
        plt.colorbar(scatter, ax=axes[1, 1], label='Rebalancing Days')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/configuration_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Saved configuration analysis plot to {output_dir}/configuration_analysis.png")
    
    plt.show()

def main():
    """Main analysis function"""
    print("Loading portfolio data...")
    
    try:
        portfolio_df, ticker_data, ticker_configs = load_portfolio_data()
        
        print(f"Successfully loaded data for {len(portfolio_df)} tickers")
        
        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(portfolio_df)
        
        # Print comprehensive summary
        print_portfolio_summary(portfolio_df, metrics)
        
        # Create plots
        print("\nGenerating portfolio visualization plots...")
        create_portfolio_plots(portfolio_df, ticker_data)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("Check the 'plots' directory for visualization files.")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
