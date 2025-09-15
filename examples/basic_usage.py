#!/usr/bin/env python3
"""
Basic Usage Examples for Black-Scholes Options Pricer
=====================================================

This script demonstrates the basic functionality of the options pricer:
1. Black-Scholes pricing
2. Greeks computation  
3. Monte Carlo simulation
4. Dynamic vs static hedging comparison
5. Risk management
"""

import torch
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api import (
    bs_price, get_all_greeks, generate_heston_paths,
    compare_hedging_strategies, generate_comparison_report,
    HFTRiskManager, RiskLimits, PositionData
)

def example_1_basic_pricing():
    """Example 1: Basic Black-Scholes pricing and Greeks"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Black-Scholes Pricing and Greeks")
    print("="*60)
    
    # Option parameters
    S = 100.0      # Current stock price
    K = 105.0      # Strike price
    r = 0.05       # Risk-free rate
    q = 0.02       # Dividend yield
    sigma = 0.20   # Volatility
    T = 0.25       # Time to expiry (3 months)
    
    print(f"Option Parameters:")
    print(f"  Stock Price (S): ${S}")
    print(f"  Strike Price (K): ${K}")
    print(f"  Risk-free Rate (r): {r:.1%}")
    print(f"  Dividend Yield (q): {q:.1%}")
    print(f"  Volatility (σ): {sigma:.1%}")
    print(f"  Time to Expiry (T): {T:.2f} years")
    
    # Calculate option prices
    call_price = bs_price(S, K, r, q, sigma, T, 'call')
    put_price = bs_price(S, K, r, q, sigma, T, 'put')
    
    print(f"\nOption Prices:")
    print(f"  Call Price: ${float(call_price):.4f}")
    print(f"  Put Price: ${float(put_price):.4f}")
    
    # Calculate Greeks
    call_greeks = get_all_greeks(S, K, r, q, sigma, T, 'call', method='analytical')
    put_greeks = get_all_greeks(S, K, r, q, sigma, T, 'put', method='analytical')
    
    print(f"\nCall Option Greeks:")
    for greek, value in call_greeks.items():
        print(f"  {greek.capitalize()}: {float(value):.6f}")
    
    print(f"\nPut Option Greeks:")
    for greek, value in put_greeks.items():
        print(f"  {greek.capitalize()}: {float(value):.6f}")
    
    # Verify put-call parity
    synthetic_call = float(put_price) + S - K * np.exp(-r * T)
    parity_error = abs(float(call_price) - synthetic_call)
    print(f"\nPut-Call Parity Check:")
    print(f"  Call Price: ${float(call_price):.6f}")
    print(f"  Synthetic Call: ${synthetic_call:.6f}")
    print(f"  Parity Error: ${parity_error:.8f}")


def example_2_monte_carlo_simulation():
    """Example 2: Monte Carlo simulation with Heston model"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Monte Carlo Simulation with Heston Model")
    print("="*60)
    
    # Market parameters
    S0 = 100.0
    r = 0.05
    q = 0.02
    T = 0.25
    
    # Heston model parameters
    kappa = 2.0      # Mean reversion speed
    theta = 0.04     # Long-term variance
    sigma_v = 0.3    # Volatility of volatility
    rho = -0.7       # Correlation
    v0 = 0.04        # Initial variance
    
    # Simulation parameters
    n_paths = 10000
    n_steps = 63  # Approximately daily steps for 3 months
    
    print(f"Heston Model Parameters:")
    print(f"  κ (mean reversion): {kappa}")
    print(f"  θ (long-term variance): {theta}")
    print(f"  σ_v (vol of vol): {sigma_v}")
    print(f"  ρ (correlation): {rho}")
    print(f"  v₀ (initial variance): {v0}")
    print(f"\nSimulation: {n_paths} paths, {n_steps} steps")
    
    # Generate paths
    print("Generating Monte Carlo paths...")
    S_paths, v_paths = generate_heston_paths(
        S0, r, q, T, kappa, theta, sigma_v, rho, v0, n_paths, n_steps
    )
    
    times = torch.linspace(0, T, n_steps + 1)
    
    # Analyze paths
    final_prices = S_paths[:, -1]
    print(f"\nPath Analysis:")
    print(f"  Final Stock Price - Mean: ${float(torch.mean(final_prices)):.2f}")
    print(f"  Final Stock Price - Std: ${float(torch.std(final_prices)):.2f}")
    print(f"  Final Stock Price - Min: ${float(torch.min(final_prices)):.2f}")
    print(f"  Final Stock Price - Max: ${float(torch.max(final_prices)):.2f}")
    
    # Calculate option payoffs
    K = 105.0
    call_payoffs = torch.clamp(final_prices - K, min=0.0)
    put_payoffs = torch.clamp(K - final_prices, min=0.0)
    
    # Monte Carlo prices
    discount_factor = np.exp(-r * T)
    mc_call_price = float(torch.mean(call_payoffs)) * discount_factor
    mc_put_price = float(torch.mean(put_payoffs)) * discount_factor
    
    print(f"\nMonte Carlo Option Prices (K=${K}):")
    print(f"  Call Price: ${mc_call_price:.4f}")
    print(f"  Put Price: ${mc_put_price:.4f}")
    
    return S_paths, v_paths, times


def example_3_hedging_comparison(S_paths, v_paths, times):
    """Example 3: Dynamic vs Static Hedging Comparison"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Dynamic vs Static Hedging Comparison")
    print("="*60)
    
    # Option parameters
    K = 105.0
    r = 0.05
    q = 0.02
    
    # Hedging parameters
    transaction_cost = 0.001  # 0.1% per trade
    impact_lambda = 1e-6     # Market impact
    
    print(f"Comparing hedging strategies for ATM call option:")
    print(f"  Strike: ${K}")
    print(f"  Transaction Cost: {transaction_cost:.3%}")
    print(f"  Market Impact: {impact_lambda:.2e}")
    
    # Run comparison
    print("\nRunning hedging comparison...")
    results = compare_hedging_strategies(
        S_paths, v_paths, times, K, r, q, 'call',
        dynamic_hedge_params={'rebal_freq': 1.0, 'exposure_scale': 1.0},
        static_strategies=['static_delta', 'buy_and_hold', 'no_hedge'],
        transaction_cost=transaction_cost,
        impact_lambda=impact_lambda
    )
    
    # Analyze results
    print(f"\nHedging Strategy Results:")
    print(f"{'Strategy':<15} {'Mean P&L':<12} {'Std P&L':<12} {'Sharpe':<10} {'Trades':<8}")
    print("-" * 65)
    
    for strategy_name, strategy_result in results.items():
        if 'error' in strategy_result:
            print(f"{strategy_name:<15} ERROR: {strategy_result['error']}")
            continue
            
        pnl = strategy_result['pnl']
        if isinstance(pnl, torch.Tensor):
            mean_pnl = float(torch.mean(pnl))
            std_pnl = float(torch.std(pnl))
        else:
            mean_pnl = np.mean(pnl)
            std_pnl = np.std(pnl)
        
        sharpe = mean_pnl / max(std_pnl, 1e-8)
        
        trades = strategy_result.get('trades_count', torch.zeros(1))
        if isinstance(trades, torch.Tensor):
            avg_trades = float(torch.mean(trades))
        else:
            avg_trades = np.mean(trades)
        
        print(f"{strategy_name:<15} ${mean_pnl:<11.2f} ${std_pnl:<11.2f} {sharpe:<9.3f} {avg_trades:<7.1f}")
    
    return results


def example_4_risk_management():
    """Example 4: HFT Risk Management"""
    print("\n" + "="*60)
    print("EXAMPLE 4: HFT Risk Management")
    print("="*60)
    
    # Create risk manager with custom limits
    risk_limits = RiskLimits(
        max_delta_exposure=500.0,
        max_gamma_exposure=50.0,
        max_vega_exposure=200.0,
        max_position_size=100
    )
    
    risk_manager = HFTRiskManager(risk_limits=risk_limits)
    
    # Add some positions
    positions = [
        PositionData(
            symbol="AAPL", option_type="call", strike=150.0, expiry=0.25,
            quantity=50, mark_price=5.50, bid=5.45, ask=5.55,
            underlying_price=148.0, implied_vol=0.25,
            greeks={'delta': 0.55, 'gamma': 0.025, 'theta': -0.08, 'vega': 0.12, 'rho': 0.06}
        ),
        PositionData(
            symbol="AAPL", option_type="put", strike=145.0, expiry=0.25,
            quantity=-30, mark_price=3.20, bid=3.15, ask=3.25,
            underlying_price=148.0, implied_vol=0.22,
            greeks={'delta': -0.35, 'gamma': 0.028, 'theta': -0.06, 'vega': 0.10, 'rho': -0.04}
        )
    ]
    
    print("Adding positions to risk manager...")
    for i, position in enumerate(positions):
        success = risk_manager.add_position(position)
        print(f"  Position {i+1}: {'✓' if success else '✗'} {position.symbol} {position.option_type} {position.strike} ({position.quantity} contracts)")
    
    # Generate risk report
    risk_report = risk_manager.get_risk_report()
    
    print(f"\nRisk Report:")
    print(f"  Portfolio Greeks:")
    for greek, value in risk_report['portfolio_greeks'].items():
        print(f"    {greek.capitalize()}: {value:.4f}")
    
    print(f"\n  Risk Limit Checks:")
    for limit, status in risk_report['risk_limit_checks'].items():
        print(f"    {limit}: {'✓' if status else '✗'}")
    
    print(f"\n  Portfolio Summary:")
    print(f"    Positions: {risk_report['positions_count']}")
    print(f"    Total Notional: ${risk_report['total_notional']:,.2f}")
    print(f"    Concentration: {risk_report['concentration']:.1%}")
    print(f"    All Limits OK: {'✓' if risk_report['all_limits_ok'] else '✗'}")
    
    # Stress testing
    print(f"\nPerforming stress tests...")
    stress_scenarios = [
        {'underlying_move': 0.05, 'vol_move': 0.2},    # 5% up, 20% vol spike
        {'underlying_move': -0.05, 'vol_move': 0.2},   # 5% down, 20% vol spike
        {'underlying_move': 0.02, 'vol_move': -0.3}    # 2% up, 30% vol crush
    ]
    
    stress_results = risk_manager.stress_test(stress_scenarios)
    
    print(f"  Stress Test Results:")
    for scenario_name, result in stress_results.items():
        scenario = result['scenario']
        print(f"    {scenario_name}: {scenario['underlying_move']:+.1%} underlying, {scenario['vol_move']:+.1%} vol")
        print(f"      Estimated P&L: ${result['estimated_pnl']:,.2f}")
        if result['risk_limit_violations']:
            print(f"      Violations: {', '.join(result['risk_limit_violations'])}")
        else:
            print(f"      No violations")
    
    # Risk reduction suggestions
    suggestions = risk_manager.suggest_risk_reduction()
    if suggestions:
        print(f"\n  Risk Reduction Suggestions:")
        for suggestion in suggestions:
            print(f"    {suggestion['type']}: {suggestion.get('reason', 'No reason provided')}")
    else:
        print(f"\n  No risk reduction needed")


def main():
    """Run all examples"""
    print("Black-Scholes Options Pricer - Comprehensive Examples")
    print("=" * 60)
    
    # Example 1: Basic pricing and Greeks
    example_1_basic_pricing()
    
    # Example 2: Monte Carlo simulation  
    S_paths, v_paths, times = example_2_monte_carlo_simulation()
    
    # Example 3: Hedging comparison
    hedging_results = example_3_hedging_comparison(S_paths, v_paths, times)
    
    # Example 4: Risk management
    example_4_risk_management()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()