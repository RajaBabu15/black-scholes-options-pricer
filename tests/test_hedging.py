#!/usr/bin/env python3
"""
Unit Tests for Hedging Modules
==============================

Test static hedging, dynamic hedging comparison, and related functionality.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optlib.hedge.static import (
    static_delta_hedge, buy_and_hold_hedge, static_gamma_hedge,
    protective_put_strategy, covered_call_strategy, no_hedge_strategy
)
from optlib.hedge.comparison import (
    compare_hedging_strategies, analyze_strategy_performance,
    hedging_efficiency_analysis, cost_benefit_analysis
)
from optlib.sim.paths import generate_heston_paths


class TestStaticHedging(unittest.TestCase):
    """Test cases for static hedging strategies"""
    
    def setUp(self):
        """Set up test parameters and paths"""
        self.S0 = 100.0
        self.K = 105.0
        self.r = 0.05
        self.q = 0.02
        self.T = 0.25
        
        # Generate test paths
        n_paths = 1000
        n_steps = 50
        
        self.S_paths, self.v_paths = generate_heston_paths(
            self.S0, self.r, self.q, self.T,
            kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04,
            n_paths=n_paths, n_steps=n_steps
        )
        
        self.times = torch.linspace(0, self.T, n_steps + 1)
    
    def test_static_delta_hedge(self):
        """Test static delta hedge strategy"""
        result = static_delta_hedge(
            self.S_paths, self.v_paths, self.times, 
            self.K, self.r, self.q, 'call'
        )
        
        # Check that all required fields are present
        required_fields = ['pnl', 'initial_premium', 'hedge_ratio', 'final_cash', 
                          'final_stock_value', 'final_payoff', 'trades_count', 
                          'total_transaction_costs']
        
        for field in required_fields:
            self.assertIn(field, result)
        
        # Check shapes
        n_paths = self.S_paths.shape[0]
        self.assertEqual(result['pnl'].shape, (n_paths,))
        self.assertEqual(result['trades_count'].shape, (n_paths,))
        
        # Check that hedge ratio is reasonable (between 0 and 1 for call)
        hedge_ratios = result['hedge_ratio']
        self.assertTrue(torch.all(hedge_ratios >= 0))
        self.assertTrue(torch.all(hedge_ratios <= 1))
        
        # Static strategy should have exactly 1 trade per path
        self.assertTrue(torch.all(result['trades_count'] == 1))
    
    def test_buy_and_hold_hedge(self):
        """Test buy and hold hedge strategy"""
        result = buy_and_hold_hedge(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q, 'call'
        )
        
        # Should have hedge ratio of exactly 1.0
        self.assertTrue(torch.all(result['hedge_ratio'] == 1.0))
        
        # Check P&L is reasonable
        pnl = result['pnl']
        self.assertFalse(torch.isnan(pnl).any())
        self.assertFalse(torch.isinf(pnl).any())
    
    def test_no_hedge_strategy(self):
        """Test no hedge (naked short) strategy"""
        result = no_hedge_strategy(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q, 'call'
        )
        
        # No hedge should have zero trades
        self.assertTrue(torch.all(result['trades_count'] == 0))
        
        # No transaction costs
        self.assertTrue(torch.all(result['total_transaction_costs'] == 0))
        
        # P&L should be premium + interest - payoff
        expected_pnl = result['final_cash'] - result['final_payoff']
        torch.testing.assert_close(result['pnl'], expected_pnl, rtol=1e-6, atol=1e-6)
    
    def test_protective_put_strategy(self):
        """Test protective put strategy"""
        result = protective_put_strategy(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q
        )
        
        # Should have 2 trades (stock + put)
        self.assertTrue(torch.all(result['trades_count'] == 2))
        
        # Final value should never be less than put strike
        # (This is the insurance property of protective puts)
        final_values = result['final_value']
        # Allow small numerical errors
        self.assertTrue(torch.all(final_values >= self.K - 1e-6))
    
    def test_covered_call_strategy(self):
        """Test covered call strategy"""
        result = covered_call_strategy(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q
        )
        
        # Should have 2 trades (stock + call)
        self.assertTrue(torch.all(result['trades_count'] == 2))
        
        # Final value should be capped at call strike (when stock > strike)
        final_stock_values = result['final_stock_value']
        final_values = result['final_value']
        
        # Where stock is above strike, final value should be approximately the strike
        above_strike = final_stock_values > self.K
        if torch.any(above_strike):
            capped_values = final_values[above_strike]
            # Allow for small numerical errors
            self.assertTrue(torch.all(capped_values <= self.K + 1e-3))
    
    def test_static_gamma_hedge(self):
        """Test static gamma hedge strategy"""
        K_hedge = 110.0  # OTM call for gamma hedge
        
        result = static_gamma_hedge(
            self.S_paths, self.v_paths, self.times,
            self.K, K_hedge, self.r, self.q, 'call', 'call'
        )
        
        # Should have both option and stock hedge ratios
        self.assertIn('options_hedge_ratio', result)
        self.assertIn('stock_hedge_ratio', result)
        
        # Should have 2 trades (hedge option + stock)
        self.assertTrue(torch.all(result['trades_count'] == 2))
        
        # Hedge ratios should be reasonable
        options_ratio = result['options_hedge_ratio']
        stock_ratio = result['stock_hedge_ratio']
        
        self.assertFalse(torch.isnan(options_ratio).any())
        self.assertFalse(torch.isnan(stock_ratio).any())
        self.assertFalse(torch.isinf(options_ratio).any())
        self.assertFalse(torch.isinf(stock_ratio).any())


class TestHedgingComparison(unittest.TestCase):
    """Test cases for hedging strategy comparison"""
    
    def setUp(self):
        """Set up test parameters and paths"""
        self.S0 = 100.0
        self.K = 105.0
        self.r = 0.05
        self.q = 0.02
        self.T = 0.25
        
        # Generate smaller set of paths for faster testing
        n_paths = 500
        n_steps = 30
        
        self.S_paths, self.v_paths = generate_heston_paths(
            self.S0, self.r, self.q, self.T,
            kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04,
            n_paths=n_paths, n_steps=n_steps
        )
        
        self.times = torch.linspace(0, self.T, n_steps + 1)
    
    def test_compare_hedging_strategies(self):
        """Test hedging strategy comparison function"""
        results = compare_hedging_strategies(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q, 'call',
            transaction_cost=0.001,
            impact_lambda=1e-6
        )
        
        # Should have both dynamic and static strategies
        strategy_types = set()
        for strategy_name, strategy_result in results.items():
            if 'error' not in strategy_result:
                strategy_types.add(strategy_result.get('strategy_type', 'unknown'))
        
        self.assertIn('dynamic', strategy_types)
        self.assertIn('static', strategy_types)
        
        # Each strategy should have required fields
        for strategy_name, strategy_result in results.items():
            if 'error' in strategy_result:
                continue
            
            required_fields = ['pnl', 'trades_count', 'total_transaction_costs', 'strategy_type']
            for field in required_fields:
                self.assertIn(field, strategy_result, f"Missing {field} in {strategy_name}")
    
    def test_analyze_strategy_performance(self):
        """Test strategy performance analysis"""
        # First get comparison results
        results = compare_hedging_strategies(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q, 'call'
        )
        
        # Analyze performance
        performance_df = analyze_strategy_performance(results)
        
        # Check DataFrame structure
        expected_columns = ['Strategy', 'Mean_PnL', 'Std_PnL', 'Sharpe_Like', 
                           'Profit_Prob', 'VaR_5pct', 'Avg_Trades']
        
        for col in expected_columns:
            self.assertIn(col, performance_df.columns)
        
        # Should have multiple strategies
        self.assertGreater(len(performance_df), 1)
        
        # Check that values are reasonable
        for _, row in performance_df.iterrows():
            # Probabilities should be between 0 and 1
            self.assertGreaterEqual(row['Profit_Prob'], 0.0)
            self.assertLessEqual(row['Profit_Prob'], 1.0)
            
            # Standard deviation should be non-negative
            self.assertGreaterEqual(row['Std_PnL'], 0.0)
            
            # VaR should be non-positive (loss)
            self.assertLessEqual(row['VaR_5pct'], 0.0)
    
    def test_hedging_efficiency_analysis(self):
        """Test hedging efficiency analysis"""
        # Get comparison results
        results = compare_hedging_strategies(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q, 'call'
        )
        
        # Separate dynamic and static results
        dynamic_results = {k: v for k, v in results.items() 
                          if v.get('strategy_type') == 'dynamic'}
        static_results = {k: v for k, v in results.items() 
                         if v.get('strategy_type') == 'static'}
        
        # Analyze efficiency
        efficiency_metrics = hedging_efficiency_analysis(
            dynamic_results, static_results, benchmark_strategy='no_hedge'
        )
        
        # Should have variance reduction metrics
        self.assertIn('benchmark_variance', efficiency_metrics)
        
        # Check that variance reduction metrics are reasonable 
        for key, value in efficiency_metrics.items():
            if 'variance_reduction' in key and value is not None:
                # Variance reduction can be negative if hedge increases variance
                # But should be bounded (allow for very bad hedges)
                self.assertGreaterEqual(value, -1000.0)  # Very loose bound
                self.assertLessEqual(value, 1.0)         # Perfect hedge reduces variance by 100%
    
    def test_cost_benefit_analysis(self):
        """Test cost-benefit analysis"""
        # Get comparison results
        results = compare_hedging_strategies(
            self.S_paths, self.v_paths, self.times,
            self.K, self.r, self.q, 'call',
            transaction_cost=0.001
        )
        
        # Perform cost-benefit analysis
        cost_benefit_df = cost_benefit_analysis(results)
        
        # Check DataFrame structure
        expected_columns = ['Strategy', 'Risk_Reduction_Benefit', 'Avg_Transaction_Cost',
                           'Operational_Complexity', 'Cost_Benefit_Ratio']
        
        for col in expected_columns:
            self.assertIn(col, cost_benefit_df.columns)
        
        # Check that values make sense
        for _, row in cost_benefit_df.iterrows():
            # Risk reduction benefit should be positive
            self.assertGreater(row['Risk_Reduction_Benefit'], 0.0)
            
            # Transaction costs should be non-negative
            self.assertGreaterEqual(row['Avg_Transaction_Cost'], 0.0)
            
            # Operational complexity should be 1, 2, or 3
            self.assertIn(row['Operational_Complexity'], [1, 2, 3])
            
            # Cost-benefit ratio should be non-negative
            self.assertGreaterEqual(row['Cost_Benefit_Ratio'], 0.0)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with very small number of paths
        small_S_paths = self.S_paths[:10, :]
        small_v_paths = self.v_paths[:10, :]
        
        results = compare_hedging_strategies(
            small_S_paths, small_v_paths, self.times,
            self.K, self.r, self.q, 'call'
        )
        
        # Should still work with small sample
        self.assertGreater(len(results), 0)
        
        # Test with extreme parameters
        extreme_K = 200.0  # Very OTM
        results_extreme = compare_hedging_strategies(
            self.S_paths, self.v_paths, self.times,
            extreme_K, self.r, self.q, 'call'
        )
        
        # Should handle extreme strikes
        self.assertGreater(len(results_extreme), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)