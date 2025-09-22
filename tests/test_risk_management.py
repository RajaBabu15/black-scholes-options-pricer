#!/usr/bin/env python3
"""
Unit Tests for Risk Management Module
====================================

Test HFT risk management functionality including position monitoring,
Greeks limits, and stress testing.
"""

import unittest
import torch
import numpy as np
import sys
import os
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optlib.risk.hft import HFTRiskManager, RiskLimits, PositionData


class TestRiskManagement(unittest.TestCase):
    """Test cases for risk management functionality"""
    
    def setUp(self):
        """Set up test risk manager and sample positions"""
        self.risk_limits = RiskLimits(
            max_delta_exposure=100.0,
            max_gamma_exposure=10.0,
            max_vega_exposure=50.0,
            max_theta_exposure=-5.0,
            max_position_size=50,
            max_daily_loss=1000.0,
            max_drawdown=0.15
        )
        
        self.risk_manager = HFTRiskManager(
            risk_limits=self.risk_limits,
            risk_free_rate=0.05,
            dividend_yield=0.02
        )
        
        # Sample positions
        self.sample_positions = [
            PositionData(
                symbol="AAPL",
                option_type="call",
                strike=150.0,
                expiry=0.25,
                quantity=10,
                mark_price=5.50,
                bid=5.45,
                ask=5.55,
                underlying_price=148.0,
                implied_vol=0.25,
                greeks={'delta': 0.55, 'gamma': 0.025, 'theta': -0.08, 'vega': 0.12, 'rho': 0.06}
            ),
            PositionData(
                symbol="AAPL",
                option_type="put",
                strike=145.0,
                expiry=0.25,
                quantity=-5,
                mark_price=3.20,
                bid=3.15,
                ask=3.25,
                underlying_price=148.0,
                implied_vol=0.22,
                greeks={'delta': -0.35, 'gamma': 0.028, 'theta': -0.06, 'vega': 0.10, 'rho': -0.04}
            )
        ]
    
    def test_risk_limits_initialization(self):
        """Test risk limits initialization"""
        limits = RiskLimits()
        
        # Check default values
        self.assertGreater(limits.max_delta_exposure, 0)
        self.assertGreater(limits.max_gamma_exposure, 0)
        self.assertGreater(limits.max_vega_exposure, 0)
        self.assertLess(limits.max_theta_exposure, 0)  # Theta is negative
        self.assertGreater(limits.max_position_size, 0)
    
    def test_add_position_success(self):
        """Test successful position addition"""
        position = self.sample_positions[0]
        success = self.risk_manager.add_position(position)
        
        self.assertTrue(success)
        self.assertEqual(len(self.risk_manager.positions), 1)
        
        # Check portfolio Greeks updated
        expected_delta = position.quantity * position.greeks['delta']
        self.assertAlmostEqual(
            self.risk_manager.portfolio_greeks['delta'], 
            expected_delta, 
            places=6
        )
    
    def test_add_position_size_limit(self):
        """Test position size limit enforcement"""
        # Create position that exceeds size limit
        large_position = PositionData(
            symbol="AAPL",
            option_type="call",
            strike=150.0,
            expiry=0.25,
            quantity=100,  # Exceeds limit of 50
            mark_price=5.50,
            bid=5.45,
            ask=5.55,
            underlying_price=148.0,
            implied_vol=0.25,
            greeks={'delta': 0.55, 'gamma': 0.025, 'theta': -0.08, 'vega': 0.12, 'rho': 0.06}
        )
        
        success = self.risk_manager.add_position(large_position)
        
        self.assertFalse(success)
        self.assertEqual(len(self.risk_manager.positions), 0)
        self.assertGreater(len(self.risk_manager.risk_violations), 0)
    
    def test_add_multiple_positions(self):
        """Test adding multiple positions and portfolio Greeks calculation"""
        # Add both sample positions
        for position in self.sample_positions:
            success = self.risk_manager.add_position(position)
            self.assertTrue(success)
        
        self.assertEqual(len(self.risk_manager.positions), 2)
        
        # Calculate expected portfolio Greeks
        expected_delta = (self.sample_positions[0].quantity * self.sample_positions[0].greeks['delta'] +
                         self.sample_positions[1].quantity * self.sample_positions[1].greeks['delta'])
        
        expected_gamma = (self.sample_positions[0].quantity * self.sample_positions[0].greeks['gamma'] +
                         self.sample_positions[1].quantity * self.sample_positions[1].greeks['gamma'])
        
        self.assertAlmostEqual(
            self.risk_manager.portfolio_greeks['delta'], 
            expected_delta, 
            places=6
        )
        self.assertAlmostEqual(
            self.risk_manager.portfolio_greeks['gamma'], 
            expected_gamma, 
            places=6
        )
    
    def test_greeks_limit_enforcement(self):
        """Test Greeks limits enforcement"""
        # Create position that would violate delta limit
        high_delta_position = PositionData(
            symbol="AAPL",
            option_type="call",
            strike=150.0,
            expiry=0.25,
            quantity=200,  # 200 * 0.9 = 180 delta exposure > 100 limit
            mark_price=5.50,
            bid=5.45,
            ask=5.55,
            underlying_price=148.0,
            implied_vol=0.25,
            greeks={'delta': 0.9, 'gamma': 0.025, 'theta': -0.08, 'vega': 0.12, 'rho': 0.06}
        )
        
        success = self.risk_manager.add_position(high_delta_position)
        
        # Should fail due to delta exposure limit (200 * 0.9 = 180 > 100 limit)
        self.assertFalse(success)
        self.assertEqual(len(self.risk_manager.positions), 0)
    
    def test_remove_position(self):
        """Test position removal"""
        # Add a position first
        position = self.sample_positions[0]
        success = self.risk_manager.add_position(position)
        self.assertTrue(success)
        
        # Get position ID
        position_id = f"{position.symbol}_{position.option_type}_{position.strike}_{position.expiry}"
        
        # Remove position
        removed = self.risk_manager.remove_position(position_id)
        self.assertTrue(removed)
        self.assertEqual(len(self.risk_manager.positions), 0)
        
        # Portfolio Greeks should be reset
        for greek_name, greek_value in self.risk_manager.portfolio_greeks.items():
            self.assertAlmostEqual(greek_value, 0.0, places=6)
    
    def test_update_market_data(self):
        """Test market data updates and Greeks recalculation"""
        # Add a position
        position = self.sample_positions[0]
        self.risk_manager.add_position(position)
        
        # Update market data
        new_underlying_price = 155.0
        new_option_prices = {
            f"{position.symbol}_{position.option_type}_{position.strike}_{position.expiry}": {
                'bid': 8.45, 'ask': 8.55, 'mark': 8.50
            }
        }
        
        self.risk_manager.update_market_data(
            position.symbol, new_underlying_price, new_option_prices
        )
        
        # Check that underlying price was updated
        updated_position = list(self.risk_manager.positions.values())[0]
        self.assertEqual(updated_position.underlying_price, new_underlying_price)
        self.assertEqual(updated_position.mark_price, 8.50)
    
    def test_check_risk_limits(self):
        """Test comprehensive risk limit checking"""
        # Add positions within limits
        for position in self.sample_positions:
            self.risk_manager.add_position(position)
        
        # Check limits
        limit_checks = self.risk_manager.check_risk_limits()
        
        # Should have all required checks
        expected_checks = ['delta_limit', 'gamma_limit', 'vega_limit', 'theta_limit', 'drawdown_limit']
        for check in expected_checks:
            self.assertIn(check, limit_checks)
            self.assertIsInstance(limit_checks[check], bool)
    
    def test_stress_testing(self):
        """Test stress testing functionality"""
        # Add positions
        for position in self.sample_positions:
            self.risk_manager.add_position(position)
        
        # Define stress scenarios
        scenarios = [
            {'underlying_move': 0.05, 'vol_move': 0.2},    # 5% up, 20% vol spike
            {'underlying_move': -0.05, 'vol_move': 0.2},   # 5% down, 20% vol spike
            {'underlying_move': 0.0, 'vol_move': -0.3}     # No move, vol crush
        ]
        
        # Run stress tests
        stress_results = self.risk_manager.stress_test(scenarios)
        
        # Should have results for each scenario
        self.assertEqual(len(stress_results), len(scenarios))
        
        # Each result should have required fields
        for scenario_name, result in stress_results.items():
            self.assertIn('scenario', result)
            self.assertIn('estimated_pnl', result)
            self.assertIn('portfolio_greeks', result)
            self.assertIn('risk_limit_violations', result)
            
            # P&L should be a number
            self.assertIsInstance(result['estimated_pnl'], (int, float))
            
            # Greeks should be a dictionary
            self.assertIsInstance(result['portfolio_greeks'], dict)
    
    def test_risk_report_generation(self):
        """Test risk report generation"""
        # Add positions
        for position in self.sample_positions:
            self.risk_manager.add_position(position)
        
        # Generate report
        report = self.risk_manager.get_risk_report()
        
        # Check required fields
        required_fields = [
            'timestamp', 'portfolio_greeks', 'risk_limit_checks',
            'positions_count', 'total_notional', 'concentration',
            'cumulative_pnl', 'current_drawdown', 'all_limits_ok'
        ]
        
        for field in required_fields:
            self.assertIn(field, report)
        
        # Check data types
        self.assertIsInstance(report['timestamp'], pd.Timestamp)
        self.assertIsInstance(report['portfolio_greeks'], dict)
        self.assertIsInstance(report['risk_limit_checks'], dict)
        self.assertIsInstance(report['positions_count'], int)
        self.assertIsInstance(report['all_limits_ok'], bool)
        
        # Positions count should match
        self.assertEqual(report['positions_count'], len(self.sample_positions))
    
    def test_risk_reduction_suggestions(self):
        """Test risk reduction suggestions"""
        # Add positions that might trigger suggestions
        for position in self.sample_positions:
            self.risk_manager.add_position(position)
        
        # Get suggestions
        suggestions = self.risk_manager.suggest_risk_reduction()
        
        # Should be a list
        self.assertIsInstance(suggestions, list)
        
        # Each suggestion should have required fields
        for suggestion in suggestions:
            self.assertIn('type', suggestion)
            self.assertIn('reason', suggestion)
    
    def test_var_calculation(self):
        """Test VaR calculation"""
        # Add some fake P&L history
        self.risk_manager.daily_pnl = [100, -50, 75, -25, 150, -80, 30, -120, 90, -40]
        
        # Calculate VaR
        var_metrics = self.risk_manager.calculate_var(confidence_level=0.05)
        
        # Check required fields
        self.assertIn('var', var_metrics)
        self.assertIn('cvar', var_metrics)
        self.assertIn('confidence_level', var_metrics)
        
        # VaR should be negative (representing loss)
        self.assertLess(var_metrics['var'], 0)
        
        # CVaR should be <= VaR
        self.assertLessEqual(var_metrics['cvar'], var_metrics['var'])
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty portfolio
        empty_report = self.risk_manager.get_risk_report()
        self.assertEqual(empty_report['positions_count'], 0)
        self.assertEqual(empty_report['total_notional'], 0)
        
        # Test VaR with insufficient data
        var_empty = self.risk_manager.calculate_var()
        self.assertIn('note', var_empty)
        
        # Test stress testing with empty portfolio
        stress_empty = self.risk_manager.stress_test([{'underlying_move': 0.05, 'vol_move': 0.2}])
        self.assertEqual(len(stress_empty), 1)
        
        # Test position with zero Greeks
        zero_greeks_position = PositionData(
            symbol="TEST",
            option_type="call",
            strike=100.0,
            expiry=0.1,
            quantity=1,
            mark_price=1.0,
            bid=0.95,
            ask=1.05,
            underlying_price=100.0,
            implied_vol=0.2,
            greeks={'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        )
        
        success = self.risk_manager.add_position(zero_greeks_position)
        self.assertTrue(success)  # Should succeed with zero Greeks


class TestPositionData(unittest.TestCase):
    """Test cases for PositionData dataclass"""
    
    def test_position_data_creation(self):
        """Test PositionData creation and field access"""
        position = PositionData(
            symbol="AAPL",
            option_type="call",
            strike=150.0,
            expiry=0.25,
            quantity=10,
            mark_price=5.50,
            bid=5.45,
            ask=5.55,
            underlying_price=148.0,
            implied_vol=0.25,
            greeks={'delta': 0.55, 'gamma': 0.025, 'theta': -0.08, 'vega': 0.12, 'rho': 0.06}
        )
        
        # Check all fields are accessible
        self.assertEqual(position.symbol, "AAPL")
        self.assertEqual(position.option_type, "call")
        self.assertEqual(position.strike, 150.0)
        self.assertEqual(position.quantity, 10)
        self.assertIsInstance(position.greeks, dict)
        
        # Check Greeks
        self.assertEqual(position.greeks['delta'], 0.55)
        self.assertEqual(position.greeks['gamma'], 0.025)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)