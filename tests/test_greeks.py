#!/usr/bin/env python3
"""
Unit Tests for Greeks Computation Module
========================================

Test analytical and numerical Greeks calculations for accuracy and consistency.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optlib.pricing.greeks import (
    bs_greeks_analytical, bs_greeks_numerical, 
    monte_carlo_greeks, get_all_greeks
)
from optlib.pricing.bs import bs_price
from optlib.sim.paths import generate_heston_paths


class TestGreeksComputation(unittest.TestCase):
    """Test cases for Greeks computation"""
    
    def setUp(self):
        """Set up test parameters"""
        self.S = 100.0
        self.K = 105.0
        self.r = 0.05
        self.q = 0.02
        self.sigma = 0.20
        self.T = 0.25
        self.tolerance = 1e-4  # Numerical tolerance
    
    def test_analytical_greeks_call(self):
        """Test analytical Greeks for call options"""
        greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call'
        )
        
        # Check that all Greeks are present
        expected_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in expected_greeks:
            self.assertIn(greek, greeks)
            self.assertIsInstance(greeks[greek], torch.Tensor)
        
        # Check reasonable ranges for call option
        self.assertGreater(float(greeks['delta']), 0)  # Call delta > 0
        self.assertLess(float(greeks['delta']), 1)     # Call delta < 1
        self.assertGreater(float(greeks['gamma']), 0)  # Gamma > 0
        self.assertLess(float(greeks['theta']), 0)     # Theta < 0 (time decay)
        self.assertGreater(float(greeks['vega']), 0)   # Vega > 0
        self.assertGreater(float(greeks['rho']), 0)    # Call rho > 0
    
    def test_analytical_greeks_put(self):
        """Test analytical Greeks for put options"""
        greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'put'
        )
        
        # Check reasonable ranges for put option
        self.assertLess(float(greeks['delta']), 0)     # Put delta < 0
        self.assertGreater(float(greeks['delta']), -1) # Put delta > -1
        self.assertGreater(float(greeks['gamma']), 0)  # Gamma > 0
        self.assertLess(float(greeks['theta']), 0)     # Theta < 0 (time decay)
        self.assertGreater(float(greeks['vega']), 0)   # Vega > 0
        self.assertLess(float(greeks['rho']), 0)       # Put rho < 0
    
    def test_numerical_vs_analytical_greeks(self):
        """Test that numerical and analytical Greeks are consistent"""
        analytical = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call'
        )
        numerical = bs_greeks_numerical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call'
        )
        
        # Compare each Greek with tolerance
        for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']:
            analytical_val = float(analytical[greek])
            numerical_val = float(numerical[greek])
            
            # Allow larger tolerance for higher-order derivatives
            tol = self.tolerance if greek in ['delta', 'vega', 'rho'] else self.tolerance * 10
            
            self.assertAlmostEqual(
                analytical_val, numerical_val, delta=tol,
                msg=f"{greek} mismatch: analytical={analytical_val:.6f}, numerical={numerical_val:.6f}"
            )
    
    def test_put_call_parity_greeks(self):
        """Test that put-call parity holds for Greeks"""
        call_greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call'
        )
        put_greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'put'
        )
        
        # Delta: call_delta - put_delta = exp(-q*T)
        expected_delta_diff = np.exp(-self.q * self.T)
        actual_delta_diff = float(call_greeks['delta'] - put_greeks['delta'])
        self.assertAlmostEqual(actual_delta_diff, expected_delta_diff, places=5)
        
        # Gamma should be the same for calls and puts
        self.assertAlmostEqual(
            float(call_greeks['gamma']), float(put_greeks['gamma']), places=5
        )
        
        # Vega should be the same for calls and puts
        self.assertAlmostEqual(
            float(call_greeks['vega']), float(put_greeks['vega']), places=5
        )
    
    def test_greeks_at_expiry(self):
        """Test Greeks behavior at expiry"""
        # ATM option at expiry
        greeks_atm = bs_greeks_analytical(
            100.0, 100.0, self.r, self.q, self.sigma, 1e-10, 'call'
        )
        
        # ITM option at expiry
        greeks_itm = bs_greeks_analytical(
            110.0, 100.0, self.r, self.q, self.sigma, 1e-10, 'call'
        )
        
        # OTM option at expiry
        greeks_otm = bs_greeks_analytical(
            90.0, 100.0, self.r, self.q, self.sigma, 1e-10, 'call'
        )
        
        # At expiry, gamma, theta, vega should be near zero
        for greeks in [greeks_atm, greeks_itm, greeks_otm]:
            self.assertAlmostEqual(float(greeks['gamma']), 0.0, places=3)
            self.assertAlmostEqual(float(greeks['theta']), 0.0, places=3)
            self.assertAlmostEqual(float(greeks['vega']), 0.0, places=3)
        
        # Delta should be 1 for ITM, 0 for OTM at expiry
        self.assertAlmostEqual(float(greeks_itm['delta']), 1.0, places=3)
        self.assertAlmostEqual(float(greeks_otm['delta']), 0.0, places=3)
    
    def test_vectorized_greeks(self):
        """Test that Greeks computation works with vectorized inputs"""
        S_vec = torch.tensor([95.0, 100.0, 105.0])
        K_vec = torch.tensor([100.0, 100.0, 100.0])
        
        greeks = bs_greeks_analytical(
            S_vec, K_vec, self.r, self.q, self.sigma, self.T, 'call'
        )
        
        # Check output shapes
        for greek_name, greek_values in greeks.items():
            self.assertEqual(greek_values.shape, (3,))
        
        # Check that OTM has lower delta than ATM than ITM
        deltas = greeks['delta']
        self.assertLess(float(deltas[0]), float(deltas[1]))  # OTM < ATM
        self.assertLess(float(deltas[1]), float(deltas[2]))  # ATM < ITM
    
    def test_monte_carlo_greeks(self):
        """Test Monte Carlo Greeks computation"""
        # Generate paths
        n_paths = 5000
        n_steps = 50
        
        S_paths, v_paths = generate_heston_paths(
            self.S, self.r, self.q, self.T, 
            kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04,
            n_paths=n_paths, n_steps=n_steps
        )
        
        times = torch.linspace(0, self.T, n_steps + 1)
        
        # Compute MC Greeks
        mc_greeks = monte_carlo_greeks(
            S_paths, v_paths, times, self.K, self.r, self.q, 'call'
        )
        
        # Check that all Greeks are computed
        expected_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        for greek in expected_greeks:
            self.assertIn(greek, mc_greeks)
        
        # Compare with analytical (should be approximately similar)
        analytical_greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call'
        )
        
        # Delta should be reasonably close (MC has noise)
        mc_delta = float(mc_greeks['delta'])
        analytical_delta = float(analytical_greeks['delta'])
        
        # Allow 20% tolerance for MC estimation
        self.assertAlmostEqual(mc_delta, analytical_delta, delta=0.2)
    
    def test_get_all_greeks_convenience_function(self):
        """Test the convenience function for getting all Greeks"""
        analytical_greeks = get_all_greeks(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call', 'analytical'
        )
        numerical_greeks = get_all_greeks(
            self.S, self.K, self.r, self.q, self.sigma, self.T, 'call', 'numerical'
        )
        
        # Should have same keys
        self.assertEqual(set(analytical_greeks.keys()), set(numerical_greeks.keys()))
        
        # Should be approximately equal
        for greek in analytical_greeks.keys():
            self.assertAlmostEqual(
                float(analytical_greeks[greek]), 
                float(numerical_greeks[greek]), 
                delta=self.tolerance * 10
            )
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Very high volatility
        high_vol_greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, 2.0, self.T, 'call'
        )
        for greek_name, greek_value in high_vol_greeks.items():
            self.assertFalse(torch.isnan(greek_value).any(), f"NaN in {greek_name} with high vol")
            self.assertFalse(torch.isinf(greek_value).any(), f"Inf in {greek_name} with high vol")
        
        # Very low volatility
        low_vol_greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, 0.01, self.T, 'call'
        )
        for greek_name, greek_value in low_vol_greeks.items():
            self.assertFalse(torch.isnan(greek_value).any(), f"NaN in {greek_name} with low vol")
            self.assertFalse(torch.isinf(greek_value).any(), f"Inf in {greek_name} with low vol")
        
        # Very short time to expiry
        short_time_greeks = bs_greeks_analytical(
            self.S, self.K, self.r, self.q, self.sigma, 1e-6, 'call'
        )
        for greek_name, greek_value in short_time_greeks.items():
            self.assertFalse(torch.isnan(greek_value).any(), f"NaN in {greek_name} with short time")
            self.assertFalse(torch.isinf(greek_value).any(), f"Inf in {greek_name} with short time")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)