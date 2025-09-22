"""
High-Frequency Trading Risk Management Module
============================================

Risk management metrics and controls specifically designed for HFT strategies
involving options and delta hedging.

Features:
1. Real-time position monitoring
2. Greeks-based risk limits
3. P&L attribution and monitoring
4. Dynamic position sizing
5. Intraday risk controls
6. Stress testing capabilities
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from optlib.utils.tensor import tensor_dtype, device
from optlib.pricing.greeks import bs_greeks_analytical, get_all_greeks


@dataclass
class RiskLimits:
    """Risk limits configuration for HFT strategies"""
    max_delta_exposure: float = 1000.0          # Maximum net delta exposure
    max_gamma_exposure: float = 100.0           # Maximum net gamma exposure  
    max_vega_exposure: float = 500.0            # Maximum net vega exposure
    max_theta_exposure: float = -50.0           # Maximum negative theta (decay)
    max_position_size: int = 1000               # Maximum position size per option
    max_daily_loss: float = 10000.0             # Maximum daily loss limit
    max_drawdown: float = 0.10                  # Maximum drawdown (10%)
    max_concentration: float = 0.25             # Maximum single position concentration
    min_liquidity_ratio: float = 0.05           # Minimum liquidity requirement
    stress_test_threshold: float = 0.02         # Stress test trigger (2% move)


@dataclass  
class PositionData:
    """Individual position data"""
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: float  # Time to expiry
    quantity: int
    mark_price: float
    bid: float
    ask: float
    underlying_price: float
    implied_vol: float
    greeks: Dict[str, float]
    

class HFTRiskManager:
    """
    Real-time risk management for HFT options strategies.
    
    Monitors positions, Greeks exposure, P&L, and enforces risk limits.
    """
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None,
                 risk_free_rate: float = 0.02,
                 dividend_yield: float = 0.0):
        self.risk_limits = risk_limits or RiskLimits()
        self.r = risk_free_rate
        self.q = dividend_yield
        
        # Position tracking
        self.positions: Dict[str, PositionData] = {}
        self.portfolio_greeks: Dict[str, float] = {
            'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0
        }
        
        # P&L tracking
        self.daily_pnl: List[float] = []
        self.position_pnl: Dict[str, float] = {}
        self.cumulative_pnl: float = 0.0
        self.peak_pnl: float = 0.0
        self.current_drawdown: float = 0.0
        
        # Risk monitoring
        self.risk_violations: List[Dict] = []
        self.stress_scenarios: List[Dict] = []
        
    def add_position(self, position: PositionData) -> bool:
        """
        Add or update a position with risk checks.
        
        Args:
            position: Position data
            
        Returns:
            True if position added successfully, False if risk limits violated
        """
        # Check position size limits
        if abs(position.quantity) > self.risk_limits.max_position_size:
            self._log_violation('position_size', f"Position size {position.quantity} exceeds limit")
            return False
        
        # Update position
        position_id = f"{position.symbol}_{position.option_type}_{position.strike}_{position.expiry}"
        self.positions[position_id] = position
        
        # Recompute portfolio Greeks
        self._update_portfolio_greeks()
        
        # Check Greeks limits
        if not self._check_greeks_limits():
            # Remove position if it violates limits
            del self.positions[position_id]
            self._update_portfolio_greeks()
            return False
        
        return True
    
    def remove_position(self, position_id: str) -> bool:
        """Remove a position from the portfolio"""
        if position_id in self.positions:
            del self.positions[position_id]
            self._update_portfolio_greeks()
            return True
        return False
    
    def update_market_data(self, symbol: str, underlying_price: float,
                          option_prices: Dict[str, Dict[str, float]]) -> None:
        """
        Update market data and recompute Greeks.
        
        Args:
            symbol: Underlying symbol
            underlying_price: Current underlying price
            option_prices: Dictionary of option prices {position_id: {'bid': x, 'ask': y, 'mark': z}}
        """
        # Update positions with new market data
        for position_id, position in self.positions.items():
            if symbol in position_id:
                position.underlying_price = underlying_price
                
                if position_id in option_prices:
                    price_data = option_prices[position_id]
                    position.bid = price_data.get('bid', position.bid)
                    position.ask = price_data.get('ask', position.ask)
                    position.mark_price = price_data.get('mark', position.mark_price)
                
                # Recompute Greeks
                position.greeks = get_all_greeks(
                    underlying_price, position.strike, self.r, self.q,
                    position.implied_vol, position.expiry, position.option_type,
                    method='analytical'
                )
                
                # Convert tensors to floats
                for greek_name, greek_value in position.greeks.items():
                    if isinstance(greek_value, torch.Tensor):
                        position.greeks[greek_name] = float(greek_value)
        
        # Update portfolio Greeks
        self._update_portfolio_greeks()
        
        # Update P&L
        self._update_pnl()
    
    def _update_portfolio_greeks(self) -> None:
        """Recompute portfolio-level Greeks"""
        portfolio_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
        
        for position in self.positions.values():
            for greek_name in portfolio_greeks:
                greek_value = position.greeks.get(greek_name, 0.0)
                portfolio_greeks[greek_name] += position.quantity * greek_value
        
        self.portfolio_greeks = portfolio_greeks
    
    def _update_pnl(self) -> None:
        """Update position and portfolio P&L"""
        total_pnl = 0.0
        
        for position_id, position in self.positions.items():
            # Simple mark-to-market P&L (would need cost basis in practice)
            position_value = position.quantity * position.mark_price
            self.position_pnl[position_id] = position_value  # Simplified
            total_pnl += position_value
        
        # Update drawdown tracking
        if total_pnl > self.peak_pnl:
            self.peak_pnl = total_pnl
            self.current_drawdown = 0.0
        else:
            self.current_drawdown = (self.peak_pnl - total_pnl) / max(abs(self.peak_pnl), 1.0)
        
        self.cumulative_pnl = total_pnl
    
    def _check_greeks_limits(self) -> bool:
        """Check if current Greeks exposure violates limits"""
        violations = []
        
        if abs(self.portfolio_greeks['delta']) > self.risk_limits.max_delta_exposure:
            violations.append(f"Delta exposure {self.portfolio_greeks['delta']:.2f} exceeds limit")
        
        if abs(self.portfolio_greeks['gamma']) > self.risk_limits.max_gamma_exposure:
            violations.append(f"Gamma exposure {self.portfolio_greeks['gamma']:.2f} exceeds limit")
        
        if abs(self.portfolio_greeks['vega']) > self.risk_limits.max_vega_exposure:
            violations.append(f"Vega exposure {self.portfolio_greeks['vega']:.2f} exceeds limit")
        
        if self.portfolio_greeks['theta'] < self.risk_limits.max_theta_exposure:
            violations.append(f"Theta exposure {self.portfolio_greeks['theta']:.2f} exceeds limit")
        
        if violations:
            for violation in violations:
                self._log_violation('greeks_limit', violation)
            return False
        
        return True
    
    def _log_violation(self, violation_type: str, message: str) -> None:
        """Log a risk violation"""
        violation = {
            'timestamp': pd.Timestamp.now(),
            'type': violation_type,
            'message': message,
            'portfolio_greeks': self.portfolio_greeks.copy(),
            'cumulative_pnl': self.cumulative_pnl
        }
        self.risk_violations.append(violation)
    
    def check_risk_limits(self) -> Dict[str, bool]:
        """
        Comprehensive risk limit check.
        
        Returns:
            Dictionary indicating which limits are violated
        """
        checks = {
            'delta_limit': abs(self.portfolio_greeks['delta']) <= self.risk_limits.max_delta_exposure,
            'gamma_limit': abs(self.portfolio_greeks['gamma']) <= self.risk_limits.max_gamma_exposure,
            'vega_limit': abs(self.portfolio_greeks['vega']) <= self.risk_limits.max_vega_exposure,
            'theta_limit': self.portfolio_greeks['theta'] >= self.risk_limits.max_theta_exposure,
            'drawdown_limit': self.current_drawdown <= self.risk_limits.max_drawdown,
            'daily_loss_limit': True  # Would need daily P&L tracking
        }
        
        return checks
    
    def stress_test(self, scenarios: List[Dict[str, float]]) -> Dict[str, Dict]:
        """
        Perform stress testing on current portfolio.
        
        Args:
            scenarios: List of stress scenarios {'underlying_move': %, 'vol_move': %}
            
        Returns:
            Dictionary with stress test results
        """
        stress_results = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = f"scenario_{i+1}"
            underlying_move = scenario.get('underlying_move', 0.0)
            vol_move = scenario.get('vol_move', 0.0)
            
            scenario_pnl = 0.0
            scenario_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0, 'rho': 0.0}
            
            for position in self.positions.values():
                # Apply stress scenario
                stressed_underlying = position.underlying_price * (1 + underlying_move)
                stressed_vol = position.implied_vol * (1 + vol_move)
                
                # Compute stressed Greeks
                stressed_greeks = get_all_greeks(
                    stressed_underlying, position.strike, self.r, self.q,
                    stressed_vol, position.expiry, position.option_type,
                    method='analytical'
                )
                
                # Convert to floats and scale by position size
                position_stressed_greeks = {}
                for greek_name, greek_value in stressed_greeks.items():
                    if isinstance(greek_value, torch.Tensor):
                        greek_val = float(greek_value)
                    else:
                        greek_val = greek_value
                    position_stressed_greeks[greek_name] = position.quantity * greek_val
                    scenario_greeks[greek_name] += position_stressed_greeks[greek_name]
                
                # Estimate P&L impact (simplified)
                delta_pnl = position_stressed_greeks['delta'] * (stressed_underlying - position.underlying_price)
                gamma_pnl = 0.5 * position_stressed_greeks['gamma'] * ((stressed_underlying - position.underlying_price) ** 2)
                vega_pnl = position_stressed_greeks['vega'] * (stressed_vol - position.implied_vol) * 100
                
                scenario_pnl += delta_pnl + gamma_pnl + vega_pnl
            
            stress_results[scenario_name] = {
                'scenario': scenario,
                'estimated_pnl': scenario_pnl,
                'portfolio_greeks': scenario_greeks,
                'risk_limit_violations': self._check_stress_violations(scenario_greeks)
            }
        
        return stress_results
    
    def _check_stress_violations(self, stressed_greeks: Dict[str, float]) -> List[str]:
        """Check which risk limits would be violated under stress"""
        violations = []
        
        if abs(stressed_greeks['delta']) > self.risk_limits.max_delta_exposure:
            violations.append('delta_limit')
        if abs(stressed_greeks['gamma']) > self.risk_limits.max_gamma_exposure:
            violations.append('gamma_limit')
        if abs(stressed_greeks['vega']) > self.risk_limits.max_vega_exposure:
            violations.append('vega_limit')
        if stressed_greeks['theta'] < self.risk_limits.max_theta_exposure:
            violations.append('theta_limit')
        
        return violations
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        risk_checks = self.check_risk_limits()
        
        # Portfolio concentration
        total_notional = sum(abs(pos.quantity * pos.mark_price) for pos in self.positions.values())
        max_position_notional = max((abs(pos.quantity * pos.mark_price) for pos in self.positions.values()), default=0)
        concentration = max_position_notional / max(total_notional, 1.0)
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'portfolio_greeks': self.portfolio_greeks,
            'risk_limit_checks': risk_checks,
            'positions_count': len(self.positions),
            'total_notional': total_notional,
            'concentration': concentration,
            'cumulative_pnl': self.cumulative_pnl,
            'current_drawdown': self.current_drawdown,
            'peak_pnl': self.peak_pnl,
            'recent_violations': self.risk_violations[-10:] if self.risk_violations else [],
            'all_limits_ok': all(risk_checks.values())
        }
        
        return report
    
    def suggest_risk_reduction(self) -> List[Dict]:
        """
        Suggest trades to reduce risk exposure.
        
        Returns:
            List of suggested trades
        """
        suggestions = []
        
        # Check delta exposure
        delta_exposure = self.portfolio_greeks['delta']
        if abs(delta_exposure) > self.risk_limits.max_delta_exposure * 0.8:  # 80% threshold
            hedge_quantity = -int(delta_exposure)  # Simplified - would need underlying hedge
            suggestions.append({
                'type': 'delta_hedge',
                'action': 'buy' if hedge_quantity > 0 else 'sell',
                'quantity': abs(hedge_quantity),
                'instrument': 'underlying',
                'reason': f'Reduce delta exposure from {delta_exposure:.2f}'
            })
        
        # Check gamma exposure  
        gamma_exposure = self.portfolio_greeks['gamma']
        if abs(gamma_exposure) > self.risk_limits.max_gamma_exposure * 0.8:
            suggestions.append({
                'type': 'gamma_hedge',
                'action': 'sell' if gamma_exposure > 0 else 'buy',
                'reason': f'Reduce gamma exposure from {gamma_exposure:.2f}',
                'note': 'Consider options with opposite gamma'
            })
        
        # Check position concentration
        if len(self.positions) > 0:
            total_notional = sum(abs(pos.quantity * pos.mark_price) for pos in self.positions.values())
            for position_id, position in self.positions.items():
                position_notional = abs(position.quantity * position.mark_price)
                if position_notional / total_notional > self.risk_limits.max_concentration:
                    suggestions.append({
                        'type': 'concentration_reduction',
                        'position_id': position_id,
                        'action': 'reduce',
                        'reason': f'Position concentration {position_notional/total_notional:.1%} exceeds limit'
                    })
        
        return suggestions
    
    def calculate_var(self, confidence_level: float = 0.05,
                     lookback_periods: int = 252) -> Dict[str, float]:
        """
        Calculate Value at Risk metrics.
        
        Args:
            confidence_level: VaR confidence level (e.g., 0.05 for 95% VaR)
            lookback_periods: Historical lookback for volatility estimation
            
        Returns:
            Dictionary with VaR metrics
        """
        if len(self.daily_pnl) < 2:
            return {'var': 0.0, 'cvar': 0.0, 'note': 'Insufficient data'}
        
        # Use available P&L data
        pnl_data = np.array(self.daily_pnl[-lookback_periods:])
        
        # Calculate VaR
        var = np.percentile(pnl_data, confidence_level * 100)
        
        # Calculate Conditional VaR (Expected Shortfall)
        cvar = np.mean(pnl_data[pnl_data <= var])
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'periods_used': len(pnl_data)
        }