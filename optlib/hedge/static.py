"""
Static Hedging Strategies Module
===============================

Implementation of various static hedging strategies for options.
These strategies involve setting up a hedge at initiation and holding
it until expiry without rebalancing.

Static strategies include:
1. Buy and hold stock hedge
2. Static delta hedge  
3. Static gamma hedge using other options
4. Protective put/covered call strategies
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from optlib.utils.tensor import tensor_dtype, device
from optlib.pricing.bs import bs_price
from optlib.pricing.greeks import bs_greeks_analytical


def static_delta_hedge(S_paths: torch.Tensor,
                      v_paths: torch.Tensor,
                      times: torch.Tensor,
                      K: Union[float, torch.Tensor],
                      r: Union[float, torch.Tensor],
                      q: Union[float, torch.Tensor],
                      option_type: str = 'call',
                      hedge_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
    """
    Static delta hedge: buy initial delta amount of stock and hold to expiry.
    
    Args:
        S_paths: Stock price paths (n_paths, n_steps+1)
        v_paths: Variance paths (n_paths, n_steps+1)
        times: Time grid
        K, r, q: Option parameters
        option_type: 'call' or 'put'
        hedge_ratio: Fixed hedge ratio (if None, use initial delta)
        
    Returns:
        Dictionary with P&L analysis
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    T = times[-1]
    
    # Convert parameters
    K = torch.as_tensor(K, dtype=tensor_dtype, device=device)
    r = torch.as_tensor(r, dtype=tensor_dtype, device=device) 
    q = torch.as_tensor(q, dtype=tensor_dtype, device=device)
    
    # Initial conditions
    S0 = S_paths[:, 0]
    v0 = v_paths[:, 0]
    sigma0 = torch.sqrt(torch.clamp(v0, min=1e-8))
    
    # Initial option price
    C0 = bs_price(S0, K, r, q, sigma0, T, option_type)
    
    # Determine hedge ratio
    if hedge_ratio is None:
        # Use initial delta as hedge ratio
        greeks = bs_greeks_analytical(S0, K, r, q, sigma0, T, option_type)
        hedge_ratio_tensor = greeks['delta']
    else:
        hedge_ratio_tensor = torch.full_like(S0, hedge_ratio)
    
    # Initial cash position (short option premium, buy hedge_ratio shares)
    initial_stock_cost = hedge_ratio_tensor * S0
    cash = C0 - initial_stock_cost
    
    # Apply interest to cash position over time
    final_cash = cash * torch.exp(r * T)
    
    # Final stock value
    final_stock_value = hedge_ratio_tensor * S_paths[:, -1]
    
    # Final option payoff
    if option_type == 'call':
        final_payoff = torch.clamp(S_paths[:, -1] - K, min=0.0)
    else:
        final_payoff = torch.clamp(K - S_paths[:, -1], min=0.0)
    
    # Total P&L: cash + stock value - option payoff
    pnl = final_cash + final_stock_value - final_payoff
    
    return {
        'pnl': pnl,
        'initial_premium': C0,
        'hedge_ratio': hedge_ratio_tensor,
        'final_cash': final_cash,
        'final_stock_value': final_stock_value,
        'final_payoff': final_payoff,
        'trades_count': torch.ones(n_paths),  # Only initial trade
        'total_transaction_costs': torch.zeros(n_paths)
    }


def buy_and_hold_hedge(S_paths: torch.Tensor,
                      v_paths: torch.Tensor, 
                      times: torch.Tensor,
                      K: Union[float, torch.Tensor],
                      r: Union[float, torch.Tensor],
                      q: Union[float, torch.Tensor],
                      option_type: str = 'call') -> Dict[str, torch.Tensor]:
    """
    Simple buy-and-hold: buy 1 share of stock per option sold.
    
    This is the simplest hedge but often not optimal.
    """
    return static_delta_hedge(S_paths, v_paths, times, K, r, q, 
                            option_type, hedge_ratio=1.0)


def static_gamma_hedge(S_paths: torch.Tensor,
                      v_paths: torch.Tensor,
                      times: torch.Tensor, 
                      K_primary: Union[float, torch.Tensor],
                      K_hedge: Union[float, torch.Tensor],
                      r: Union[float, torch.Tensor],
                      q: Union[float, torch.Tensor],
                      option_type_primary: str = 'call',
                      option_type_hedge: str = 'call') -> Dict[str, torch.Tensor]:
    """
    Static gamma hedge using another option to hedge gamma exposure.
    
    Args:
        S_paths, v_paths, times: Path data
        K_primary: Strike of primary option being hedged
        K_hedge: Strike of hedging option
        r, q: Market parameters
        option_type_primary: Type of primary option
        option_type_hedge: Type of hedging option
        
    Returns:
        P&L analysis with gamma hedge
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    T = times[-1]
    
    # Convert parameters
    K_primary = torch.as_tensor(K_primary, dtype=tensor_dtype, device=device)
    K_hedge = torch.as_tensor(K_hedge, dtype=tensor_dtype, device=device)
    r = torch.as_tensor(r, dtype=tensor_dtype, device=device)
    q = torch.as_tensor(q, dtype=tensor_dtype, device=device)
    
    # Initial conditions
    S0 = S_paths[:, 0]
    v0 = v_paths[:, 0] 
    sigma0 = torch.sqrt(torch.clamp(v0, min=1e-8))
    
    # Initial Greeks for both options
    greeks_primary = bs_greeks_analytical(S0, K_primary, r, q, sigma0, T, option_type_primary)
    greeks_hedge = bs_greeks_analytical(S0, K_hedge, r, q, sigma0, T, option_type_hedge)
    
    # Determine hedge ratios to neutralize gamma
    gamma_primary = greeks_primary['gamma']
    gamma_hedge = greeks_hedge['gamma']
    
    # Avoid division by zero
    gamma_hedge_safe = torch.where(torch.abs(gamma_hedge) > 1e-8, gamma_hedge, torch.ones_like(gamma_hedge) * 1e-8)
    options_hedge_ratio = -gamma_primary / gamma_hedge_safe
    
    # Calculate remaining delta to hedge with stock
    delta_primary = greeks_primary['delta']
    delta_hedge = greeks_hedge['delta']
    remaining_delta = delta_primary + options_hedge_ratio * delta_hedge
    
    # Initial prices
    C0_primary = bs_price(S0, K_primary, r, q, sigma0, T, option_type_primary)
    C0_hedge = bs_price(S0, K_hedge, r, q, sigma0, T, option_type_hedge)
    
    # Initial cash position
    # Short primary option, buy hedge options, buy stock for remaining delta
    initial_cash = (C0_primary - 
                   options_hedge_ratio * C0_hedge - 
                   remaining_delta * S0)
    
    # Apply interest
    final_cash = initial_cash * torch.exp(r * T)
    
    # Final values
    if option_type_primary == 'call':
        final_payoff_primary = torch.clamp(S_paths[:, -1] - K_primary, min=0.0)
    else:
        final_payoff_primary = torch.clamp(K_primary - S_paths[:, -1], min=0.0)
    
    if option_type_hedge == 'call':
        final_payoff_hedge = torch.clamp(S_paths[:, -1] - K_hedge, min=0.0)
    else:
        final_payoff_hedge = torch.clamp(K_hedge - S_paths[:, -1], min=0.0)
    
    final_stock_value = remaining_delta * S_paths[:, -1]
    
    # Total P&L
    pnl = (final_cash + 
           options_hedge_ratio * final_payoff_hedge + 
           final_stock_value - 
           final_payoff_primary)
    
    return {
        'pnl': pnl,
        'initial_premium_primary': C0_primary,
        'initial_premium_hedge': C0_hedge,
        'options_hedge_ratio': options_hedge_ratio,
        'stock_hedge_ratio': remaining_delta,
        'final_cash': final_cash,
        'trades_count': torch.ones(n_paths) * 2,  # Options + stock trade
        'total_transaction_costs': torch.zeros(n_paths)
    }


def protective_put_strategy(S_paths: torch.Tensor,
                           v_paths: torch.Tensor,
                           times: torch.Tensor,
                           K_put: Union[float, torch.Tensor],
                           r: Union[float, torch.Tensor],
                           q: Union[float, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Protective put: own stock + buy put option for downside protection.
    
    Args:
        S_paths, v_paths, times: Path data
        K_put: Put strike price
        r, q: Market parameters
        
    Returns:
        P&L analysis for protective put
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    T = times[-1]
    
    # Convert parameters
    K_put = torch.as_tensor(K_put, dtype=tensor_dtype, device=device)
    r = torch.as_tensor(r, dtype=tensor_dtype, device=device)
    q = torch.as_tensor(q, dtype=tensor_dtype, device=device)
    
    # Initial conditions
    S0 = S_paths[:, 0]
    v0 = v_paths[:, 0]
    sigma0 = torch.sqrt(torch.clamp(v0, min=1e-8))
    
    # Initial put price
    P0 = bs_price(S0, K_put, r, q, sigma0, T, 'put')
    
    # Initial cost: buy stock + buy put
    initial_cost = S0 + P0
    
    # Final values
    final_stock_value = S_paths[:, -1]
    final_put_payoff = torch.clamp(K_put - S_paths[:, -1], min=0.0)
    
    # Total final value
    final_value = final_stock_value + final_put_payoff
    
    # P&L
    pnl = final_value - initial_cost
    
    return {
        'pnl': pnl,
        'initial_stock_cost': S0,
        'initial_put_cost': P0,
        'total_initial_cost': initial_cost,
        'final_stock_value': final_stock_value,
        'final_put_payoff': final_put_payoff,
        'final_value': final_value,
        'trades_count': torch.ones(n_paths) * 2,  # Stock + put
        'total_transaction_costs': torch.zeros(n_paths)
    }


def covered_call_strategy(S_paths: torch.Tensor,
                         v_paths: torch.Tensor,
                         times: torch.Tensor,
                         K_call: Union[float, torch.Tensor],
                         r: Union[float, torch.Tensor],
                         q: Union[float, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Covered call: own stock + sell call option for additional income.
    
    Args:
        S_paths, v_paths, times: Path data
        K_call: Call strike price  
        r, q: Market parameters
        
    Returns:
        P&L analysis for covered call
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    T = times[-1]
    
    # Convert parameters
    K_call = torch.as_tensor(K_call, dtype=tensor_dtype, device=device)
    r = torch.as_tensor(r, dtype=tensor_dtype, device=device)
    q = torch.as_tensor(q, dtype=tensor_dtype, device=device)
    
    # Initial conditions
    S0 = S_paths[:, 0]
    v0 = v_paths[:, 0]
    sigma0 = torch.sqrt(torch.clamp(v0, min=1e-8))
    
    # Initial call price  
    C0 = bs_price(S0, K_call, r, q, sigma0, T, 'call')
    
    # Initial net cost: buy stock - sell call
    initial_cost = S0 - C0
    
    # Final values
    final_stock_value = S_paths[:, -1]
    final_call_payoff = torch.clamp(S_paths[:, -1] - K_call, min=0.0)
    
    # Net final value (stock value minus call obligation)
    final_value = final_stock_value - final_call_payoff
    
    # P&L
    pnl = final_value - initial_cost
    
    return {
        'pnl': pnl,
        'initial_stock_cost': S0,
        'initial_call_premium': C0,
        'net_initial_cost': initial_cost,
        'final_stock_value': final_stock_value,
        'final_call_payoff': final_call_payoff,
        'final_value': final_value,
        'trades_count': torch.ones(n_paths) * 2,  # Stock + call
        'total_transaction_costs': torch.zeros(n_paths)
    }


def no_hedge_strategy(S_paths: torch.Tensor,
                     v_paths: torch.Tensor,
                     times: torch.Tensor,
                     K: Union[float, torch.Tensor],
                     r: Union[float, torch.Tensor],
                     q: Union[float, torch.Tensor],
                     option_type: str = 'call') -> Dict[str, torch.Tensor]:
    """
    No hedge: just sell the option and hold cash.
    
    This represents the unhedged P&L for comparison purposes.
    """
    n_paths, n_steps_plus_1 = S_paths.shape
    T = times[-1]
    
    # Convert parameters
    K = torch.as_tensor(K, dtype=tensor_dtype, device=device)
    r = torch.as_tensor(r, dtype=tensor_dtype, device=device)
    q = torch.as_tensor(q, dtype=tensor_dtype, device=device)
    
    # Initial conditions
    S0 = S_paths[:, 0]
    v0 = v_paths[:, 0]
    sigma0 = torch.sqrt(torch.clamp(v0, min=1e-8))
    
    # Initial option price
    C0 = bs_price(S0, K, r, q, sigma0, T, option_type)
    
    # Earn interest on premium
    final_cash = C0 * torch.exp(r * T)
    
    # Final option payoff
    if option_type == 'call':
        final_payoff = torch.clamp(S_paths[:, -1] - K, min=0.0)
    else:
        final_payoff = torch.clamp(K - S_paths[:, -1], min=0.0)
    
    # P&L = premium + interest - payoff
    pnl = final_cash - final_payoff
    
    return {
        'pnl': pnl,
        'initial_premium': C0,
        'final_cash': final_cash,
        'final_payoff': final_payoff,
        'trades_count': torch.zeros(n_paths),  # No hedging trades
        'total_transaction_costs': torch.zeros(n_paths)
    }