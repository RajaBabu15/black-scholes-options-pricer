# simulator.py
import numpy as np
from pricers import BlackScholesPricer # We use our BSM pricer for hedging decisions

class DynamicHedgingSimulator:
    def __init__(self, S0, K, T, r, option_type="call", transaction_cost=0.001):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.option_type = option_type
        self.tc = transaction_cost
        self.pricer = BlackScholesPricer()

    def _generate_heston_path(self, v0, kappa, theta, xi, rho, num_steps):
        """
        Generates a single stock and volatility path using the Heston model.
        Uses a simple Euler-Maruyama discretization scheme.
        """
        dt = self.T / num_steps
        S_path = np.zeros(num_steps + 1)
        v_path = np.zeros(num_steps + 1)
        S_path[0], v_path[0] = self.S0, v0

        # Correlated random variables
        Z_s = np.random.normal(size=num_steps)
        Z_v = rho * Z_s + np.sqrt(1 - rho**2) * np.random.normal(size=num_steps)

        for i in range(1, num_steps + 1):
            v_path[i] = np.maximum(v_path[i-1] + kappa * (theta - v_path[i-1]) * dt +
                                 xi * np.sqrt(v_path[i-1] * dt) * Z_v[i-1], 0) # Reflection to keep v non-negative
            S_path[i] = S_path[i-1] * np.exp((self.r - 0.5 * v_path[i-1]) * dt +
                                           np.sqrt(v_path[i-1] * dt) * Z_s[i-1])
        return S_path, np.sqrt(v_path) # Return stock path and vol path

    def run_simulation(self, hedge_sigma, heston_params, num_steps):
        """
        Runs the main delta hedging simulation.

        hedge_sigma: The constant volatility the hedger *thinks* is true.
        heston_params: A dict with the true market parameters (v0, kappa, theta, xi, rho).
        """
        # 1. Generate the true market path
        S_path, vol_path = self._generate_heston_path(**heston_params, num_steps=num_steps)
        dt = self.T / num_steps

        # 2. Initial Setup (at t=0)
        # We sell the option
        initial_option_price = self.pricer.price(self.S0, self.K, self.T, self.r, hedge_sigma, self.option_type)
        cash = initial_option_price
        
        # Initial hedge
        delta_t = self.pricer.delta(self.S0, self.K, self.T, self.r, hedge_sigma, self.option_type)
        stock_units = delta_t
        cash -= stock_units * self.S0 * (1 + self.tc) # Buy stock, pay cost
        
        portfolio_pnl_path = np.zeros(num_steps + 1)
        portfolio_pnl_path[0] = 0 # P&L starts at zero

        # 3. Rebalancing Loop
        for i in range(1, num_steps):
            time_to_maturity = self.T - i * dt
            S_t = S_path[i]
            
            # P&L from previous period's position
            interest_earned = cash * self.r * dt
            cash += interest_earned
            
            # Calculate new delta for rebalancing
            delta_new = self.pricer.delta(S_t, self.K, time_to_maturity, self.r, hedge_sigma, self.option_type)
            
            # Rebalance
            trade_units = delta_new - stock_units
            cash -= trade_units * S_t * (1 + np.sign(trade_units) * self.tc)
            stock_units += trade_units
            
            # Log current portfolio P&L
            # Portfolio value = cash + stocks held. Option liability is handled at the end.
            portfolio_pnl_path[i] = cash + stock_units * S_t - initial_option_price

        # 4. Final Settlement (at T)
        final_S = S_path[-1]
        
        # Settle the option payoff
        if self.option_type == "call":
            option_payoff = np.maximum(final_S - self.K, 0)
        else:
            option_payoff = np.maximum(self.K - final_S, 0)
        
        # Liquidate stock position
        cash += stock_units * final_S * (1 - self.tc)
        stock_units = 0
        
        # Final hedging P&L
        hedging_pnl = cash - option_payoff
        portfolio_pnl_path[-1] = hedging_pnl

        return hedging_pnl, S_path, portfolio_pnl_path

    def run_static_hedge(self, heston_params, num_steps):
        """A naive hedge: hedge once at the beginning and never rebalance."""
        S_path, _ = self._generate_heston_path(**heston_params, num_steps=num_steps)
        
        initial_option_price = self.pricer.price(self.S0, self.K, self.T, self.r, np.sqrt(heston_params['v0']), self.option_type)
        cash = initial_option_price
        delta0 = self.pricer.delta(self.S0, self.K, self.T, self.r, np.sqrt(heston_params['v0']), self.option_type)
        stock_units = delta0
        cash -= stock_units * self.S0 * (1 + self.tc)
        
        # Let cash grow at risk-free rate
        final_cash = cash * np.exp(self.r * self.T)
        
        # Settle option
        final_S = S_path[-1]
        option_payoff = np.maximum(final_S - self.K, 0) if self.option_type == "call" else np.maximum(self.K - final_S, 0)
        
        # Liquidate stock
        final_cash += stock_units * final_S * (1 - self.tc)
        
        return final_cash - option_payoff
