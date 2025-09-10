# pricers.py
import numpy as np
from scipy.stats import norm

class BlackScholesPricer:
    """
    A class to calculate European option prices and Greeks using the Black-Scholes model.
    """
    def _d1(self, S, K, T, r, sigma):
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    def _d2(self, S, K, T, r, sigma):
        return self._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    def price(self, S, K, T, r, sigma, option_type="call"):
        """Calculates the BS price for a call or put option."""
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)

        d1 = self._d1(S, K, T, r, sigma)
        d2 = self._d2(S, K, T, r, sigma)

        if option_type == "call":
            price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type == "put":
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type. Choose 'call' or 'put'.")
        return price

    def delta(self, S, K, T, r, sigma, option_type="call"):
        """Calculates the option's Delta."""
        if T <= 0:
            if option_type == "call":
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = self._d1(S, K, T, r, sigma)
        if option_type == "call":
            return norm.cdf(d1)
        else: # put
            return norm.cdf(d1) - 1

    def gamma(self, S, K, T, r, sigma):
        """Calculates the option's Gamma."""
        if T <= 0:
            return 0.0
        d1 = self._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    def vega(self, S, K, T, r, sigma):
        """Calculates the option's Vega."""
        if T <= 0:
            return 0.0
        d1 = self._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T)

class MonteCarloPricer:
    """
    A class for European option pricing using Monte Carlo simulation.
    """
    def __init__(self, num_sims=10000, num_steps=100):
        self.num_sims = num_sims
        self.num_steps = num_steps

    def price(self, S, K, T, r, sigma, option_type="call"):
        """Calculates MC price for a call or put option under GBM."""
        if T <= 0:
            return max(S - K, 0) if option_type == "call" else max(K - S, 0)

        dt = T / self.num_steps
        # Generate random paths
        Z = np.random.standard_normal((self.num_steps, self.num_sims))
        paths = np.zeros((self.num_steps + 1, self.num_sims))
        paths[0] = S

        for t in range(1, self.num_steps + 1):
            paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])

        # Calculate payoffs
        final_prices = paths[-1]
        if option_type == "call":
            payoffs = np.maximum(final_prices - K, 0)
        elif option_type == "put":
            payoffs = np.maximum(K - final_prices, 0)
        else:
            raise ValueError("Invalid option type.")

        # Discount and average
        return np.mean(payoffs) * np.exp(-r * T)

    def delta(self, S, K, T, r, sigma, option_type="call", ds=0.01):
        """Calculates Delta using finite difference."""
        price_up = self.price(S + ds, K, T, r, sigma, option_type)
        price_down = self.price(S - ds, K, T, r, sigma, option_type)
        return (price_up - price_down) / (2 * ds)

    def gamma(self, S, K, T, r, sigma, option_type="call", ds=0.01):
        """Calculates Gamma using finite difference."""
        price_up = self.price(S + ds, K, T, r, sigma, option_type)
        price_mid = self.price(S, K, T, r, sigma, option_type)
        price_down = self.price(S - ds, K, T, r, sigma, option_type)
        return (price_up - 2 * price_mid + price_down) / (ds**2)

    def vega(self, S, K, T, r, sigma, option_type="call", dv=0.001):
        """Calculates Vega using finite difference."""
        price_up = self.price(S, K, T, r, sigma + dv, option_type)
        price_down = self.price(S, K, T, r, sigma - dv, option_type)
        return (price_up - price_down) / (2 * dv)
