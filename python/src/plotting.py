# python/src/plotting.py

import numpy as np
import matplotlib.pyplot as plt
from . import black_scholes as bs # Use relative import within the package

def plot_sensitivity(param_to_vary, param_range, S, K, T, r, sigma, option_type, greek=None):
    """
    Plots option price or a specific Greek against a range of one input parameter.

    Args:
        param_to_vary (str): Name of the parameter to vary (e.g., 'S', 'K', 'T', 'r', 'sigma').
        param_range (np.array): Array of values for the parameter being varied.
        S, K, T, r, sigma: Fixed values for the other parameters.
        option_type (str): 'call' or 'put'.
        greek (str, optional): Name of the Greek to plot ('Delta', 'Gamma', 'Vega', 'Theta', 'Rho').
                               If None, plots the option price.
    """
    prices = []
    greek_values = []

    fixed_params = {'S': S, 'K': K, 'T': T, 'r': r, 'sigma': sigma}

    for val in param_range:
        current_params = fixed_params.copy()
        current_params[param_to_vary] = val

        # Recalculate price
        if option_type == 'call':
            price = bs.black_scholes_call_price(**current_params)
        else:
            price = bs.black_scholes_put_price(**current_params)
        prices.append(price)

        # Recalculate requested Greek (if any)
        if greek:
            if greek == 'Delta':
                g_val = bs.delta(**current_params, option_type=option_type)
            elif greek == 'Gamma':
                 g_val = bs.gamma(**current_params) # Gamma is same for call/put
            elif greek == 'Vega':
                 g_val = bs.vega(**current_params) # Vega is same for call/put
            elif greek == 'Theta':
                 g_val = bs.theta(**current_params, option_type=option_type)
            elif greek == 'Rho':
                 g_val = bs.rho(**current_params, option_type=option_type)
            else:
                raise ValueError(f"Unknown Greek requested: {greek}")
            greek_values.append(g_val)

    # Plotting
    plt.figure(figsize=(10, 6))

    if greek:
        plt.plot(param_range, greek_values, label=f'{greek} ({option_type.capitalize()})')
        plt.ylabel(greek)
        plot_title = f'{option_type.capitalize()} Option {greek} vs. {param_to_vary.capitalize()}'
    else:
        plt.plot(param_range, prices, label=f'Price ({option_type.capitalize()})')
        plt.ylabel('Option Price')
        plot_title = f'{option_type.capitalize()} Option Price vs. {param_to_vary.capitalize()}'

    plt.xlabel(param_to_vary.capitalize())
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()