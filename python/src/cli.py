# python/src/cli.py

import argparse
import numpy as np
from . import black_scholes as bs # Relative import
from . import plotting         # Relative import

def main():
    parser = argparse.ArgumentParser(description="Calculate Black-Scholes option price and Greeks.")

    # Required arguments
    parser.add_argument('-S', type=float, required=True, help="Current underlying asset price (e.g., 100)")
    parser.add_argument('-K', type=float, required=True, help="Option strike price (e.g., 105)")
    parser.add_argument('-T', type=float, required=True, help="Time to expiration in years (e.g., 0.5 for 6 months)")
    parser.add_argument('-r', type=float, required=True, help="Risk-free interest rate (annualized, e.g., 0.05 for 5%)")
    parser.add_argument('-v', '--volatility', type=float, required=True, dest='sigma',
                        help="Volatility of the underlying asset (annualized standard deviation, e.g., 0.2 for 20%)")
    parser.add_argument('-t', '--type', type=str, required=True, choices=['call', 'put'],
                        help="Option type ('call' or 'put')")

    # Optional arguments for Greeks and Plotting
    parser.add_argument('--greeks', action='store_true', help="Calculate and display all Greeks.")
    parser.add_argument('--plot', type=str, choices=['S', 'K', 'T', 'r', 'sigma'], default=None,
                        help="Plot sensitivity analysis by varying the specified parameter (e.g., --plot S)")
    parser.add_argument('--plot-greek', type=str, choices=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'], default=None,
                        help="Specify which Greek to plot sensitivity for (requires --plot)")
    parser.add_argument('--plot-range', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help="Specify the range for the varying parameter in the plot (e.g., --plot-range 80 120)")
    parser.add_argument('--plot-steps', type=int, default=50,
                        help="Number of steps for the parameter range in the plot (default: 50)")


    args = parser.parse_args()

    # Validate inputs
    if args.T <= 0:
        print("Warning: Time to expiration (T) must be positive for meaningful BS calculations beyond intrinsic value.")
        # Allow proceeding, BS functions handle T=0, but Greeks might be zero/undefined
    if args.sigma < 0:
        parser.error("Volatility (sigma) cannot be negative.")
    if args.r < 0:
        print("Warning: Negative risk-free rate (r) is unusual but allowed.")
    if args.S <= 0 or args.K <= 0 :
         print("Warning: Stock price (S) and Strike (K) are typically positive.")


    # --- Plotting Mode ---
    if args.plot:
        if args.plot_range is None:
            # Define default reasonable ranges if not provided
            if args.plot == 'S':
                min_val, max_val = args.S * 0.8, args.S * 1.2
            elif args.plot == 'K':
                min_val, max_val = args.K * 0.8, args.K * 1.2
            elif args.plot == 'T':
                 min_val, max_val = max(0.01, args.T * 0.1), args.T * 1.5 # Avoid T=0 for plots
            elif args.plot == 'r':
                 min_val, max_val = max(0.0, args.r - 0.04), args.r + 0.04
            elif args.plot == 'sigma':
                 min_val, max_val = max(0.01, args.sigma * 0.5), args.sigma * 1.5 # Avoid sigma=0
            print(f"Plot range not specified, using default: {min_val:.2f} to {max_val:.2f}")
        else:
            min_val, max_val = args.plot_range
            if min_val >= max_val:
                parser.error("--plot-range MIN must be less than MAX.")
            # Prevent non-positive T or sigma in plots
            if args.plot == 'T' and min_val <= 0: min_val = 0.01
            if args.plot == 'sigma' and min_val <= 0: min_val = 0.01


        param_range = np.linspace(min_val, max_val, args.plot_steps)

        plotting.plot_sensitivity(
            param_to_vary=args.plot,
            param_range=param_range,
            S=args.S, K=args.K, T=args.T, r=args.r, sigma=args.sigma,
            option_type=args.type,
            greek=args.plot_greek
        )

    # --- Calculation Mode ---
    else:
        # Calculate Price
        if args.type == 'call':
            price = bs.black_scholes_call_price(args.S, args.K, args.T, args.r, args.sigma)
        else: # put
            price = bs.black_scholes_put_price(args.S, args.K, args.T, args.r, args.sigma)

        print("\n--- Black-Scholes Calculation Results ---")
        print(f"Option Type:       {args.type.capitalize()}")
        print(f"Underlying Price:  {args.S:.4f}")
        print(f"Strike Price:      {args.K:.4f}")
        print(f"Time to Expiry (Y):{args.T:.4f}")
        print(f"Risk-Free Rate:    {args.r:.4f} ({args.r*100:.2f}%)")
        print(f"Volatility:        {args.sigma:.4f} ({args.sigma*100:.2f}%)")
        print("-----------------------------------------")
        print(f"Theoretical Price: {price:.4f}")
        print("-----------------------------------------")

        # Calculate Greeks if requested
        if args.greeks:
            greeks = bs.calculate_all_greeks(args.S, args.K, args.T, args.r, args.sigma, args.type)
            print("Greeks:")
            for name, value in greeks.items():
                print(f"  {name:<6}: {value:>12.6f}") # Right align and format
            print("-----------------------------------------")

if __name__ == "__main__":
    # This allows running the CLI directly using `python -m src.cli ...`
    # from the `python/` directory if needed, though run scripts are better.
    main()