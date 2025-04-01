#!/bin/bash

# chmod +x run_python_cli.sh

# Activate your virtual environment if you have one
# source venv/bin/activate

echo "Running Black-Scholes Python CLI..."
echo ""

# Example Calculation: Call Option + Greeks
echo "--- Call Option + Greeks ---"
python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call --greeks
echo ""


# Example Calculation: Put Option (Price only)
echo "--- Put Option (Price only) ---"
python -m src.cli -S 50 -K 55 -T 1.0 -r 0.03 -v 0.25 --type put
echo ""


# Example Plotting: Call Price vs. Underlying Price
echo "--- Plotting Call Price vs. Underlying Price (Window will open) ---"
python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call --plot S --plot-range 80 120
echo ""


# Example Plotting: Put Delta vs. Volatility
echo "--- Plotting Put Delta vs. Volatility (Window will open) ---"
python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type put --plot sigma --plot-greek Delta --plot-range 0.05 0.40
echo ""


echo "Done."

# Deactivate environment if activated
# deactivate