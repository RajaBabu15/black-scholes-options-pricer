@echo off
REM Activate your virtual environment if you have one
REM CALL venv\Scripts\activate

echo Running Black-Scholes Python CLI...

REM Example Calculation: Call Option + Greeks
python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call --greeks

echo.
echo Example Calculation: Put Option (Price only)
python -m src.cli -S 50 -K 55 -T 1.0 -r 0.03 -v 0.25 --type put

echo.
echo Example Plotting: Call Price vs. Underlying Price
python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call --plot S --plot-range 80 120

echo.
echo Example Plotting: Put Delta vs. Volatility
python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type put --plot sigma --plot-greek Delta --plot-range 0.05 0.40

echo Done.

REM Deactivate environment if activated
REM deactivate
pause