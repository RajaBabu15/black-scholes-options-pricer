# Black-Scholes Option Pricing & Greeks Calculator

This project provides implementations of the Black-Scholes-Merton model for pricing European options and calculating their associated Greeks (Delta, Gamma, Vega, Theta, Rho). Implementations are available in both **Python** and **C#**, featuring command-line interfaces (CLIs) for ease of use. The Python version also includes basic sensitivity analysis plotting.


Below is a revised version of your README.md formulas with some minor formatting improvements and clarifications. For instance, I’ve clarified that Vega and Rho are typically scaled for a 1 percentage point change, and I’ve formatted the equations using clearer LaTeX formatting:

## Features

- Calculates theoretical prices for European Call and Put options.
- Computes the primary option Greeks:
  - **Delta:** Sensitivity to underlying price change.
  - **Gamma:** Sensitivity of Delta to underlying price change.
  - **Vega:** Sensitivity to volatility change (per 1 percentage point change).
  - **Theta:** Sensitivity to time decay (per day).
  - **Rho:** Sensitivity to interest rate change (per 1 percentage point change).
- Dual implementations:
  - **Python:** Using NumPy and SciPy for numerical calculations and Matplotlib for plotting.
  - **C#:** Using standard libraries and `MathNet.Numerics` for statistical functions (CDF/PDF).
- Command-Line Interfaces (CLIs) for both implementations to input parameters and view results.
- Python CLI includes optional sensitivity analysis plotting (e.g., Price vs. Stock Price, Delta vs. Volatility).
- Includes basic input validation and handling of edge cases (e.g., \(T = 0\), \(\sigma = 0\)).

## Black-Scholes Formulas Used

Let:
- \( S \) = Current price of the underlying asset  
- \( K \) = Strike price of the option  
- \( T \) = Time to expiration (in years)  
- \( r \) = Risk-free interest rate (annualized)  
- \( \sigma \) = Volatility of the underlying asset's returns (annualized standard deviation)  
- \( N(x) \) = Cumulative distribution function (CDF) of the standard normal distribution  
- \( n(x) \) = Probability density function (PDF) of the standard normal distribution  

Then:

$$
d_1 = \frac{\ln\left(\frac{S}{K}\right) + \left(r + \frac{1}{2}\sigma^2\right)T}{\sigma\sqrt{T}}
$$

$$
d_2 = d_1 - \sigma\sqrt{T}
$$

**Pricing:**

- **Call Price:**

  $$
  C = S \, N(d_1) - K \, e^{-rT} \, N(d_2)
  $$

- **Put Price:**

  $$
  P = K \, e^{-rT} \, N(-d_2) - S \, N(-d_1)
  $$

**Greeks:**

- **Delta (Call):**

  $$
  \Delta_C = N(d_1)
  $$

- **Delta (Put):**

  $$
  \Delta_P = N(d_1) - 1
  $$

- **Gamma:**

  $$
  \Gamma = \frac{n(d_1)}{S \, \sigma \, \sqrt{T}}
  $$

- **Vega:**

  $$
  \mathcal{V} = S \, n(d_1) \, \sqrt{T}
  $$  
  *Often scaled by 0.01 for a 1 percentage point change.*

- **Theta (Call):**

  $$
  \Theta_C = -\frac{S \, n(d_1) \, \sigma}{2\sqrt{T}} - r \, K \, e^{-rT} \, N(d_2)
  $$  
  *Often divided by 365 for a per-day change.*

- **Theta (Put):**

  $$
  \Theta_P = -\frac{S \, n(d_1) \, \sigma}{2\sqrt{T}} + r \, K \, e^{-rT} \, N(-d_2)
  $$  
  *Often divided by 365 for a per-day change.*

- **Rho (Call):**

  $$
  \rho_C = K \, T \, e^{-rT} \, N(d_2)
  $$  
  *Often scaled by 0.01 for a 1 percentage point change.*

- **Rho (Put):**

  $$
  \rho_P = -K \, T \, e^{-rT} \, N(-d_2)
  $$  
  *Often scaled by 0.01 for a 1 percentage point change.*


## Project Structure

```
black-scholes-options-pricer/
|-- python/                      # Python Implementation
|   |-- src/
|   |   |-- __init__.py
|   |   |-- black_scholes.py     # Core BS formulas and Greeks
|   |   |-- cli.py               # Command-line interface logic
|   |   `-- plotting.py          # Sensitivity analysis plots
|   |-- requirements.txt
|   `-- run_python_cli.[bat|sh]  # Example run scripts
|
|-- csharp/                      # C# Implementation
|   |-- BlackScholesCalculator/  # .NET Project folder
|   |   |-- BlackScholesCalculator.csproj
|   |   |-- BlackScholesModel.cs # Class with BS formulas and Greeks
|   |   `-- Program.cs           # Main CLI entry point
|   `-- BlackScholesCalculator.sln # .NET Solution file
|
|-- .gitignore
`-- README.md                    # Project documentation
```

## Setup

**1. Prerequisites:**
   *   Git (for cloning)
   *   **For Python:** Python 3.7+, pip
   *   **For C#:** .NET SDK (6.0 or newer recommended)

**2. Clone Repository:**
   ```bash
   git clone https://github.com/RajaBabu15/black-scholes-options-pricer.git
   cd black-scholes-options-pricer
   ```

## Python Setup:
   ```bash
   cd python
   # Create and activate a virtual environment (recommended)
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/macOS: source venv/bin/activate
   pip install -r requirements.txt
   cd ..
   ```

## C# Setup:
   ```bash
   cd csharp
   # Restore packages (including MathNet.Numerics)
   dotnet restore BlackScholesCalculator.sln
   # Build the project (optional, running will build it)
   dotnet build BlackScholesCalculator.sln
   cd ..
   ```

## Usage

**Python CLI:**

*   Navigate to the `python/` directory in your terminal (and activate the virtual environment if you created one).
*   Use `python -m src.cli [arguments]` or the example run scripts (`run_python_cli.bat` / `./run_python_cli.sh`).
*   **Basic Calculation:**
    ```bash
    python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call
    ```
*   **Calculate with Greeks:**
    ```bash
    python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type put --greeks
    ```
*   **Plot Price Sensitivity (vs. Underlying Price S):**
    ```bash
    # A plot window will open
    python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call --plot S --plot-range 80 120
    ```
*   **Plot Greek Sensitivity (Delta vs. Time T):**
    ```bash
    # A plot window will open
    python -m src.cli -S 100 -K 105 -T 0.5 -r 0.05 -v 0.2 --type call --plot T --plot-greek Delta --plot-range 0.01 1.0
    ```
*   See `python -m src.cli --help` for all options.

**C# CLI:**

*   Navigate to the `csharp/` directory in your terminal.
*   Run the calculator using `dotnet run`:
    ```bash
    dotnet run --project BlackScholesCalculator/BlackScholesCalculator.csproj
    ```
*   The program will prompt you interactively to enter the required parameters (S, K, T, r, volatility, type).
*   It will then calculate and display the option price and all Greeks.

## Technologies Used

*   **Python:**
    *   Python 3
    *   NumPy (Numerical operations)
    *   SciPy (Statistical functions - CDF)
    *   Matplotlib (Plotting)
*   **C#:**
    *   .NET (Core) SDK (6.0+)
    *   MathNet.Numerics (For Normal Distribution CDF/PDF)

## Validation

The results generated by these implementations should be compared against established online Black-Scholes calculators or financial software (e.g., Bloomberg terminal, QuantLib) using the same inputs to verify accuracy. Small differences may occur due to floating-point precision or slight variations in CDF/PDF approximations.

## Limitations

*   **European Options Only:** The Black-Scholes model as implemented here is valid only for European options (options that can only be exercised at expiration). It is not suitable for American options (exercisable anytime before expiration) without modifications (like binomial models or partial differential equation solvers).
*   **Constant Assumptions:** The model assumes constant volatility (\( \sigma \)), constant risk-free interest rate (\( r \)), and no dividend payments during the option's life. These assumptions often do not hold true in real markets.
*   **Lognormal Distribution:** Assumes the underlying asset price follows a geometric Brownian motion, resulting in a lognormal distribution of prices at expiration.
*   **No Transaction Costs:** Does not account for commissions or bid-ask spreads.
*   **Market Frictions:** Ignores margin requirements, liquidity constraints, etc.

This calculator serves as an educational tool for understanding the Black-Scholes model and its sensitivities (Greeks).