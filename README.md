# Black-Scholes Options Pricing and Dynamic Hedging Analysis

A comprehensive quantitative finance project demonstrating option pricing theory, Monte Carlo simulation, and dynamic hedging strategies under stochastic volatility models.

![Options Trading](https://img.shields.io/badge/Finance-Options%20Trading-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 Project Overview

This project implements and analyzes fundamental concepts in quantitative finance, specifically focusing on:

- **Black-Scholes-Merton Model**: Analytical option pricing with Greeks calculation
- **Monte Carlo Simulation**: Path-dependent pricing methods for validation
- **Dynamic Delta Hedging**: Portfolio rebalancing strategies with transaction costs
- **Stochastic Volatility**: Heston model for realistic market dynamics
- **Model Risk Analysis**: Impact of parameter misspecification on hedging performance

## 🏗️ Project Structure

```
black-scholes-options-pricer/
│
├── pricers.py              # Black-Scholes and Monte Carlo pricing engines
├── simulator.py            # Dynamic hedging simulation engine
├── hedging_analysis.ipynb  # Main analysis notebook with results
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore patterns
```

## 🚀 Key Features

### 1. **Pricing Engines**
- **BlackScholesPricer**: Analytical closed-form solutions for European options
- **MonteCarloPricer**: Simulation-based pricing with configurable parameters
- **Greeks Calculation**: Delta, Gamma, Vega with both analytical and numerical methods

### 2. **Market Simulation**
- **Heston Stochastic Volatility Model**: Realistic market dynamics with mean-reverting volatility
- **Transaction Cost Integration**: Realistic trading friction modeling
- **Path Generation**: Correlated Brownian motion for stock and volatility processes

### 3. **Hedging Analysis**
- **Dynamic Delta Hedging**: Continuous rebalancing simulation
- **Static Hedging Comparison**: Benchmark against buy-and-hold strategies
- **Performance Metrics**: Hedging error, P&L distribution analysis

### 4. **Sensitivity Analysis**
- **Transaction Cost Impact**: How trading costs affect hedging performance
- **Rebalancing Frequency**: Optimal rebalancing frequency analysis
- **Model Risk**: Volatility misspecification impact on hedging

## 📊 Key Results & Insights

Our comprehensive analysis reveals:

- **Dynamic hedging reduces risk by ~60-80%** compared to static hedging
- **Transaction costs create a trade-off** between hedging frequency and cost
- **Model risk is significant**: Wrong volatility assumptions create systematic bias
- **Gamma risk** is the primary source of hedging error between rebalances
- **Stochastic volatility** introduces additional complexity beyond Black-Scholes

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Conda or pip package manager

### Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/black-scholes-options-pricer.git
   cd black-scholes-options-pricer
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n options-pricer python=3.10 -y
   conda activate options-pricer
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter notebook**:
   ```bash
   jupyter notebook hedging_analysis.ipynb
   ```

## 📈 Usage Examples

### Basic Option Pricing
```python
from pricers import BlackScholesPricer, MonteCarloPricer

# Initialize pricers
bs_pricer = BlackScholesPricer()
mc_pricer = MonteCarloPricer(num_sims=100000)

# Price a call option
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20
price = bs_pricer.price(S, K, T, r, sigma, option_type="call")
delta = bs_pricer.delta(S, K, T, r, sigma, option_type="call")

print(f"Option Price: ${price:.2f}")
print(f"Delta: {delta:.3f}")
```

### Dynamic Hedging Simulation
```python
from simulator import DynamicHedgingSimulator

# Setup parameters
heston_params = {
    'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 
    'xi': 0.3, 'rho': -0.7
}

# Run simulation
simulator = DynamicHedgingSimulator(S, K, T, r, "call", transaction_cost=0.001)
pnl, stock_path, pnl_path = simulator.run_simulation(
    hedge_sigma=0.20, 
    heston_params=heston_params, 
    num_steps=252
)

print(f"Hedging P&L: ${pnl:.2f}")
```

## 🧮 Mathematical Foundation

### Black-Scholes Formula
For a European call option:
```
C(S,t) = S₀N(d₁) - Ke^(-r(T-t))N(d₂)
```
Where:
```
d₁ = [ln(S/K) + (r + σ²/2)(T-t)] / [σ√(T-t)]
d₂ = d₁ - σ√(T-t)
```

### Heston Stochastic Volatility Model
```
dS = rS dt + √v S dW₁
dv = κ(θ - v)dt + ξ√v dW₂
```
With correlation `ρ` between `dW₁` and `dW₂`.

### Greeks
- **Delta (Δ)**: ∂C/∂S - sensitivity to underlying price
- **Gamma (Γ)**: ∂²C/∂S² - sensitivity of delta to underlying price  
- **Vega (ν)**: ∂C/∂σ - sensitivity to volatility

## 📊 Analysis Highlights

The notebook provides comprehensive analysis including:

1. **Model Validation**: Comparison between analytical and Monte Carlo results
2. **Single Path Analysis**: Detailed examination of hedging mechanics
3. **Statistical Analysis**: Distribution of hedging P&L across 1000+ simulations
4. **Sensitivity Studies**: Impact of transaction costs, rebalancing frequency, and model parameters
5. **Visualization Suite**: 10+ charts showing key insights and relationships

## 🔬 Technical Details

### Performance Optimizations
- Vectorized NumPy operations for Monte Carlo simulations
- Efficient random number generation for correlated processes
- Modular design for easy extension and testing

### Numerical Methods
- Euler-Maruyama discretization for Heston model
- Finite difference methods for Greeks calculation
- Antithetic variate techniques for variance reduction

## 🚀 Extensions & Future Work

Potential enhancements include:

- **Gamma Hedging**: Delta-gamma neutral portfolios
- **Jump-Diffusion Models**: Merton jump-diffusion implementation
- **American Options**: Early exercise features
- **Portfolio Analysis**: Multiple options hedging
- **Machine Learning**: Optimal rebalancing strategies
- **Real Data Integration**: Market data backtesting

## 📚 References & Learning Resources

### Books
- Hull, J. C. "Options, Futures, and Other Derivatives"
- Wilmott, P. "Paul Wilmott on Quantitative Finance"
- Shreve, S. "Stochastic Calculus for Finance"

### Papers
- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Heston, S. L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility"

### Online Resources
- [QuantLib Documentation](https://www.quantlib.org/)
- [Wilmott Forums](https://www.wilmott.com/)
- [Risk.net](https://www.risk.net/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This project is for educational purposes only. It is not intended as financial advice or for actual trading. Options trading involves substantial risk and may not be suitable for all investors.

## 📞 Contact

For questions or collaboration opportunities, please reach out via GitHub issues or discussions.

---

**Built with ❤️ for quantitative finance education**
