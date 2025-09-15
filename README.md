# Black-Scholes Options Pricer

A comprehensive Python library for options pricing, Greeks computation, Monte Carlo simulation, dynamic hedging, and risk management, specifically designed for high-frequency trading strategies.

## Features

### 1. **Black-Scholes Pricing Model** ✅
- Analytical Black-Scholes option pricing for European calls and puts
- Support for dividend yields and risk-free rates
- Vectorized computations using PyTorch for high performance

### 2. **Greeks Computation** ✅
- **Analytical Greeks**: Delta, Gamma, Theta, Vega, Rho using closed-form formulas
- **Numerical Greeks**: Finite difference methods for validation
- **Monte Carlo Greeks**: Greeks computation from stochastic paths
- Vectorized calculations for portfolio-level Greeks

### 3. **Monte Carlo Simulation** ✅
- **Heston Stochastic Volatility Model**: Full implementation with Euler scheme
- **Ultra-fast vectorized path generation**: GPU-accelerated when available
- **Multiple maturity support**: Batch generation for different expiries
- Supports correlation between stock and volatility processes

### 4. **Dynamic Delta Hedging** ✅
- **Vectorized delta hedging simulation**: Maximum performance implementation
- **Transaction costs**: Bid-ask spreads and market impact modeling
- **Flexible rebalancing**: Configurable frequencies (intraday to daily)
- **Exposure scaling**: Dynamic position sizing
- **P&L tracking**: Step-by-step returns and performance metrics

### 5. **Static Hedging Strategies** ✅
- **Static Delta Hedge**: Initial delta hedge held to expiry
- **Buy and Hold**: Simple stock hedge
- **Static Gamma Hedge**: Multi-option gamma-neutral strategies
- **Protective Put**: Downside protection strategies
- **Covered Call**: Income generation strategies
- **No Hedge**: Unhedged option selling for comparison

### 6. **Hedging Strategy Comparison** ✅
- **Performance Analysis**: Comprehensive metrics comparison
- **Risk-Adjusted Returns**: Sharpe ratios, VaR, CVaR analysis
- **Cost-Benefit Analysis**: Transaction costs vs risk reduction
- **Hedging Efficiency**: Variance reduction measurements
- **Automated Report Generation**: Excel exports with detailed analysis

### 7. **HFT Risk Management** ✅
- **Real-time Position Monitoring**: Greeks exposure tracking
- **Dynamic Risk Limits**: Configurable exposure limits
- **Stress Testing**: Multi-scenario portfolio analysis
- **P&L Attribution**: Position-level and portfolio-level tracking
- **Risk Reduction Suggestions**: Automated hedge recommendations
- **VaR/CVaR Calculation**: Historical and parametric methods

## Installation

### Requirements
- Python 3.8+
- PyTorch
- NumPy
- Pandas
- SciPy
- Matplotlib (for visualization)

### Install Dependencies
```bash
pip install torch numpy pandas scipy matplotlib yfinance
```

### Clone Repository
```bash
git clone https://github.com/RajaBabu15/black-scholes-options-pricer.git
cd black-scholes-options-pricer
```

## Quick Start

### Basic Option Pricing and Greeks
```python
from api import bs_price, get_all_greeks

# Option parameters
S = 100.0    # Stock price
K = 105.0    # Strike price  
r = 0.05     # Risk-free rate
q = 0.02     # Dividend yield
sigma = 0.20 # Volatility
T = 0.25     # Time to expiry

# Price options
call_price = bs_price(S, K, r, q, sigma, T, 'call')
put_price = bs_price(S, K, r, q, sigma, T, 'put')

# Calculate Greeks
greeks = get_all_greeks(S, K, r, q, sigma, T, 'call')
print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
```

### Monte Carlo Simulation
```python
from api import generate_heston_paths

# Generate paths with Heston model
S_paths, v_paths = generate_heston_paths(
    S0=100.0, r=0.05, q=0.02, T=0.25,
    kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7, v0=0.04,
    n_paths=10000, n_steps=63
)
```

### Hedging Strategy Comparison
```python
from api import compare_hedging_strategies
import torch

# Create time grid
times = torch.linspace(0, 0.25, 64)

# Compare strategies
results = compare_hedging_strategies(
    S_paths, v_paths, times, K=105.0, r=0.05, q=0.02, 
    option_type='call', transaction_cost=0.001
)

# Analyze results
for strategy, result in results.items():
    pnl = result['pnl']
    print(f"{strategy}: Mean P&L = ${float(torch.mean(pnl)):.2f}")
```

### Risk Management
```python
from api import HFTRiskManager, RiskLimits, PositionData

# Create risk manager
risk_manager = HFTRiskManager(
    risk_limits=RiskLimits(max_delta_exposure=500.0)
)

# Add position
position = PositionData(
    symbol="AAPL", option_type="call", strike=150.0, expiry=0.25,
    quantity=50, mark_price=5.50, bid=5.45, ask=5.55,
    underlying_price=148.0, implied_vol=0.25,
    greeks={'delta': 0.55, 'gamma': 0.025, 'theta': -0.08, 'vega': 0.12, 'rho': 0.06}
)

success = risk_manager.add_position(position)
report = risk_manager.get_risk_report()
```

## Examples

Run the comprehensive examples:
```bash
python examples/basic_usage.py
```

This demonstrates:
- Black-Scholes pricing and Greeks
- Monte Carlo simulation with Heston model
- Dynamic vs static hedging comparison
- HFT risk management

## Testing

Run the test suite:
```bash
# Test Greeks computation
python tests/test_greeks.py

# Test hedging strategies
python tests/test_hedging.py

# Test risk management
python tests/test_risk_management.py
```

## Module Structure

```
optlib/
├── pricing/
│   ├── bs.py           # Black-Scholes pricing
│   ├── greeks.py       # Greeks computation
│   ├── iv.py           # Implied volatility
│   └── cos.py          # COS method pricing
├── models/
│   └── heston.py       # Heston stochastic volatility model
├── sim/
│   └── paths.py        # Monte Carlo path generation
├── hedge/
│   ├── delta.py        # Dynamic delta hedging
│   ├── static.py       # Static hedging strategies
│   └── comparison.py   # Strategy comparison framework
├── risk/
│   └── hft.py          # HFT risk management
├── metrics/
│   └── performance.py  # Performance metrics
└── utils/
    └── tensor.py       # PyTorch utilities
```

## Performance Features

- **GPU Acceleration**: PyTorch tensors with CUDA support
- **Vectorized Operations**: Batch processing for multiple options/paths
- **Memory Efficient**: Optimized tensor operations
- **Parallel Processing**: Multi-core path generation

## Risk Management Features

### Position Monitoring
- Real-time Greeks exposure tracking
- Portfolio-level risk aggregation
- Position concentration monitoring

### Risk Limits
- Delta, Gamma, Vega, Theta exposure limits
- Position size limits
- Drawdown monitoring
- Daily loss limits

### Stress Testing
- Multi-scenario analysis
- Underlying price and volatility shocks
- Risk limit violation detection

### Reporting
- Comprehensive risk reports
- VaR/CVaR calculations
- Risk reduction suggestions

## Advanced Features

### Dynamic Hedging
- **Sub-daily rebalancing**: Intraday hedging support
- **Transaction costs**: Realistic cost modeling
- **Market impact**: Quadratic impact functions
- **Exposure scaling**: Dynamic position sizing

### Static Strategies
- **Multiple strategies**: Delta, gamma, protective strategies
- **Performance comparison**: Risk-adjusted metrics
- **Cost analysis**: Transaction costs vs benefits

### HFT Alignment
- **Low-latency**: Optimized for speed
- **Real-time monitoring**: Live position tracking
- **Automated alerts**: Risk limit violations
- **Integration ready**: API-first design

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Authors

- **RajaBabu15** - Initial implementation and HFT optimization

## Acknowledgments

- Black-Scholes-Merton option pricing model
- Heston stochastic volatility model
- PyTorch for numerical computations