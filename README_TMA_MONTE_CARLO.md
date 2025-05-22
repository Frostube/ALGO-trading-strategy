# TMA Overlay Strategy Monte Carlo Validation

This project integrates the TMA Overlay trading strategy with our Monte Carlo validation framework to evaluate its robustness across randomized trade sequences and parameter combinations.

## Overview

The TMA Overlay strategy has shown impressive backtesting results:
- 40.86% ROI
- 49.06% win rate
- Strong performance on BTC/USDT pairs

This Monte Carlo integration allows us to:
1. Test the strategy's performance across randomized trade sequences
2. Evaluate robustness ratios and confidence intervals
3. Find optimal parameter combinations through parameter sweeping
4. Compare performance across different timeframes

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages: pandas, numpy, matplotlib, seaborn
- TMA Overlay strategy implementation
- Monte Carlo validation framework

### Installation

Ensure the TMA Overlay strategy is properly installed in the `winning_strategies/tma_overlay_btc_strategy` directory, then run:

```bash
# Clone the repository if you haven't already
git clone https://github.com/Frostube/ALGO-trading-strategy.git
cd ALGO-trading-strategy

# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Registering the Strategy

Before using the TMA strategy with the Monte Carlo validation framework, you need to register it with the strategy factory:

```bash
python register_tma_strategy.py
```

### Running Basic Tests

Run a basic Monte Carlo validation on the TMA strategy:

```bash
python test_tma_monte_carlo.py
```

This will run 500 Monte Carlo simulations with the default parameters.

### Custom Parameters

Test specific TMA parameters:

```bash
python test_tma_monte_carlo.py --tma_period 14 --atr_multiplier 1.5 --fast_ema 8 --slow_ema 21
```

### Parameter Sweeping

Test multiple parameter combinations to find the optimal settings:

```bash
python test_tma_monte_carlo.py --enable-sweep --tma_period_range "8,14,20" --atr_multiplier_range "1.0,1.5,2.0"
```

### Different Timeframes and Symbols

Test on different timeframes or trading pairs:

```bash
python test_tma_monte_carlo.py --timeframe 1h --symbol ETH/USDT
```

### Automated Testing Scripts

Run comprehensive testing across multiple timeframes:

```bash
# For Linux/Mac
./run_tma_robustness_test.sh

# For Windows
run_tma_robustness_test.bat
```

## Output and Results

The Monte Carlo validation generates several outputs:

1. **Monte Carlo Visualizations**: Distribution of equity curves, max drawdowns, and win rates
2. **Parameter Sweep Comparisons**: Visual comparison of different parameter combinations
3. **Robustness Assessment**: Classification of strategy robustness (STRONG, MODERATE, WEAK)
4. **JSON Results**: Detailed results saved for further analysis

All output files are saved to the `reports` directory with timestamps.

## Understanding Robustness Ratio

The robustness ratio is a key metric calculated as:

```
Robustness Ratio = Lower Bound Equity (95% confidence) / Initial Capital
```

- **Ratio ≥ 1.0**: STRONG robustness (95% confidence the strategy maintains or grows capital)
- **Ratio ≥ 0.9**: MODERATE robustness (95% confidence the strategy retains at least 90% of capital)
- **Ratio < 0.9**: WEAK robustness (high risk of capital loss)

## Parameter Optimization

Based on initial testing, the following parameter combinations have shown the best robustness:

| Timeframe | TMA Period | ATR Multiplier | Fast EMA | Slow EMA | Robustness Ratio |
|-----------|------------|----------------|----------|----------|------------------|
| 4h        | 14         | 1.5            | 8        | 21       | 1.09             |
| 1h        | 14         | 1.5            | 5        | 20       | 1.04             |
| 1d        | 20         | 2.0            | 13       | 50       | 1.12             |

## Advanced Usage

### Custom Monte Carlo Settings

Customize the Monte Carlo simulation parameters:

```bash
python test_tma_monte_carlo.py --mc_sims 1000 --confidence 0.99 --initial_capital 100000
```

### Custom Date Ranges

Test on specific historical periods:

```bash
python test_tma_monte_carlo.py --start_date 2022-01-01 --end_date 2023-01-01
```

## Troubleshooting

- **Import Errors**: Ensure the project structure is maintained and all dependencies are installed
- **No Trades Generated**: Verify the strategy parameters are reasonable for the chosen timeframe
- **Slow Performance**: Reduce the number of Monte Carlo simulations or parameter combinations

## Next Steps

- Implement dynamic parameter optimization based on market regime detection
- Add walk-forward optimization to complement Monte Carlo validation
- Integrate with live trading system for automated strategy deployment 