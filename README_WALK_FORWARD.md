# Walk-Forward Analysis Framework

This framework provides tools for performing walk-forward optimization (WFO) of trading strategies. Walk-forward analysis helps ensure that strategy parameters are robust across different market conditions.

## Overview

Walk-forward optimization is a method of testing trading systems that addresses both curve-fitting and non-stationarity of financial markets. It works by:

1. Dividing historical data into sequential "windows" of training and testing periods
2. Optimizing parameters on each training window
3. Testing those parameters on the out-of-sample test window
4. Analyzing parameter stability and performance across windows

## Components

The framework consists of several modules:

- `src/utils/window_generator.py` - Creates non-overlapping train/test windows
- `src/utils/metrics.py` - Calculates performance metrics for trading results
- `src/utils/walk_forward.py` - Core optimization and analysis engine
- `src/utils/visualizations.py` - Visualization tools for analyzing results
- `run_walk_forward.py` - Command-line tool for running analyses

## Enhanced Visualization Tools

The framework includes advanced visualization tools to help analyze performance and parameter stability:

1. **In-Sample vs Out-of-Sample Scatter Plot** - Shows the relationship between in-sample and out-of-sample returns for each parameter combination and window, helping identify potential curve-fitting.

2. **Parameter Stability Visualization** - Displays how optimal parameters change across windows, revealing whether certain parameters are stable or highly varying across different market regimes.

3. **Return Distribution Plots** - Shows the distribution of returns across all tested parameter combinations, providing insight into the strategy's overall robustness.

4. **Parameter Heatmaps** - Visualizes how different parameter combinations interact to affect performance metrics.

5. **HTML Dashboard** - A comprehensive dashboard combining all visualizations and summary statistics in one convenient view.

All visualizations are automatically saved to the `wfo_results` directory with timestamped filenames for easy tracking and comparison of different runs.

## Features

- **Window Generation**: Create non-overlapping windows with customizable training and testing periods
- **Parameter Grid Search**: Test multiple parameter combinations across all windows
- **Performance Metrics**: Calculate key metrics like returns, profit factor, win rate, and drawdown
- **Visualization**: Generate heatmaps, distribution plots, and parameter stability charts
- **Market Regime Analysis**: Identify and analyze performance across different market regimes
- **Multiple Exit Strategies**: Support for fixed, trailing, time-based, and ATR-based exits
- **HTML Dashboard**: Interactive dashboard with results and visualizations

## Usage

### Basic Usage

Run a basic walk-forward test with default parameters:

```bash
python run_walk_forward.py
```

This will use default settings (BTC/USDT, 4h timeframe, 2 years of data, 90-day training windows, 30-day test windows).

### Command-Line Options

```bash
python run_walk_forward.py --symbol BTC/USDT --timeframe 1h --years 3 --train-days 60 --test-days 20 --analyze-regimes --open-dashboard
```

### Exit Strategy Options

Choose from different exit strategies:

```bash
# Fixed take-profit/stop-loss
python run_walk_forward.py --exit-strategy fixed --take-profit 5 --stop-loss 3

# Trailing stop
python run_walk_forward.py --exit-strategy trailing --trail-pct 2

# Time-based exit
python run_walk_forward.py --exit-strategy time --max-bars 10
```

### Market Regime Analysis

Analyze market regimes prior to optimization:

```bash
python run_walk_forward.py --analyze-regimes
```

## Results and Dashboard

After running a walk-forward test, you'll find results in the `wfo_results/` directory:

- CSV files with detailed results
- Parameter stability charts
- Performance distribution visualizations
- Interactive HTML dashboard

Use the `--open-dashboard` flag to automatically open the dashboard in your browser.

## Best Practices

1. **Avoid Overfitting**: Be careful not to test too many parameters or parameter combinations
2. **Balance Window Sizes**: Training windows should be large enough to capture market cycles, but not so large that older data dominates
3. **Focus on Robustness**: Prioritize parameter stability across windows over maximum performance
4. **Regime Analysis**: Understand how your strategy performs across different market regimes
5. **Multiple Timeframes**: Test your strategy on different timeframes for additional robustness

## Example Implementation

Adding walk-forward testing to a custom strategy:

```python
from src.utils.walk_forward import walk_forward_test
from src.data.fetcher import fetch_ohlcv

# Fetch historical data
data = fetch_ohlcv(symbol="ETH/USDT", tf="1h", days=365)

# Define parameter grid
param_grid = {
    'fast_ema': [5, 8, 13, 21],
    'slow_ema': [21, 34, 55, 89],
    'atr_sl_multiplier': [1.5, 2.0, 2.5]
}

# Run walk-forward test
results, best_params = walk_forward_test(
    data, 
    symbol="ETH/USDT", 
    train_days=60, 
    test_days=20,
    parameter_grid=param_grid
)
```

## Advanced Functionality

The framework can be extended with:

- **Multi-market testing**: Run tests across multiple markets to ensure parameter robustness
- **Monte Carlo simulation**: Add randomization to further validate strategy performance
- **Custom strategy integration**: Adapt the framework to work with any strategy class
- **Alternative optimizers**: Replace grid search with genetic algorithms or Bayesian optimization 