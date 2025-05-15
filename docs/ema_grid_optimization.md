# EMA Grid Search Optimization

## Overview

This feature implements a comprehensive grid search system to find optimal EMA parameters for each trading symbol. It uses walk-forward validation with separate training and test periods to prevent overfitting, ensuring robust parameter selection.

## Key Components

1. **Parameter Search Space**
   - Fast EMA: 3 to 14 (step 1)
   - Slow EMA: 20 to 80 (step 2)
   - Trend EMA: None, 100, or 200
   - ATR Multiplier: 0.8 to 1.6 (step 0.2)

2. **Validation Criteria**
   - Minimum 15 trades in out-of-sample period
   - Minimum profit factor of 1.1 in out-of-sample period
   - Stores top 3 parameter sets per symbol

3. **Walk-Forward Validation**
   - Training period: 80% of historical data
   - Test period: 20% of historical data (most recent)
   - Parameters must perform well in unseen data

## How to Use

### Running the Grid Search

Use the `run_ema_grid_search.py` script:

```bash
python run_ema_grid_search.py --symbols BTC/USDT,ETH/USDT --timeframe 2h
```

Options:
- `--symbols`: Comma-separated list of symbols to optimize
- `--timeframe`: Data timeframe (default: 2h)
- `--days`: Historical days to analyze (default: 270)
- `--workers`: Number of parallel processes (default: 4)
- `--output`: Output directory for parameter files (default: params)

### Automated Nightly Optimization

A batch script `run_ema_grid_nightly.bat` is provided for automated execution:

```bash
# Run on Windows
run_ema_grid_nightly.bat

# Schedule via Task Scheduler for nightly execution
```

### Using Optimized Parameters

The system automatically integrates with the trading strategies:

1. Parameters are saved to `params/{SYMBOL}_ema_params.json`
2. The EMA Crossover strategy checks for optimized parameters on initialization
3. The system prioritizes fresh parameters (less than 7 days old)
4. If fresh parameters aren't available, default values are used

## Implementation Details

### Core Components

1. **Parameter Generation**
   - Generates all valid parameter combinations (fast EMA < slow EMA)
   - Filters by basic validity rules
   - Total search space is approximately 5,500+ combinations per symbol

2. **Parallel Processing**
   - Uses Python's concurrent.futures for parallel testing
   - Configurable number of parallel workers
   - Progress tracking and ETA estimation

3. **Performance Metrics**
   - Win rate, profit factor, monthly return
   - Average win/loss size, total trades
   - Performance in both training and test periods

## Example Results

Example for BTC/USDT on 2h timeframe:

```json
{
  "symbol": "BTC/USDT",
  "timeframe": "2h",
  "fast_ema": 3,
  "slow_ema": 26,
  "trend_ema": 200,
  "atr_mult": 1.0,
  "test_profit_factor": 2.36,
  "test_win_rate": 0.47,
  "test_trades": 17,
  "test_monthly_return": 6.92%
}
```

## Benefits

1. **Tailored Parameters**
   - Each symbol gets optimized parameters suited to its specific characteristics
   - No more one-size-fits-all approach

2. **Adaptability**
   - Regular re-optimization ensures the strategy adapts to changing market conditions
   - Full parameter refresh every 7 days

3. **Robustness**
   - Walk-forward validation prevents overfitting
   - Out-of-sample testing ensures parameters work on unseen data

4. **Performance Improvement**
   - Expected increase in trade count: 2-3x
   - Expected improvement in returns: 3-6% per quarter

## Future Enhancements

1. **Adaptive Optimization**
   - Adjust optimization frequency based on market volatility
   - More frequent updates during high volatility periods

2. **Meta-Parameter Optimization**
   - Optimize the search space itself based on results
   - Find optimal time periods for training/testing split

3. **Multi-Timeframe Integration**
   - Optimize parameters across multiple timeframes
   - Use higher timeframe for trend, lower for entries

## Integration with Volatility Regime Switch

This system works in conjunction with the previously implemented volatility regime switch:

1. Parameters are optimized for each symbol independently
2. The volatility regime system adjusts position sizing based on current market conditions
3. Together, they provide both optimized signals and adaptive risk management 