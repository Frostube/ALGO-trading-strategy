# BTC/USDT Intra-Day Scalping Strategy

An algorithmic trading strategy for BTC/USDT with enhanced entry filters, adaptive thresholds, optimized stop logic, and comprehensive performance tracking.

## Project Structure

```
src/
  backtest/          ← run_backtest.py, optimize.py
  config.py          ← Strategy configuration parameters
  data/              ← Data fetching and management
  indicators/        ← Technical indicators
  strategy/          ← Scalping strategy implementation
  execution/         ← Order execution
  db/                ← Database models
  api/               ← API endpoints
  utils/             ← Utility functions
  dashboard/         ← Performance dashboard and interactive backtesting
  tradingview/       ← Integration with TradingView
tests/               ← pytest unit tests
run_paper_trading.py ← Paper trading script
run_new_backtest.py  ← Backtest runner
run_dashboard.py     ← Streamlit dashboard runner
strategy_summary.py  ← Strategy configuration summary
```

## Binance Testnet Support

This project supports Binance testnet for development and testing without requiring KYC verification:

- **No KYC Required**: Use Binance testnet API keys to test without verification
- **Mock Data Generation**: Automatically generates realistic BTC/USDT data when testnet data is unavailable
- **Simple Setup**: Follow instructions in `TESTNET_SETUP.md` to configure

To use the testnet:
```bash
# Run dashboard with testnet support (enabled by default)
python -m streamlit run src/dashboard/backtest_dashboard.py
```

## Strategy Features

The BTC/USDT intra-day scalping strategy implements several advanced techniques:

1. **Enhanced Entry Filters**
   - Momentum confirmation (break above previous 5-min high for longs)
   - Micro-trend detection using EMA(50) slope
   - Multiple consecutive bars agreement requirement

2. **Adaptive Thresholds**
   - Percentile-based RSI thresholds calculated over rolling periods
   - Dynamic volume spike detection based on recent market conditions

3. **Overtrading Prevention**
   - Minimum bar gap between trades (5 bars)
   - Hourly trade limits (max 3 trades per hour)

4. **Sophisticated Stop Logic**
   - Two-leg stop strategy with initial ATR-based stop loss
   - Trailing stop that activates after 0.15% profit at 0.5× ATR
   - Soft stop alerts for manual intervention

5. **Performance Analytics**
   - FalsePositive tracking for trades that timeout
   - Comprehensive visualization reports
   - Time-series performance statistics

## Features

- EMA crossover strategy with dynamic parameter optimization
- Multiple timeframe analysis for trend confirmation
- Advanced backtesting with walk-forward optimization
- Health monitoring system to detect strategy degradation
- Volatility targeting for position sizing
- Volatility regime detection with adaptive parameter adjustment
- Multiple asset trading with portfolio risk management
- Performance logging and visualization

## Volatility Regime Detection

The strategy now includes a volatility regime detection system that adapts trading parameters based on market conditions:

### Regime Classification

- **QUIET (<3% volatility)**: Lower position sizes (50% of base risk), tighter stops
- **NORMAL (3-6% volatility)**: Standard position sizes (75% of base risk)
- **EXPLOSIVE (>6% volatility)**: Full position sizes (100% of base risk), pyramiding enabled

### Adaptive Parameters

The strategy automatically adjusts the following parameters based on the current volatility regime:

- **Position Sizing**: Scales with volatility regime (lower in quiet markets, higher in volatile markets)
- **Pyramiding**: Only enabled in explosive regimes to capture trending moves
- **Stop Loss Distance**: Adaptive based on recent volatility
- **Strategy Selection**: Different approaches optimized for each regime type

### How to Test Volatility Regimes

Run the volatility regime checker to see the current state of the market:

```
python check_volatility_regimes.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" --plot
```

This will generate volatility classification reports and regime visualizations in the `reports/` directory.

## Installation

```bash
git clone https://github.com/Frostube/ALGO-trading-strategy.git
cd ALGO-trading-strategy
pip install -r requirements.txt
cp .env-example .env   # Fill in your API keys
```

### API Key Setup Options:

1. **Binance Main Network**: Requires KYC verification
   - Complete Binance's identity verification process
   - Create API keys with trading permissions

2. **Binance Testnet**: No KYC required
   - Run `python create_env.py` to create a template .env file
   - Follow instructions in `TESTNET_SETUP.md` to get testnet keys
   - Add your testnet keys to the .env file

## Configuration

Key parameters in `src/config.py`:

```python
# Indicator parameters
EMA_FAST = 12  
EMA_SLOW = 26
RSI_PERIOD = 5
USE_ADAPTIVE_THRESHOLDS = True  

# Risk management
RISK_PER_TRADE = 0.01  # 1% of account per trade
USE_ATR_STOPS = True   # Use ATR-based stops
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0

# Two-leg stop settings
USE_TWO_LEG_STOP = True
TRAIL_ACTIVATION_PCT = 0.15  # Activate trailing at 0.15% profit
TRAIL_ATR_MULTIPLIER = 0.5   # Trail at 0.5× ATR
```

## Usage

### Interactive Dashboard

```bash
# Run the Streamlit dashboard
python run_dashboard.py

# Specify a custom port
python run_dashboard.py --port 8502
```

The Streamlit dashboard includes:
- Performance tracking of live/paper trading
- Interactive backtesting with parameter tweaking
- Trade analysis and visualization
- Overfitting detection between training/testing datasets

### Backtest the Strategy

```bash
# Default: last 30 days, $10k balance
python run_new_backtest.py --plot

# Customized backtest
python run_new_backtest.py --days 60 --balance 5000 --output results.json
```

### View Strategy Summary

```bash
python strategy_summary.py
```

### Run Paper Trading

```bash
python run_paper_trading.py
```

## Performance Metrics

Key metrics tracked:
- Profit/Loss
- Profit Factor
- Win Rate
- Max Drawdown
- Sharpe Ratio

## License

MIT
