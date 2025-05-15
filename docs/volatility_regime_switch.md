# Volatility Regime Switch

## Overview

This feature implements a dynamic volatility regime classification system that adjusts trading parameters based on current market conditions. By detecting different volatility regimes (calm, normal, storm), the strategy can adapt its position sizing, pyramiding rules, and risk profile to better match market behavior.

## Volatility Regime Classifications

The system classifies market volatility into three distinct regimes:

1. **Calm Regime** (vol < 3%)
   - Low volatility, typically ranging markets
   - Position sizes are reduced by 50%
   - Pyramiding is disabled
   - Risk is lowered to prevent overtrading in narrow ranges

2. **Normal Regime** (vol between 3-6%)
   - Medium volatility, balanced markets
   - Standard position sizing (75% of maximum)
   - Standard risk parameters
   - Default trading behavior

3. **Storm Regime** (vol > 6%)
   - High volatility, typically trending markets
   - Full position sizing
   - Pyramiding is enabled
   - Maximize opportunity during strong trends

## Implementation Details

### Core Components

1. **Volatility Monitor** (`src/risk/volatility_monitor.py`)
   - Calculates realized volatility over configurable timeframes
   - Classifies market regimes based on volatility thresholds
   - Caches results to avoid redundant calculations
   - Provides visualization tools for volatility history

2. **Portfolio Risk Manager** (`src/risk/portfolio_manager.py`)
   - Integrates volatility regime data into position sizing
   - Dynamically adjusts risk parameters based on market regime
   - Determines whether pyramiding should be enabled
   - Scales position sizes according to volatility regime

3. **Strategy Adaptations**
   - Both EMA Crossover and RSI Oscillator strategies check market regime
   - Pyramiding rules are dynamically enabled/disabled
   - Position sizes are scaled based on current regime
   - Market regime is logged with trade information

## Usage

### Checking Current Volatility Regimes

Use the `check_volatility.py` script to view current market regimes:

```bash
python check_volatility.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" --plot
```

Options:
- `--symbols`: Comma-separated list of symbols to check
- `--timeframe`: Data timeframe (default: "1d")
- `--lookback`: Lookback period in days (default: 30)
- `--plot`: Generate volatility plots
- `--plot-days`: Days to include in plot (default: 90)
- `--force-update`: Force recalculation of volatility metrics

### Visualization

The system can generate volatility plots showing historical volatility patterns with regime bands:

```bash
python check_volatility.py --symbols "BTC/USDT" --plot --plot-days 180
```

This will create plots in the `reports/volatility/` directory, showing 180 days of volatility history with regime threshold bands.

## Benefit and Impact

1. **Risk Management**
   - Reduces position sizes in low-volatility environments to avoid overtrading
   - Increases exposure during high-volatility trending markets
   - Adapts to changing market conditions automatically

2. **Performance Optimization**
   - Enables pyramiding only in appropriate market conditions
   - Prevents position buildup in ranging markets
   - Maximizes capital efficiency across different market regimes

3. **Decision Support**
   - Provides clear visibility into current market conditions
   - Helps inform manual trading decisions
   - Documents market regime during each trade for post-analysis

## Configuration Parameters

The key parameters that control the regime switch behavior are defined in the risk manager:

```python
# Volatility regime thresholds
VOL_CALM = 0.03  # 3% realized volatility threshold for calm markets
VOL_STORM = 0.06  # 6% realized volatility threshold for volatile markets
```

These can be adjusted based on the specific characteristics of the markets being traded. 