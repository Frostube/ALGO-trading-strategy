# ATR Trailing Stop Improvements

## Overview

This document outlines the implementation of tighter ATR trailing stops for losing trades across both the EMA Crossover and RSI Oscillator strategies. This adaptation helps cut losing trades faster while still allowing winning trades to run with a wider trailing stop.

## Changes Made

1. **Configurable ATR Trailing Stop Parameters**
   - Added constants to both strategy classes:
     - `ATR_TRAIL_START = 1.25` - Normal trailing stop multiplier (unchanged)
     - `ATR_TRAIL_LOSER = 1.00` - Tighter trailing stop for losing trades
     - `R_TRIGGER = 0.5` - Threshold in R-multiples to trigger the tighter trail

2. **ATR Trail Tightening Logic**
   - Added functionality to detect when a trade's unrealized loss exceeds 0.5R
   - When this occurs, the trailing stop is tightened to 1.0 × ATR (instead of 1.25 × ATR)
   - The tightening only happens once per trade, tracked with a `trail_tightened` flag
   - Trail tightening works for both long and short positions

3. **Benefits Expected**
   - Faster exit from losing trades that have moved against the position by 0.5R
   - Improved average win-to-loss ratio by reducing the size of losing trades
   - Decreased drawdown by cutting losses more quickly
   - Maintaining the full trail size for winning trades, allowing them to run with the standard 1.25 × ATR trail

## Testing

The improvements can be tested with the following command:

```bash
python run_adaptive_strategy.py \
  --mode backtest \
  --symbols "BTC/USDT" \
  --timeframe 4h \
  --days 30 \
  --initial-balance 10000
```

Expected results:
- Smaller average loss size
- Slightly higher win percentage
- Reduced drawdown
- Overall improved strategy performance

## Implementation Details

The key implementation involves calculating the unrealized PnL in terms of R-multiples (risk units), and when it drops below the trigger threshold, adjusting the trailing stop to be tighter:

```python
# Calculate PnL in R-multiples
pnl_r = (current_price - trade['entry_price']) / trade['atr']
if trade['side'] == 'sell':
    pnl_r = -pnl_r  # Invert for short trades
    
# Check if trade is losing more than R_TRIGGER and needs tighter trail
if (not trade.get('trail_tightened')) and pnl_r < -R_TRIGGER:
    # Apply tighter trail logic
    # Set trail_tightened = True to avoid repeated tightening
```

This change affects both the EMA crossover and RSI oscillator trading strategies. 