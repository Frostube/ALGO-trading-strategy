# Edge-Weighted Position Sizing Improvements

## Overview

This document outlines the implementation of edge-weighted position sizing in the trading strategy, which dynamically adjusts risk allocation based on the recent performance (profit factor) of each strategy.

## Changes Made

1. **Added PF Stabilizer (metrics.py)**
   - Implemented a minimum loss threshold to prevent unrealistically high profit factors
   - Added a `MIN_LOSS = 0.01` floor value to stabilize profit factor calculations
   - Prevents division by tiny numbers and ensures realistic strategy evaluation

2. **Edge-Weighted Position Sizing (portfolio_manager.py)**
   - Added `EDGE_WEIGHTS` configuration to define risk levels for each strategy
   - Implemented `dynamic_risk_pct` method to calculate risk percentage based on profit factor
   - Enhanced `calculate_position_size` to adjust risk based on recent edge (PF)
   - Risk parameters:
     - EMA Crossover: 1.0% risk when PF ≥ 1.2, 0.5% risk otherwise
     - RSI Oscillator: 1.2% risk when PF ≥ 1.2, 0.7% risk otherwise

3. **Extended Health Monitor (health_monitor.py)**
   - Added `get_profit_factor_last_n` method to calculate profit factor for the last N trades
   - Provides a rolling window assessment of strategy performance

4. **Updated Strategy Classes**
   - Modified EMACrossoverStrategy and RSIOscillatorStrategy to pass profit factor to position sizing
   - Added code to extract recent profit factor from health monitor
   - Updated position sizing calls to include strategy name and profit factor

## Benefits

- Allocates more capital to strategies with proven edge (PF > 1.2)
- Reduces exposure to underperforming strategies
- Prevents distortion of statistics from outlier trades
- Adapts to changing market conditions by using recent performance data

## Configuration

```python
EDGE_WEIGHTS = {
    "ema_crossover": {"pf_hi": 1.2, "risk_hi": 0.010, "risk_lo": 0.005},
    "rsi_oscillator": {"pf_hi": 1.2, "risk_hi": 0.012, "risk_lo": 0.007},
}
```

## Usage Example

```python
# Get recent profit factor from health monitor
pf_recent = health_monitor.get_profit_factor_last_n(40)

# Calculate position size with edge weighting
position_size, _, _ = portfolio_manager.calculate_position_size(
    symbol=symbol,
    current_price=price,
    atr_value=atr,
    strat_name="ema_crossover",
    pf_recent=pf_recent,
    side=side
)
```

## Performance Impact

Edge-weighted position sizing contributes to the overall goal of achieving:
- PF ≥ 1.2
- Win rate ≥ 55%
- Net return ≥ +3%
- Max drawdown ≤ 10% 