# Realized Volatility Regime Switch

## Overview

The Realized Volatility Regime Switch is designed to dynamically adapt trading strategies based on current market volatility conditions. By measuring 30-day realized volatility, the system can automatically switch between different trading modes optimized for specific market environments.

## Key Components

### 1. Volatility Calculation

```python
# 30-day realized volatility calculation
df['returns'] = df['close'].pct_change()
df['volatility_30d'] = df['returns'].rolling(window=30*24).std() * np.sqrt(365)  # Annualized
```

### 2. Regime Classification

```
σ < 3%       → "quiet"     → enable Mean-Reversion mode
3% ≤ σ ≤ 6%  → "normal"    → keep EMA trend mode
σ > 6%       → "explosive" → switch to Breakout mode
```

### 3. Strategy Modes

1. **Mean-Reversion Mode (Quiet Markets)**
   - RSI(2) extreme readings
   - Bollinger Band touches
   - Target: Short-term scalps (0.5 ATR)
   - Position sizing: 50% of normal size

2. **Trend Mode (Normal Markets)**
   - EMA crossovers (optimized parameters)
   - Normal ATR trailing stops
   - Target: Medium-term trends
   - Position sizing: 75% of normal size 

3. **Breakout Mode (Explosive Markets)**
   - Donchian channel (20-period) breakouts
   - Volume spike confirmation
   - Target: Catching major moves
   - Position sizing: 100% of normal size
   - Trailing stop: 1.5x ATR

## Implementation Plan

### Phase 1: Volatility Calculation and Monitoring

1. Create `src/risk/volatility_regimes.py` module with:
   - 30-day realized volatility calculation
   - Regime classification
   - Visualization tools

2. Create `check_volatility_regimes.py` CLI tool to:
   - Display current volatility for monitored symbols
   - Show regime transitions
   - Generate regime history charts

### Phase 2: Strategy Integration

1. Modify strategy selection logic to:
   - Check current volatility regime
   - Select appropriate strategy based on regime
   - Adjust position sizing proportionally

2. Implement Mean-Reversion and Breakout strategies:
   - `src/strategy/rsi_mean_reversion.py`
   - `src/strategy/donchian_breakout.py`

### Phase 3: Backtest Verification

1. Test each regime independently:
   - Filter historical data by regime
   - Verify each strategy performs best in its targeted regime

2. End-to-end testing:
   - Compare performance with and without regime switching
   - Verify smooth transitions between regimes

## Expected Impact

| Market Type | % of Time | Strategy | Expected Improvement |
|-------------|-----------|----------|----------------------|
| Quiet (<3%) | ~30%      | Mean-Reversion | +3-5% quarterly |
| Normal (3-6%) | ~55%    | Trend Following | Base case |
| Explosive (>6%) | ~15%  | Breakout | +4-8% quarterly |

Overall expected system improvement: **+2-4% per quarter** in net return

## Next Steps

After implementing the realized volatility regime switch:

1. **Fee & Funding Model** - Add accurate exchange fee and funding rate modeling
2. **Performance Verification** - Run comprehensive backtests on all symbols
3. **Multi-Timeframe Integration** - Consider adding multi-timeframe confirmation 