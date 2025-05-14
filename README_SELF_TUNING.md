# Self-Tuning Trading Strategy

This document explains the self-tuning mechanism implemented in the strategy to keep it adaptive and maintain its edge over time.

## System Components

The self-tuning mechanism consists of four concentric safety nets:

1. **Rolling Parameter Re-fit**: Daily optimization to ensure parameters keep pace with changing market rhythms.
2. **Walk-forward OOS Check**: Validation to prevent overfitting parameters before they go live.
3. **Live Health Monitor**: Detection of edge degradation during trading with automatic pausing if needed.
4. **Regime Switch & Portfolio Guard**: Dynamic position sizing based on volatility and portfolio risk caps.

## How to Use

### 1. Daily Parameter Optimization

The system automatically optimizes strategy parameters every night using walk-forward validation:

```bash
# Run manually
python -m src.optimization.daily_optimizer --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT"

# With Windows Task Scheduler
scheduled_tasks.bat
```

This process:
- Runs a grid search over 18 months of data (550 days)
- Uses 80% of data for training, 20% for out-of-sample testing
- Only accepts parameter sets where training performance matches testing performance (±20%)
- Saves the top 3 parameter sets to `params/{symbol}_{timeframe}.json`
- The trading system loads the best parameter set at the start of each trading day

### 2. Health Monitoring

The health monitor tracks recent performance and automatically pauses trading if the edge degrades:

```bash
# Enable health monitoring when running the strategy
python run_adaptive_strategy.py --symbols "BTC/USDT,ETH/USDT" --health-monitor
```

The monitor:
- Tracks a rolling 40-trade window for each symbol/strategy
- Pauses trading for 24 hours if Profit Factor < 1.0 or Win Rate < 35%
- Sends alerts via Slack when trading is paused
- Automatically resumes trading after the pause period

### 3. Volatility-Targeted Position Sizing

The system automatically adjusts position sizes based on market volatility:

```python
# Position sizing calculation
def position_qty(account_equity, atr_usdt, risk_pct=0.0075):
    dollar_risk = account_equity * risk_pct
    return round(dollar_risk / atr_usdt, 3)  # contracts on USDT-margin
```

This approach:
- Uses smaller positions in high-volatility markets
- Uses larger positions in low-volatility markets
- Always targets the same dollar risk (0.75% of account)

### 4. Portfolio Risk Cap

The system enforces a global risk cap across all positions:

```python
# Before opening a new position
def can_open(new_risk):
    total_risk = sum(pos.open_risk for pos in portfolio.positions)
    return (total_risk + new_risk) <= 0.015 * account_equity
```

This ensures:
- Total portfolio risk never exceeds 1.5% of account equity
- New positions that would breach the cap are not opened
- Risk is properly distributed across multiple assets

## Adaptive Behavior by Regime

The strategy also adapts based on market volatility regimes:

| Regime | Volatility | Position Size | Pyramiding | Risk |
|--------|------------|--------------|------------|------|
| Low    | < 3%       | Half size    | Disabled   | 0.5× |
| Normal | 3-6%       | Normal size  | Enabled    | 0.75× |
| High   | > 6%       | Normal size  | Enabled    | 1.0× |

## Running the Adaptive Strategy

```bash
# Backtest mode
python run_adaptive_strategy.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" --mode backtest --optimize

# Live trading mode with all safety features
python run_adaptive_strategy.py --symbols "BTC/USDT,ETH/USDT" --mode live --health-monitor --risk-cap 0.015 --notifications
```

## Maintenance and Monitoring

The system logs all performance metrics and parameter changes in:

- `docs/performance_log.md`: Summary of strategy performance
- `logs/notifications.log`: Record of all alerts and pauses
- `params/{symbol}_{timeframe}.json`: Optimized parameter sets

## Benefits

This self-tuning system provides:
- **Adaptive Edge**: Parameters automatically adjust to changing market conditions
- **Protection**: Multiple safeguards prevent catastrophic losses
- **Consistency**: Performance stays within target ranges (PF ≥ 1.3, DD ≤ 15%)
- **Hands-off Operation**: Minimal manual intervention required 