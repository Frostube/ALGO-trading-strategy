# TradingView Advanced Scalping Strategy

This directory contains the PineScript implementation of our enhanced BTC/USDT scalping strategy for TradingView. This implementation includes all the advanced features from our Python codebase.

## How to Use

1. Go to [TradingView](https://www.tradingview.com/)
2. Open a chart for BTC/USDT futures (recommend BTCUSDT Perpetual on Binance)
3. Switch to the 1-minute timeframe
4. Click on "Pine Editor" at the bottom of the page
5. Copy and paste the entire content of `strategy.pine` into the editor
6. Click "Save" and then "Add to Chart"
7. Configure settings via the "Settings" button in the strategy panel
8. Click on "Strategy Tester" panel at the bottom to see performance statistics

## Key Advanced Features

### 1. Enhanced Entry Filters
- **Momentum Confirmation**: Requires price to break above/below previous high/low
- **Micro-trend Detection**: Uses EMA(50) slope to identify short-term trends
- **Consecutive Bars Agreement**: Requires multiple consecutive bars to confirm signals

### 2. Adaptive Thresholds
- **Percentile-based RSI Thresholds**: Dynamically adjusts based on recent market conditions
- **Dynamic Volume Spike Detection**: Adapts volume threshold to market liquidity

### 3. Overtrading Prevention
- **Minimum Bar Gap**: Enforces minimum bars between trades (default: 5)
- **Hourly Trade Limits**: Caps trades per hour (default: 3)
- **Consecutive Bar Agreement**: Requires multiple bars confirming the same signal

### 4. Two-Leg Stop Strategy
- **Initial ATR-based Stop**: First leg uses volatility-based stop loss
- **Trailing Stop**: Second leg activates at 0.15% profit, trails with ATR multiplier
- **Visual Indicators**: Shows trailing stop level on chart

### 5. Advanced Visualization
- **Adaptive Threshold Display**: Shows current dynamic thresholds
- **Market Information Panel**: Displays all strategy indicators
- **Trade Control Panel**: Shows bars since last signal and trades this hour

### 6. Enhanced Performance Tracking
- **Real-time Statistics Panel**: Displays key performance metrics during backtesting and paper trading
- **Direction Analysis**: Shows performance differences between long and short trades
- **Risk Metrics**: Risk-reward ratio, expectancy, and drawdown tracking
- **Consecutive Trade Tracking**: Monitors winning and losing streaks
- **Current Position Stats**: Real-time P&L and position information

## Strategy Parameters

### EMAs
- Fast EMA: 12 (default)
- Slow EMA: 26 (default)
- Trend EMA: 200 (default)
- Micro-Trend EMA: 50 (default)

### RSI & Volume
- RSI Length: 5 (default)
- RSI Oversold: 30 (default, or adaptive)
- RSI Overbought: 70 (default, or adaptive)
- Volume MA Period: 20 (default)
- Volume Threshold: 1.5× average (default, or adaptive)

### Risk Management
- ATR Period: 14 (default)
- ATR Stop Loss Multiplier: 1.5× (default)
- ATR Take Profit Multiplier: 3.0× (default)
- Fixed Stop Loss: 0.15% (fallback)
- Fixed Take Profit: 0.30% (fallback)

### Trailing Stop
- Trail Activation: 0.15% profit (default)
- Trail ATR Multiplier: 0.5× (default)

### Optimization Tips

For the best results:

1. **Entry Filters Optimization**:
   - Tune the momentum period (default: 5) to match the asset's volatility
   - Adjust consecutive bars requirement based on timeframe (higher for lower timeframes)

2. **Adaptive Settings**:
   - Try different lookback periods (default: 100) to balance responsiveness and stability
   - The 10/90 percentiles for RSI can be adjusted for different market regimes

3. **Stop Management**:
   - ATR multipliers should be tuned to the asset's volatility
   - Trail activation threshold depends on typical profit targets

4. **Overtrading Controls**:
   - Minimum bars between trades should be adjusted for timeframe
   - Hourly trade limits depend on your risk tolerance and strategy aggressiveness

## Performance Analysis

When analyzing the strategy performance in TradingView, pay attention to:

1. **Win Rate**: Target >50% with the enhanced filters
2. **Profit Factor**: Should be >1.5 with the two-leg stop strategy
3. **Average Trade Duration**: Should be longer with the trailing stop feature
4. **Maximum Drawdown**: Should be reduced with the stricter entry criteria
5. **Trades per Day**: Should be optimal with the overtrading prevention

## Troubleshooting

If you experience issues with the script:

1. Try updating to the latest TradingView Pine Script version
2. Ensure your chart has enough historical data loaded
3. For percentile functions, use PineScript v5 or later
4. If experiencing performance issues, try reducing the adaptive lookback period 