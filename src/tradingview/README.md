# TradingView Strategy Implementation

This directory contains a Pine Script implementation of the BTC/USDT EMA+RSI+Volume Scalping Strategy for use on TradingView.

## How to Use

1. Go to [TradingView](https://www.tradingview.com/)
2. Open a chart for BTC/USDT futures (recommend BTCUSDT Perpetual on Binance)
3. Switch to the 1-minute timeframe
4. Click on "Pine Editor" at the bottom of the page
5. Copy and paste the entire content of `strategy.pine` into the editor
6. Click "Save" and then "Add to Chart"
7. Click on "Strategy Tester" panel at the bottom to see performance statistics

## Strategy Settings

The strategy is configurable through the TradingView settings panel:

- **EMA Parameters**: Fast (9) and Slow (21) periods
- **RSI Parameters**: Length (2), Oversold (10), Overbought (90) levels
- **Volume Parameters**: MA period (20), Spike threshold (1.5)
- **Risk Management**: Stop-loss (0.15%), Take-profit (0.30%)
- **Trailing Stop**: Toggle on/off, Activation % (0.15%), Trail distance % (0.10%)

## Testing Tips

1. **Historical Performance**:
   - Test over different market conditions (bull, bear, sideways)
   - Pay attention to drawdowns and consistency

2. **Optimization**:
   - Use TradingView's "Strategy Tester" to optimize parameters
   - Avoid overfitting by testing on different time periods

3. **Paper Trading**:
   - TradingView offers paper trading on the strategy
   - Click "Paper Trading" button when the strategy is active on the chart

4. **Comparison to Python Version**:
   - Results should be similar to the Python backtest
   - Some differences may occur due to TradingView's data or execution model

## Key Performance Metrics to Monitor

- **Win Rate**: Aim for >50%
- **Profit Factor**: Target >1.5
- **Max Drawdown**: Should be less than 15% of account
- **Average Trade Duration**: Should be 3-5 bars as expected

## Adjusting for Different Market Conditions

- In high volatility: Increase stop-loss and take-profit percentages
- In low volatility: Consider reducing the RSI thresholds
- In strong trends: The strategy may generate false signals, consider using a trend filter 