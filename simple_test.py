#!/usr/bin/env python3
"""
Simple test script to verify the backtest can generate trades
"""

# Standard imports
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Make sure we can import from src directory
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    # Modify config directly
    from src.config import SYMBOL, TIMEFRAME, EMA_FAST, EMA_SLOW
    import src.config as config
    
    # Set less restrictive parameters
    config.USE_ML_FILTER = False
    config.USE_TIME_FILTERS = False
    config.WEEKEND_TRADING = True
    config.AVOID_MIDNIGHT_HOURS = False
    config.RSI_LONG_THRESHOLD = 40
    config.RSI_SHORT_THRESHOLD = 60
    config.USE_ADAPTIVE_THRESHOLDS = False
    config.MIN_BARS_BETWEEN_TRADES = 1
    
    print("Successfully imported and modified config")
    
    # Create simple test function
    def generate_mock_data(periods=1000):
        """Generate mock OHLCV data for testing"""
        print("Generating mock OHLCV data...")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)
        dates = pd.date_range(start=start_date, end=end_date, periods=periods)
        
        # Generate prices with some randomness
        base_price = 30000  # Starting price for BTC
        np.random.seed(42)  # For reproducibility
        
        # Create price series with drift and volatility
        returns = np.random.normal(0, 0.002, periods).cumsum()
        price_series = base_price * (1 + returns)
        
        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['open'] = price_series
        data['close'] = price_series * (1 + np.random.normal(0, 0.001, periods))
        data['high'] = np.maximum(data['open'], data['close']) * (1 + np.abs(np.random.normal(0, 0.0015, periods)))
        data['low'] = np.minimum(data['open'], data['close']) * (1 - np.abs(np.random.normal(0, 0.0015, periods)))
        data['volume'] = np.random.normal(1000, 200, periods) * (1 + np.abs(returns) * 10)
        
        return data
    
    # Import technical indicators module
    from src.indicators.technical import apply_indicators, get_signal
    
    # Create mock data
    mock_data = generate_mock_data(1000)
    
    # Apply indicators
    print("Applying technical indicators...")
    data_with_indicators = apply_indicators(mock_data)
    
    # Count signals
    print("Checking for signals...")
    long_signals = 0
    short_signals = 0
    neutral_signals = 0
    
    # Check for signals
    check_bars = len(data_with_indicators)
    for i in range(check_bars):
        signal = get_signal(data_with_indicators, i)
        if signal['signal'] == 'buy':
            long_signals += 1
            print(f"Buy signal at bar {i} - RSI: {signal['rsi']:.1f}, Strategy: {signal.get('strategy', 'unknown')}")
        elif signal['signal'] == 'sell':
            short_signals += 1
            print(f"Sell signal at bar {i} - RSI: {signal['rsi']:.1f}, Strategy: {signal.get('strategy', 'unknown')}")
        else:
            neutral_signals += 1
    
    # Print summary
    print("\nSignal generation summary:")
    print(f"Long signals: {long_signals}")
    print(f"Short signals: {short_signals}")
    print(f"Neutral signals: {neutral_signals}")
    print(f"Total signals: {long_signals + short_signals} out of {check_bars} bars ({(long_signals + short_signals) / check_bars * 100:.1f}%)")
    
    # Try to run backtest with mock data
    if long_signals + short_signals > 0:
        print("\nRunning backtest with mock data...")
        from src.backtest.backtest import Backtester
        
        backtester = Backtester(data=data_with_indicators, initial_balance=10000)
        results = backtester.run()
        
        print("\nBacktest results:")
        print(f"Total trades: {results.get('total_trades', 0)}")
        print(f"Total return: {results.get('total_return', 0) * 100:.2f}%")
        print(f"Win rate: {results.get('win_rate', 0) * 100:.2f}%")
        print(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"Max drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
        
        # Print some sample trades
        trades = results.get('trades', [])
        if trades:
            print(f"\nSample trades (showing {min(5, len(trades))} of {len(trades)}):")
            for i, trade in enumerate(trades[:5]):
                print(f"{i+1}. {trade.get('side', 'unknown')} entry: ${trade.get('entry_price', 0):.2f}, "
                      f"exit: ${trade.get('exit_price', 0):.2f}, PnL: ${trade.get('pnl', 0):.2f}")
    
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
    
print("\nTest script completed.") 