#!/usr/bin/env python3
"""
Test script to verify signal generation with modified parameters
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add parent directories to path so imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
root_dir = os.path.dirname(parent_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Change working directory to root for relative imports
os.chdir(root_dir)

# Import from relative
from src.data.fetcher import DataFetcher
from src.indicators.technical import apply_indicators, get_signal
from src.config import SYMBOL, TIMEFRAME

# Modify config settings to be less restrictive
import src.config as config
config.USE_ML_FILTER = False
config.USE_TIME_FILTERS = False
config.WEEKEND_TRADING = True
config.AVOID_MIDNIGHT_HOURS = False
config.RSI_LONG_THRESHOLD = 35  # Higher value = easier to trigger
config.RSI_SHORT_THRESHOLD = 65  # Lower value = easier to trigger
config.USE_ADAPTIVE_THRESHOLDS = False  # Disable adaptive thresholds
config.MIN_BARS_BETWEEN_TRADES = 1  # Allow back-to-back signals

def test_signal_generation():
    """Test if we can generate signals with our modified parameters"""
    print(f"Testing signal generation for {SYMBOL}...")
    
    # Create data fetcher and get data
    data_fetcher = DataFetcher(use_testnet=True)
    df = data_fetcher.fetch_historical_data(days=5, timeframe=TIMEFRAME)
    
    if df is None or df.empty:
        print("Error: Failed to fetch data")
        return
    
    print(f"Fetched {len(df)} bars of data")
    
    # Apply indicators
    df = apply_indicators(df)
    
    # Count signals
    long_signals = 0
    short_signals = 0
    neutral_signals = 0
    
    # Check the last 100 bars for signals
    check_bars = min(100, len(df))
    
    print(f"\nTesting last {check_bars} bars for signals...")
    
    for i in range(-check_bars, 0):
        signal = get_signal(df, i)
        
        if signal['signal'] == 'buy':
            long_signals += 1
            print(f"Buy signal at {df.index[i]} - RSI: {signal['rsi']:.1f}, Strategy: {signal.get('strategy', 'unknown')}")
        elif signal['signal'] == 'sell':
            short_signals += 1
            print(f"Sell signal at {df.index[i]} - RSI: {signal['rsi']:.1f}, Strategy: {signal.get('strategy', 'unknown')}")
        else:
            neutral_signals += 1
    
    print("\nSignal generation summary:")
    print(f"Long signals: {long_signals}")
    print(f"Short signals: {short_signals}")
    print(f"Neutral signals: {neutral_signals}")
    print(f"Total signals: {long_signals + short_signals} / {check_bars} bars ({(long_signals + short_signals) / check_bars * 100:.1f}%)")
    
    # Plot indicators for visual verification
    if long_signals + short_signals > 0:
        print("\nPlotting some indicators and signals for verification...")
        
        # Get a subset of data for plotting
        plot_df = df.iloc[-check_bars:]
        
        # Create figure and axis
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price
        ax[0].plot(plot_df.index, plot_df['close'], label='Price')
        
        # Plot EMAs if available
        if 'ema_9' in plot_df.columns:
            ax[0].plot(plot_df.index, plot_df['ema_9'], label='EMA9')
        if 'ema_21' in plot_df.columns:
            ax[0].plot(plot_df.index, plot_df['ema_21'], label='EMA21')
        
        # Plot Hull MAs if available
        if 'hma_9' in plot_df.columns:
            ax[0].plot(plot_df.index, plot_df['hma_9'], label='HMA9')
        
        # Mark signals
        buy_signals = []
        sell_signals = []
        
        for i in range(len(plot_df)):
            signal = get_signal(plot_df, i)
            if signal['signal'] == 'buy':
                buy_signals.append(i)
            elif signal['signal'] == 'sell':
                sell_signals.append(i)
        
        # Plot signal markers
        if buy_signals:
            ax[0].scatter(
                [plot_df.index[i] for i in buy_signals],
                [plot_df['close'].iloc[i] * 0.998 for i in buy_signals],
                marker='^', color='green', s=100, label='Buy Signals'
            )
            
        if sell_signals:
            ax[0].scatter(
                [plot_df.index[i] for i in sell_signals],
                [plot_df['close'].iloc[i] * 1.002 for i in sell_signals],
                marker='v', color='red', s=100, label='Sell Signals'
            )
        
        # Plot RSI in subplot
        ax[1].plot(plot_df.index, plot_df['rsi'], label='RSI', color='purple')
        ax[1].axhline(y=30, color='green', linestyle='-', alpha=0.5)
        ax[1].axhline(y=70, color='red', linestyle='-', alpha=0.5)
        ax[1].set_ylim(0, 100)
        ax[1].set_title('RSI')
        
        # Set labels and title
        ax[0].set_title(f'{SYMBOL} Price and Signals')
        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Price')
        ax[0].legend()
        
        # Save the plot
        save_path = os.path.join(root_dir, "signal_test_plot.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    print("\nSignal test completed.")

if __name__ == "__main__":
    test_signal_generation() 