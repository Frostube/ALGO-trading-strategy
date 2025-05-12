#!/usr/bin/env python3
"""
Diagnostic script to check data and indicators for the BTC/USDT scalping strategy.
This script doesn't rely on the backtesting engine or dashboard, just the basic data
fetching and indicator calculation.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Script started")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Try importing modules one by one
try:
    print("Importing DataFetcher...")
    from src.data.fetcher import DataFetcher
    print("DataFetcher imported successfully")
except Exception as e:
    print(f"Error importing DataFetcher: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing apply_indicators...")
    from src.indicators.technical import apply_indicators
    print("apply_indicators imported successfully")
except Exception as e:
    print(f"Error importing apply_indicators: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("Importing config...")
    from src.config import (
        SYMBOL, EMA_FAST, EMA_SLOW, RSI_PERIOD, 
        RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD, VOLUME_PERIOD
    )
    print("Config imported successfully")
    print(f"SYMBOL: {SYMBOL}")
    print(f"EMA_FAST: {EMA_FAST}")
    print(f"EMA_SLOW: {EMA_SLOW}")
except Exception as e:
    print(f"Error importing config: {e}")
    traceback.print_exc()
    sys.exit(1)

def main():
    """Check data and indicators."""
    print(f"\nChecking data and indicators for {SYMBOL} strategy...")
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize DataFetcher
    print("Initializing DataFetcher...")
    try:
        fetcher = DataFetcher()
        print("DataFetcher initialized")
    except Exception as e:
        print(f"Error initializing DataFetcher: {e}")
        traceback.print_exc()
        return
    
    try:
        # Fetch a small amount of data (3 days)
        days = 3
        print(f"\nFetching {days} days of historical data...")
        data = fetcher.fetch_historical_data(days=days)
        
        if data is None or len(data) == 0:
            print("Error: No data returned from data fetcher")
            return
        
        print(f"Successfully fetched {len(data)} data points")
        print(f"Data range: {data.index[0]} to {data.index[-1]}")
        
        # Display data sample
        print("\nData sample (first 5 rows):")
        print(data.head())
        
        # Apply indicators
        print("\nCalculating indicators...")
        data_with_indicators = apply_indicators(data)
        
        # Display indicators sample
        print("\nData with indicators sample (first 5 rows):")
        print(data_with_indicators.head())
        
        # Count trade signals
        long_signals = data_with_indicators[(data_with_indicators['ema_trend'] > 0) & 
                                           (data_with_indicators['rsi'] < RSI_LONG_THRESHOLD) &
                                           (data_with_indicators['volume_spike'])]
        
        short_signals = data_with_indicators[(data_with_indicators['ema_trend'] < 0) & 
                                            (data_with_indicators['rsi'] > RSI_SHORT_THRESHOLD) &
                                            (data_with_indicators['volume_spike'])]
        
        print(f"\nFound {len(long_signals)} potential long signals and {len(short_signals)} potential short signals")
        
        # Save a sample to CSV for inspection
        sample_path = 'data/indicator_sample.csv'
        data_with_indicators.head(100).to_csv(sample_path)
        print(f"\nSaved 100 rows of indicator data to {sample_path}")
        
        # Create a simple plot of price and indicators
        print("\nCreating plot of price and indicators...")
        plot_data(data_with_indicators.iloc[-200:])
        
        print("\nDiagnostic check completed successfully!")
    
    except Exception as e:
        print(f"Error during data check: {str(e)}")
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            print("Closing DataFetcher...")
            fetcher.close()
            print("DataFetcher closed")
        except Exception as e:
            print(f"Error closing DataFetcher: {e}")
            traceback.print_exc()

def plot_data(data):
    """Create a simple plot of price and indicators."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot price and EMAs
    ax1.plot(data.index, data['close'], label='Close Price')
    if 'ema_fast' in data.columns:
        ax1.plot(data.index, data['ema_fast'], label=f'EMA {EMA_FAST}')
    if 'ema_slow' in data.columns:
        ax1.plot(data.index, data['ema_slow'], label=f'EMA {EMA_SLOW}')
    
    ax1.set_title(f'{SYMBOL} Price and EMAs')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot RSI
    if 'rsi' in data.columns:
        ax2.plot(data.index, data['rsi'], color='purple')
        ax2.axhline(y=RSI_LONG_THRESHOLD, color='green', linestyle='--')
        ax2.axhline(y=RSI_SHORT_THRESHOLD, color='red', linestyle='--')
        ax2.set_title(f'RSI ({RSI_PERIOD})')
        ax2.set_ylabel('RSI')
        ax2.grid(True)
    
    # Plot volume
    ax3.bar(data.index, data['volume'], color='blue', alpha=0.5)
    if 'volume_ma' in data.columns:
        ax3.plot(data.index, data['volume_ma'], color='orange', label=f'Volume MA ({VOLUME_PERIOD})')
    
    # Mark volume spikes
    if 'volume_spike' in data.columns:
        spikes = data[data['volume_spike']]
        if len(spikes) > 0:
            ax3.scatter(spikes.index, spikes['volume'], color='red', marker='^', s=50, label='Volume Spike')
    
    ax3.set_title('Volume')
    ax3.set_ylabel('Volume')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('data/indicator_plot.png')
    print(f"Saved plot to data/indicator_plot.png")

if __name__ == "__main__":
    main() 