#!/usr/bin/env python3
"""
EMA Crossover Signal Generator

Generates basic EMA crossover signals and visualizes them on a chart.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import data fetcher
from src.data.fetcher import DataFetcher
from src.simulation.market_conditions import MarketConditionDetector, MarketCondition

def generate_ema_crossover_signals(df, fast_ema=8, slow_ema=21, trend_ema=50):
    """
    Generate basic EMA crossover signals.
    
    Args:
        df: OHLCV DataFrame
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        trend_ema: Trend EMA period
        
    Returns:
        DataFrame with signals
    """
    # Calculate EMAs
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast_ema, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_ema, adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=trend_ema, adjust=False).mean()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Calculate crossover signals
    for i in range(1, len(df)):
        # Bullish crossover: Fast EMA crosses above Slow EMA
        if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
            df.loc[df.index[i], 'signal'] = 1
            
        # Bearish crossover: Fast EMA crosses below Slow EMA
        elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
            df.loc[df.index[i], 'signal'] = -1
    
    return df

def visualize_signals(df, symbol, timeframe):
    """
    Create a visualization of price and signals.
    
    Args:
        df: DataFrame with OHLCV data and signals
        symbol: Trading symbol
        timeframe: Timeframe string
    """
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Plot price and EMAs
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['close'], label='Price', color='black', alpha=0.5)
    plt.plot(df.index, df['ema_fast'], label=f'Fast EMA ({df["ema_fast"].name})', color='blue')
    plt.plot(df.index, df['ema_slow'], label=f'Slow EMA ({df["ema_slow"].name})', color='orange')
    plt.plot(df.index, df['ema_trend'], label=f'Trend EMA ({df["ema_trend"].name})', color='red')
    
    # Plot buy signals
    buy_signals = df[df['signal'] == 1]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    
    # Plot sell signals
    sell_signals = df[df['signal'] == -1]
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    # Configure plot
    plt.title(f'{symbol} {timeframe} - EMA Crossover Signals')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot market conditions
    plt.subplot(2, 1, 2)
    
    # Detect market conditions
    detector = MarketConditionDetector()
    market_conditions = detector.detect_condition(df)
    
    # Create a numerical mapping for market conditions
    condition_map = {}
    for i, condition in enumerate(MarketCondition):
        condition_map[condition] = i
        
    # Convert condition enum to numerical value for plotting
    condition_values = [condition_map[condition] for condition in market_conditions]
    
    # Create colormap
    colors = {
        MarketCondition.STRONG_BULL: 'darkgreen',
        MarketCondition.NORMAL_BULL: 'green',
        MarketCondition.WEAK_BULL: 'lightgreen',
        MarketCondition.SIDEWAYS: 'gray',
        MarketCondition.WEAK_BEAR: 'pink',
        MarketCondition.NORMAL_BEAR: 'red',
        MarketCondition.STRONG_BEAR: 'darkred',
        MarketCondition.VOLATILE_UPTREND: 'turquoise',
        MarketCondition.VOLATILE_DOWNTREND: 'purple',
        MarketCondition.UNKNOWN: 'blue'
    }
    
    # Plot market conditions
    for i in range(len(df)):
        condition = market_conditions[i]
        plt.bar(df.index[i], 1, width=0.8, color=colors[condition], align='center')
    
    # Create custom legend for market conditions
    import matplotlib.patches as mpatches
    patches = []
    for condition, color in colors.items():
        patches.append(mpatches.Patch(color=color, label=condition.value))
    plt.legend(handles=patches, loc='upper left', fontsize=8)
    
    # Configure plot
    plt.title('Market Conditions')
    plt.ylabel('Condition')
    plt.ylim(0, 1)
    plt.grid(False)
    
    # Configure figure
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    # Save figure
    filename = f'charts/{symbol.replace("/", "_")}_{timeframe}_signals.png'
    plt.savefig(filename)
    print(f'Chart saved to {filename}')
    
    # Display statistics
    print(f"\nSignal Statistics:")
    print(f"Total Buy Signals: {len(buy_signals)}")
    print(f"Total Sell Signals: {len(sell_signals)}")
    
    # Analyze signals by market condition
    condition_stats = {}
    
    for i, row in df.iterrows():
        if row['signal'] != 0:
            idx = df.index.get_loc(i)
            condition = market_conditions[idx].value
            
            if condition not in condition_stats:
                condition_stats[condition] = {'buy': 0, 'sell': 0}
                
            if row['signal'] == 1:
                condition_stats[condition]['buy'] += 1
            elif row['signal'] == -1:
                condition_stats[condition]['sell'] += 1
    
    # Print market condition analysis
    print("\nSignals by Market Condition:")
    print("-" * 60)
    print(f"{'Condition':<20} | {'Buy':<5} | {'Sell':<5} | {'Total':<5}")
    print("-" * 60)
    
    for condition, stats in condition_stats.items():
        total = stats['buy'] + stats['sell']
        print(f"{condition:<20} | {stats['buy']:<5} | {stats['sell']:<5} | {total:<5}")
    
    # Market condition distribution
    condition_counts = {}
    for condition in market_conditions:
        if condition.value not in condition_counts:
            condition_counts[condition.value] = 0
        condition_counts[condition.value] += 1
    
    print("\nMarket Condition Distribution:")
    print("-" * 60)
    print(f"{'Condition':<20} | {'Count':<6} | {'Percentage':<10}")
    print("-" * 60)
    
    for condition, count in condition_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{condition:<20} | {count:<6} | {percentage:<10.2f}%")

def main(symbol='BTC/USDT', timeframe='4h', days=90, fast_ema=8, slow_ema=21, trend_ema=50):
    """
    Main function to generate and visualize signals.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe to analyze
        days: Number of days to look back
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        trend_ema: Trend EMA period
    """
    print(f"Generating EMA crossover signals for {symbol} ({timeframe})")
    print(f"Parameters: Fast EMA = {fast_ema}, Slow EMA = {slow_ema}, Trend EMA = {trend_ema}")
    print(f"Timeframe: {timeframe}, Lookback: {days} days")
    
    # Create data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch historical data
    data = data_fetcher.fetch_historical_data(
        symbol=symbol, 
        days=days, 
        timeframe=timeframe
    )
    
    if data is None or data.empty:
        print(f"ERROR: Failed to fetch data for {symbol} ({timeframe})")
        return
    
    print(f"Received {len(data)} candles from {data.index[0]} to {data.index[-1]}")
    
    # Generate signals
    signals = generate_ema_crossover_signals(data, fast_ema, slow_ema, trend_ema)
    
    # Visualize signals
    visualize_signals(signals, symbol, timeframe)
    
    # Test output
    save_test_data(signals, symbol, timeframe)

def save_test_data(signals, symbol, timeframe):
    """Save a sample of signal data for testing"""
    # Create a smaller sample of the data with signals
    signal_data = signals[signals['signal'] != 0]
    
    if len(signal_data) > 10:
        signal_data = signal_data.tail(10)
    
    # Convert to dictionary for JSON serialization
    sample_data = []
    
    for idx, row in signal_data.iterrows():
        sample_data.append({
            'timestamp': idx.strftime('%Y-%m-%d %H:%M:%S') if isinstance(idx, pd.Timestamp) else str(idx),
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume']),
            'signal': int(row['signal']),
            'ema_fast': float(row['ema_fast']),
            'ema_slow': float(row['ema_slow']),
            'ema_trend': float(row['ema_trend'])
        })
    
    # Save to file
    os.makedirs('tests', exist_ok=True)
    filename = f'tests/{symbol.replace("/", "_")}_{timeframe}_signals.json'
    
    with open(filename, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Sample signal data saved to {filename}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate and visualize EMA crossover signals")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="4h", help="Timeframe to analyze")
    parser.add_argument("--days", type=int, default=90, help="Number of days to look back")
    parser.add_argument("--fast-ema", type=int, default=8, help="Fast EMA period")
    parser.add_argument("--slow-ema", type=int, default=21, help="Slow EMA period")
    parser.add_argument("--trend-ema", type=int, default=50, help="Trend EMA period")
    
    args = parser.parse_args()
    
    main(args.symbol, args.timeframe, args.days, args.fast_ema, args.slow_ema, args.trend_ema) 