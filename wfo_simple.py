#!/usr/bin/env python3
"""
Simplified Walk-Forward Optimization for High-Leverage Strategy

This version eliminates the problematic EMA optimization and focuses on basic strategy evaluation.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json

from src.strategy.high_leverage_strategy import HighLeverageStrategy

def load_data_from_file(filepath):
    """Load OHLCV data from a JSON file."""
    print(f"Loading data from {filepath}")
    try:
        # Load the JSON data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
        
        print(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_metrics(trades):
    """Calculate performance metrics from a list of trades."""
    if not trades or len(trades) == 0:
        return {
            'total_return': 0.0,
            'roi': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0
        }
    
    # Extract profits and calculate cumulative returns
    profits = [trade['profit_pct'] for trade in trades]
    cum_returns = np.cumsum(profits)
    
    # Calculate metrics
    total_return = cum_returns[-1] if len(cum_returns) > 0 else 0
    win_rate = sum(1 for p in profits if p > 0) / len(profits) if profits else 0
    
    # Profit factor
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss = abs(sum(p for p in profits if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Maximum drawdown
    peak = 0
    max_dd = 0
    for ret in cum_returns:
        peak = max(peak, ret)
        dd = (peak - ret) / (1 + peak) if peak > 0 else 0
        max_dd = max(max_dd, dd)
    
    # Sharpe ratio approximation (simplified)
    sharpe = 0
    if len(profits) > 1:
        returns_mean = np.mean(profits)
        returns_std = np.std(profits)
        sharpe = (returns_mean / returns_std) * np.sqrt(len(profits)) if returns_std > 0 else 0
    
    return {
        'total_return': total_return,
        'roi': total_return * 100,  # Convert to percentage
        'win_rate': win_rate * 100,  # Convert to percentage
        'profit_factor': profit_factor,
        'max_drawdown': -max_dd * 100,  # Convert to percentage and make negative
        'sharpe_ratio': sharpe,
        'total_trades': len(trades)
    }

def run_full_backtest():
    """Run a simple full backtest on the 4h data to test strategy functionality."""
    # Parameters
    data_file = 'data/BTC_USDT_4h_366d.json'
    
    # Load data
    print(f"Loading data for full backtest")
    df = load_data_from_file(data_file)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create strategy with fixed parameters - no optimization
    strategy = HighLeverageStrategy(
        # Base parameters - using fixed values to avoid optimization
        fast_ema=8,
        slow_ema=21,
        trend_ema=55,
        risk_per_trade=0.02,
        
        # Signal filters
        use_mtf_filter=True,
        mtf_signal_mode='any',
        
        use_momentum_filter=True,
        momentum_threshold=30,  # Made more permissive
        
        use_pattern_filter=True,
        pattern_strictness='loose',  # Made more permissive
        
        use_volume_filter=True,
        volume_threshold=1.2,
        
        # Exit strategy
        use_trailing_stop=True,
        take_profit_r=2.0,
        use_partial_exit=True,
        max_hold_periods=15
    )
    
    # Run backtest
    print("Running backtest on entire dataset...")
    results = strategy.backtest(df)
    
    # Calculate metrics
    metrics = calculate_metrics(results.get('trades', []))
    
    # Print results
    print("\nBacktest Results:")
    print(f"Total Return: {metrics['roi']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
    # Save trades to CSV
    trades_df = pd.DataFrame(results.get('trades', []))
    if not trades_df.empty:
        trades_df.to_csv('results/full_backtest_trades.csv', index=False)
        print("Trades saved to results/full_backtest_trades.csv")
        
        # Print sample trades
        print("\nSample Trades:")
        print(trades_df.head(5))
    else:
        print("\nNo trades were generated in the backtest.")

def analyze_trading_filters():
    """Analyze why the strategy might not be generating trades."""
    # Parameters
    data_file = 'data/BTC_USDT_4h_366d.json'
    
    # Load data
    print("Loading data for filter analysis")
    df = load_data_from_file(data_file)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Create the most permissive strategy settings
    strategy = HighLeverageStrategy(
        # Base parameters
        fast_ema=8,
        slow_ema=21,
        trend_ema=55,
        risk_per_trade=0.02,
        
        # Make all filters extremely permissive
        use_mtf_filter=False,        # Turn off MTF filter
        
        use_momentum_filter=False,   # Turn off momentum filter
        
        use_pattern_filter=False,    # Turn off pattern filter
        
        use_volume_filter=False,     # Turn off volume filter
        
        # Make exits very generous
        use_trailing_stop=True,
        take_profit_r=1.5,           # Lower take profit
        use_partial_exit=True,
        max_hold_periods=20,         # Longer hold time
        
        # Minimum confirmations
        min_confirmations=0.0        # Require zero confirmations
    )
    
    # Run backtest
    print("Running backtest with minimally restrictive filters...")
    results = strategy.backtest(df)
    
    # Calculate metrics
    metrics = calculate_metrics(results.get('trades', []))
    
    # Print results
    print("\nMinimally Restrictive Backtest Results:")
    print(f"Total Return: {metrics['roi']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")
    
    if metrics['total_trades'] == 0:
        print("\nSTRATEGY DIAGNOSIS:")
        print("No trades were generated even with the most permissive settings!")
        print("This indicates a fundamental issue with either:")
        print("1. The underlying OHLCV data (e.g., simulated/future data)")
        print("2. The signal generation mechanism in the strategy")
        print("3. A bug in the strategy implementation")
    
        # Check for crossovers directly in the data
        print("\nAnalyzing for potential crossovers in data:")
        
        # Calculate simple EMAs
        df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()
        
        # Check for potential crossovers (basic check)
        df['potential_crossover'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) | \
                                   (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
        
        crossover_count = df['potential_crossover'].sum()
        print(f"Detected {crossover_count} simple EMA crossovers in the data")
        
        if crossover_count > 0:
            print("Crossovers exist, but the strategy isn't trading them.")
            print("This suggests issues in the signal implementation or filter layers.")
        else:
            print("No crossovers detected! The data might be simulated/manipulated.")

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("=" * 50)
    print("RUNNING FULL BACKTEST")
    print("=" * 50)
    run_full_backtest()
    
    print("\n" + "=" * 50)
    print("ANALYZING TRADING FILTERS")
    print("=" * 50)
    analyze_trading_filters() 