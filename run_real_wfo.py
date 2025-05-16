#!/usr/bin/env python3
"""
Walk-Forward Optimization Script for High-Leverage Strategy Using Real Historical Data

This script loads real historical BTC/USDT data and performs walk-forward testing
with the high-leverage strategy using reasonable parameters.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import json

from src.strategy.high_leverage_strategy import HighLeverageStrategy

def print_separator():
    print("\n" + "-" * 70 + "\n")

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

def generate_windows(df, train_days=60, test_days=30):
    """Generate walk-forward windows."""
    windows = []
    end_date = df.index.max()
    current_test_end = end_date
    
    while True:
        # Calculate window dates
        test_start = current_test_end - timedelta(days=test_days)
        train_end = test_start - timedelta(days=1)  # 1 day before test start
        train_start = train_end - timedelta(days=train_days)
        
        # Create window if we have enough data
        if train_start >= df.index.min():
            window = {
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': current_test_end
            }
            windows.append(window)
            
            # Move to next window
            current_test_end = train_start - timedelta(days=1)
        else:
            break
            
        # Limit to 6 windows
        if len(windows) >= 6:
            break
    
    # Print window info
    for i, w in enumerate(windows):
        print(f"Window {i+1}: Train {w['train_start'].strftime('%Y-%m-%d')}–{w['train_end'].strftime('%Y-%m-%d')}, "
              f"Test {w['test_start'].strftime('%Y-%m-%d')}–{w['test_end'].strftime('%Y-%m-%d')}")
    
    return windows

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

def run_wfo():
    # Parameters
    data_file = 'data/BTC_USDT_4h_366d.json'
    output_csv = 'results/real_wfo_results.csv'
    output_plot = 'results/real_wfo_plot.png'
    
    # Step 1: Load data
    print_separator()
    print("STEP 1: LOADING HISTORICAL DATA")
    print_separator()
    
    df = load_data_from_file(data_file)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Step 2: Generate windows
    print_separator()
    print("STEP 2: GENERATING WALK-FORWARD WINDOWS")
    print_separator()
    
    windows = generate_windows(df, train_days=60, test_days=30)
    
    # Step 3: Run walk-forward optimization
    print_separator()
    print("STEP 3: RUNNING WALK-FORWARD TESTS")
    print_separator()
    
    results = []
    
    for i, window in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}")
        
        # Get data slices
        train_df = df.loc[window['train_start']:window['train_end']]
        test_df = df.loc[window['test_start']:window['test_end']]
        
        print(f"Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        # Create strategy with reasonable parameters
        strategy = HighLeverageStrategy(
            # Base parameters
            fast_ema=8,  # Short-term EMA
            slow_ema=21,  # Medium-term EMA
            trend_ema=55,  # Longer-term trend filter
            risk_per_trade=0.02,  # 2% risk per trade
            
            # Signal filters with reasonable settings
            use_mtf_filter=True,
            mtf_signal_mode='any',  # More permissive
            
            use_momentum_filter=True,
            momentum_threshold=35,
            
            use_pattern_filter=True,
            pattern_strictness='medium',
            
            use_volume_filter=True,
            volume_threshold=1.25,
            
            # Exit strategy
            use_trailing_stop=True,
            take_profit_r=2.0,
            use_partial_exit=True,
            max_hold_periods=15
        )
        
        # Run backtest on training data
        print("Running backtest on training data...")
        train_results = strategy.backtest(train_df)
        
        # Run backtest on test data
        print("Running backtest on test data...")
        test_results = strategy.backtest(test_df)
        
        # Calculate metrics
        train_metrics = calculate_metrics(train_results.get('trades', []))
        test_metrics = calculate_metrics(test_results.get('trades', []))
        
        # Add window info
        result = {
            'window_id': i+1,
            'train_start': window['train_start'].strftime('%Y-%m-%d'),
            'train_end': window['train_end'].strftime('%Y-%m-%d'),
            'test_start': window['test_start'].strftime('%Y-%m-%d'),
            'test_end': window['test_end'].strftime('%Y-%m-%d'),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_roi': train_metrics['roi'],
            'train_trades': train_metrics['total_trades'],
            'train_win_rate': train_metrics['win_rate'],
            'train_sharpe': train_metrics['sharpe_ratio'],
            'train_drawdown': train_metrics['max_drawdown'],
            'test_roi': test_metrics['roi'],
            'test_trades': test_metrics['total_trades'],
            'test_win_rate': test_metrics['win_rate'],
            'test_sharpe': test_metrics['sharpe_ratio'],
            'test_drawdown': test_metrics['max_drawdown'],
            'test_profit_factor': test_metrics['profit_factor']
        }
        
        print(f"Window {i+1} results:")
        print(f"  Train ROI: {train_metrics['roi']:.2f}% ({train_metrics['total_trades']} trades)")
        print(f"  Test ROI: {test_metrics['roi']:.2f}% ({test_metrics['total_trades']} trades)")
        print(f"  Test Win Rate: {test_metrics['win_rate']:.2f}%")
        print(f"  Test Profit Factor: {test_metrics['profit_factor']:.2f}")
        
        results.append(result)
    
    # Step 4: Save results
    print_separator()
    print("STEP 4: SAVING RESULTS")
    print_separator()
    
    # Create DataFrame from results
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Create summary plot
    plt.figure(figsize=(12, 10))
    
    # Plot ROI for training and test
    plt.subplot(2, 2, 1)
    plt.bar([r-0.2 for r in range(1, len(results)+1)], results_df['train_roi'], width=0.4, label='Train', color='blue')
    plt.bar([r+0.2 for r in range(1, len(results)+1)], results_df['test_roi'], width=0.4, label='Test', color='green')
    plt.axhline(y=results_df['test_roi'].mean(), color='red', linestyle='--', label=f'Avg Test: {results_df["test_roi"].mean():.2f}%')
    plt.title('Return on Investment by Window')
    plt.xlabel('Window')
    plt.ylabel('ROI (%)')
    plt.legend()
    
    # Plot win rate
    plt.subplot(2, 2, 2)
    plt.bar(range(1, len(results)+1), results_df['test_win_rate'], color='purple')
    plt.axhline(y=results_df['test_win_rate'].mean(), color='red', linestyle='--', 
                label=f'Avg: {results_df["test_win_rate"].mean():.2f}%')
    plt.title('Win Rate by Window (Test)')
    plt.xlabel('Window')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    
    # Plot number of trades
    plt.subplot(2, 2, 3)
    plt.bar(range(1, len(results)+1), results_df['test_trades'], color='orange')
    plt.axhline(y=results_df['test_trades'].mean(), color='red', linestyle='--', 
                label=f'Avg: {results_df["test_trades"].mean():.1f}')
    plt.title('Number of Trades by Window (Test)')
    plt.xlabel('Window')
    plt.ylabel('Trades')
    plt.legend()
    
    # Plot profit factor
    plt.subplot(2, 2, 4)
    plt.bar(range(1, len(results)+1), results_df['test_profit_factor'].clip(0, 5), color='green')
    plt.axhline(y=min(5, results_df['test_profit_factor'].mean()), color='red', linestyle='--', 
                label=f'Avg: {results_df["test_profit_factor"].mean():.2f}')
    plt.title('Profit Factor by Window (Test)')
    plt.xlabel('Window')
    plt.ylabel('Profit Factor (capped at 5)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Step 5: Show summary statistics
    print_separator()
    print("STEP 5: SUMMARY STATISTICS")
    print_separator()
    
    print(f"Average Test ROI: {results_df['test_roi'].mean():.2f}%")
    print(f"Average Test Win Rate: {results_df['test_win_rate'].mean():.2f}%")
    print(f"Average Test Trades: {results_df['test_trades'].mean():.1f}")
    print(f"Average Test Profit Factor: {results_df['test_profit_factor'].mean():.2f}")
    print(f"Percentage of Profitable Windows: {(results_df['test_roi'] > 0).mean() * 100:.1f}%")
    print(f"Average Test Drawdown: {results_df['test_drawdown'].mean():.2f}%")

if __name__ == '__main__':
    run_wfo() 