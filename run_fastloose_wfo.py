#!/usr/bin/env python3
"""
Walk-Forward Optimization Script for Fast & Loose High-Leverage Strategy

This script runs a 6-window walk-forward optimization on the high-leverage strategy
using the "Fast & Loose" configuration to verify robustness across different market periods.
"""

import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

from src.utils.window_generator import generate_windows
from src.data.fetcher import fetch_ohlcv
from src.strategy.high_leverage_strategy import HighLeverageStrategy
from src.utils.metrics import calculate_metrics

def plot_wfo_results(results, output_path):
    """Generate visualization of WFO results."""
    plt.figure(figsize=(12, 10))
    
    # Plot ROI by window
    plt.subplot(2, 2, 1)
    plt.bar(results['window_id'], results['roi'], color='green')
    plt.axhline(y=results['roi'].mean(), color='r', linestyle='--', label=f'Avg: {results["roi"].mean():.2f}%')
    plt.title('Return on Investment by Window')
    plt.xlabel('Window ID')
    plt.ylabel('ROI (%)')
    plt.legend()
    
    # Plot Sharpe ratio by window
    plt.subplot(2, 2, 2)
    plt.bar(results['window_id'], results['sharpe_ratio'], color='blue')
    plt.axhline(y=results['sharpe_ratio'].mean(), color='r', linestyle='--', 
                label=f'Avg: {results["sharpe_ratio"].mean():.2f}')
    plt.title('Sharpe Ratio by Window')
    plt.xlabel('Window ID')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    
    # Plot Max Drawdown by window
    plt.subplot(2, 2, 3)
    plt.bar(results['window_id'], results['max_drawdown'], color='red')
    plt.axhline(y=results['max_drawdown'].mean(), color='black', linestyle='--', 
                label=f'Avg: {results["max_drawdown"].mean():.2f}%')
    plt.title('Maximum Drawdown by Window')
    plt.xlabel('Window ID')
    plt.ylabel('Max Drawdown (%)')
    plt.legend()
    
    # Plot Win Rate by window
    plt.subplot(2, 2, 4)
    plt.bar(results['window_id'], results['win_rate'], color='purple')
    plt.axhline(y=results['win_rate'].mean(), color='black', linestyle='--', 
                label=f'Avg: {results["win_rate"].mean():.2f}%')
    plt.title('Win Rate by Window')
    plt.xlabel('Window ID')
    plt.ylabel('Win Rate (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"WFO visualization saved to {output_path}")

def load_historical_data(symbol, timeframe, days):
    """
    Load historical OHLCV data from cache or fetch from exchange.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for analysis (e.g., '4h')
        days: Number of days of history to load
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Loading {days} days of {timeframe} data for {symbol}")
    
    # File format is symbol_timeframe_days.json
    # Use cached data if available, otherwise fetch
    df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
    
    print(f"Loaded data from {df.index[0]} to {df.index[-1]} ({len(df)} samples)")
    
    # Check if data is too recent or in the future
    now = pd.Timestamp.now()
    if df.index[-1] > now:
        print("Warning: Data contains future dates, which suggests simulated data")
        
    return df

def run_fastloose_wfo(symbol, timeframe, days,
                     train_days, test_days, output_csv, plot_results=True):
    """
    Run walk-forward optimization for the Fast & Loose high-leverage strategy.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        timeframe: Timeframe for analysis (e.g., '4h')
        days: Total number of days of historical data to use
        train_days: Number of days in each training window
        test_days: Number of days in each test window
        output_csv: Output file path for results
        plot_results: Whether to generate visualization
    """
    print(f"Running WFO for {symbol} {timeframe}")
    print(f"Training window: {train_days} days, Test window: {test_days} days")
    
    # 1) Load historical data - use real historical data
    df = load_historical_data(symbol, timeframe, days)
    
    # 2) Create windows - generate non-overlapping windows
    windows = generate_windows(df, train_days=train_days, test_days=test_days)
    print(f"Generated {len(windows)} windows")
    
    # Limit to just 6 windows if we have more
    if len(windows) > 6:
        print(f"Limiting to 6 windows for WFO analysis")
        windows = windows[:6]
    
    results = []
    # 3) Loop windows
    for i, w in enumerate(windows):
        print(f"\nProcessing Window {i+1}/{len(windows)}")
        train_df = df.loc[w['train_start']:w['train_end']]
        test_df = df.loc[w['test_start']:w['test_end']]
        
        print(f"Train period: {w['train_start']} to {w['train_end']} ({len(train_df)} samples)")
        print(f"Test period: {w['test_start']} to {w['test_end']} ({len(test_df)} samples)")
        
        # 4) Instantiate Fast & Loose strategy with more aggressive parameters
        # to generate more trade signals
        strat = HighLeverageStrategy(
            # Base parameters
            fast_ema=8,           # Faster EMA for more crossovers
            slow_ema=21,          # Standard slow EMA
            trend_ema=50,         # Shorter trend EMA for more trades
            risk_per_trade=0.03,  # Increased risk (3% per trade)
            
            # Fast & Loose configuration
            use_mtf_filter=True,
            mtf_timeframes=['1h', '4h'],  # Lower timeframes for more signals
            mtf_signal_mode='any',        # Only one higher-timeframe needs to agree
            
            # Very permissive momentum settings
            momentum_threshold=30,        # Lowered from default 35
            use_momentum_filter=True,     # Keep momentum filter but make it more permissive
            
            # Relax pattern requirements
            use_pattern_filter=True,
            pattern_strictness='loose',   # Make pattern detection less strict
            
            # Volume confirmation
            use_volume_filter=True,
            volume_threshold=1.1,         # Lower threshold (110% of avg volume)
            
            # Disable volatility sizing to allow more trades
            use_volatility_sizing=False,
            
            # More aggressive exit strategy
            use_trailing_stop=True,
            take_profit_r=1.5,            # Lower take profit for more frequent wins
            use_partial_exit=True,
            partial_exit_r=0.75,          # Take partial exit sooner
            max_hold_periods=12,          # Shorter max hold time
            
            # Adaptive scaling
            adaptive_vol_scaling=True,
            
            # Minimum confirmations
            min_confirmations=0.25,       # Require only 25% of filters to pass
        )
        
        # 5) "Train" step (no optimization, just establish state)
        print("Running backtest on training data...")
        train_results = strat.backtest(train_df)
        
        # 6) Out-of-sample test
        print("Running backtest on out-of-sample test data...")
        test_results = strat.backtest(test_df)
        
        # Calculate metrics
        metrics = calculate_metrics(test_results)
        
        # Ensure all required metrics have default values
        default_metrics = {
            'roi': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0
        }
        
        # Add any missing metrics with defaults
        for key, default_value in default_metrics.items():
            if key not in metrics:
                metrics[key] = default_value
        
        metrics.update({
            'window_id': i+1,
            'train_start': w['train_start'].strftime('%Y-%m-%d'),
            'train_end': w['train_end'].strftime('%Y-%m-%d'),
            'test_start': w['test_start'].strftime('%Y-%m-%d'),
            'test_end': w['test_end'].strftime('%Y-%m-%d'),
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'train_roi': train_results.get('roi', 0),
            'train_sharpe': train_results.get('sharpe_ratio', 0),
        })
        
        # Extract trade count
        num_trades = len(test_results.get('trades', []))
        metrics['num_trades'] = num_trades
        
        print(f"Test window {i+1} results:")
        print(f"  ROI: {metrics['roi']:.2f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Trades: {num_trades}")
        
        results.append(metrics)
    
    # 7) Save summary to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nWFO summary saved to {output_csv}")
    
    # 8) Plot results if requested
    if plot_results:
        output_plot = os.path.splitext(output_csv)[0] + '_plot.png'
        plot_wfo_results(results_df, output_plot)
    
    # 9) Print aggregate statistics
    print("\nAggregate WFO Statistics:")
    print(f"Average ROI: {results_df['roi'].mean():.2f}%")
    print(f"Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
    print(f"Average Max Drawdown: {results_df['max_drawdown'].mean():.2f}%")
    print(f"Average Win Rate: {results_df['win_rate'].mean():.2f}%")
    print(f"Average Trades per Window: {results_df['num_trades'].mean():.1f}")
    print(f"Consistency (% of windows with positive returns): {(results_df['roi'] > 0).mean() * 100:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Walk-Forward Optimization for Fast & Loose strategy')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', default='4h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=366, help='Days of historical data to use')
    parser.add_argument('--train-days', type=int, default=60, help='Training window size in days')
    parser.add_argument('--test-days', type=int, default=30, help='Testing window size in days')
    parser.add_argument('--output-csv', default='results/fastloose_wfo_summary.csv', help='Output CSV file')
    parser.add_argument('--no-plot', action='store_true', help='Disable results visualization')
    
    args = parser.parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    run_fastloose_wfo(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        train_days=args.train_days,
        test_days=args.test_days,
        output_csv=args.output_csv,
        plot_results=not args.no_plot
    ) 