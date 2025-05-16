#!/usr/bin/env python3
"""
Simple EMA Crossover Walk-Forward Optimization

A clean implementation of walk-forward optimization for EMA crossover strategy
using only pandas, without the problematic strategy code.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import os

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

def generate_windows(df, train_days=60, test_days=30, max_windows=6):
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
                'window_id': len(windows) + 1,
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
            
        # Limit to max_windows
        if len(windows) >= max_windows:
            break
    
    # Print window info
    for w in windows:
        print(f"Window {w['window_id']}: Train {w['train_start'].strftime('%Y-%m-%d')}–{w['train_end'].strftime('%Y-%m-%d')}, "
              f"Test {w['test_start'].strftime('%Y-%m-%d')}–{w['test_end'].strftime('%Y-%m-%d')}")
    
    return windows

def prepare_data(df):
    """Prepare data for strategy by adding basic indicators."""
    # Make a copy to avoid modifying the original
    prepared_df = df.copy()
    
    # Calculate basic EMAs
    prepared_df['ema8'] = prepared_df['close'].ewm(span=8, adjust=False).mean()
    prepared_df['ema21'] = prepared_df['close'].ewm(span=21, adjust=False).mean()
    prepared_df['ema55'] = prepared_df['close'].ewm(span=55, adjust=False).mean()
    
    # Calculate RSI for momentum
    delta = prepared_df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
    prepared_df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate ATR for position sizing and stops
    high_low = prepared_df['high'] - prepared_df['low']
    high_close = (prepared_df['high'] - prepared_df['close'].shift()).abs()
    low_close = (prepared_df['low'] - prepared_df['close'].shift()).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    prepared_df['atr'] = tr.rolling(window=14).mean()
    
    # Volume indicator
    prepared_df['volume_ma'] = prepared_df['volume'].rolling(window=20).mean()
    prepared_df['vol_ratio'] = prepared_df['volume'] / prepared_df['volume_ma']
    
    # Basic trend indicator
    prepared_df['trend'] = np.where(prepared_df['ema55'] > prepared_df['ema55'].shift(1), 1, -1)
    
    return prepared_df

def generate_signals(df):
    """Generate trading signals based on EMA crossover."""
    # Make a copy
    signals = df.copy()
    
    # Initialize signal columns
    signals['signal'] = 0
    signals['position'] = 0
    signals['entry_price'] = np.nan
    signals['stop_loss'] = np.nan
    signals['take_profit'] = np.nan
    
    # Generate crossover signals with trend filter
    signals.loc[(signals['ema8'] > signals['ema21']) & 
                (signals['ema8'].shift(1) <= signals['ema21'].shift(1)) &
                (signals['trend'] > 0) &
                (signals['vol_ratio'] > 1.2), 'signal'] = 1  # Buy signal
    
    signals.loc[(signals['ema8'] < signals['ema21']) & 
                (signals['ema8'].shift(1) >= signals['ema21'].shift(1)) &
                (signals['trend'] < 0) &
                (signals['vol_ratio'] > 1.2), 'signal'] = -1  # Sell signal
    
    # Set position size (constant for simplicity)
    signals.loc[signals['signal'] != 0, 'position'] = signals['signal']
    
    # Fill positions forward
    signals['position'] = signals['position'].replace(to_replace=0, method='ffill')
    
    # Calculate entry prices where signals occur
    signals.loc[signals['signal'] != 0, 'entry_price'] = signals['close']
    
    # Calculate stop loss and take profit levels for each entry
    for i in range(1, len(signals)):
        if signals['signal'].iloc[i] == 1:  # Buy signal
            signals.loc[signals.index[i], 'stop_loss'] = signals['close'].iloc[i] - 2 * signals['atr'].iloc[i]
            signals.loc[signals.index[i], 'take_profit'] = signals['close'].iloc[i] + 3 * signals['atr'].iloc[i]
        elif signals['signal'].iloc[i] == -1:  # Sell signal
            signals.loc[signals.index[i], 'stop_loss'] = signals['close'].iloc[i] + 2 * signals['atr'].iloc[i]
            signals.loc[signals.index[i], 'take_profit'] = signals['close'].iloc[i] - 3 * signals['atr'].iloc[i]
    
    return signals

def backtest_signals(signals, initial_capital=10000):
    """Backtest signals and generate trades with returns."""
    # Initialize values
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_date = None
    trades = []
    
    # Process each candle
    for i in range(1, len(signals)):
        current_date = signals.index[i]
        current_price = signals['close'].iloc[i]
        
        # Check if we need to enter a trade
        if position == 0 and signals['signal'].iloc[i] != 0:
            position = signals['signal'].iloc[i]
            entry_price = signals['close'].iloc[i]
            stop_loss = signals['stop_loss'].iloc[i]
            take_profit = signals['take_profit'].iloc[i]
            entry_date = current_date
        
        # Check if we need to exit a trade
        elif position != 0:
            exit_signal = False
            exit_reason = ''
            
            # Check for exit conditions
            if position > 0:  # Long position
                # Stop loss hit
                if signals['low'].iloc[i] <= stop_loss:
                    exit_price = stop_loss
                    exit_signal = True
                    exit_reason = 'stop'
                # Take profit hit
                elif signals['high'].iloc[i] >= take_profit:
                    exit_price = take_profit
                    exit_signal = True
                    exit_reason = 'target'
                # Reverse signal
                elif signals['signal'].iloc[i] == -1:
                    exit_price = signals['close'].iloc[i]
                    exit_signal = True
                    exit_reason = 'signal'
            
            elif position < 0:  # Short position
                # Stop loss hit
                if signals['high'].iloc[i] >= stop_loss:
                    exit_price = stop_loss
                    exit_signal = True
                    exit_reason = 'stop'
                # Take profit hit
                elif signals['low'].iloc[i] <= take_profit:
                    exit_price = take_profit
                    exit_signal = True
                    exit_reason = 'target'
                # Reverse signal
                elif signals['signal'].iloc[i] == 1:
                    exit_price = signals['close'].iloc[i]
                    exit_signal = True
                    exit_reason = 'signal'
            
            # Process trade exit
            if exit_signal:
                # Calculate profit/loss
                if position > 0:
                    profit_pct = (exit_price - entry_price) / entry_price
                else:
                    profit_pct = (entry_price - exit_price) / entry_price
                
                # Record the trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'position': position,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'reason': exit_reason
                })
                
                # Reset position
                position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0
    
    # Close any open positions at the end
    if position != 0:
        exit_price = signals['close'].iloc[-1]
        
        # Calculate profit/loss
        if position > 0:
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - exit_price) / entry_price
        
        # Record the trade
        trades.append({
            'entry_date': entry_date,
            'exit_date': signals.index[-1],
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_pct': profit_pct,
            'reason': 'end'
        })
    
    # Calculate overall returns
    if trades:
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(trades)
        
        # Calculate cumulative returns
        cumulative_return = (1 + trades_df['profit_pct']).prod() - 1
        
        # Calculate other metrics
        win_rate = (trades_df['profit_pct'] > 0).mean()
        profit_factor = trades_df.loc[trades_df['profit_pct'] > 0, 'profit_pct'].sum() / abs(trades_df.loc[trades_df['profit_pct'] < 0, 'profit_pct'].sum()) if abs(trades_df.loc[trades_df['profit_pct'] < 0, 'profit_pct'].sum()) > 0 else float('inf')
        
        results = {
            'trades': trades,
            'num_trades': len(trades),
            'win_rate': win_rate * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'total_return': cumulative_return * 100,  # Convert to percentage
            'final_equity': initial_capital * (1 + cumulative_return)
        }
    else:
        results = {
            'trades': [],
            'num_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_return': 0,
            'final_equity': initial_capital
        }
    
    return results

def run_walk_forward_optimization():
    """Run a walk-forward optimization for the EMA crossover strategy."""
    # Parameters
    data_file = 'data/BTC_USDT_4h_366d.json'
    output_csv = 'results/simple_wfo_results.csv'
    output_plot = 'results/simple_wfo_plot.png'
    
    # Step 1: Load data
    print("\n" + "=" * 70)
    print("STEP 1: LOADING HISTORICAL DATA")
    print("=" * 70)
    
    df = load_data_from_file(data_file)
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Print data sample to verify
    print("\nData Sample:")
    print(df.head())
    
    # Step 2: Generate windows for walk-forward testing
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING WALK-FORWARD WINDOWS")
    print("=" * 70)
    
    windows = generate_windows(df, train_days=60, test_days=30)
    
    # Step 3: Run walk-forward optimization
    print("\n" + "=" * 70)
    print("STEP 3: RUNNING WALK-FORWARD TESTS")
    print("=" * 70)
    
    results = []
    
    for window in windows:
        print(f"\nProcessing Window {window['window_id']}")
        
        # Get data slices
        train_df = df.loc[window['train_start']:window['train_end']]
        test_df = df.loc[window['test_start']:window['test_end']]
        
        print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
        
        # Prepare data with indicators
        print("  Preparing data with technical indicators...")
        train_prepared = prepare_data(train_df)
        test_prepared = prepare_data(test_df)
        
        # Generate signals
        print("  Generating trading signals...")
        train_signals = generate_signals(train_prepared)
        test_signals = generate_signals(test_prepared)
        
        # Run backtest
        print("  Running backtest on train data...")
        train_results = backtest_signals(train_signals)
        
        print("  Running backtest on test data...")
        test_results = backtest_signals(test_signals)
        
        # Display window results
        print(f"  Train results: {train_results['num_trades']} trades, {train_results['total_return']:.2f}% return")
        print(f"  Test results: {test_results['num_trades']} trades, {test_results['total_return']:.2f}% return")
        
        # Store results
        result = {
            'window_id': window['window_id'],
            'train_start': window['train_start'].strftime('%Y-%m-%d'),
            'train_end': window['train_end'].strftime('%Y-%m-%d'),
            'test_start': window['test_start'].strftime('%Y-%m-%d'),
            'test_end': window['test_end'].strftime('%Y-%m-%d'),
            'train_trades': train_results['num_trades'],
            'train_return': train_results['total_return'],
            'train_win_rate': train_results['win_rate'],
            'train_profit_factor': train_results['profit_factor'],
            'test_trades': test_results['num_trades'],
            'test_return': test_results['total_return'],
            'test_win_rate': test_results['win_rate'],
            'test_profit_factor': test_results['profit_factor'],
        }
        
        results.append(result)
    
    # Step 4: Analyze results
    print("\n" + "=" * 70)
    print("STEP 4: ANALYZING RESULTS")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    # Create summary plots
    plt.figure(figsize=(12, 10))
    
    # Plot returns
    plt.subplot(2, 2, 1)
    plt.bar([r-0.2 for r in results_df['window_id']], results_df['train_return'], width=0.4, label='Train', color='lightblue')
    plt.bar([r+0.2 for r in results_df['window_id']], results_df['test_return'], width=0.4, label='Test', color='darkblue')
    plt.axhline(y=results_df['test_return'].mean(), color='red', linestyle='--', label=f'Avg Test: {results_df["test_return"].mean():.2f}%')
    plt.title('Return by Window')
    plt.xlabel('Window')
    plt.ylabel('Return (%)')
    plt.xticks(results_df['window_id'])
    plt.legend()
    
    # Plot win rates
    plt.subplot(2, 2, 2)
    plt.bar([r-0.2 for r in results_df['window_id']], results_df['train_win_rate'], width=0.4, label='Train', color='lightgreen')
    plt.bar([r+0.2 for r in results_df['window_id']], results_df['test_win_rate'], width=0.4, label='Test', color='darkgreen')
    plt.axhline(y=results_df['test_win_rate'].mean(), color='red', linestyle='--', label=f'Avg Test: {results_df["test_win_rate"].mean():.2f}%')
    plt.title('Win Rate by Window')
    plt.xlabel('Window')
    plt.ylabel('Win Rate (%)')
    plt.xticks(results_df['window_id'])
    plt.legend()
    
    # Plot number of trades
    plt.subplot(2, 2, 3)
    plt.bar([r-0.2 for r in results_df['window_id']], results_df['train_trades'], width=0.4, label='Train', color='lightcoral')
    plt.bar([r+0.2 for r in results_df['window_id']], results_df['test_trades'], width=0.4, label='Test', color='darkred')
    plt.axhline(y=results_df['test_trades'].mean(), color='black', linestyle='--', label=f'Avg Test: {results_df["test_trades"].mean():.1f}')
    plt.title('Number of Trades by Window')
    plt.xlabel('Window')
    plt.ylabel('Trades')
    plt.xticks(results_df['window_id'])
    plt.legend()
    
    # Plot profit factors
    plt.subplot(2, 2, 4)
    profit_factors_train = results_df['train_profit_factor'].clip(0, 5)  # Cap at 5 for better visualization
    profit_factors_test = results_df['test_profit_factor'].clip(0, 5)
    
    plt.bar([r-0.2 for r in results_df['window_id']], profit_factors_train, width=0.4, label='Train', color='lightyellow')
    plt.bar([r+0.2 for r in results_df['window_id']], profit_factors_test, width=0.4, label='Test', color='gold')
    mean_pf = min(5, results_df['test_profit_factor'].mean())
    plt.axhline(y=mean_pf, color='red', linestyle='--', label=f'Avg Test: {results_df["test_profit_factor"].mean():.2f}')
    plt.title('Profit Factor by Window (capped at 5)')
    plt.xlabel('Window')
    plt.ylabel('Profit Factor')
    plt.xticks(results_df['window_id'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    
    # Step 5: Show summary statistics
    print("\n" + "=" * 70)
    print("STEP 5: SUMMARY STATISTICS")
    print("=" * 70)
    
    print(f"Average Test Return: {results_df['test_return'].mean():.2f}%")
    print(f"Average Test Win Rate: {results_df['test_win_rate'].mean():.2f}%")
    print(f"Average Test Trades: {results_df['test_trades'].mean():.1f}")
    print(f"Average Test Profit Factor: {results_df['test_profit_factor'].mean():.2f}")
    
    # Calculate consistency metrics
    profitable_windows = (results_df['test_return'] > 0).mean() * 100
    print(f"Percentage of Profitable Windows: {profitable_windows:.1f}%")
    
    # Calculate degradation from in-sample to out-of-sample
    avg_train_return = results_df['train_return'].mean()
    avg_test_return = results_df['test_return'].mean()
    degradation = (avg_train_return - avg_test_return) / abs(avg_train_return) * 100 if avg_train_return != 0 else 0
    print(f"Performance Degradation (Train to Test): {degradation:.2f}%")
    
    # Print recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    if results_df['test_trades'].mean() < 5:
        print("- Strategy generates too few trades for reliable conclusions.")
        print("- Consider using shorter-term EMAs or relaxed entry criteria.")
    
    if degradation > 50:
        print("- Large degradation from in-sample to out-of-sample performance suggests curve-fitting.")
        print("- Simplify strategy parameters or improve robustness.")
    
    if profitable_windows < 60:
        print("- Strategy lacks consistency across different market regimes.")
        print("- Consider adding adaptive parameters based on market volatility.")

if __name__ == '__main__':
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("=" * 70)
    print("SIMPLE EMA CROSSOVER WALK-FORWARD OPTIMIZATION")
    print("=" * 70)
    
    run_walk_forward_optimization() 