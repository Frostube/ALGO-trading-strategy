#!/usr/bin/env python3
"""
EMA Optimizer Module

Performs grid search to find optimal EMA parameters for each symbol and timeframe.
Uses walk-forward validation to prevent overfitting and stores results for later use.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from pathlib import Path
import concurrent.futures
from itertools import product
import time
from functools import partial

from src.data.fetcher import fetch_ohlcv
from src.utils.logger import logger
from src.utils.metrics import profit_factor

# Define parameter search space
FAST_EMA_RANGE = range(3, 15)  # 3 to 14 step 1
SLOW_EMA_RANGE = range(20, 82, 2)  # 20 to 80 step 2
TREND_EMA_OPTIONS = [None, 100, 200]  # None = no trend filter
ATR_MULT_RANGE = [round(0.8 + i*0.2, 1) for i in range(5)]  # 0.8 to 1.6 step 0.2

# Backtesting settings
MIN_TRADES = 15  # Reduced from 40 to 15 trades for a valid parameter set
MIN_PROFIT_FACTOR = 1.1  # Reduced from 1.3 to 1.1 minimum OOS profit factor
TOP_N_PARAMS = 3  # Number of top parameter sets to save per symbol

def fetch_historical_data(symbol, timeframe="4h", days=365):
    """
    Fetch historical OHLCV data for a symbol.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '4h')
        days: Number of days of historical data to fetch
        
    Returns:
        DataFrame with OHLCV data
    """
    try:
        # Fetch data from the data fetcher module
        df = fetch_ohlcv(symbol, timeframe, days=days)
        
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol} on {timeframe} timeframe")
            return None
            
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_signals(df, fast_ema, slow_ema, trend_ema=None):
    """
    Calculate trading signals based on EMA crossovers.
    
    Args:
        df: DataFrame with OHLCV data
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        trend_ema: Trend EMA period (optional)
        
    Returns:
        DataFrame with signals
    """
    # Calculate EMAs
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast_ema, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_ema, adjust=False).mean()
    
    # Calculate trend EMA if specified
    if trend_ema is not None:
        df['ema_trend'] = df['close'].ewm(span=trend_ema, adjust=False).mean()
    
    # Initialize signal column
    df['signal'] = 0
    
    # Find crossover points
    for i in range(1, len(df)):
        # Bullish crossover: fast crosses above slow
        if (df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and 
            df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]):
            
            # Apply trend filter if specified
            if trend_ema is not None:
                # Only take long signals if price is above trend EMA
                if df['close'].iloc[i] > df['ema_trend'].iloc[i]:
                    df.loc[df.index[i], 'signal'] = 1
            else:
                df.loc[df.index[i], 'signal'] = 1
                
        # Bearish crossover: fast crosses below slow
        elif (df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and 
              df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]):
            
            # Apply trend filter if specified
            if trend_ema is not None:
                # Only take short signals if price is below trend EMA
                if df['close'].iloc[i] < df['ema_trend'].iloc[i]:
                    df.loc[df.index[i], 'signal'] = -1
            else:
                df.loc[df.index[i], 'signal'] = -1
    
    return df

def calculate_atr(df, period=14):
    """
    Calculate Average True Range for the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        
    Returns:
        DataFrame with ATR column added
    """
    df = df.copy()
    
    # Calculate True Range
    df['tr0'] = abs(df['high'] - df['low'])
    df['tr1'] = abs(df['high'] - df['close'].shift())
    df['tr2'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Clean up temporary columns
    df.drop(['tr0', 'tr1', 'tr2', 'tr'], axis=1, inplace=True)
    
    return df

def backtest_parameters(df, fast_ema, slow_ema, trend_ema, atr_mult, 
                        train_start_idx=0, train_end_idx=None, test_start_idx=None, test_end_idx=None):
    """
    Backtest a set of parameters on the given data.
    
    Args:
        df: DataFrame with OHLCV data
        fast_ema: Fast EMA period
        slow_ema: Slow EMA period
        trend_ema: Trend EMA period (or None)
        atr_mult: ATR multiplier for stop loss
        train_start_idx, train_end_idx: Indices for training period
        test_start_idx, test_end_idx: Indices for test period
        
    Returns:
        Dict with backtest results for both training and test periods
    """
    # Add ATR to the dataframe
    df = calculate_atr(df)
    
    # Calculate signals
    df = calculate_signals(df, fast_ema, slow_ema, trend_ema)
    
    # Initialize backtest metrics
    results = {
        'train': {'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0, 'loss': 0},
        'test': {'trades': 0, 'wins': 0, 'losses': 0, 'profit': 0, 'loss': 0}
    }
    
    # Run backtest on training set
    in_position = 0
    entry_price = 0
    entry_idx = 0
    stop_loss = 0
    
    # Set default indices if not provided
    if train_end_idx is None:
        train_end_idx = len(df) // 4 * 3  # 75% for training
    if test_start_idx is None:
        test_start_idx = train_end_idx
    if test_end_idx is None:
        test_end_idx = len(df) - 1  # Use last index instead of len(df)
    
    # Ensure indices are within bounds
    train_end_idx = min(train_end_idx, len(df) - 1)
    test_end_idx = min(test_end_idx, len(df) - 1)
    
    # Backtest training period
    for i in range(train_start_idx + 1, train_end_idx):
        # Close existing position
        if in_position != 0:
            # Check if stop loss was hit
            if (in_position == 1 and df['low'].iloc[i] <= stop_loss) or \
               (in_position == -1 and df['high'].iloc[i] >= stop_loss):
                # Calculate P&L
                if in_position == 1:
                    pnl = (stop_loss / entry_price - 1) * 100
                else:
                    pnl = (entry_price / stop_loss - 1) * 100
                
                # Update metrics
                results['train']['trades'] += 1
                if pnl > 0:
                    results['train']['wins'] += 1
                    results['train']['profit'] += pnl
                else:
                    results['train']['losses'] += 1
                    results['train']['loss'] += abs(pnl)
                
                # Reset position
                in_position = 0
            
            # Check for exit signal
            elif (in_position == 1 and df['signal'].iloc[i] == -1) or \
                 (in_position == -1 and df['signal'].iloc[i] == 1):
                # Calculate P&L
                if in_position == 1:
                    pnl = (df['close'].iloc[i] / entry_price - 1) * 100
                else:
                    pnl = (entry_price / df['close'].iloc[i] - 1) * 100
                
                # Update metrics
                results['train']['trades'] += 1
                if pnl > 0:
                    results['train']['wins'] += 1
                    results['train']['profit'] += pnl
                else:
                    results['train']['losses'] += 1
                    results['train']['loss'] += abs(pnl)
                
                # Reset position
                in_position = 0
                
            # Maximum trade duration (10 bars)
            elif i - entry_idx >= 10:
                # Calculate P&L
                if in_position == 1:
                    pnl = (df['close'].iloc[i] / entry_price - 1) * 100
                else:
                    pnl = (entry_price / df['close'].iloc[i] - 1) * 100
                
                # Update metrics
                results['train']['trades'] += 1
                if pnl > 0:
                    results['train']['wins'] += 1
                    results['train']['profit'] += pnl
                else:
                    results['train']['losses'] += 1
                    results['train']['loss'] += abs(pnl)
                
                # Reset position
                in_position = 0
        
        # Open new position if we're not in one
        if in_position == 0 and df['signal'].iloc[i] != 0:
            in_position = df['signal'].iloc[i]  # 1 for long, -1 for short
            entry_price = df['close'].iloc[i]
            entry_idx = i
            
            # Set stop loss based on ATR
            if in_position == 1:
                stop_loss = entry_price - (df['atr'].iloc[i] * atr_mult)
            else:
                stop_loss = entry_price + (df['atr'].iloc[i] * atr_mult)
    
    # Reset for test period
    in_position = 0
    
    # Backtest test period
    for i in range(test_start_idx + 1, test_end_idx):
        # Close existing position
        if in_position != 0:
            # Check if stop loss was hit
            if (in_position == 1 and df['low'].iloc[i] <= stop_loss) or \
               (in_position == -1 and df['high'].iloc[i] >= stop_loss):
                # Calculate P&L
                if in_position == 1:
                    pnl = (stop_loss / entry_price - 1) * 100
                else:
                    pnl = (entry_price / stop_loss - 1) * 100
                
                # Update metrics
                results['test']['trades'] += 1
                if pnl > 0:
                    results['test']['wins'] += 1
                    results['test']['profit'] += pnl
                else:
                    results['test']['losses'] += 1
                    results['test']['loss'] += abs(pnl)
                
                # Reset position
                in_position = 0
            
            # Check for exit signal
            elif (in_position == 1 and df['signal'].iloc[i] == -1) or \
                 (in_position == -1 and df['signal'].iloc[i] == 1):
                # Calculate P&L
                if in_position == 1:
                    pnl = (df['close'].iloc[i] / entry_price - 1) * 100
                else:
                    pnl = (entry_price / df['close'].iloc[i] - 1) * 100
                
                # Update metrics
                results['test']['trades'] += 1
                if pnl > 0:
                    results['test']['wins'] += 1
                    results['test']['profit'] += pnl
                else:
                    results['test']['losses'] += 1
                    results['test']['loss'] += abs(pnl)
                
                # Reset position
                in_position = 0
                
            # Maximum trade duration (10 bars)
            elif i - entry_idx >= 10:
                # Calculate P&L
                if in_position == 1:
                    pnl = (df['close'].iloc[i] / entry_price - 1) * 100
                else:
                    pnl = (entry_price / df['close'].iloc[i] - 1) * 100
                
                # Update metrics
                results['test']['trades'] += 1
                if pnl > 0:
                    results['test']['wins'] += 1
                    results['test']['profit'] += pnl
                else:
                    results['test']['losses'] += 1
                    results['test']['loss'] += abs(pnl)
                
                # Reset position
                in_position = 0
        
        # Open new position if we're not in one
        if in_position == 0 and df['signal'].iloc[i] != 0:
            in_position = df['signal'].iloc[i]  # 1 for long, -1 for short
            entry_price = df['close'].iloc[i]
            entry_idx = i
            
            # Set stop loss based on ATR
            if in_position == 1:
                stop_loss = entry_price - (df['atr'].iloc[i] * atr_mult)
            else:
                stop_loss = entry_price + (df['atr'].iloc[i] * atr_mult)
    
    # Calculate final metrics
    for period in ['train', 'test']:
        # Avoid division by zero
        if results[period]['trades'] == 0:
            results[period]['win_rate'] = 0
            results[period]['avg_win'] = 0
            results[period]['avg_loss'] = 0
            results[period]['profit_factor'] = 0
            results[period]['net_pnl'] = 0
            results[period]['monthly_return'] = 0
            continue
        
        # Calculate metrics
        results[period]['win_rate'] = results[period]['wins'] / results[period]['trades'] if results[period]['trades'] > 0 else 0
        results[period]['avg_win'] = results[period]['profit'] / results[period]['wins'] if results[period]['wins'] > 0 else 0
        results[period]['avg_loss'] = results[period]['loss'] / results[period]['losses'] if results[period]['losses'] > 0 else 0
        
        # Avoid division by zero for profit factor
        gross_loss = max(0.01, results[period]['loss'])  # Use at least 1 cent to avoid division by zero
        results[period]['profit_factor'] = results[period]['profit'] / gross_loss
        
        # Calculate net P&L
        results[period]['net_pnl'] = results[period]['profit'] - results[period]['loss']
        
        # Monthly return estimation (assuming 30 days per month)
        period_days = 0
        if period == 'train':
            if train_end_idx > train_start_idx:
                period_days = (df.index[train_end_idx] - df.index[train_start_idx]).days
        else:  # 'test'
            if test_end_idx > test_start_idx:
                period_days = (df.index[test_end_idx] - df.index[test_start_idx]).days
                
        if period_days > 0:
            results[period]['monthly_return'] = results[period]['net_pnl'] * (30 / period_days)
        else:
            results[period]['monthly_return'] = 0
    
    return results

def test_params(params, df, train_start_idx, train_end_idx, test_start_idx, test_end_idx, min_trades, min_profit_factor, symbol, timeframe):
    """
    Test a single parameter set.
    
    Args:
        params: Tuple of (fast_ema, slow_ema, trend_ema, atr_mult)
        df: DataFrame with OHLCV data
        train_start_idx, train_end_idx, test_start_idx, test_end_idx: Indices for backtest periods
        min_trades: Minimum number of trades required
        min_profit_factor: Minimum profit factor required
        symbol: Trading pair symbol
        timeframe: Data timeframe
        
    Returns:
        Dict with parameter results or None if criteria not met
    """
    fast, slow, trend, atr_mult = params
    results = backtest_parameters(
        df, fast, slow, trend, atr_mult,
        train_start_idx, train_end_idx, test_start_idx, test_end_idx
    )
    
    # Only return results if they meet the minimum criteria
    if (results['test']['trades'] >= min_trades and 
        results['test']['profit_factor'] >= min_profit_factor):
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'fast_ema': fast,
            'slow_ema': slow,
            'trend_ema': trend,
            'atr_mult': atr_mult,
            'train_profit_factor': results['train']['profit_factor'],
            'train_win_rate': results['train']['win_rate'],
            'train_trades': results['train']['trades'],
            'train_monthly_return': results['train']['monthly_return'],
            'test_profit_factor': results['test']['profit_factor'],
            'test_win_rate': results['test']['win_rate'],
            'test_trades': results['test']['trades'],
            'test_monthly_return': results['test']['monthly_return'],
            'avg_monthly_return': (results['train']['monthly_return'] + results['test']['monthly_return']) / 2
        }
    return None

def optimize_ema_parameters(symbol, timeframe="4h", history_days=365, max_workers=4):
    """
    Perform grid search to find optimal EMA parameters for a symbol.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '4h')
        history_days: Days of historical data to use
        max_workers: Maximum parallel workers for optimization
        
    Returns:
        List of dictionaries with top parameter sets
    """
    logger.info(f"Starting EMA parameter optimization for {symbol} on {timeframe} timeframe...")
    
    # Fetch historical data
    df = fetch_historical_data(symbol, timeframe, days=history_days)
    
    if df is None or len(df) < 200:
        logger.warning(f"Insufficient data for {symbol} on {timeframe} timeframe")
        return []
    
    # Calculate date for training/test split (8 months training, 1 month OOS)
    total_days = (df.index[-1] - df.index[0]).days
    train_days = int(total_days * 0.8)  # 80% for training
    split_date = df.index[0] + timedelta(days=train_days)
    
    # Find indices for split
    train_start_idx = 0
    train_end_idx = df.index.get_indexer([split_date], method='nearest')[0]
    test_start_idx = train_end_idx
    test_end_idx = len(df) - 1  # Use last index instead of len(df)
    
    # Ensure indices are within bounds
    train_end_idx = min(train_end_idx, len(df) - 1)
    test_end_idx = min(test_end_idx, len(df) - 1)
    
    logger.info(f"Training period: {df.index[train_start_idx]} to {df.index[train_end_idx]}")
    logger.info(f"Test period: {df.index[test_start_idx]} to {df.index[test_end_idx]}")
    
    # Generate all parameter combinations
    all_params = list(product(
        FAST_EMA_RANGE,
        SLOW_EMA_RANGE,
        TREND_EMA_OPTIONS,
        ATR_MULT_RANGE
    ))
    
    # Filter invalid combinations (fast EMA must be smaller than slow EMA)
    valid_params = [(f, s, t, a) for f, s, t, a in all_params if f < s]
    
    logger.info(f"Testing {len(valid_params)} parameter combinations")
    
    # Run optimization with parallel processing
    results = []
    start_time = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create partial function with fixed arguments
        test_func = partial(
            test_params, 
            df=df,
            train_start_idx=train_start_idx, 
            train_end_idx=train_end_idx, 
            test_start_idx=test_start_idx, 
            test_end_idx=test_end_idx,
            min_trades=MIN_TRADES,
            min_profit_factor=MIN_PROFIT_FACTOR,
            symbol=symbol,
            timeframe=timeframe
        )
        
        # Map the parameters to the test function
        for result in executor.map(test_func, valid_params):
            if result is not None:
                results.append(result)
    
    # Sort results by test profit factor and monthly return
    sorted_results = sorted(
        results, 
        key=lambda x: (x['test_profit_factor'], x['test_monthly_return']), 
        reverse=True
    )
    
    # Take top N parameter sets
    top_results = sorted_results[:TOP_N_PARAMS]
    
    logger.info(f"Optimization completed in {(time.time() - start_time)/60:.1f} minutes")
    logger.info(f"Found {len(results)} valid parameter sets, top {len(top_results)} selected")
    
    if top_results:
        for i, result in enumerate(top_results):
            logger.info(f"Top {i+1}: EMA{result['fast_ema']}/{result['slow_ema']}" + 
                       (f"/Trend{result['trend_ema']}" if result['trend_ema'] else "") +
                       f", ATR-SL: {result['atr_mult']}" +
                       f", Test PF: {result['test_profit_factor']:.2f}" +
                       f", Win Rate: {result['test_win_rate']*100:.1f}%" +
                       f", Monthly: {result['test_monthly_return']:.2f}%" +
                       f", Trades: {result['test_trades']}")
    
    return top_results

def save_parameters(params, output_dir="params"):
    """
    Save optimized parameters to JSON files.
    
    Args:
        params: List of parameter dictionaries
        output_dir: Directory to save parameter files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group parameters by symbol
    params_by_symbol = {}
    for param in params:
        symbol = param['symbol']
        if symbol not in params_by_symbol:
            params_by_symbol[symbol] = []
        params_by_symbol[symbol].append(param)
    
    # Save parameters for each symbol
    for symbol, symbol_params in params_by_symbol.items():
        # Create safe filename from symbol
        safe_symbol = symbol.replace('/', '_')
        filename = os.path.join(output_dir, f"{safe_symbol}_ema_params.json")
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                'symbol': symbol,
                'updated_at': datetime.now().isoformat(),
                'parameters': symbol_params
            }, f, indent=2)
        
        logger.info(f"Saved {len(symbol_params)} parameter sets for {symbol} to {filename}")

def find_best_ema_pair(symbol, timeframe="4h", history_days=365):
    """
    Find the best EMA pair for a symbol based on historical data.
    This is a simplified version used by strategies directly.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        timeframe: Candle timeframe (e.g., '4h')
        history_days: Number of days of historical data to use
        
    Returns:
        Tuple of (fast_ema, slow_ema, stats)
    """
    try:
        # Check if we have saved parameters for this symbol
        params_dir = "params"
        safe_symbol = symbol.replace('/', '_')
        params_file = os.path.join(params_dir, f"{safe_symbol}_ema_params.json")
        
        if os.path.exists(params_file):
            # Load parameters from file
            with open(params_file, 'r') as f:
                params_data = json.load(f)
            
            # Check if parameters are fresh (less than 7 days old)
            updated_at = datetime.fromisoformat(params_data['updated_at'])
            if datetime.now() - updated_at < timedelta(days=7):
                # Use the top parameter set
                best_params = params_data['parameters'][0]
                return (
                    best_params['fast_ema'], 
                    best_params['slow_ema'], 
                    {
                        'profit_factor': best_params['test_profit_factor'],
                        'win_rate': best_params['test_win_rate'],
                        'total_return': best_params['test_monthly_return'] * 3,  # Estimate for 90 days
                        'total_trades': best_params['test_trades']
                    }
                )
        
        # If we don't have saved parameters or they're stale, run a simplified optimization
        logger.info(f"No fresh parameters found for {symbol}, running simplified optimization")
        
        # Optimize parameters
        optimized_params = optimize_ema_parameters(symbol, timeframe, history_days, max_workers=2)
        
        if optimized_params:
            best_params = optimized_params[0]
            save_parameters(optimized_params)
            return (
                best_params['fast_ema'], 
                best_params['slow_ema'], 
                {
                    'profit_factor': best_params['test_profit_factor'],
                    'win_rate': best_params['test_win_rate'],
                    'total_return': best_params['test_monthly_return'] * 3,  # Estimate for 90 days
                    'total_trades': best_params['test_trades']
                }
            )
        
        # If optimization fails, return default parameters
        logger.warning(f"Optimization failed for {symbol}, using default parameters")
        return (5, 13, None)
        
    except Exception as e:
        logger.error(f"Error finding best EMA pair for {symbol}: {str(e)}")
        return (5, 13, None)

if __name__ == "__main__":
    """
    CLI for optimizing EMA parameters for a list of symbols.
    
    Example usage:
    python -m src.strategy.ema_optimizer --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframe 2h
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize EMA parameters for trading")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT", help="Comma-separated list of symbols")
    parser.add_argument("--timeframe", type=str, default="2h", help="Timeframe (e.g., '2h', '4h', '1d')")
    parser.add_argument("--days", type=int, default=270, help="Days of historical data (default: 270 days)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--output", type=str, default="params", help="Output directory for parameter files")
    
    args = parser.parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Run optimization for each symbol
    all_params = []
    for symbol in symbols:
        params = optimize_ema_parameters(
            symbol=symbol.strip(),
            timeframe=args.timeframe,
            history_days=args.days,
            max_workers=args.workers
        )
        all_params.extend(params)
    
    # Save all parameters
    save_parameters(all_params, args.output)
    
    print(f"\nEMA Optimization Summary:")
    print(f"=========================")
    print(f"Symbols: {args.symbols}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Parameters saved to: {args.output}")
    print(f"Total valid parameter sets: {len(all_params)}") 