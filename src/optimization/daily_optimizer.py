#!/usr/bin/env python3
"""
Daily Parameter Optimization Script

This script performs daily parameter optimization for the EMA crossover strategy
using historical data and a walk-forward validation approach. It should be scheduled
to run at midnight UTC.

Usage:
    python -m src.optimization.daily_optimizer --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT"
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.fetcher import fetch_ohlcv, DataFetcher
from src.data.data_loader import load_data
from src.utils.logger import logger
from src.utils.performance_logger import append_result, log_performance_results
from src.utils.git_commit import auto_commit_log

# Default settings
DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
DEFAULT_TIMEFRAME = "4h"
DEFAULT_WINDOW_DAYS = 550  # ~18 months
TRAIN_SPLIT = 0.8  # 80% training, 20% out-of-sample testing

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Daily parameter optimization for trading strategies')
    parser.add_argument('--symbols', type=str, default=','.join(DEFAULT_SYMBOLS),
                       help=f'Comma-separated list of symbols to optimize (default: {",".join(DEFAULT_SYMBOLS)})')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                       help=f'Timeframe to use (default: {DEFAULT_TIMEFRAME})')
    parser.add_argument('--days', type=int, default=DEFAULT_WINDOW_DAYS,
                       help=f'Number of days of historical data to use (default: {DEFAULT_WINDOW_DAYS})')
    parser.add_argument('--param-sets', type=int, default=3,
                       help='Number of parameter sets to save per symbol (default: 3)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--auto-commit', action='store_true',
                       help='Automatically commit performance logs to git')
    return parser.parse_args()

def grid_search_ema_params(df, train_split=TRAIN_SPLIT):
    """
    Perform grid search for optimal EMA parameters using walk-forward validation.
    
    Args:
        df: DataFrame with OHLCV data
        train_split: Fraction of data to use for training (default: 0.8)
        
    Returns:
        List of parameter sets sorted by out-of-sample profit factor
    """
    from src.backtest.backtest import Backtester
    
    # Define parameter ranges to search
    fast_range = range(8, 21, 2)  # 8, 10, 12, 14, 16, 18, 20
    slow_range = range(20, 61, 5)  # 20, 25, 30, 35, 40, 45, 50, 55, 60
    trend_range = [100, 150, 200, 250]
    atr_sl_range = [0.75, 1.0, 1.25, 1.5]
    
    # Split data into training and testing sets
    split_idx = int(len(df) * train_split)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Ensure both datasets have enough data
    if len(train_df) < 100 or len(test_df) < 50:
        logger.warning(f"Insufficient data for robust optimization: train={len(train_df)}, test={len(test_df)}")
        return []
    
    results = []
    total_combos = len(fast_range) * len(slow_range) * len(trend_range) * len(atr_sl_range)
    current_combo = 0
    
    logger.info(f"Starting grid search with {total_combos} parameter combinations")
    
    # Perform grid search
    for fast in fast_range:
        for slow in slow_range:
            # Skip invalid fast/slow combinations
            if fast >= slow:
                continue
                
            for trend in trend_range:
                for atr_sl in atr_sl_range:
                    current_combo += 1
                    if current_combo % 10 == 0:
                        progress = (current_combo / total_combos) * 100
                        logger.info(f"Progress: {progress:.1f}% ({current_combo}/{total_combos})")
                    
                    # Create parameter set for this combination
                    params = {
                        'ema_fast': fast,
                        'ema_slow': slow,
                        'ema_trend': trend,
                        'atr_sl_multiplier': atr_sl,
                        'enable_pyramiding': True,
                        'max_pyramid_entries': 2,
                        'pyramid_threshold': 0.5,
                        'pyramid_position_scale': 0.5,
                        'risk_per_trade': 0.0075,
                        'vol_target_pct': 0.0075,
                        'use_volatility_sizing': True,
                    }
                    
                    # Train on training set
                    train_backtester = Backtester(data=train_df, initial_balance=10000, params=params)
                    train_results = train_backtester.run(strategies=['ema_crossover'])
                    
                    # Test on testing set
                    test_backtester = Backtester(data=test_df, initial_balance=10000, params=params)
                    test_results = test_backtester.run(strategies=['ema_crossover'])
                    
                    # Extract metrics from both runs
                    if 'ema_crossover' in train_results and 'ema_crossover' in test_results:
                        train_stats = train_results['ema_crossover']
                        test_stats = test_results['ema_crossover']
                        
                        # Calculate key metrics
                        pf_train = train_stats.get('profit_factor', 0)
                        pf_oos = test_stats.get('profit_factor', 0)
                        win_train = train_stats.get('win_rate', 0) * 100
                        win_oos = test_stats.get('win_rate', 0) * 100
                        dd_train = train_stats.get('max_drawdown', 0) * 100
                        dd_oos = test_stats.get('max_drawdown', 0) * 100
                        ret_train = train_stats.get('total_return', 0) * 100
                        ret_oos = test_stats.get('total_return', 0) * 100
                        trades_train = train_stats.get('total_trades', 0)
                        trades_oos = test_stats.get('total_trades', 0)
                        
                        # Only consider parameter sets with sufficient trades
                        if trades_train >= 15 and trades_oos >= 8:
                            results.append({
                                'fast': fast,
                                'slow': slow,
                                'trend': trend,
                                'atr_sl': atr_sl,
                                'pf_train': pf_train,
                                'pf_oos': pf_oos,
                                'win_train': win_train,
                                'win_oos': win_oos,
                                'dd_train': dd_train,
                                'dd_oos': dd_oos,
                                'ret_train': ret_train,
                                'ret_oos': ret_oos,
                                'trades_train': trades_train,
                                'trades_oos': trades_oos
                            })
    
    # Sort results by out-of-sample profit factor
    results.sort(key=lambda x: x['pf_oos'], reverse=True)
    
    return results

def optimize_symbol(symbol, timeframe, days):
    """
    Optimize parameters for a single symbol and save the results.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe to analyze
        days: Number of days of historical data to use
        
    Returns:
        List of optimized parameter sets
    """
    logger.info(f"Starting optimization for {symbol} on {timeframe} timeframe")
    
    # Fetch historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = load_data(symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date)
    
    if df.empty:
        logger.error(f"Failed to load data for {symbol}. Skipping optimization.")
        return []
    
    logger.info(f"Loaded {len(df)} candles for {symbol}")
    
    # Perform grid search
    best_params = grid_search_ema_params(df)
    
    if not best_params:
        logger.warning(f"No valid parameter sets found for {symbol}")
        return []
    
    # Filter for robust parameter sets (train PF ≈ OOS PF)
    robust_params = []
    for param_set in best_params:
        # Check if train and OOS profit factors are within 20% of each other
        pf_delta_pct = abs(param_set['pf_train'] - param_set['pf_oos']) / max(param_set['pf_oos'], 0.1)
        
        if pf_delta_pct < 0.2:
            robust_params.append(param_set)
            logger.info(f"Robust parameter set found: EMA {param_set['fast']}/{param_set['slow']}/{param_set['trend']} "
                       f"with PF_train={param_set['pf_train']:.2f}, PF_oos={param_set['pf_oos']:.2f}, "
                       f"Win_oos={param_set['win_oos']:.1f}%, Return_oos={param_set['ret_oos']:.1f}%")
    
    return robust_params

def save_params(symbol, timeframe, param_sets, max_sets=3):
    """
    Save the optimized parameter sets to a JSON file.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe used
        param_sets: List of parameter sets
        max_sets: Maximum number of parameter sets to save
    """
    # Create directory if it doesn't exist
    params_dir = Path("params")
    params_dir.mkdir(parents=True, exist_ok=True)
    
    # Format symbol for filename (replace / with _)
    symbol_formatted = symbol.replace('/', '_')
    
    # Prepare data to save
    params = param_sets[:max_sets]  # Take top N sets
    if not params:
        logger.warning(f"No parameter sets to save for {symbol}")
        return
    
    # Convert parameters to format expected by strategy
    formatted_params = []
    for p in params:
        formatted_params.append({
            'ema_fast': p['fast'],
            'ema_slow': p['slow'],
            'ema_trend': p['trend'],
            'atr_sl_multiplier': p['atr_sl'],
            'enable_pyramiding': True,
            'max_pyramid_entries': 2,
            'pyramid_threshold': 0.5,
            'pyramid_position_scale': 0.5,
            'risk_per_trade': 0.0075,
            'vol_target_pct': 0.0075,
            'use_volatility_sizing': True,
            'meta': {
                'pf_train': p['pf_train'],
                'pf_oos': p['pf_oos'],
                'win_rate_oos': p['win_oos'],
                'max_dd_oos': p['dd_oos'],
                'return_oos': p['ret_oos'],
                'trades_oos': p['trades_oos'],
                'optimized_date': datetime.now().strftime('%Y-%m-%d')
            }
        })
    
    # Save to JSON file
    filename = params_dir / f"{symbol_formatted}_{timeframe}.json"
    with open(filename, 'w') as f:
        json.dump(formatted_params, f, indent=2)
    
    logger.info(f"Saved {len(formatted_params)} parameter sets to {filename}")
    
    # Log performance of the best parameter set
    top_param = param_sets[0]
    append_result(
        "optimizer",
        f"{symbol} {timeframe}",
        f"EMA {top_param['fast']}/{top_param['slow']}/{top_param['trend']} ATR {top_param['atr_sl']}",
        top_param['pf_oos'],
        top_param['win_oos'],
        top_param['dd_oos'],
        top_param['ret_oos']
    )

def main():
    args = parse_args()
    
    # Parse symbols list
    symbols = args.symbols.split(',')
    
    # Create params directory if it doesn't exist
    params_dir = Path("params")
    params_dir.mkdir(parents=True, exist_ok=True)
    
    # Optimize each symbol
    for symbol in symbols:
        try:
            # Optimize parameters
            param_sets = optimize_symbol(symbol, args.timeframe, args.days)
            
            # Save parameters
            if param_sets:
                save_params(symbol, args.timeframe, param_sets, args.param_sets)
            else:
                logger.warning(f"No valid parameter sets found for {symbol}")
        except Exception as e:
            logger.error(f"Error optimizing {symbol}: {str(e)}")
    
    # Commit logs to git if requested
    if args.auto_commit:
        auto_commit_log()

if __name__ == "__main__":
    main() 