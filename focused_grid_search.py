#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import argparse
import itertools
from tqdm import tqdm
import multiprocessing
from functools import partial
import logging
import csv

from src.data.fetcher import DataFetcher, fetch_ohlcv
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.backtest.backtest import Backtester, MockAccount
from src.utils.logger import logger
from src.utils.metrics import profit_factor, calculate_expectancy, calculate_r_multiples  # Import the new function

# Set up logging for detailed output
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)

# Minimum number of trades required for a valid backtest
MIN_TRADES = 5   # Lower threshold for discovery phase
# Minimum profit factor required to consider a parameter set
MIN_PROFIT_FACTOR = 1.0

# Percentage of data to use for training
# Remaining data used for validation
TRAIN_PCT = 0.7

# Global variables to be shared with the worker processes
_data_by_symbol = {}
_daily_data_by_symbol = {}
_symbols = []
_timeframe = ""
_days = 0
_initial_balance = 10000

def process_param_set(param_set):
    """
    Process a single parameter set for backtesting.
    This function needs to be at the module level for multiprocessing.
    
    Args:
        param_set: Dictionary of strategy parameters
        
    Returns:
        dict: Results dictionary or None on error
    """
    try:
        # Log the parameters being tested
        logger.debug(f"Testing parameters: {param_set}")
        
        result = run_backtest(
            symbols=_symbols,
            data_by_symbol=_data_by_symbol,
            daily_data_by_symbol=_daily_data_by_symbol,
            param_set=param_set,
            timeframe=_timeframe,
            days=_days,
            initial_balance=_initial_balance
        )
        return result
    except Exception as e:
        logger.error(f"Error in backtest with params {param_set}: {str(e)}")
        return None

def focused_grid_search():
    """Run a focused grid search over a smaller set of parameter combinations"""
    global _data_by_symbol, _daily_data_by_symbol, _symbols, _timeframe, _days, _initial_balance, MIN_TRADES
    
    parser = argparse.ArgumentParser(description='Run a focused grid search for optimal strategy parameters')
    parser.add_argument('--symbols', type=str, default='BTC/USDT', help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for backtest')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for parallel execution')
    parser.add_argument('--debug_mode', action='store_true', help='Run with a single permissive test case to check for any trades')
    parser.add_argument('--min_trades', type=int, default=MIN_TRADES, help='Minimum number of trades required for valid results')
    parser.add_argument('--top_n', type=int, default=5, help='Number of top parameter sets to display')
    args = parser.parse_args()
    
    _symbols = args.symbols.split(',')
    _timeframe = args.timeframe
    _days = args.days
    _initial_balance = args.initial_balance
    num_workers = max(1, min(args.workers, multiprocessing.cpu_count()))
    
    # Update MIN_TRADES if specified by command line
    MIN_TRADES = args.min_trades
    
    # Define parameter ranges for grid search
    ema_fast_range = [8, 12, 16, 21]
    ema_slow_range = [21, 30, 50, 89]
    rsi_period_range = [14]  # Standard
    rsi_oversold_range = [35, 40, 45]
    rsi_overbought_range = [55, 60, 65, 70]
    volume_threshold_range = [1.0, 1.2, 1.5]

    # New pyramiding parameters
    enable_pyramiding_range = [True, False]
    max_pyramid_entries_range = [2, 3]
    pyramid_threshold_range = [0.5, 1.0, 1.5]
    pyramid_position_scale_range = [0.5]

    # Risk parameters
    risk_per_trade_range = [0.005, 0.0075, 0.01]

    # Strategy features
    use_trend_filter_range = [True]
    use_volatility_sizing_range = [True]

    # Exits
    atr_sl_multiplier_range = [1.0, 1.2, 1.4, 1.6, 2.0]
    breakeven_trigger_r_range = [0.5, 1.0, 1.2, 1.5]
    atr_trail_multiplier_range = [1.5, 2.0, 2.5, 3.0]
    # A value of None means no take profit
    atr_tp_multiplier_range = [None, 3, 4, 6]

    # Minimum bars (time) to hold a trade
    min_hold_bars_range = [0, 3, 5, 8]
    
    # Filtering options
    use_rsi_filter_range = [True, False]  # Option to disable RSI filtering
    use_volume_filter_range = [True, False]  # Option to disable volume filtering
    
    # Debug mode with highly permissive parameters to check if any trades are generated
    if args.debug_mode:
        logger.info("Running in debug mode with permissive parameters")
        # Use a single parameter set with wide filters that should generate trades
        param_combinations = [{
            'ema_fast': 3,
            'ema_slow': 15,
            'rsi_period': 14,
            'rsi_oversold': 30,  # Extremely permissive RSI thresholds
            'rsi_overbought': 70,
            'volume_threshold': 0.0,  # No volume filter
            'enable_pyramiding': False,  # Disable pyramiding
            'max_pyramid_entries': 1,
            'pyramid_threshold': 0.5,
            'pyramid_position_scale': 0.5,
            'risk_per_trade': 0.01,  # Higher risk
            'use_trend_filter': False,  # No trend filter
            'use_volatility_sizing': False,  # Fixed position sizing
            'vol_target_pct': 0.01,
            'min_bars_between_trades': 0,  # No bar restrictions
            'atr_sl_multiplier': 2.0,  # Wide stop loss
            'breakeven_trigger_r': 0.1,  # Quick breakeven
            'atr_trail_multiplier': 3.0,  # Wide trailing stop
            'atr_tp_multiplier': None,  # No take profit
            'min_hold_bars': 0,  # No minimum hold time
            'use_rsi_filter': False,  # Disable RSI filtering entirely
            'use_volume_filter': False  # Disable volume filtering entirely
        }]
        logger.info(f"Debug mode - using single parameter set: {param_combinations[0]}")
    else:
        # Generate all parameter combinations
        param_combinations = []
        for params in itertools.product(
            ema_fast_range, ema_slow_range, rsi_period_range, rsi_oversold_range, 
            rsi_overbought_range, volume_threshold_range, enable_pyramiding_range,
            max_pyramid_entries_range, pyramid_threshold_range, pyramid_position_scale_range,
            risk_per_trade_range, use_trend_filter_range, use_volatility_sizing_range,
            min_hold_bars_range, use_rsi_filter_range, use_volume_filter_range
        ):
            # Skip parameter sets where fast EMA >= slow EMA
            if params[0] >= params[1]:
                continue
            
            # Create parameter dictionary for this combination
            param_dict = {
                'ema_fast': params[0],
                'ema_slow': params[1],
                'rsi_period': params[2],
                'rsi_oversold': params[3],
                'rsi_overbought': params[4],
                'volume_threshold': params[5],
                'enable_pyramiding': params[6],
                'max_pyramid_entries': params[7],
                'pyramid_threshold': params[8],
                'pyramid_position_scale': params[9],
                'risk_per_trade': params[10],
                'use_trend_filter': params[11],
                'use_volatility_sizing': params[12],
                'min_hold_bars': params[13],
                'use_rsi_filter': params[14],
                'use_volume_filter': params[15],
            }
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} parameter sets to evaluate")
    
    # Fetch data once for all parameter combinations
    _data_by_symbol = {}
    _daily_data_by_symbol = {}
    
    for symbol in _symbols:
        # Fetch primary timeframe data
        _data_by_symbol[symbol] = fetch_ohlcv(
            symbol=symbol,
            tf=_timeframe,
            days=_days
        )
        
        # Fetch daily data for higher timeframe confirmation
        _daily_data_by_symbol[symbol] = fetch_ohlcv(
            symbol=symbol,
            tf='1d',
            days=_days*2  # Fetch more days for the higher timeframe
        )
    
    # Run backtests in parallel if workers > 1
    all_results = []
    if num_workers > 1:
        logger.info(f"Running parallel optimization with {num_workers} workers")
        with multiprocessing.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_param_set, param_combinations),
                total=len(param_combinations),
                desc="Optimizing"
            ))
            all_results = [r for r in results if r is not None]
    else:
        # Single-process execution
        for param_set in tqdm(param_combinations, desc="Optimizing"):
            result = process_param_set(param_set)
            if result is not None:
                all_results.append(result)
    
    # Sort results by different metrics
    if all_results:
        # Calculate expectancy scores for all results
        for r in all_results:
            # Extract all trades from all symbols
            all_trades = []
            for symbol_result in r['symbol_results'].values():
                if 'trades' in symbol_result.get('train', {}):
                    trades = symbol_result['train'].get('trades', [])
                    
                    # Diagnostic: Check for trades missing pnl
                    for i, tr in enumerate(trades):
                        if "pnl" not in tr:
                            logger.error(f"Trade {i} is missing pnl: {tr}")
                            
                    all_trades.extend(trades)
            
            # Calculate R metrics if not already calculated
            if 'train_r_metrics' not in r:
                r['train_r_metrics'] = calculate_r_multiples(all_trades)
            
            # Calculate weighted score if not already done
            if 'train_score' not in r:
                # Convert max drawdown to R units
                max_dd_pct = r.get('max_drawdown_pct', 0)
                risk_per_trade = r['param_set'].get('risk_per_trade', 0.01)
                max_dd_r = max_dd_pct / risk_per_trade
                
                # R-based weighted score
                r['train_score'] = r['train_r_metrics']['expectancy'] - 0.5 * max_dd_r
        
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = f"grid_search_results_{timestamp}.csv"
        
        try:
            with open(csv_file, 'w', newline='') as f:
                fieldnames = ['timestamp', 'symbols', 'timeframe', 'params', 'score', 'expectancy', 
                             'win_rate', 'avg_r', 'total_trades', 'total_r_risked',
                             'max_dd_r', 'pct_trades_over_2r', 'pct_trades_over_5r']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for r in sorted(all_results, key=lambda x: x.get('train_score', 0), reverse=True):
                    writer.writerow({
                        'timestamp': timestamp,
                        'symbols': ','.join(r['valid_symbols']),
                        'timeframe': args.timeframe,
                        'params': str(r['param_set']),
                        'score': r.get('train_score', 0),
                        'expectancy': r['train_r_metrics'].get('expectancy', 0),
                        'win_rate': r['train_r_metrics'].get('win_rate', 0),
                        'avg_r': r['train_r_metrics'].get('avg_r', 0),
                        'total_trades': len(r.get('all_train_trades', [])),
                        'total_r_risked': r['train_r_metrics'].get('total_r_risked', 0),
                        'max_dd_r': r.get('train_max_dd_r', 0),
                        'pct_trades_over_2r': r['train_r_metrics'].get('pct_trades_over_2r', 0),
                        'pct_trades_over_5r': r['train_r_metrics'].get('pct_trades_over_5r', 0)
                    })
                    
            logger.info(f"Results saved to {csv_file}")
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            
        # Sort by expectancy (from R-metrics)
        all_results.sort(key=lambda x: x.get('train_score', 0), reverse=True)
        
        # Display top results
        logger.info(f"Top {min(args.top_n, len(all_results))} parameter sets by R-weighted score:")
        for i, r in enumerate(all_results[:args.top_n]):
            param_summary = {k: r['param_set'][k] for k in ['ema_fast', 'ema_slow', 'rsi_oversold', 'rsi_overbought', 
                                                           'enable_pyramiding', 'atr_sl_multiplier', 'breakeven_trigger_r',
                                                           'atr_trail_multiplier', 'atr_tp_multiplier']}
            logger.info(f"{i+1}. Score: {r.get('train_score', 0):.3f}, "
                       f"Expectancy: {r['train_r_metrics'].get('expectancy', 0):.3f}, "
                       f"Win Rate: {r['train_r_metrics'].get('win_rate', 0)*100:.1f}%, "
                       f"Avg R: {r['train_r_metrics'].get('avg_r', 0):.2f}, "
                       f"Max DD R: {r.get('train_max_dd_r', 0):.2f}, "
                       f"Params: {param_summary}")
    else:
        logger.info("No valid parameter sets found that meet the minimum trade criteria.")
    
    return all_results

def run_backtest(symbols, data_by_symbol, daily_data_by_symbol, param_set, timeframe, days, initial_balance):
    """
    Run backtest for a parameter set across multiple symbols
    
    Args:
        symbols: List of symbols to test
        data_by_symbol: Dictionary of primary timeframe data by symbol
        daily_data_by_symbol: Dictionary of daily timeframe data for confirmation
        param_set: Dictionary of strategy parameters
        timeframe: Timeframe string
        days: Number of days to backtest
        initial_balance: Initial account balance
        
    Returns:
        dict: Results dictionary or None if criteria not met
    """
    # Print key parameter settings to help debug
    print_params = {k: v for k, v in param_set.items() if k in ['ema_fast', 'ema_slow', 'rsi_oversold', 'rsi_overbought', 
                                                              'enable_pyramiding', 'atr_sl_multiplier', 'breakeven_trigger_r', 
                                                              'atr_trail_multiplier', 'atr_tp_multiplier']}
    logger.debug(f"Testing parameters: {param_set}")
    
    results = {}
    valid_symbols = []
    
    for symbol in symbols:
        if symbol not in data_by_symbol or symbol not in daily_data_by_symbol:
            logger.debug(f"Skipping {symbol} - missing data")
            continue
            
        # Get data for this symbol
        candles = data_by_symbol[symbol]
        daily_candles = daily_data_by_symbol[symbol]
        
        # Skip if not enough data
        if len(candles) < 50:
            logger.debug(f"Skipping {symbol} - not enough candles")
            continue
            
        # Split into train/test sets
        split_idx = int(len(candles) * TRAIN_PCT)
        
        # Train period
        train_candles = candles.iloc[:split_idx].copy()
        train_daily_candles = daily_candles.copy()  # Use all daily candles for confirmation
        
        # Test period (validation)
        test_candles = candles.iloc[split_idx:].copy()
        
        # Initialize backtester
        backtester = Backtester(initial_balance=initial_balance)
        
        # Create strategy instance with parameters
        strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            config=param_set
        )
        
        # Run backtest on training data
        train_result = backtester.run_single_backtest(
            strategy=strategy,
            data=train_candles, 
            higher_tf_df=train_daily_candles
        )
        
        # Check if training period didn't produce valid results
        if train_result is None:
            logger.debug(f"Skipping {symbol} - no training results")
            continue
            
        if 'trades' not in train_result:
            logger.debug(f"Skipping {symbol} - no trades found in training period")
            continue
        
        # Debug: check the structure of train_result
        logger.debug(f"Train result keys: {train_result.keys()}")
        if 'pnl' not in train_result:
            logger.error(f"Missing 'pnl' in train_result. Available keys: {train_result.keys()}")
            # Add pnl key based on trade results
            total_pnl = sum(t.get('pnl', 0) for t in train_result.get('trades', []))
            train_result['pnl'] = total_pnl
            logger.info(f"Added missing 'pnl' key to train_result: {total_pnl}")
        
        # Ensure each trade has a pnl field and compute r_multiples
        for i, trade in enumerate(train_result.get('trades', [])):
            if 'pnl' not in trade:
                logger.error(f"Trade {i} in training is missing pnl: {trade}")
                # If pnl is missing, calculate it from entry and exit prices
                if all(k in trade for k in ['entry_price', 'exit_price', 'size']):
                    try:
                        # Determine if long or short
                        if trade.get('type', '').lower() == 'long' or trade.get('side', '').lower() == 'buy':
                            trade['pnl'] = (float(trade['exit_price']) - float(trade['entry_price'])) * float(trade['size'])
                        else:  # short
                            trade['pnl'] = (float(trade['entry_price']) - float(trade['exit_price'])) * float(trade['size'])
                        logger.info(f"Added missing 'pnl' to trade {i}: {trade['pnl']}")
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error calculating pnl for trade {i}: {e}")
        
        # Calculate r-multiples for the training set
        r_metrics = calculate_r_multiples(train_result.get('trades', []))
        train_result['r_metrics'] = r_metrics
        
        # If we have enough training trades, continue to validation
        if len(train_result.get('trades', [])) >= MIN_TRADES:
            # Create a new strategy instance for test data
            test_strategy = EMACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                config=param_set
            )
            
            # Run validation only if training was successful
            test_result = backtester.run_single_backtest(
                strategy=test_strategy,
                data=test_candles, 
                higher_tf_df=train_daily_candles  # Same daily data
            )
            
            # Validate test results
            if test_result is not None and 'trades' in test_result:
                # Check if we're at end of test without a trade closed
                # Instead of accessing backtester.strategy, check the test_result to see if the last trade is complete
                # Look for any open trade by checking if the last candle timestamp matches the last trade timestamp
                last_candle_time = test_candles.index[-1]
                
                # Check for any incomplete trades (missing exit_time or exit_price)
                incomplete_trade = False
                if 'trades' in test_result and len(test_result['trades']) > 0:
                    last_trade = test_result['trades'][-1]
                    if 'exit_time' not in last_trade or 'exit_price' not in last_trade:
                        incomplete_trade = True
                
                if incomplete_trade:
                    # Force close the trade
                    final_price = test_candles["close"].iloc[-1]
                    # Add a new trade with exit info
                    closed_trade = last_trade.copy()
                    closed_trade['exit_time'] = last_candle_time
                    closed_trade['exit_price'] = final_price
                    closed_trade['exit_reason'] = "session_end"
                    
                    # Calculate PnL
                    if closed_trade['side'] == 'buy':
                        pnl = (final_price - closed_trade['entry_price']) * closed_trade['amount']
                    else:  # short
                        pnl = (closed_trade['entry_price'] - final_price) * closed_trade['amount'] 
                    
                    closed_trade['pnl'] = pnl
                    
                    # Replace the incomplete trade
                    test_result['trades'][-1] = closed_trade
                    logger.info(f"Added force-closed trade with PnL: {pnl}")
                
                # Check if test set also meets minimum criteria
                if len(test_result.get('trades', [])) > 0:
                    # Add PNL to test results if missing
                    if 'pnl' not in test_result:
                        test_result['pnl'] = sum(t.get('pnl', 0) for t in test_result.get('trades', []))
                    
                    # Check for missing pnl in test trades
                    for i, trade in enumerate(test_result.get('trades', [])):
                        if 'pnl' not in trade:
                            logger.error(f"Test trade {i} missing pnl: {trade}")
                            # Add pnl if possible
                            if all(k in trade for k in ['entry_price', 'exit_price', 'size']):
                                try:
                                    is_long = trade.get('type', '').lower() == 'long' or trade.get('side', '').lower() == 'buy'
                                    if is_long:
                                        trade['pnl'] = (float(trade['exit_price']) - float(trade['entry_price'])) * float(trade['size'])
                                    else:  # short
                                        trade['pnl'] = (float(trade['entry_price']) - float(trade['exit_price'])) * float(trade['size'])
                                    logger.info(f"Added pnl to test trade {i}: {trade['pnl']}")
                                except (ValueError, KeyError) as e:
                                    logger.error(f"Error calculating pnl for test trade {i}: {e}")
                    
                    # Calculate R metrics for the test set
                    test_r_metrics = calculate_r_multiples(test_result.get('trades', []))
                    test_result['r_metrics'] = test_r_metrics
                    
                    valid_symbols.append(symbol)
                    results[symbol] = {
                        'train': train_result,
                        'train_r_metrics': r_metrics,
                        'test': test_result,
                        'test_r_metrics': test_r_metrics
                    }
    
    if not valid_symbols:
        logger.debug(f"No valid symbols found for parameter set {param_set}")
        return None
    
    # Calculate combined metrics across all valid symbols
    all_train_trades = []
    all_test_trades = []
    train_equity = 0
    test_equity = 0
    
    for symbol in valid_symbols:
        symbol_data = results[symbol]
        all_train_trades.extend(symbol_data['train'].get('trades', []))
        all_test_trades.extend(symbol_data['test'].get('trades', []))
        train_equity += symbol_data['train'].get('pnl', 0)
        test_equity += symbol_data['test'].get('pnl', 0)
    
    # Calculate overall R-metrics
    combined_train_r = calculate_r_multiples(all_train_trades)
    combined_test_r = calculate_r_multiples(all_test_trades)
    
    # Calculate max drawdown in R units
    train_max_dd = max([results[s]['train'].get('max_drawdown_pct', 0) for s in valid_symbols])
    test_max_dd = max([results[s]['test'].get('max_drawdown_pct', 0) for s in valid_symbols])
    
    # Convert to R units (assuming risk_per_trade represents 1R)
    train_max_dd_r = train_max_dd / param_set.get('risk_per_trade', 0.01)
    test_max_dd_r = test_max_dd / param_set.get('risk_per_trade', 0.01)
    
    # Calculate R-based weighted score
    train_score = combined_train_r['expectancy'] - 0.5 * train_max_dd_r
    test_score = combined_test_r['expectancy'] - 0.5 * test_max_dd_r
    
    return {
        'param_set': param_set,
        'valid_symbols': valid_symbols,
        'num_symbols': len(valid_symbols),
        'symbol_results': results,
        'all_train_trades': all_train_trades,
        'all_test_trades': all_test_trades,
        'train_r_metrics': combined_train_r,
        'test_r_metrics': combined_test_r,
        'train_max_dd_r': train_max_dd_r,
        'test_max_dd_r': test_max_dd_r,
        'train_equity': train_equity,
        'test_equity': test_equity,
        'train_score': train_score,
        'test_score': test_score,
        'avg_train_trades': len(all_train_trades) / len(valid_symbols),
        'avg_test_trades': len(all_test_trades) / len(valid_symbols)
    }

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Required for Windows multiprocessing
    focused_grid_search() 