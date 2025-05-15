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

from src.data.fetcher import DataFetcher, fetch_ohlcv
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.backtest.backtest import Backtester, MockAccount
from src.utils.logger import logger
from src.utils.metrics import profit_factor  # Import the stabilized profit_factor

# Set up logging for detailed output
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)

# Minimum number of trades required for reliable statistics
MIN_TRADES = 3  # Reduced from 5 to 3 to get more parameter sets to explore

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
    global _data_by_symbol, _daily_data_by_symbol, _symbols, _timeframe, _days, _initial_balance
    
    parser = argparse.ArgumentParser(description='Run a focused grid search for optimal strategy parameters')
    parser.add_argument('--symbols', type=str, default='BTC/USDT', help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for backtest')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes for parallel execution')
    args = parser.parse_args()
    
    _symbols = args.symbols.split(',')
    _timeframe = args.timeframe
    _days = args.days
    _initial_balance = args.initial_balance
    num_workers = max(1, min(args.workers, multiprocessing.cpu_count()))
    
    # Define parameter ranges (more focused based on preliminary results)
    ema_fast_range = [3, 5, 8]
    ema_slow_range = [15, 20, 25, 30]
    rsi_period_range = [14]
    rsi_oversold_range = [35, 40]
    rsi_overbought_range = [60, 65]
    volume_threshold_range = [1.2]
    
    # Pyramiding parameters
    enable_pyramiding_range = [True]
    max_pyramid_entries_range = [2]
    pyramid_threshold_range = [0.5]
    pyramid_position_scale_range = [0.5]

    # Risk management parameters
    risk_per_trade_range = [0.0075]  # 0.75% risk per trade
    use_trend_filter_range = [True]
    use_volatility_sizing_range = [True]
    vol_target_pct_range = [0.0075]  # 0.75% volatility target
    min_bars_between_trades_range = [1]
    atr_sl_multiplier_range = [1.0, 1.2, 1.4]
    
    # Additional optimization parameters
    breakeven_trigger_r_range = [0.5, 1.0, 1.2]  # R multiples to trigger breakeven
    initial_trail_mult_range = [1.25, 1.5, 2.0]  # Initial trail ATR multiplier
    tp_multiplier_range = [None, 3, 4]  # Take profit multiplier (None for trail only)
    min_hold_bars_range = [0, 3]  # Minimum bars to hold a position
    
    # Generate all parameter combinations
    param_combinations = []
    for params in itertools.product(
        ema_fast_range, ema_slow_range, rsi_period_range, rsi_oversold_range, 
        rsi_overbought_range, volume_threshold_range, enable_pyramiding_range,
        max_pyramid_entries_range, pyramid_threshold_range, pyramid_position_scale_range,
        risk_per_trade_range, use_trend_filter_range, use_volatility_sizing_range,
        vol_target_pct_range, min_bars_between_trades_range, atr_sl_multiplier_range,
        breakeven_trigger_r_range, initial_trail_mult_range, tp_multiplier_range,
        min_hold_bars_range
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
            'vol_target_pct': params[13],
            'min_bars_between_trades': params[14],
            'atr_sl_multiplier': params[15],
            'breakeven_trigger_r': params[16],
            'atr_trail_multiplier': params[17],
            'atr_tp_multiplier': params[18],
            'min_hold_bars': params[19],
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
                if 'trades' in symbol_result:
                    all_trades.extend(symbol_result.get('trades', []))
            
            # Calculate win rate
            win_rate = r['avg_test_win_rate'] / 100  # Convert from percentage
            
            # Calculate average win and loss
            win_pnls = [t['pnl'] for t in all_trades if t['pnl'] > 0]
            loss_pnls = [abs(t['pnl']) for t in all_trades if t['pnl'] <= 0]  # Store as positive values
            
            avg_win = np.mean(win_pnls) if win_pnls else 0.0
            avg_loss = np.mean(loss_pnls) if loss_pnls else 0.0
            
            # Expectancy calculation
            if avg_loss == 0:
                expectancy = avg_win * win_rate
            else:
                expectancy = (avg_win * win_rate) - (avg_loss * (1 - win_rate))
            
            # Calculate trades per month
            total_days = r['avg_test_days']
            total_trades = len(all_trades)
            trades_per_month = (total_trades / total_days) * 30 if total_days > 0 else 0
            
            # Expectancy score (expectancy × trades per month)
            r['expectancy_score'] = expectancy * trades_per_month
            
            # Calculate CAGR/DD ratio - return divided by max drawdown
            max_dd = r['avg_test_max_dd']
            cagr = r['avg_test_cagr']
            r['avg_test_cagr_dd'] = cagr / max(0.01, abs(max_dd))  # Avoid division by zero
            
            # Consistency score - based on difference between train and test results
            train_pf = r['avg_train_pf']
            test_pf = r['avg_test_pf']
            train_return = r['avg_train_return']
            test_return = r['avg_test_return']
            
            # Penalize large differences between train and test
            pf_ratio = min(train_pf, test_pf) / max(1.01, max(train_pf, test_pf))
            return_ratio = min(train_return, test_return) / max(0.01, max(train_return, test_return))
            
            # Consistency score: 1 = perfect consistency, 0 = completely inconsistent
            r['consistency_score'] = (pf_ratio + return_ratio) / 2
        
        # Get maximum values for normalization
        max_expectancy = max(r['expectancy_score'] for r in all_results) or 1e-6
        max_cagr_dd = max(r['avg_test_cagr_dd'] for r in all_results) or 1e-6
        
        # Calculate weighted scores
        for r in all_results:
            r['weighted_score'] = (
                0.6 * r['expectancy_score'] / max_expectancy +
                0.3 * r['avg_test_cagr_dd'] / max_cagr_dd +
                0.1 * r['consistency_score']
            )
        
        # Sort by weighted score
        all_results.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        # Print top results
        logger.info(f"Top 5 parameter sets by weighted score:")
        for i, result in enumerate(all_results[:5]):
            logger.info(f"{i+1}. {result['params']} - Score: {result['weighted_score']:.4f}, "
                       f"Win Rate: {result['avg_test_win_rate']:.2f}%, "
                       f"PF: {result['avg_test_pf']:.2f}, "
                       f"Return: {result['avg_test_return']:.2f}%, "
                       f"Trades: {result['avg_test_trades']:.1f}")
        
        # Save best parameters to file
        if all_results:
            best_params = all_results[0]['params']
            with open('optimal_params.txt', 'w') as f:
                f.write("Optimal parameters found via grid search:\n\n")
                for k, v in best_params.items():
                    f.write(f"{k} = {v}\n")
                
                f.write("\nPerformance metrics:\n")
                f.write(f"Win Rate: {all_results[0]['avg_test_win_rate']:.2f}%\n")
                f.write(f"Profit Factor: {all_results[0]['avg_test_pf']:.2f}\n")
                f.write(f"Return: {all_results[0]['avg_test_return']:.2f}%\n")
                f.write(f"Max Drawdown: {all_results[0]['avg_test_max_dd']:.2f}%\n")
                f.write(f"Avg Trades: {all_results[0]['avg_test_trades']:.1f}\n")
                f.write(f"Expectancy Score: {all_results[0]['expectancy_score']:.4f}\n")
                f.write(f"CAGR/DD Ratio: {all_results[0]['avg_test_cagr_dd']:.2f}\n")
                f.write(f"Consistency Score: {all_results[0]['consistency_score']:.2f}\n")
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
    # Split data for walk-forward validation - 80% training, 20% testing
    train_test_splits = {}
    for symbol in symbols:
        df = data_by_symbol[symbol]
        split_idx = int(len(df) * 0.8)
        
        train_data = df.iloc[:split_idx]
        test_data = df.iloc[split_idx:]
        
        # Get dates for the split points
        if len(train_data) > 0 and len(test_data) > 0:
            train_start = train_data.index[0]
            train_end = train_data.index[-1]
            test_start = test_data.index[0]
            test_end = test_data.index[-1]
            
            train_test_splits[symbol] = (train_start, train_end, test_start, test_end)
            logger.debug(f"Split data for {symbol}: Train {train_start} to {train_end}, Test {test_start} to {test_end}")
    
    # Results for each symbol
    symbol_results = {}
    
    # Training period metrics
    total_train_pf = 0
    total_train_win_rate = 0
    total_train_return = 0
    total_train_trades = 0
    total_train_max_dd = 0
    
    # Testing period metrics
    total_test_pf = 0
    total_test_win_rate = 0
    total_test_return = 0
    total_test_trades = 0
    total_test_max_dd = 0
    total_test_days = 0
    total_test_cagr = 0
    
    valid_symbols = 0
    
    for symbol in symbols:
        # Skip symbols without enough data
        if symbol not in train_test_splits:
            logger.debug(f"Skipping {symbol} - insufficient data for split")
            continue
        
        train_start, train_end, test_start, test_end = train_test_splits[symbol]
        
        # Create strategy with parameters for training period
        train_strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            fast_ema=param_set['ema_fast'],
            slow_ema=param_set['ema_slow'],
            enable_pyramiding=param_set['enable_pyramiding'],
            max_pyramid_entries=param_set['max_pyramid_entries'],
            risk_per_trade=param_set['risk_per_trade'],
            use_volatility_sizing=param_set['use_volatility_sizing'],
            vol_target_pct=param_set['vol_target_pct'],
            atr_sl_multiplier=param_set['atr_sl_multiplier'],
            rsi_period=param_set['rsi_period'],
            rsi_overbought=param_set['rsi_overbought'],
            rsi_oversold=param_set['rsi_oversold'],
            volume_threshold=param_set['volume_threshold'],
            use_trend_filter=param_set['use_trend_filter'],
            min_bars_between_trades=param_set['min_bars_between_trades'],
            trend_ema=200  # Fixed trend EMA
        )
        
        # Create backtester for training period
        train_backtester = Backtester(
            initial_balance=initial_balance,
            params=param_set
        )
        
        # Run backtest on training period
        train_df = data_by_symbol[symbol][(data_by_symbol[symbol].index >= train_start) & 
                                         (data_by_symbol[symbol].index <= train_end)]
        train_htf_df = daily_data_by_symbol[symbol]
        
        train_result = None
        
        # Only proceed if we have data
        if len(train_df) > 0:
            train_result = train_backtester.run_single_backtest(train_strategy, train_df, train_htf_df)
        
        # Skip if training period didn't produce valid results
        if train_result is None:
            logger.debug(f"Skipping {symbol} - no training results")
            continue
            
        if 'trades' not in train_result:
            logger.debug(f"Skipping {symbol} - no trades found in training period")
            continue
            
        train_trades_count = len(train_result['trades'])
        if train_trades_count < MIN_TRADES:
            logger.debug(f"Skipping {symbol} - insufficient trades in training period: {train_trades_count} < {MIN_TRADES}")
            continue
            
        logger.debug(f"Training period for {symbol}: {train_trades_count} trades, PnL: ${train_result.get('pnl', 0):.2f}")
        
        # Create strategy with same parameters for testing period
        test_strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            fast_ema=param_set['ema_fast'],
            slow_ema=param_set['ema_slow'],
            enable_pyramiding=param_set['enable_pyramiding'],
            max_pyramid_entries=param_set['max_pyramid_entries'],
            risk_per_trade=param_set['risk_per_trade'],
            use_volatility_sizing=param_set['use_volatility_sizing'],
            vol_target_pct=param_set['vol_target_pct'],
            atr_sl_multiplier=param_set['atr_sl_multiplier'],
            rsi_period=param_set['rsi_period'],
            rsi_overbought=param_set['rsi_overbought'],
            rsi_oversold=param_set['rsi_oversold'],
            volume_threshold=param_set['volume_threshold'],
            use_trend_filter=param_set['use_trend_filter'],
            min_bars_between_trades=param_set['min_bars_between_trades'],
            trend_ema=200  # Fixed trend EMA
        )
        
        # Create backtester for testing period
        test_backtester = Backtester(
            initial_balance=initial_balance,
            params=param_set
        )
        
        # Run backtest on testing period
        test_df = data_by_symbol[symbol][(data_by_symbol[symbol].index >= test_start) & 
                                        (data_by_symbol[symbol].index <= test_end)]
        test_htf_df = daily_data_by_symbol[symbol]
        
        test_result = None
        
        # Only proceed if we have data
        if len(test_df) > 0:
            test_result = test_backtester.run_single_backtest(test_strategy, test_df, test_htf_df)
        
        # Skip if testing period didn't produce valid results
        if test_result is None:
            logger.debug(f"Skipping {symbol} - no test results")
            continue
            
        if 'trades' not in test_result:
            logger.debug(f"Skipping {symbol} - no trades found in testing period")
            continue
            
        test_trades_count = len(test_result['trades'])
        if test_trades_count < MIN_TRADES:
            logger.debug(f"Skipping {symbol} - insufficient trades in testing period: {test_trades_count} < {MIN_TRADES}")
            continue
            
        logger.debug(f"Testing period for {symbol}: {test_trades_count} trades, PnL: ${test_result.get('pnl', 0):.2f}")
        
        # Calculate key metrics for this symbol
        train_trades = len(train_result['trades'])
        train_wins = sum(1 for t in train_result['trades'] if t['pnl'] > 0)
        train_losses = sum(1 for t in train_result['trades'] if t['pnl'] <= 0)
        
        train_win_rate = (train_wins / train_trades * 100) if train_trades > 0 else 0
        train_gross_profit = sum(t['pnl'] for t in train_result['trades'] if t['pnl'] > 0)
        train_gross_loss = sum(t['pnl'] for t in train_result['trades'] if t['pnl'] <= 0)
        train_profit_factor = profit_factor(
            [t['pnl'] for t in train_result['trades'] if t['pnl'] > 0],
            [t['pnl'] for t in train_result['trades'] if t['pnl'] <= 0]
        )
        
        test_trades = len(test_result['trades'])
        test_wins = sum(1 for t in test_result['trades'] if t['pnl'] > 0)
        test_losses = sum(1 for t in test_result['trades'] if t['pnl'] <= 0)
        
        test_win_rate = (test_wins / test_trades * 100) if test_trades > 0 else 0
        test_gross_profit = sum(t['pnl'] for t in test_result['trades'] if t['pnl'] > 0)
        test_gross_loss = sum(t['pnl'] for t in test_result['trades'] if t['pnl'] <= 0)
        test_profit_factor = profit_factor(
            [t['pnl'] for t in test_result['trades'] if t['pnl'] > 0],
            [t['pnl'] for t in test_result['trades'] if t['pnl'] <= 0]
        )
        
        # Calculate test period duration in days
        test_days = (test_end - test_start).days
        
        # Calculate annualized return (CAGR)
        test_return_pct = test_result['pnl'] / initial_balance * 100
        annual_multiplier = 365 / max(1, test_days)
        test_cagr = ((1 + test_return_pct/100) ** annual_multiplier - 1) * 100
        
        # Store results for this symbol
        symbol_results[symbol] = {
            'train_win_rate': train_win_rate,
            'train_pnl': train_result['pnl'],
            'train_return': train_result['pnl'] / initial_balance * 100,
            'train_pf': train_profit_factor,
            'train_trades': train_trades,
            'train_max_dd': train_result.get('max_drawdown', 0),
            
            'test_win_rate': test_win_rate,
            'test_pnl': test_result['pnl'],
            'test_return': test_return_pct,
            'test_pf': test_profit_factor,
            'test_trades': test_trades,
            'test_max_dd': test_result.get('max_drawdown', 0),
            'test_days': test_days,
            'test_cagr': test_cagr,
            
            'trades': test_result['trades'],  # Store test trades for further analysis
        }
        
        # Accumulate metrics for averaging
        total_train_pf += train_profit_factor
        total_train_win_rate += train_win_rate
        total_train_return += train_result['pnl'] / initial_balance * 100
        total_train_trades += train_trades
        total_train_max_dd += train_result.get('max_drawdown', 0)
        
        total_test_pf += test_profit_factor
        total_test_win_rate += test_win_rate
        total_test_return += test_return_pct
        total_test_trades += test_trades
        total_test_max_dd += test_result.get('max_drawdown', 0)
        total_test_days += test_days
        total_test_cagr += test_cagr
        
        valid_symbols += 1
        logger.debug(f"Valid results for {symbol} - added to results")
    
    # Return None if no valid symbols were found
    if valid_symbols == 0:
        logger.debug(f"No valid symbols found for parameter set {param_set}")
        return None
    
    # Calculate averages
    avg_train_pf = total_train_pf / valid_symbols
    avg_train_win_rate = total_train_win_rate / valid_symbols
    avg_train_return = total_train_return / valid_symbols
    avg_train_trades = total_train_trades / valid_symbols
    avg_train_max_dd = total_train_max_dd / valid_symbols
    
    avg_test_pf = total_test_pf / valid_symbols
    avg_test_win_rate = total_test_win_rate / valid_symbols
    avg_test_return = total_test_return / valid_symbols
    avg_test_trades = total_test_trades / valid_symbols
    avg_test_max_dd = total_test_max_dd / valid_symbols
    avg_test_days = total_test_days / valid_symbols
    avg_test_cagr = total_test_cagr / valid_symbols
    
    logger.debug(f"Found valid parameter set with {valid_symbols} symbols, avg test trades: {avg_test_trades:.1f}, avg profit: {avg_test_return:.2f}%")
    
    # Return results
    return {
        'params': param_set,
        'symbol_results': symbol_results,
        'avg_train_pf': avg_train_pf,
        'avg_train_win_rate': avg_train_win_rate,
        'avg_train_return': avg_train_return,
        'avg_train_trades': avg_train_trades,
        'avg_train_max_dd': avg_train_max_dd,
        
        'avg_test_pf': avg_test_pf,
        'avg_test_win_rate': avg_test_win_rate,
        'avg_test_return': avg_test_return,
        'avg_test_trades': avg_test_trades,
        'avg_test_max_dd': avg_test_max_dd,
        'avg_test_days': avg_test_days,
        'avg_test_cagr': avg_test_cagr,
    }

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()  # Required for Windows multiprocessing
    focused_grid_search() 