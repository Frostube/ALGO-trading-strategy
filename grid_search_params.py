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

from src.data.fetcher import DataFetcher, fetch_ohlcv
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.backtest.backtest import Backtester, MockAccount
from src.utils.logger import logger

def grid_search():
    """Run a grid search over multiple parameter combinations"""
    parser = argparse.ArgumentParser(description='Run grid search over strategy parameters')
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT", help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--min-trades', type=int, default=8, help='Minimum trades per symbol to target')
    parser.add_argument('--max-trades', type=int, default=12, help='Maximum trades per symbol to target')
    args = parser.parse_args()
    
    # Define grid search parameters
    ema_fast_values = [3, 5, 8, 10]
    ema_slow_values = [10, 12, 15, 21]
    rsi_oversold_values = [30, 35, 40, 45]
    rsi_overbought_values = [55, 60, 65, 70]
    volume_threshold_values = [1.2, 1.5]
    min_bars_values = [1, 2]
    
    # Only test valid combinations where fast < slow
    ema_pairs = [(fast, slow) for fast in ema_fast_values for slow in ema_slow_values if fast < slow]
    
    # Define symbols and other parameters
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    days = args.days
    initial_balance = args.initial_balance
    
    # Load data for all symbols once (outside the parameter search loop)
    symbol_data = {}
    for symbol in symbols:
        try:
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            if df is not None and len(df) > 0:
                symbol_data[symbol] = df
                logger.info(f"Loaded {len(df)} candles for {symbol}")
            else:
                logger.error(f"No data found for {symbol}")
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {str(e)}")
    
    # Store results
    all_results = []
    
    # Create a cartesian product of all parameter combinations
    param_combinations = list(itertools.product(
        ema_pairs,
        rsi_oversold_values,
        rsi_overbought_values,
        volume_threshold_values,
        min_bars_values
    ))
    
    logger.info(f"Running grid search with {len(param_combinations)} parameter combinations...")
    
    # Run backtest for each parameter combination
    for params in tqdm(param_combinations):
        ema_pair, rsi_oversold, rsi_overbought, volume_threshold, min_bars = params
        ema_fast, ema_slow = ema_pair
        
        # Skip if RSI thresholds are invalid
        if rsi_oversold >= rsi_overbought:
            continue
        
        # Create parameter set
        param_set = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi_period': 14,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volume_threshold': volume_threshold,
            'enable_pyramiding': False,
            'use_trend_filter': False,
            'min_bars_between_trades': min_bars
        }
        
        # Run backtest for each symbol with these parameters
        symbol_results = {}
        for symbol, df in symbol_data.items():
            try:
                # Create strategy with parameters
                strategy = EMACrossoverStrategy(
                    symbol=symbol,
                    timeframe=timeframe,
                    fast_ema=param_set['ema_fast'],
                    slow_ema=param_set['ema_slow'],
                    enable_pyramiding=param_set['enable_pyramiding'],
                    risk_per_trade=0.0075,
                    use_volatility_sizing=True,
                    vol_target_pct=0.0075
                )
                
                # Override min_bars_between_trades
                strategy.min_bars_between_trades = param_set['min_bars_between_trades']
                
                # Create backtester with this data and strategy
                backtester = Backtester(df, initial_balance)
                results = backtester._backtest_strategy(strategy, df)
                
                # Store results for this symbol
                symbol_results[symbol] = {
                    'return': results['total_return']*100,  # Convert to percentage
                    'profit_factor': results['profit_factor'],
                    'win_rate': results['win_rate'],
                    'trades': results['total_trades'],
                    'max_drawdown': results['max_drawdown']*100  # Convert to percentage
                }
                
            except Exception as e:
                logger.error(f"Error in backtest for {symbol} with params {param_set}: {str(e)}")
                symbol_results[symbol] = {
                    'return': 0,
                    'profit_factor': 0,
                    'win_rate': 0,
                    'trades': 0,
                    'max_drawdown': 0
                }
        
        # Calculate aggregate metrics
        if len(symbol_results) > 0:
            # Only include symbols with trades in averages
            symbols_with_trades = [s for s, r in symbol_results.items() if r['trades'] > 0]
            if symbols_with_trades:
                avg_return = np.mean([symbol_results[s]['return'] for s in symbols_with_trades])
                
                # Handle inf profit factors
                profit_factors = [symbol_results[s]['profit_factor'] for s in symbols_with_trades 
                                 if symbol_results[s]['profit_factor'] != float('inf')]
                avg_pf = np.mean(profit_factors) if profit_factors else 0
                
                avg_wr = np.mean([symbol_results[s]['win_rate'] for s in symbols_with_trades])
                avg_trades = np.mean([symbol_results[s]['trades'] for s in symbols_with_trades])
                max_dd = np.max([symbol_results[s]['max_drawdown'] for s in symbols_with_trades])
                total_trades = sum([symbol_results[s]['trades'] for s in symbol_results.keys()])
            else:
                # No symbols with trades
                avg_return = 0
                avg_pf = 0
                avg_wr = 0
                avg_trades = 0
                max_dd = 0
                total_trades = 0
            
            # Store results for this parameter set
            result = {
                'params': param_set,
                'avg_return': avg_return,
                'avg_profit_factor': avg_pf,
                'avg_win_rate': avg_wr * 100,  # Convert to percentage
                'avg_trades_per_symbol': avg_trades,
                'total_trades': total_trades,
                'max_drawdown': max_dd,
                'symbols_tested': len(symbols),
                'symbols_with_trades': len(symbols_with_trades),
                'symbol_results': symbol_results
            }
            
            all_results.append(result)
    
    # Sort results by different metrics
    results_by_trades = sorted(all_results, key=lambda x: -x['avg_trades_per_symbol'])
    results_by_pf = sorted(all_results, key=lambda x: -x['avg_profit_factor'])
    results_by_return = sorted(all_results, key=lambda x: -x['avg_return'])
    
    # Filter for combinations that meet the trade count criteria
    target_trades = [r for r in all_results if args.min_trades <= r['avg_trades_per_symbol'] <= args.max_trades]
    
    # If we found parameter sets in the target range, sort by profit factor
    if target_trades:
        target_trades = sorted(target_trades, key=lambda x: (-x['avg_profit_factor'], -x['avg_win_rate']))
    
    # Print top 10 results by different metrics
    logger.info("\n===== TOP RESULTS BY PROFIT FACTOR =====")
    for i, r in enumerate(results_by_pf[:10]):
        params = r['params']
        logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}, MinBars {params['min_bars_between_trades']}")
        logger.info(f"   PF: {r['avg_profit_factor']:.2f}, Win%: {r['avg_win_rate']:.1f}%, Return: {r['avg_return']:.2f}%, Trades/Symbol: {r['avg_trades_per_symbol']:.1f}, DD: {r['max_drawdown']:.2f}%")
    
    logger.info("\n===== TOP RESULTS BY AVERAGE TRADES PER SYMBOL =====")
    for i, r in enumerate(results_by_trades[:10]):
        params = r['params']
        logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}, MinBars {params['min_bars_between_trades']}")
        logger.info(f"   Trades/Symbol: {r['avg_trades_per_symbol']:.1f}, PF: {r['avg_profit_factor']:.2f}, Win%: {r['avg_win_rate']:.1f}%, Return: {r['avg_return']:.2f}%, DD: {r['max_drawdown']:.2f}%")
    
    # If we found parameter sets in the target range, print them
    if target_trades:
        logger.info(f"\n===== BEST PARAMETERS IN TARGET RANGE ({args.min_trades}-{args.max_trades} trades/symbol) =====")
        for i, r in enumerate(target_trades[:10]):
            params = r['params']
            logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}, MinBars {params['min_bars_between_trades']}")
            logger.info(f"   PF: {r['avg_profit_factor']:.2f}, Win%: {r['avg_win_rate']:.1f}%, Return: {r['avg_return']:.2f}%, Trades/Symbol: {r['avg_trades_per_symbol']:.1f}, DD: {r['max_drawdown']:.2f}%")
    else:
        logger.info(f"\nNo parameter combinations found in target range of {args.min_trades}-{args.max_trades} trades/symbol")
    
    # Save the best parameters to a file
    if target_trades:
        best_params = target_trades[0]['params']
        best_results = target_trades[0]
        
        with open('optimal_params.txt', 'w') as f:
            f.write(f"# Optimal Parameters (Target: {args.min_trades}-{args.max_trades} trades/symbol)\n")
            f.write(f"EMA_FAST = {best_params['ema_fast']}\n")
            f.write(f"EMA_SLOW = {best_params['ema_slow']}\n")
            f.write(f"RSI_PERIOD = {best_params['rsi_period']}\n")
            f.write(f"RSI_OVERSOLD = {best_params['rsi_oversold']}\n")
            f.write(f"RSI_OVERBOUGHT = {best_params['rsi_overbought']}\n")
            f.write(f"VOL_RATIO_MIN = {best_params['volume_threshold']}\n")
            f.write(f"MIN_BARS_BETWEEN_TRADES = {best_params['min_bars_between_trades']}\n\n")
            f.write(f"# Performance Metrics\n")
            f.write(f"Average Profit Factor: {best_results['avg_profit_factor']:.2f}\n")
            f.write(f"Average Win Rate: {best_results['avg_win_rate']:.1f}%\n")
            f.write(f"Average Return: {best_results['avg_return']:.2f}%\n")
            f.write(f"Average Trades Per Symbol: {best_results['avg_trades_per_symbol']:.1f}\n")
            f.write(f"Maximum Drawdown: {best_results['max_drawdown']:.2f}%\n")
            f.write(f"Total Trades: {best_results['total_trades']}\n")
        
        logger.info(f"\nOptimal parameters saved to optimal_params.txt")
        
        # Run a final backtest with the optimal parameters and log to performance_log.md
        run_final_backtest(best_params, symbols, timeframe, days, initial_balance)
    
    return all_results

def run_final_backtest(params, symbols, timeframe, days, initial_balance):
    """Run a final backtest with the optimal parameters and log the results"""
    logger.info(f"\nRunning final backtest with optimal parameters...")
    
    # Run backtest for each symbol
    symbol_results = {}
    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                continue
            
            # Create strategy with parameters
            strategy = EMACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_ema=params['ema_fast'],
                slow_ema=params['ema_slow'],
                enable_pyramiding=params['enable_pyramiding'],
                risk_per_trade=0.0075,
                use_volatility_sizing=True,
                vol_target_pct=0.0075
            )
            
            # Override min_bars_between_trades
            strategy.min_bars_between_trades = params['min_bars_between_trades']
            
            # Create backtester with this data and strategy
            backtester = Backtester(df, initial_balance)
            results = backtester._backtest_strategy(strategy, df)
            
            # Store results for this symbol
            symbol_results[symbol] = {
                'return': results['total_return']*100,  # Convert to percentage
                'profit_factor': results['profit_factor'],
                'win_rate': results['win_rate'],
                'trades': results['total_trades'],
                'max_drawdown': results['max_drawdown']*100  # Convert to percentage
            }
            
            logger.info(f"Backtest results for {symbol}:")
            logger.info(f"  Return: {results['total_return']*100:.2f}%")
            logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
            logger.info(f"  Win Rate: {results['win_rate']*100:.2f}%")
            logger.info(f"  Total Trades: {results['total_trades']}")
            logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
            
        except Exception as e:
            logger.error(f"Error in backtest for {symbol}: {str(e)}")
            continue
    
    # Calculate aggregate metrics
    if len(symbol_results) > 0:
        # Only include symbols with trades in averages
        symbols_with_trades = [s for s, r in symbol_results.items() if r['trades'] > 0]
        if symbols_with_trades:
            avg_return = np.mean([symbol_results[s]['return'] for s in symbols_with_trades])
            
            # Handle inf profit factors
            profit_factors = [symbol_results[s]['profit_factor'] for s in symbols_with_trades 
                             if symbol_results[s]['profit_factor'] != float('inf')]
            avg_pf = np.mean(profit_factors) if profit_factors else 0
            
            avg_wr = np.mean([symbol_results[s]['win_rate'] for s in symbols_with_trades])
            avg_trades = np.mean([symbol_results[s]['trades'] for s in symbols_with_trades])
            max_dd = np.max([symbol_results[s]['max_drawdown'] for s in symbols_with_trades])
            total_trades = sum([symbol_results[s]['trades'] for s in symbol_results.keys()])
        else:
            # No symbols with trades
            avg_return = 0
            avg_pf = 0
            avg_wr = 0
            avg_trades = 0
            max_dd = 0
            total_trades = 0
        
        # Log to performance file
        with open('docs/performance_log.md', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d')
            symbols_str = '+'.join(symbols)
            f.write(f"| {now} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {avg_pf:.2f} | {avg_wr*100:.2f}% | {max_dd:.2f}% | {total_trades} | Grid Search: EMAs={params['ema_fast']}/{params['ema_slow']}, RSI={params['rsi_oversold']}/{params['rsi_overbought']}, Vol={params['volume_threshold']}, MinBars={params['min_bars_between_trades']} |\n")
            
        logger.info(f"\nFinal results logged to docs/performance_log.md")

if __name__ == "__main__":
    grid_search() 