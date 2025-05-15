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
from src.backtest.backtest import Backtester
from src.utils.logger import logger
from src.utils.metrics import profit_factor  # Import the stabilized profit_factor

# Minimum number of trades required for reliable statistics
MIN_TRADES = 3  # Reduced from 5 to 3 to get more parameter sets to explore

def focused_grid_search():
    """Run a focused grid search over a smaller set of parameter combinations"""
    parser = argparse.ArgumentParser(description='Run a focused grid search for optimal strategy parameters')
    parser.add_argument('--symbols', type=str, default='BTC/USDT', help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for backtest')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    args = parser.parse_args()
    
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    days = args.days
    initial_balance = args.initial_balance
    
    # Define parameter ranges (more focused on areas that worked well in previous tests)
    param_ranges = {
        'ema_fast': [3, 5, 8],              # Fast EMA values
        'ema_slow': [12, 15, 20],           # Slow EMA values
        'atr_sl_multiplier': [1.0, 1.2, 1.4], # ATR stop loss multiplier
        'rsi_oversold': [35, 40],           # RSI oversold level
        'rsi_overbought': [60, 65],         # RSI overbought level
        'enable_pyramiding': [True],        # Always enable pyramiding
        'max_pyramid_entries': [2],         # Maximum additional entries
        'risk_per_trade': [0.0075],         # Risk per trade (0.75%)
        'use_trend_filter': [True],         # Always use trend filter
        'use_volatility_sizing': [True],    # Always use volatility sizing
        'vol_target_pct': [0.0075],         # Target volatility of 0.75%
        'volume_threshold': [1.2],          # Volume threshold
        'min_bars_between_trades': [1],     # Min bars between trades
    }
    
    # Load data for all symbols once (outside the parameter search loop)
    data_by_symbol = {}
    daily_data_by_symbol = {}
    for symbol in symbols:
        # Get last N days of data
        data = fetch_ohlcv(symbol, timeframe, days=days)
        daily_data = fetch_ohlcv(symbol, '1d', days=days)
        
        if data is not None and daily_data is not None:
            data_by_symbol[symbol] = data
            daily_data_by_symbol[symbol] = daily_data
            logger.info(f"Loaded {len(data)} candles for {symbol} on {timeframe} and {len(daily_data)} daily candles")
        else:
            logger.error(f"Failed to load data for {symbol}")
            symbols.remove(symbol)
    
    # Create a cartesian product of all parameter combinations
    param_combinations = list(itertools.product(
        *[param_ranges[param] for param in param_ranges]
    ))
    
    logger.info(f"Running focused grid search with {len(param_combinations)} parameter combinations...")
    
    all_results = []
    
    # Run backtest for each parameter combination
    for params in tqdm(param_combinations):
        ema_fast, ema_slow, atr_sl_multiplier, rsi_oversold, rsi_overbought, enable_pyramiding, max_pyramid_entries, risk_per_trade, use_trend_filter, use_volatility_sizing, vol_target_pct, volume_threshold, min_bars_between_trades = params
        
        # Create parameter set
        param_set = {
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'rsi_period': 14,
            'rsi_oversold': rsi_oversold,
            'rsi_overbought': rsi_overbought,
            'volume_threshold': volume_threshold,
            'enable_pyramiding': enable_pyramiding,
            'max_pyramid_entries': max_pyramid_entries,
            'pyramid_threshold': 0.5,
            'pyramid_position_scale': 0.5,
            'risk_per_trade': risk_per_trade,
            'use_trend_filter': use_trend_filter,
            'use_volatility_sizing': use_volatility_sizing,
            'vol_target_pct': vol_target_pct,
            'min_bars_between_trades': min_bars_between_trades,
            'atr_sl_multiplier': atr_sl_multiplier
        }
        
        result = run_backtest(symbols, data_by_symbol, daily_data_by_symbol, param_set, timeframe, days, initial_balance)
        if result:
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
            
            # Calculate expectancy: (win_rate * avg_win) - ((1-win_rate) * avg_loss)
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Calculate trades per month (30 days)
            test_period_days = 90  # Assuming 90-day test period, adjust if needed
            trades_per_month = r['avg_test_trades'] * (30 / test_period_days)
            
            # Store expectancy score (expectancy * trades per month)
            r['expectancy_score'] = expectancy * trades_per_month
        
        # Get max values for normalization
        max_expectancy = max((r['expectancy_score'] for r in all_results if r['expectancy_score'] > 0), default=1e-6)
        max_cagr_dd = max((r['avg_test_cagr_dd'] for r in all_results if r['avg_test_cagr_dd'] > 0), default=1e-6)
        
        # Calculate weighted scores
        for r in all_results:
            # Create weighted score: 60% Expectancy*Trades/Month + 30% CAGR/DD + 10% Consistency
            r['weighted_score'] = (
                0.6 * r['expectancy_score'] / max_expectancy +
                0.3 * r['avg_test_cagr_dd'] / max_cagr_dd +
                0.1 * r['consistency_score']
            )
        
        # Sort by the new weighted score (primarily expectancy-based)
        results_by_weighted = sorted(all_results, key=lambda x: -x['weighted_score'])
        
        # Also keep the other sorting methods for reference
        results_by_cagr_dd = sorted(all_results, key=lambda x: -x['avg_test_cagr_dd'])
        results_by_pf = sorted(all_results, key=lambda x: -x['avg_test_profit_factor'])
        results_by_consistency = sorted(all_results, key=lambda x: -x['consistency_score'])
        
        # Print top results by expectancy score (new primary metric)
        logger.info("\n===== TOP RESULTS BY EXPECTANCY SCORE =====")
        for i, r in enumerate(results_by_weighted[:5]):
            params = r['params']
            logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, ATR {params['atr_sl_multiplier']}x, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}")
            logger.info(f"   Expectancy Score: {r['expectancy_score']:.4f}, Trades/Month: {r['avg_test_trades'] * (30/90):.1f}, Win%: {r['avg_test_win_rate']:.1f}%, PF: {r['avg_test_profit_factor']:.2f}, Return: {r['avg_test_return']:.2f}%")
        
        # Print top results by CAGR/DD (primary optimization metric)
        logger.info("\n===== TOP RESULTS BY CAGR/DRAWDOWN =====")
        for i, r in enumerate(results_by_cagr_dd[:5]):
            params = r['params']
            logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, ATR {params['atr_sl_multiplier']}x, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}")
            logger.info(f"   CAGR/DD: {r['avg_test_cagr_dd']:.2f}, PF: {r['avg_test_profit_factor']:.2f}, Win%: {r['avg_test_win_rate']:.1f}%, Return: {r['avg_test_return']:.2f}%, Trades: {r['avg_test_trades']:.1f}, DD: {r['max_test_drawdown']:.2f}%")
        
        # Print top results by profit factor
        logger.info("\n===== TOP RESULTS BY PROFIT FACTOR =====")
        for i, r in enumerate(results_by_pf[:5]):
            params = r['params']
            logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, ATR {params['atr_sl_multiplier']}x, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}")
            logger.info(f"   PF: {r['avg_test_profit_factor']:.2f}, Win%: {r['avg_test_win_rate']:.1f}%, Return: {r['avg_test_return']:.2f}%, CAGR/DD: {r['avg_test_cagr_dd']:.2f}, Trades: {r['avg_test_trades']:.1f}, DD: {r['max_test_drawdown']:.2f}%")
        
        # Print top results by consistency score
        logger.info("\n===== TOP RESULTS BY CONSISTENCY =====")
        for i, r in enumerate(results_by_consistency[:5]):
            params = r['params']
            logger.info(f"{i+1}. EMA {params['ema_fast']}/{params['ema_slow']}, ATR {params['atr_sl_multiplier']}x, RSI {params['rsi_oversold']}/{params['rsi_overbought']}, Vol {params['volume_threshold']}")
            logger.info(f"   Consistency: {r['consistency_score']:.2f}, Train PF: {r['avg_train_profit_factor']:.2f}, Test PF: {r['avg_test_profit_factor']:.2f}, Train WR: {r['avg_train_win_rate']:.1f}%, Test WR: {r['avg_test_win_rate']:.1f}%")
        
        # Save the best parameters based on weighted score
        best_params = results_by_weighted[0]['params']
        best_results = results_by_weighted[0]
        
        with open('optimal_params.txt', 'w') as f:
            f.write(f"# Optimal Parameters\n")
            f.write(f"EMA_FAST = {best_params['ema_fast']}\n")
            f.write(f"EMA_SLOW = {best_params['ema_slow']}\n")
            f.write(f"RSI_PERIOD = {best_params['rsi_period']}\n")
            f.write(f"RSI_OVERSOLD = {best_params['rsi_oversold']}\n")
            f.write(f"RSI_OVERBOUGHT = {best_params['rsi_overbought']}\n")
            f.write(f"ATR_SL_MULTIPLIER = {best_params['atr_sl_multiplier']}\n")
            f.write(f"VOL_RATIO_MIN = {best_params['volume_threshold']}\n")
            f.write(f"MIN_BARS_BETWEEN_TRADES = {best_params['min_bars_between_trades']}\n")
            f.write(f"ENABLE_PYRAMIDING = {best_params['enable_pyramiding']}\n")
            f.write(f"MAX_PYRAMID_ENTRIES = {best_params['max_pyramid_entries']}\n")
            f.write(f"PYRAMID_THRESHOLD = {best_params['pyramid_threshold']}\n")
            f.write(f"PYRAMID_POSITION_SCALE = {best_params['pyramid_position_scale']}\n")
            f.write(f"USE_TREND_FILTER = {best_params['use_trend_filter']}\n\n")
            f.write(f"# Performance Metrics\n")
            f.write(f"Test Return: {best_results['avg_test_return']:.2f}%\n")
            f.write(f"Test Profit Factor: {best_results['avg_test_profit_factor']:.2f}\n")
            f.write(f"Test Win Rate: {best_results['avg_test_win_rate']:.1f}%\n")
            f.write(f"Test CAGR/DD: {best_results['avg_test_cagr_dd']:.2f}\n")
            f.write(f"Test Trades/Symbol: {best_results['avg_test_trades']:.1f}\n")
            f.write(f"Maximum Drawdown: {best_results['max_test_drawdown']:.2f}%\n")
            f.write(f"Consistency Score: {best_results['consistency_score']:.2f}\n")
        
        logger.info(f"\nOptimal parameters saved to optimal_params.txt")
        
        # Run a final backtest with the optimal parameters
        run_final_backtest(best_params, symbols, timeframe, days, initial_balance)
    else:
        logger.info("No valid parameter sets found that meet the minimum trade criteria.")
    
    return all_results

def run_final_backtest(params, symbols, timeframe, days, initial_balance):
    """Run a final backtest with the optimal parameters and log the results"""
    logger.info(f"\nRunning final backtest with optimal parameters...")
    
    # Run backtest for each symbol
    symbol_results = {}
    total_trades_detail = []
    
    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                continue
                
            # Fetch higher timeframe data for confirmation
            if timeframe == '4h':
                higher_df = fetch_ohlcv(symbol=symbol, tf='1d', days=days*2)
            elif timeframe == '1h':
                higher_df = fetch_ohlcv(symbol=symbol, tf='4h', days=days)
            else:
                higher_df = None
                
            # Create strategy with optimal parameters
            strategy = EMACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_ema=params['ema_fast'],
                slow_ema=params['ema_slow'],
                enable_pyramiding=params['enable_pyramiding'],
                max_pyramid_entries=params['max_pyramid_entries'],
                risk_per_trade=params['risk_per_trade'],
                use_volatility_sizing=params['use_volatility_sizing'],
                vol_target_pct=params['vol_target_pct'],
                atr_sl_multiplier=params['atr_sl_multiplier']
            )
            
            # Set pyramid attributes after creation
            strategy.pyramid_threshold = params['pyramid_threshold']
            strategy.pyramid_position_scale = params['pyramid_position_scale']
            
            # Override min_bars_between_trades
            strategy.min_bars_between_trades = params['min_bars_between_trades']
            strategy.use_trend_filter = params['use_trend_filter']
            
            # Create backtester
            backtester = Backtester(df, initial_balance)
            results = backtester._backtest_strategy(strategy, df, higher_tf_df=higher_df)
            
            # Store results
            symbol_results[symbol] = {
                'return': results['total_return']*100,
                'profit_factor': results['profit_factor'],
                'win_rate': results['win_rate']*100,
                'trades': results['total_trades'],
                'max_drawdown': results['max_drawdown']*100
            }
            
            # Log individual symbol results
            logger.info(f"Backtest results for {symbol}:")
            logger.info(f"  Return: {symbol_results[symbol]['return']:.2f}%")
            logger.info(f"  Profit Factor: {symbol_results[symbol]['profit_factor']:.2f}")
            logger.info(f"  Win Rate: {symbol_results[symbol]['win_rate']:.2f}%")
            logger.info(f"  Total Trades: {symbol_results[symbol]['trades']}")
            logger.info(f"  Max Drawdown: {symbol_results[symbol]['max_drawdown']:.2f}%")
            
            # Collect trade details for summary
            if results.get('trades'):
                total_trades_detail.extend(results['trades'])
            
        except Exception as e:
            logger.error(f"Error in final backtest for {symbol}: {str(e)}")
    
    # Calculate aggregate metrics
    if total_trades_detail:
        wins = [t['pnl'] for t in total_trades_detail if t['pnl'] > 0]
        losses = [t['pnl'] for t in total_trades_detail if t['pnl'] <= 0]
        
        win_rate = len(wins) / len(total_trades_detail) * 100 if total_trades_detail else 0
        avg_win = sum(wins) / len(wins) if wins else 0
        avg_loss = sum(losses) / len(losses) if losses else 0
        avg_duration_hrs = sum([t['duration'].total_seconds()/3600 for t in total_trades_detail]) / len(total_trades_detail) if total_trades_detail else 0
        
        logger.info(f"\nTrade Details Summary:")
        logger.info(f"  Total Trades: {len(total_trades_detail)}")
        logger.info(f"  Win Rate: {win_rate:.2f}%")
        logger.info(f"  Average Win: ${avg_win:.2f}")
        logger.info(f"  Average Loss: ${avg_loss:.2f}")
        logger.info(f"  Average Duration: {avg_duration_hrs:.1f} hours")
    
    # Log to performance file
    with open("docs/performance_log.md", "a") as f:
        f.write(f"\n## Grid Search Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"### Parameters\n")
        f.write(f"- Fast EMA: {params['ema_fast']}\n")
        f.write(f"- Slow EMA: {params['ema_slow']}\n")
        f.write(f"- RSI: {params['rsi_oversold']}/{params['rsi_overbought']}\n")
        f.write(f"- ATR Multiplier: {params['atr_sl_multiplier']}\n")
        f.write(f"- Volume Threshold: {params['volume_threshold']}\n")
        f.write(f"- Pyramiding: {'Enabled' if params['enable_pyramiding'] else 'Disabled'}\n")
        f.write(f"- Trend Filter: {'Enabled' if params['use_trend_filter'] else 'Disabled'}\n\n")
        
        f.write(f"### Results\n")
        f.write(f"| Symbol | Return | Profit Factor | Win Rate | Trades | Max DD |\n")
        f.write(f"|--------|--------|--------------|----------|--------|--------|\n")
        
        for symbol, results in symbol_results.items():
            f.write(f"| {symbol} | {results['return']:.2f}% | {results['profit_factor']:.2f} | {results['win_rate']:.2f}% | {results['trades']} | {results['max_drawdown']:.2f}% |\n")
        
        f.write(f"\n### Trade Summary\n")
        f.write(f"- Total Trades: {len(total_trades_detail)}\n")
        f.write(f"- Win Rate: {win_rate:.2f}%\n")
        f.write(f"- Average Win: ${avg_win:.2f}\n")
        f.write(f"- Average Loss: ${avg_loss:.2f}\n")
        f.write(f"- Average Duration: {avg_duration_hrs:.1f} hours\n")
    
    logger.info("\nFinal results logged to docs/performance_log.md")

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
        higher_df = daily_data_by_symbol[symbol]
        
        if df is not None and len(df) > 0:
            split_idx = int(len(df) * 0.8)
            higher_split_idx = int(len(higher_df) * 0.8) if higher_df is not None else None
            
            train_test_splits[symbol] = {
                'train_df': df.iloc[:split_idx].copy(),
                'test_df': df.iloc[split_idx:].copy(),
                'train_higher_df': higher_df.iloc[:higher_split_idx].copy() if higher_df is not None else None,
                'test_higher_df': higher_df.iloc[higher_split_idx:].copy() if higher_df is not None else None
            }
    
    # Run backtest for each symbol with these parameters
    symbol_results = {}
    param_set_valid = True  # Flag to track if this parameter set meets our criteria
    
    for symbol, split_data in train_test_splits.items():
        train_df = split_data['train_df']
        test_df = split_data['test_df']
        
        try:
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
                atr_sl_multiplier=param_set['atr_sl_multiplier']
            )
            
            # Set pyramid attributes after creation
            train_strategy.pyramid_threshold = param_set['pyramid_threshold']
            train_strategy.pyramid_position_scale = param_set['pyramid_position_scale']
            
            # Override min_bars_between_trades
            train_strategy.min_bars_between_trades = param_set['min_bars_between_trades']
            train_strategy.use_trend_filter = param_set['use_trend_filter']
            
            # Create backtester with training data and strategy
            train_backtester = Backtester(train_df, initial_balance)
            train_results = train_backtester._backtest_strategy(train_strategy, train_df, higher_tf_df=split_data['train_higher_df'])
            
            # Filter out parameter sets with insufficient trades in training
            if train_results['total_trades'] < MIN_TRADES:
                param_set_valid = False
                break
                
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
                atr_sl_multiplier=param_set['atr_sl_multiplier']
            )
            
            # Set pyramid attributes after creation
            test_strategy.pyramid_threshold = param_set['pyramid_threshold']
            test_strategy.pyramid_position_scale = param_set['pyramid_position_scale']
            
            # Override min_bars_between_trades
            test_strategy.min_bars_between_trades = param_set['min_bars_between_trades']
            test_strategy.use_trend_filter = param_set['use_trend_filter']
            
            # Create backtester with test data and strategy
            test_backtester = Backtester(test_df, initial_balance)
            test_results = test_backtester._backtest_strategy(test_strategy, test_df, higher_tf_df=split_data['test_higher_df'])
            
            # Filter out parameter sets with insufficient trades in testing
            if test_results['total_trades'] < MIN_TRADES:
                param_set_valid = False
                break
            
            # Calculate ratio of train to test profit factor (consistency check)
            train_pf = train_results['profit_factor']
            test_pf = test_results['profit_factor']
            
            # Calculate CAGR/DD as fitness function
            train_monthly_return = train_results['total_return'] * 30 / len(train_df)
            test_monthly_return = test_results['total_return'] * 30 / len(test_df)
            
            train_cagr_dd = train_monthly_return / max(train_results['max_drawdown'], 0.01)  # Minimum DD to avoid div by zero
            test_cagr_dd = test_monthly_return / max(test_results['max_drawdown'], 0.01)
            
            # Store results for this symbol
            symbol_results[symbol] = {
                'train_return': train_results['total_return']*100,
                'test_return': test_results['total_return']*100,
                'train_profit_factor': train_pf,
                'test_profit_factor': test_pf,
                'train_win_rate': train_results['win_rate'],
                'test_win_rate': test_results['win_rate'],
                'train_trades': train_results['total_trades'],
                'test_trades': test_results['total_trades'],
                'train_max_drawdown': train_results['max_drawdown']*100,
                'test_max_drawdown': test_results['max_drawdown']*100,
                'train_cagr_dd': train_cagr_dd,
                'test_cagr_dd': test_cagr_dd,
                'trades': train_results.get('trades', []) + test_results.get('trades', [])
            }
            
        except Exception as e:
            logger.error(f"Error in backtest for {symbol} with params {param_set}: {str(e)}")
            param_set_valid = False
            break
    
    # If parameter set is valid (enough trades in all symbols), calculate aggregate metrics
    if param_set_valid and len(symbol_results) > 0:
        # Calculate aggregate metrics across all symbols
        avg_train_return = np.mean([res['train_return'] for res in symbol_results.values()])
        avg_test_return = np.mean([res['test_return'] for res in symbol_results.values()])
        
        avg_train_pf = np.mean([res['train_profit_factor'] for res in symbol_results.values()])
        avg_test_pf = np.mean([res['test_profit_factor'] for res in symbol_results.values()])
        
        avg_train_wr = np.mean([res['train_win_rate'] for res in symbol_results.values()])
        avg_test_wr = np.mean([res['test_win_rate'] for res in symbol_results.values()])
        
        avg_train_trades = np.mean([res['train_trades'] for res in symbol_results.values()])
        avg_test_trades = np.mean([res['test_trades'] for res in symbol_results.values()])
        
        max_train_dd = np.max([res['train_max_drawdown'] for res in symbol_results.values()])
        max_test_dd = np.max([res['test_max_drawdown'] for res in symbol_results.values()])
        
        avg_train_cagr_dd = np.mean([res['train_cagr_dd'] for res in symbol_results.values()])
        avg_test_cagr_dd = np.mean([res['test_cagr_dd'] for res in symbol_results.values()])
        
        # Calculate consistency score (higher is better)
        pf_consistency = 1 - min(abs(avg_train_pf - avg_test_pf) / max(avg_train_pf, avg_test_pf), 1)
        wr_consistency = 1 - min(abs(avg_train_wr - avg_test_wr) / max(avg_train_wr, avg_test_wr), 1)
        
        # Calculate total consistency score (weighted average)
        consistency_score = 0.7 * pf_consistency + 0.3 * wr_consistency
        
        # Store results for this parameter set
        result = {
            'params': param_set,
            'avg_train_return': avg_train_return,
            'avg_test_return': avg_test_return,
            'avg_train_profit_factor': avg_train_pf,
            'avg_test_profit_factor': avg_test_pf,
            'avg_train_win_rate': avg_train_wr * 100,  # Convert to percentage
            'avg_test_win_rate': avg_test_wr * 100,  # Convert to percentage
            'avg_train_trades': avg_train_trades,
            'avg_test_trades': avg_test_trades,
            'max_train_drawdown': max_train_dd,
            'max_test_drawdown': max_test_dd,
            'avg_train_cagr_dd': avg_train_cagr_dd,
            'avg_test_cagr_dd': avg_test_cagr_dd,
            'consistency_score': consistency_score,
            'symbols_tested': len(symbols),
            'symbol_results': symbol_results
        }
        
        return result
    
    return None

if __name__ == "__main__":
    focused_grid_search() 