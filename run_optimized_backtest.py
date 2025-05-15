#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import argparse

from src.data.fetcher import DataFetcher, fetch_ohlcv
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.backtest.backtest import Backtester, MockAccount
from src.utils.logger import logger

def run_optimized_backtest():
    """Run backtest with optimized parameters from our grid search"""
    parser = argparse.ArgumentParser(description='Run backtest with optimized parameters')
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT", help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    args = parser.parse_args()
    
    # Optimized parameters from grid search
    optimized_params = {
        'ema_fast': 5,
        'ema_slow': 12,
        'rsi_period': 14,
        'rsi_oversold': 35,
        'rsi_overbought': 65,
        'volume_threshold': 1.2,
        'enable_pyramiding': False,
        'use_trend_filter': False,
        'min_bars_between_trades': 1
    }
    
    # Define symbols and other parameters
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    days = args.days
    initial_balance = args.initial_balance
    
    # Store results for the final summary
    symbol_results = {}
    total_trades_detail = []
    
    # Run backtest for each symbol
    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                continue
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            
            # Create strategy with optimized parameters
            strategy = EMACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_ema=optimized_params['ema_fast'],
                slow_ema=optimized_params['ema_slow'],
                enable_pyramiding=optimized_params['enable_pyramiding'],
                risk_per_trade=0.0075,
                use_volatility_sizing=True,
                vol_target_pct=0.0075
            )
            
            # Override min_bars_between_trades
            strategy.min_bars_between_trades = optimized_params['min_bars_between_trades']
            
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
            
            # Store trade details for summary
            if 'trades' in results and results['trades']:
                for trade in results['trades']:
                    trade_info = {
                        'symbol': symbol,
                        'type': trade['type'],
                        'entry_time': trade['entry_time'],
                        'exit_time': trade['exit_time'],
                        'duration_hours': (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600,
                        'entry_price': trade['entry_price'],
                        'exit_price': trade['exit_price'],
                        'pnl': trade['pnl'],
                        'pnl_pct': trade['pnl'] / trade['equity_before'] * 100 if 'equity_before' in trade else 0
                    }
                    total_trades_detail.append(trade_info)
            
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
        
        # Print the summary
        logger.info("\n===== BACKTEST SUMMARY =====")
        logger.info(f"Symbols tested: {len(symbols)}")
        logger.info(f"Symbols with trades: {len(symbols_with_trades)}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Average trades per symbol: {avg_trades:.1f}")
        logger.info(f"Average return: {avg_return:.2f}%")
        logger.info(f"Average profit factor: {avg_pf:.2f}")
        logger.info(f"Average win rate: {avg_wr*100:.2f}%")
        logger.info(f"Maximum drawdown: {max_dd:.2f}%")
        
        # Log trade details summary
        if total_trades_detail:
            logger.info("\nTrade Details Summary:")
            winning_trades = [t for t in total_trades_detail if t['pnl'] > 0]
            losing_trades = [t for t in total_trades_detail if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(total_trades_detail) if total_trades_detail else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            avg_duration = np.mean([t['duration_hours'] for t in total_trades_detail]) if total_trades_detail else 0
            
            logger.info(f"  Total Trades: {len(total_trades_detail)}")
            logger.info(f"  Win Rate: {win_rate*100:.2f}%")
            logger.info(f"  Average Win: ${avg_win:.2f}")
            logger.info(f"  Average Loss: ${avg_loss:.2f}")
            logger.info(f"  Average Duration: {avg_duration:.1f} hours")
        
        # Log to performance file
        with open('docs/performance_log.md', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d')
            symbols_str = '+'.join(symbols)
            f.write(f"| {now} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {avg_pf:.2f} | {avg_wr*100:.2f}% | {max_dd:.2f}% | {total_trades} | Optimized: EMAs={optimized_params['ema_fast']}/{optimized_params['ema_slow']}, RSI={optimized_params['rsi_oversold']}/{optimized_params['rsi_overbought']}, Vol={optimized_params['volume_threshold']}, MinBars={optimized_params['min_bars_between_trades']} |\n")
            
        logger.info(f"\nFinal results logged to docs/performance_log.md")
        
        # Print results for each symbol
        logger.info("\nResults by Symbol:")
        for symbol, results in symbol_results.items():
            logger.info(f"{symbol}: Return={results['return']:.2f}%, PF={results['profit_factor']:.2f}, Win={results['win_rate']*100:.2f}%, Trades={results['trades']}, DD={results['max_drawdown']:.2f}%")

if __name__ == "__main__":
    run_optimized_backtest() 