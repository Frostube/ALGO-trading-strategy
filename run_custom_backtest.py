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

def main():
    parser = argparse.ArgumentParser(description='Run custom backtest with specified parameters')
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT", help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    args = parser.parse_args()
    
    # Custom parameters based on your requirements
    custom_params = {
        'ema_fast': 3,                # Even faster EMA (3 instead of 5)
        'ema_slow': 10,               # Faster slow EMA (10 instead of 15)
        'rsi_period': 14,             # Standard RSI period
        'rsi_oversold': 35,           # Less strict oversold threshold (35 instead of 30)
        'rsi_overbought': 65,         # Less strict overbought threshold (65 instead of 70)
        'volume_threshold': 1.2,      # Lower volume threshold (1.2 instead of 1.5)
        'enable_pyramiding': False,   # Disable pyramiding for now
        'use_trend_filter': False     # Disable 200 EMA trend filter
    }
    
    # Define symbols and other parameters
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    days = args.days
    initial_balance = args.initial_balance
    
    # Run backtest for each symbol
    all_results = {}
    symbol_results = {}
    for symbol in symbols:
        logger.info(f"Running backtest for {symbol}...")
        
        # Fetch historical data
        try:
            # Use the fetch_ohlcv function directly instead of DataFetcher
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            
            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                continue
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            
            # Create strategy with custom parameters
            strategy = EMACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_ema=custom_params['ema_fast'],
                slow_ema=custom_params['ema_slow'],
                enable_pyramiding=custom_params['enable_pyramiding'],
                risk_per_trade=0.0075,
                use_volatility_sizing=True,
                vol_target_pct=0.0075
            )
            
            # Create backtester with this data and strategy
            backtester = Backtester(df, initial_balance)
            results = backtester._backtest_strategy(strategy, df)
            
            # Print out available keys in results for debugging
            logger.info(f"Available keys in results: {list(results.keys())}")
            
            # Log results (using appropriate keys)
            logger.info(f"Backtest results for {symbol}:")
            logger.info(f"  Return: {results['total_return']*100:.2f}%")
            logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
            logger.info(f"  Win Rate: {results['win_rate']*100:.2f}%")
            logger.info(f"  Total Trades: {results['total_trades']}")
            logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
            
            symbol_results[symbol] = {
                'return': results['total_return']*100,  # Convert to percentage
                'profit_factor': results['profit_factor'],
                'win_rate': results['win_rate'],
                'trades': results['total_trades'],
                'max_drawdown': results['max_drawdown']*100  # Convert to percentage
            }
            
            all_results[symbol] = results
            
        except Exception as e:
            logger.error(f"Error in backtest for {symbol}: {str(e)}")
            continue
    
    # Calculate aggregate metrics
    if len(symbol_results) > 0:
        avg_return = np.mean([r['return'] for r in symbol_results.values()])
        avg_pf = np.mean([r['profit_factor'] for r in symbol_results.values() if r['profit_factor'] != float('inf')])
        avg_wr = np.mean([r['win_rate'] for r in symbol_results.values()])
        total_trades = sum([r['trades'] for r in symbol_results.values()])
        max_dd = max([r['max_drawdown'] for r in symbol_results.values()])
        
        logger.info("\nOverall Performance:")
        logger.info(f"  Average Return: {avg_return:.2f}%")
        logger.info(f"  Average Profit Factor: {avg_pf:.2f}")
        logger.info(f"  Average Win Rate: {avg_wr*100:.2f}%")
        logger.info(f"  Total Trades: {total_trades}")
        logger.info(f"  Max Drawdown: {max_dd:.2f}%")
        
        # Log to performance file
        with open('docs/performance_log.md', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d')
            symbols_str = args.symbols.replace(',', '+')
            f.write(f"| {now} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {avg_pf:.2f} | {avg_wr*100:.2f}% | {max_dd:.2f}% | {total_trades} | Custom params: EMAs={custom_params['ema_fast']}/{custom_params['ema_slow']}, RSI={custom_params['rsi_oversold']}/{custom_params['rsi_overbought']}, Vol={custom_params['volume_threshold']} |\n")
            
    return symbol_results

if __name__ == "__main__":
    main() 