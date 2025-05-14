#!/usr/bin/env python3
import sys
import traceback

print("Starting EMA backtest test")

try:
    print("Importing modules...")
    from src.backtest.ema_backtest import run_ema_backtest
    print("Imports successful")
    
    print("Running EMA backtest...")
    results = run_ema_backtest(
        symbol='BTC/USDT',
        timeframe='1h',
        days=10,
        initial_balance=10000,
        plot=True,
        optimize=True
    )
    
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Fast EMA: {results['strategy_params']['fast_ema']}")
    print(f"Slow EMA: {results['strategy_params']['slow_ema']}")
    
except Exception as e:
    print(f"Error running EMA backtest: {type(e).__name__}: {str(e)}")
    print("Traceback:")
    traceback.print_exc(file=sys.stdout) 