#!/usr/bin/env python3
"""
Run a quick backtest and display results in a simple text interface 
"""
import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtest.backtest import run_backtest
from src.config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD, RSI_LONG_THRESHOLD, 
    RSI_SHORT_THRESHOLD, VOLUME_THRESHOLD, STOP_LOSS_PCT, TAKE_PROFIT_PCT
)

def print_separator():
    """Print a separator line for text UI."""
    print("\n" + "="*60 + "\n")

def main():
    """Run a quick backtest and display results."""
    print("\n" + "="*60)
    print("\tBTC/USDT Intra-Day Scalping Strategy Backtest")
    print("="*60 + "\n")
    
    print("Current Strategy Parameters:")
    print(f"- EMA Fast/Slow: {EMA_FAST}/{EMA_SLOW}")
    print(f"- RSI Period: {RSI_PERIOD}")
    print(f"- RSI Thresholds (Long/Short): {RSI_LONG_THRESHOLD}/{RSI_SHORT_THRESHOLD}")
    print(f"- Volume Spike Threshold: {VOLUME_THRESHOLD}x")
    print(f"- Stop Loss: {STOP_LOSS_PCT*100:.2f}%")
    print(f"- Take Profit: {TAKE_PROFIT_PCT*100:.2f}%")
    
    print_separator()
    
    print("Running backtest...")
    print("This may take a few minutes depending on the amount of data.")
    
    # Run backtest with train/test split
    results = run_backtest(days=30, initial_balance=10000, plot=True)
    
    print_separator()
    
    # Print training set results
    if "train" in results:
        print("Training Set Results:")
        print(f"Total Return: {results['train']['total_return']*100:.2f}%")
        print(f"Win Rate: {results['train']['win_rate']*100:.2f}%")
        print(f"Sharpe Ratio: {results['train']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['train']['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['train']['total_trades']}")
        
        if 'avg_trade_pnl' in results['train']:
            print(f"Average Trade PnL: ${results['train']['avg_trade_pnl']:.2f}")
        
        print_separator()
        
        print("Testing Set Results:")
        print(f"Total Return: {results['test']['total_return']*100:.2f}%")
        print(f"Win Rate: {results['test']['win_rate']*100:.2f}%")
        print(f"Sharpe Ratio: {results['test']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['test']['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['test']['total_trades']}")
        
        if 'avg_trade_pnl' in results['test']:
            print(f"Average Trade PnL: ${results['test']['avg_trade_pnl']:.2f}")
    else:
        # Print overall results
        print("Backtest Results:")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        if 'avg_trade_pnl' in results:
            print(f"Average Trade PnL: ${results['avg_trade_pnl']:.2f}")
    
    print_separator()
    
    # Trade analysis
    if 'trades' in results.get('train', {}) or 'trades' in results.get('test', {}) or 'trades' in results:
        trades = []
        if 'train' in results and 'trades' in results['train']:
            for trade in results['train']['trades']:
                trade['dataset'] = 'Train'
                trades.append(trade)
        if 'test' in results and 'trades' in results['test']:
            for trade in results['test']['trades']:
                trade['dataset'] = 'Test'
                trades.append(trade)
        if 'trades' in results and 'train' not in results:
            trades = results['trades']
        
        if trades:
            print(f"Last 10 Trades:")
            for i, trade in enumerate(trades[-10:]):
                side = trade.get('side', '')
                entry = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl = trade.get('pnl', 0)
                reason = trade.get('exit_reason', '')
                
                print(f"{i+1}. {side.upper()} {entry:.2f} -> {exit_price:.2f} = ${pnl:.2f} ({reason})")
    
    print_separator()
    print("Backtest completed. Check the plots for visualization.")
    
    # Wait for user input to keep plots open
    input("Press Enter to exit...")

if __name__ == "__main__":
    main() 