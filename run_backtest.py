#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.data.fetcher import DataFetcher
from src.backtest.backtest import Backtester
from src.indicators.technical import apply_indicators
from src.utils.logger import logger

def main():
    """Run a simple backtest with our modified settings."""
    logger.info("Starting backtest with modified settings...")
    
    # Create data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch 30 days of 5-minute data
    days = 30
    timeframe = '5m'
    logger.info(f"Fetching {days} days of {timeframe} data...")
    data = data_fetcher.fetch_historical_data(days=days, timeframe=timeframe)
    
    if data is None or data.empty:
        logger.error("Failed to fetch data or data is empty")
        return
    
    logger.info(f"Received {len(data)} bars of data")
    
    # Initialize backtester
    initial_balance = 10000
    backtester = Backtester(data=data, initial_balance=initial_balance)
    
    # Run backtest with 80/20 train/test split
    logger.info("Running backtest...")
    results = backtester.run(train_test_split=0.8)
    
    # Print results
    if results:
        train_results = results.get('train', {})
        test_results = results.get('test', {})
        
        print("\n============================================================")
        print("Training Set Results:")
        print(f"Total Return: {train_results.get('total_return', 0) * 100:.2f}%")
        print(f"Win Rate: {train_results.get('win_rate', 0) * 100:.2f}%")
        print(f"Sharpe Ratio: {train_results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {train_results.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Total Trades: {train_results.get('total_trades', 0)}")
        
        print("\n============================================================")
        print("Testing Set Results:")
        print(f"Total Return: {test_results.get('total_return', 0) * 100:.2f}%")
        print(f"Win Rate: {test_results.get('win_rate', 0) * 100:.2f}%")
        print(f"Sharpe Ratio: {test_results.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {test_results.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Total Trades: {test_results.get('total_trades', 0)}")
        
        # Display trades
        train_trades = train_results.get('trades', [])
        test_trades = test_results.get('trades', [])
        
        if train_trades:
            print("\n============================================================")
            print(f"Example trades from training set ({min(5, len(train_trades))} of {len(train_trades)}):")
            for i, trade in enumerate(train_trades[:5]):
                print(f"{i+1}. {trade.get('side', 'unknown')} entry: {trade.get('entry_price', 0):.2f}, "
                      f"exit: {trade.get('exit_price', 0):.2f}, PnL: ${trade.get('pnl', 0):.2f}, "
                      f"Strategy: {trade.get('strategy', 'unknown')}")
        
        if test_trades:
            print("\n============================================================")
            print(f"Example trades from test set ({min(5, len(test_trades))} of {len(test_trades)}):")
            for i, trade in enumerate(test_trades[:5]):
                print(f"{i+1}. {trade.get('side', 'unknown')} entry: {trade.get('entry_price', 0):.2f}, "
                      f"exit: {trade.get('exit_price', 0):.2f}, PnL: ${trade.get('pnl', 0):.2f}, "
                      f"Strategy: {trade.get('strategy', 'unknown')}")
    
    print("\n============================================================")
    print("Backtest completed. Check the plots for visualization.")
    
    # Save plots if desired
    try:
        backtester.plot_results(save_path="reports/backtest_results.png")
        print("Saved results plot to reports/backtest_results.png")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    print("Press Enter to exit...")
    input()

if __name__ == "__main__":
    main() 