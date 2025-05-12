#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
import sys

sys.path.append('.')  # Add current directory to path

from src.backtest.backtest import run_backtest
from src.utils.logger import logger

def main():
    """
    Run a backtest with command line arguments.
    
    Example usage:
    python run_new_backtest.py --days 30 --balance 10000 --plot
    """
    parser = argparse.ArgumentParser(description='Run backtest for BTC/USDT intra-day scalping strategy')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to use')
    parser.add_argument('--balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--plot', action='store_true', help='Generate equity curve plot')
    parser.add_argument('--output', type=str, default=None, help='Output file path for results JSON')
    
    args = parser.parse_args()
    
    logger.info(f"Starting backtest with {args.days} days of data and ${args.balance} initial balance")
    
    # Run backtest
    results = run_backtest(days=args.days, initial_balance=args.balance, plot=args.plot)
    
    # Save results to JSON if output path is provided
    if args.output:
        # Ensure directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Convert datetime objects to strings
        def json_serial(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(args.output, 'w') as f:
            json.dump(results, f, default=json_serial, indent=2)
        
        logger.info(f"Saved backtest results to {args.output}")
    
    # Print summary
    print("\nBacktest Results Summary:")
    print("========================")
    
    # Print training set results
    if "train" in results:
        print("\nTraining Set:")
        print(f"Total Return: {results['train']['total_return']*100:.2f}%")
        print(f"Win Rate: {results['train']['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results['train'].get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results['train']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['train']['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['train']['total_trades']}")
    
    # Print testing set results
    if "test" in results:
        print("\nTesting Set:")
        print(f"Total Return: {results['test']['total_return']*100:.2f}%")
        print(f"Win Rate: {results['test']['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results['test'].get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results['test']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['test']['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['test']['total_trades']}")
    
    # Print overall results if no train/test split
    if "train" not in results:
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
    
    logger.info("Backtest completed successfully")

if __name__ == '__main__':
    main() 