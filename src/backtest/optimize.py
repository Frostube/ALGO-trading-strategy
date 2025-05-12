#!/usr/bin/env python3
"""
Parameter optimization for the trading strategy using Optuna.

Usage:
    python src/backtest/optimize.py --days 30 --trials 100 --output results/optimization.json
"""
import os
import sys
import json
import argparse
import optuna
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.db.models import init_db
from src.data.fetcher import DataFetcher
from src.indicators.technical import apply_indicators
from src.backtest.backtest import backtest_strategy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('optimize')

def objective(trial, data_train, data_test=None):
    """
    Optuna objective function for optimizing strategy parameters.
    
    Args:
        trial: Optuna trial object
        data_train: Training data DataFrame
        data_test: Test data DataFrame (for cross-validation)
        
    Returns:
        float: Objective score (could be Sharpe ratio, total return, etc.)
    """
    # Define the parameter space
    params = {
        # EMA parameters
        'ema_fast': trial.suggest_int('ema_fast', 5, 20),
        'ema_slow': trial.suggest_int('ema_slow', 15, 50),
        
        # RSI parameters
        'rsi_period': trial.suggest_int('rsi_period', 2, 5),
        'rsi_long_threshold': trial.suggest_int('rsi_long_threshold', 5, 20),
        'rsi_short_threshold': trial.suggest_int('rsi_short_threshold', 80, 95),
        
        # Volume parameters
        'volume_period': trial.suggest_int('volume_period', 10, 30),
        'volume_threshold': trial.suggest_float('volume_threshold', 1.2, 2.0),
        
        # Risk parameters
        'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.001, 0.003),
        'take_profit_pct': trial.suggest_float('take_profit_pct', 0.002, 0.005),
        'risk_per_trade': 0.01  # Fixed at 1% for optimization
    }
    
    
    # Run backtest with these parameters
    results = backtest_strategy(data_train, params)
    
    # Calculate objective metric (e.g., Sharpe ratio)
    if results['total_trades'] < 5:
        # If too few trades, penalize
        return -100
    
    # Calculate Sharpe ratio
    daily_returns = results['daily_returns']
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)  # Annualized
    
    # If test data is provided, also evaluate on it
    if data_test is not None:
        # Run backtest on test data
        test_results = backtest_strategy(data_test, params)
        
        if test_results['total_trades'] < 5:
            # If too few trades in test set, penalize
            return -100
        
        # Calculate test Sharpe ratio
        test_daily_returns = test_results['daily_returns']
        test_sharpe_ratio = (test_daily_returns.mean() / test_daily_returns.std()) * np.sqrt(252)
        
        # Check for overfitting: if train performance is much better than test, penalize
        if sharpe_ratio > 2 * test_sharpe_ratio:
            return test_sharpe_ratio * 0.5  # Heavily penalize overfitting
        
        # Return average of train and test Sharpe (with more weight on test)
        return (sharpe_ratio + 2 * test_sharpe_ratio) / 3
    
    # If no test data, just return train Sharpe
    return sharpe_ratio

def optimize_parameters(data, n_trials=100, test_size=0.3):
    """
    Optimize strategy parameters using Optuna.
    
    Args:
        data: DataFrame with historical data
        n_trials: Number of optimization trials
        test_size: Fraction of data to use for testing
        
    Returns:
        dict: Best parameters found
    """
    # Split data into train and test
    split_idx = int(len(data) * (1 - test_size))
    data_train = data.iloc[:split_idx].copy()
    data_test = data.iloc[split_idx:].copy()
    
    logger.info(f"Train data: {len(data_train)} bars, Test data: {len(data_test)} bars")
    
    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data_train, data_test), n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best value (Sharpe ratio): {best_value:.4f}")
    
    # Backtest the best parameters on the full dataset
    full_results = backtest_strategy(data, best_params)
    
    logger.info(f"Full backtest results: Total trades: {full_results['total_trades']}, Win rate: {full_results['win_rate']:.2%}, PnL: {full_results['total_pnl']:.2f}")
    
    # Return the results
    return {
        'best_params': best_params,
        'best_value': best_value,
        'full_backtest': {
            'total_trades': full_results['total_trades'],
            'win_rate': full_results['win_rate'],
            'total_pnl': full_results['total_pnl']
        }
    }

def main():
    """Main optimization function."""
    parser = argparse.ArgumentParser(description='Optimize trading strategy parameters')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to use')
    parser.add_argument('--trials', type=int, default=100, help='Number of optimization trials')
    parser.add_argument('--test-size', type=float, default=0.3, help='Fraction of data to use for testing')
    parser.add_argument('--output', type=str, default='results/optimization.json', help='Output file for optimization results')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Initialize database session
    db_session = init_db()
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Fetch historical data
    logger.info(f"Fetching {args.days} days of historical data")
    historical_data = data_fetcher.fetch_historical_data(days=args.days)
    
    # Apply indicators with default parameters
    logger.info("Applying indicators")
    data_with_indicators = apply_indicators(historical_data)
    
    # Optimize parameters
    logger.info(f"Starting optimization with {args.trials} trials")
    results = optimize_parameters(data_with_indicators, n_trials=args.trials, test_size=args.test_size)
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Optimization results saved to {args.output}")

if __name__ == "__main__":
    m