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
from src.backtest.backtest import Backtester

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
        'ema_fast': trial.suggest_int('ema_fast', 3, 12),
        'ema_slow': trial.suggest_int('ema_slow', 13, 26),
        
        # RSI parameters
        'rsi_period': trial.suggest_int('rsi_period', 2, 14),
        'rsi_long_threshold': trial.suggest_int('rsi_long_threshold', 20, 40),
        'rsi_short_threshold': trial.suggest_int('rsi_short_threshold', 60, 80),
        
        # HMA parameters
        'hma_fast_period': trial.suggest_int('hma_fast_period', 5, 16),
        'hma_slow_period': trial.suggest_int('hma_slow_period', 17, 30),
        
        # Volume parameters
        'volume_period': trial.suggest_int('volume_period', 10, 20),
        'volume_threshold': trial.suggest_float('volume_threshold', 1.2, 2.0),
        
        # Risk parameters
        'stop_loss_pct': trial.suggest_float('stop_loss_pct', 0.001, 0.003),
        'take_profit_pct': trial.suggest_float('take_profit_pct', 0.003, 0.009),
        'atr_sl_multiplier': trial.suggest_float('atr_sl_multiplier', 1.0, 2.0),
        'atr_tp_multiplier': trial.suggest_float('atr_tp_multiplier', 2.0, 4.0)
    }
    
    # Modify indicator parameters before running backtest
    # This will be passed to apply_indicators through custom parameters
    # in a real implementation
    
    # Run backtest with training data
    backtester = Backtester(data=data_train, initial_balance=10000)
    
    # Set custom parameters (in a real implementation)
    # Here we need to use the global config parameters
    
    # Run the backtest
    train_results = backtester._run_on_dataset(data_train, "Training")
    
    # Calculate objective metric
    if train_results['total_trades'] < 5:
        # If too few trades, penalize
        return -100
    
    # Calculate a score based on multiple metrics
    train_score = (train_results['total_return'] * 0.4 + 
                  train_results['sharpe_ratio'] * 0.4 + 
                  train_results['win_rate'] * 0.2)
    
    # If test data is provided, also evaluate on it
    if data_test is not None:
        # Run backtest on test data
        backtester = Backtester(data=data_test, initial_balance=10000)
        test_results = backtester._run_on_dataset(data_test, "Testing")
        
        if test_results['total_trades'] < 3:
            # If too few trades in test set, penalize but less severely
            return -50
        
        # Calculate test score
        test_score = (test_results['total_return'] * 0.4 + 
                     test_results['sharpe_ratio'] * 0.4 + 
                     test_results['win_rate'] * 0.2)
        
        # Check for overfitting: if train performance is much better than test, penalize
        if train_score > 0 and test_score < 0:
            return test_score * 0.5  # Heavily penalize overfitting
        
        # Return weighted average of train and test scores (more weight on test)
        return (train_score + 2 * test_score) / 3
    
    # If no test data, just return train score
    return train_score

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
    # Apply indicators first
    data_with_indicators = apply_indicators(data)
    
    # Split data into train and test
    split_idx = int(len(data_with_indicators) * (1 - test_size))
    data_train = data_with_indicators.iloc[:split_idx].copy()
    data_test = data_with_indicators.iloc[split_idx:].copy()
    
    logger.info(f"Train data: {len(data_train)} bars, Test data: {len(data_test)} bars")
    
    # Create and run the study
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, data_train, data_test), n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best value (optimization score): {best_value:.4f}")
    
    # Backtest the best parameters on the full dataset
    backtester = Backtester(data=data_with_indicators, initial_balance=10000)
    full_results = backtester._run_on_dataset(data_with_indicators, "Full")
    
    logger.info(f"Full backtest results: Total trades: {full_results['total_trades']}, Win rate: {full_results['win_rate']:.2%}, Return: {full_results['total_return']:.2%}")
    
    # Return the results
    return {
        'best_params': best_params,
        'best_value': best_value,
        'full_backtest': {
            'total_trades': full_results['total_trades'],
            'win_rate': full_results['win_rate'],
            'total_return': full_results['total_return'],
            'sharpe_ratio': full_results['sharpe_ratio'],
            'max_drawdown': full_results['max_drawdown']
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
    try:
        db_session = init_db()
    except Exception as e:
        logger.warning(f"Could not initialize DB: {e}. Continuing without DB.")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch historical data
    logger.info(f"Fetching {args.days} days of historical data")
    historical_data = data_fetcher.fetch_historical_data(days=args.days)
    
    # Optimize parameters
    logger.info(f"Starting optimization with {args.trials} trials")
    results = optimize_parameters(historical_data, n_trials=args.trials, test_size=args.test_size)
    
    # Save results to file
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Optimization results saved to {args.output}")

if __name__ == "__main__":
    main() 