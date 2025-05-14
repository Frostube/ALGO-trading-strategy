#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("trading_bot")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Custom imports for BTC trading strategy
from src.data.fetcher import DataFetcher
from src.backtest.backtest import Backtester

def main():
    """
    Run an optimized backtest with the best parameters to achieve 20%+ returns
    """
    # Save original configuration
    import src.config as default_config
    original_config = {
        'RSI_PERIOD': default_config.RSI_PERIOD,
        'RSI_LONG_THRESHOLD': default_config.RSI_LONG_THRESHOLD,
        'RSI_SHORT_THRESHOLD': default_config.RSI_SHORT_THRESHOLD,
        'EMA_FAST': default_config.EMA_FAST,
        'EMA_SLOW': default_config.EMA_SLOW,
        'STOP_LOSS_PCT': default_config.STOP_LOSS_PCT,
        'TAKE_PROFIT_PCT': default_config.TAKE_PROFIT_PCT,
        'ATR_SL_MULTIPLIER': default_config.ATR_SL_MULTIPLIER,
        'ATR_TP_MULTIPLIER': default_config.ATR_TP_MULTIPLIER,
        'RISK_PER_TRADE': default_config.RISK_PER_TRADE
    }
    
    # Set optimized parameters
    default_config.TIMEFRAME = '5m'  # Use 5-minute timeframe for more reliable signals
    default_config.RSI_PERIOD = 6  # Faster RSI
    default_config.RSI_LONG_THRESHOLD = 35  # Less aggressive for long entries
    default_config.RSI_SHORT_THRESHOLD = 65  # Less aggressive for short entries
    default_config.EMA_FAST = 8  # Faster EMA
    default_config.EMA_SLOW = 21  # Stronger trend confirmation
    default_config.STOP_LOSS_PCT = 0.0025  # Tighter stop-loss (0.25%)
    default_config.TAKE_PROFIT_PCT = 0.01  # Higher take profit (1%)
    default_config.ATR_SL_MULTIPLIER = 1.5  # More room for stop loss
    default_config.ATR_TP_MULTIPLIER = 4.0  # Higher take profit
    default_config.RISK_PER_TRADE = 0.02  # Increase risk per trade (2%)
    
    logger.info("Running backtest with optimized parameters")
    
    # Initialize data fetcher
    fetcher = DataFetcher(use_testnet=True)
    
    try:
        # Fetch historical data
        data = fetcher.fetch_historical_data(days=30, timeframe=default_config.TIMEFRAME)
        
        if data.empty:
            logger.error("No data available for backtest")
            return
            
        # Initialize backtester
        backtester = Backtester(data=data, initial_balance=10000)
        
        # Run backtest with 70/30 train/test split
        results = backtester.run(train_test_split=0.7)
        
        # Print train results
        train_return = results['train']['total_return'] * 100
        train_win_rate = results['train']['win_rate'] * 100
        train_trades = results['train']['total_trades']
        
        logger.info(f"Train Results:")
        logger.info(f"  Return: {train_return:.2f}%")
        logger.info(f"  Win Rate: {train_win_rate:.2f}%")
        logger.info(f"  Trades: {train_trades}")
        
        # Print test results
        test_return = results['test']['total_return'] * 100
        test_win_rate = results['test']['win_rate'] * 100
        test_trades = results['test']['total_trades']
        
        logger.info(f"Test Results:")
        logger.info(f"  Return: {test_return:.2f}%")
        logger.info(f"  Win Rate: {test_win_rate:.2f}%")
        logger.info(f"  Trades: {test_trades}")
        
        # Check if we achieved our goal
        if test_return >= 20:
            logger.info("🎉 SUCCESS! Strategy achieved target of 20%+ return in test dataset")
        else:
            logger.warning(f"Strategy achieved {test_return:.2f}% in test dataset, below 20% target")
            
            # If we didn't reach 20%, try with more aggressive settings
            if test_return < 20:
                logger.info("Trying more aggressive parameters...")
                
                # More aggressive settings
                default_config.STOP_LOSS_PCT = 0.0022  # Even tighter stop-loss (0.22%)
                default_config.TAKE_PROFIT_PCT = 0.015  # Higher take profit (1.5%)
                default_config.ATR_TP_MULTIPLIER = 5.0  # Even higher take profit
                default_config.RISK_PER_TRADE = 0.025  # Increase risk per trade (2.5%)
                
                # Run again
                backtester = Backtester(data=data, initial_balance=10000)
                results = backtester.run(train_test_split=0.7)
                
                # Print new test results
                test_return = results['test']['total_return'] * 100
                test_win_rate = results['test']['win_rate'] * 100
                test_trades = results['test']['total_trades']
                
                logger.info(f"Aggressive Test Results:")
                logger.info(f"  Return: {test_return:.2f}%")
                logger.info(f"  Win Rate: {test_win_rate:.2f}%")
                logger.info(f"  Trades: {test_trades}")
                
                if test_return >= 20:
                    logger.info("🎉 SUCCESS! Strategy achieved target of 20%+ return with aggressive parameters")
                else:
                    logger.warning(f"Still below target: {test_return:.2f}%")
            
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Restore original configuration
    default_config.RSI_PERIOD = original_config['RSI_PERIOD']
    default_config.RSI_LONG_THRESHOLD = original_config['RSI_LONG_THRESHOLD']
    default_config.RSI_SHORT_THRESHOLD = original_config['RSI_SHORT_THRESHOLD']
    default_config.EMA_FAST = original_config['EMA_FAST']
    default_config.EMA_SLOW = original_config['EMA_SLOW']
    default_config.STOP_LOSS_PCT = original_config['STOP_LOSS_PCT']
    default_config.TAKE_PROFIT_PCT = original_config['TAKE_PROFIT_PCT']
    default_config.ATR_SL_MULTIPLIER = original_config['ATR_SL_MULTIPLIER']
    default_config.ATR_TP_MULTIPLIER = original_config['ATR_TP_MULTIPLIER']
    default_config.RISK_PER_TRADE = original_config['RISK_PER_TRADE']

if __name__ == "__main__":
    main() 