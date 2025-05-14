#!/usr/bin/env python3

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_bot")

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.fetcher import DataFetcher
from src.backtest.backtest import Backtester
from src.config_optimized import (
    SYMBOL, TIMEFRAME, HIGHER_TIMEFRAME, HISTORICAL_DATA_DAYS
)

def run_backtest():
    """
    Run backtest with the optimized strategy and return results
    """
    try:
        logger.info(f"Starting backtest for {SYMBOL} with optimized strategy")
        
        # Initialize data fetcher with testnet (will use mock data if testnet unavailable)
        fetcher = DataFetcher(use_testnet=True)
        
        # Fetch historical data
        data = fetcher.fetch_historical_data(
            days=HISTORICAL_DATA_DAYS,
            timeframe=TIMEFRAME
        )
        
        # Fetch higher timeframe data for confirmation
        higher_tf_data = fetcher.fetch_historical_data(
            days=HISTORICAL_DATA_DAYS,
            timeframe=HIGHER_TIMEFRAME
        )
        
        if data.empty:
            logger.error("No data available for backtest")
            return None, None
            
        logger.info(f"Fetched {len(data)} candles for {TIMEFRAME} and {len(higher_tf_data)} candles for {HIGHER_TIMEFRAME}")
        
        # Initialize backtester with optimized parameters
        backtester = Backtester(
            data=data,
            initial_balance=10000,
            use_optimized=True
        )
        
        # Run backtest with 70/30 train/test split
        results = backtester.run(train_test_split=0.7)
        
        # Print summary
        train_return = results['train']['total_return'] * 100
        test_return = results['test']['total_return'] * 100
        
        logger.info(f"Train Return: {train_return:.2f}%, Test Return: {test_return:.2f}%")
        logger.info(f"Train Win Rate: {results['train']['win_rate']*100:.2f}%, Test Win Rate: {results['test']['win_rate']*100:.2f}%")
        
        # Check if we achieved our 20% goal
        if test_return >= 20:
            logger.info("🎉 SUCCESS! Strategy achieved target of 20%+ return in test dataset")
        else:
            logger.warning(f"Strategy achieved {test_return:.2f}% in test dataset, below 20% target")
        
        return results, backtester
        
    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def main():
    """
    Main function to run the optimized strategy and launch the dashboard
    """
    parser = argparse.ArgumentParser(description="Run optimized trading strategy with enhanced dashboard")
    parser.add_argument('--port', type=int, default=8505, help='Port to run the Streamlit dashboard on')
    parser.add_argument('--backtest-only', action='store_true', help='Run backtest only without launching dashboard')
    args = parser.parse_args()
    
    # Run backtest first
    results, backtester = run_backtest()
    
    if results is None:
        logger.error("Backtest failed, exiting.")
        return
    
    # Save backtest results to file for analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/backtest_results_{timestamp}.json"
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Convert results to serializable format and save
    serializable_results = {
        'train': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                 for k, v in results['train'].items() if k != 'trades' and k != 'price_data'},
        'test': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                for k, v in results['test'].items() if k != 'trades' and k != 'price_data'}
    }
    
    import json
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f)
    
    logger.info(f"Saved backtest results to {results_file}")
    
    # Check if we just want to run the backtest
    if args.backtest_only:
        return
    
    # Launch the enhanced dashboard with the results
    port = args.port
    logger.info(f"Starting enhanced Streamlit dashboard on port {port}...")
    
    # Set environment variable to use optimized config
    os.environ['USE_OPTIMIZED_CONFIG'] = 'true'
    
    # Construct command to run the dashboard
    cmd = [
        "streamlit", "run", 
        "src/dashboard/enhanced_dashboard.py",
        "--server.port", str(port),
        "--"  # Arguments after this are passed to the script
    ]
    
    # Run the dashboard
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 