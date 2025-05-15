#!/usr/bin/env python3
"""
Debug test script for running focused grid search with permissive parameters
"""

import os
import sys
import logging
import traceback
from focused_grid_search import focused_grid_search

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Override sys.argv with our debug parameters
    sys.argv = [
        "focused_grid_search.py",
        "--symbols", "BTC/USDT", 
        "--timeframe", "1h",  # Use 1h instead of 4h for 6x more bars
        "--days", "90",  # Use more days
        "--debug_mode",  # Enable debug mode
        "--min_trades", "0"  # Accept any number of trades
    ]
    
    # Run the focused grid search with extended exception handling
    logger.info("Starting debug test with permissive parameters")
    try:
        results = focused_grid_search()
        
        # Report results
        if results:
            logger.info(f"Found {len(results)} valid parameter sets")
            for i, result in enumerate(results[:5]):
                logger.info(f"Result {i+1}:")
                logger.info(f"  Parameters: {result['params']}")
                logger.info(f"  Score: {result.get('weighted_score', 0):.4f}")
                logger.info(f"  Win Rate: {result['avg_test_win_rate']:.2f}%")
                logger.info(f"  Profit Factor: {result['avg_test_pf']:.2f}")
                logger.info(f"  Return: {result['avg_test_return']:.2f}%")
                logger.info(f"  Trades: {result['avg_test_trades']:.1f}")
        else:
            logger.info("No valid parameter sets found")
    except Exception as e:
        logger.error(f"Error running grid search: {str(e)}")
        # Print full traceback for detailed debugging
        logger.error(traceback.format_exc())
        
    logger.info("Debug test completed") 