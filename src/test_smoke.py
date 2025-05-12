#!/usr/bin/env python3
"""
Smoke test for the BTC/USDT Intra-Day Scalper.
This script tests basic functionality to ensure everything is working properly.
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db.models import init_db, OHLCV
from src.indicators.technical import apply_indicators
from src.data.fetcher import DataFetcher
from src.utils.logger import logger

def test_database_connection():
    """Test database connection and initialization."""
    logger.info("Testing database connection...")
    try:
        db_session = init_db()
        logger.info("Database connection successful!")
        return db_session
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return None

def test_data_fetching():
    """Test fetching sample data (from database or via API)."""
    logger.info("Testing data fetching...")
    try:
        # Try to fetch from database first
        db_session = init_db()
        sample_data = db_session.query(OHLCV).limit(10).all()
        
        if sample_data:
            logger.info(f"Found {len(sample_data)} candles in database")
        else:
            # If no data in database, try to fetch from API
            logger.info("No data in database, fetching from API...")
            data_fetcher = DataFetcher()
            # Fetch just 1 day of data for testing
            sample_data = data_fetcher.fetch_historical_data(days=1)
            logger.info(f"Fetched {len(sample_data)} candles from API")
            data_fetcher.close()
        
        return sample_data
    except Exception as e:
        logger.error(f"Data fetching failed: {str(e)}")
        return None

def test_indicators():
    """Test indicator calculations."""
    logger.info("Testing indicators...")
    try:
        # Create a sample dataframe
        dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=100, freq='1min')
        sample_df = pd.DataFrame({
            'open': [100.0 + i * 0.1 for i in range(100)],
            'high': [101.0 + i * 0.1 for i in range(100)],
            'low': [99.0 + i * 0.1 for i in range(100)],
            'close': [100.5 + i * 0.1 for i in range(100)],
            'volume': [1000.0 + i * 10 for i in range(100)]
        }, index=dates)
        
        # Apply indicators
        result_df = apply_indicators(sample_df)
        
        # Check if indicators were calculated
        required_columns = ['ema_9', 'ema_21', 'ema_trend', 'rsi', 'volume_spike']
        missing_columns = [col for col in required_columns if col not in result_df.columns]
        
        if missing_columns:
            logger.error(f"Missing indicator columns: {missing_columns}")
            return False
        else:
            logger.info("All indicators calculated successfully!")
            logger.info(f"Sample indicators: EMA9={result_df['ema_9'].iloc[-1]:.2f}, "
                       f"EMA21={result_df['ema_21'].iloc[-1]:.2f}, "
                       f"RSI={result_df['rsi'].iloc[-1]:.2f}")
            return True
    except Exception as e:
        logger.error(f"Indicator calculation failed: {str(e)}")
        return False

def main():
    """Run all smoke tests."""
    logger.info("Starting smoke tests...")
    
    # Test database connection
    db_session = test_database_connection()
    if not db_session:
        logger.error("Database test failed!")
    
    # Test data fetching
    data = test_data_fetching()
    if not data:
        logger.error("Data fetching test failed!")
    
    # Test indicators
    indicators_ok = test_indicators()
    if not indicators_ok:
        logger.error("Indicator test failed!")
    
    # Overall result
    if db_session and data and indicators_ok:
        logger.info("Smoke tests PASSED! System appears to be working correctly.")
        return True
    else:
        logger.error("Smoke tests FAILED! Please check the logs for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 