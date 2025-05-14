import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pathlib
import time
import logging

from src.config import (
    EXCHANGE, SYMBOL, TIMEFRAME, HIGHER_TIMEFRAME, 
    BINANCE_API_KEY, BINANCE_SECRET_KEY, FUTURES
)
from src.utils.logger import logger
from src.data.fetcher import fetch_ohlcv, DataFetcher

def load_data(symbol=SYMBOL, timeframe=TIMEFRAME, start_date=None, end_date=None, days=None):
    """
    Load OHLCV data for a given date range.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candlestick timeframe ('1m', '15m', '1h', '4h', '1d', etc.)
        start_date: Start date for data (datetime)
        end_date: End date for data (datetime)
        days: Number of days to fetch (alternative to start_date)
        
    Returns:
        DataFrame with OHLCV data for the specified date range
    """
    # Configure date range
    if end_date is None:
        end_date = datetime.now()
        
    if start_date is None:
        if days is not None:
            start_date = end_date - timedelta(days=days)
        else:
            start_date = end_date - timedelta(days=90)  # Default to 90 days
    
    days_to_fetch = (end_date - start_date).days + 1
    
    logger.info(f"Loading data for {symbol} ({timeframe}) from {start_date.date()} to {end_date.date()}")
    
    try:
        # Fetch data using the existing function
        df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days_to_fetch)
        
        if df.empty:
            logger.warning(f"No data found for {symbol} ({timeframe})")
            return df
        
        # Filter to requested date range
        mask = (df.index >= start_date) & (df.index <= end_date)
        filtered_df = df.loc[mask]
        
        logger.info(f"Loaded {len(filtered_df)} candles for {symbol} ({timeframe})")
        
        return filtered_df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        
        # Try fallback to mock data if real data fails
        try:
            logger.warning("Attempting to use mock data as fallback")
            data_fetcher = DataFetcher(use_mock=True)
            mock_df = data_fetcher.generate_mock_data(days=days_to_fetch, timeframe=timeframe)
            
            # Filter to requested date range
            mask = (mock_df.index >= start_date) & (mock_df.index <= end_date)
            filtered_mock_df = mock_df.loc[mask]
            
            logger.info(f"Generated {len(filtered_mock_df)} mock candles for {symbol} ({timeframe})")
            
            return filtered_mock_df
        except Exception as mock_error:
            logger.error(f"Error generating mock data: {str(mock_error)}")
            return pd.DataFrame()  # Return empty DataFrame on complete failure 