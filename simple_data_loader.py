#!/usr/bin/env python3
"""
Simple Data Loader

This module provides a simple data loading functionality for the TMA strategy testing.
It generates synthetic price data with realistic patterns for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(symbol='BTC/USDT', timeframe='4h', start_date='2022-01-01', end_date=None):
    """
    Load historical price data for testing.
    
    Since we don't have access to real market data in this environment,
    this function generates synthetic price data with realistic patterns.
    
    Args:
        symbol (str): Trading symbol (e.g., 'BTC/USDT')
        timeframe (str): Timeframe (e.g., '1h', '4h', '1d')
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format, or None for current date
        
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Parse dates
    start = pd.to_datetime(start_date)
    if end_date:
        end = pd.to_datetime(end_date)
    else:
        end = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    # Determine time step based on timeframe
    if timeframe == '1h':
        step = timedelta(hours=1)
    elif timeframe == '4h':
        step = timedelta(hours=4)
    elif timeframe == '1d':
        step = timedelta(days=1)
    else:
        step = timedelta(hours=4)  # Default to 4h
    
    # Create date range
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += step
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducibility
    
    # Initial price based on symbol
    if 'BTC' in symbol:
        initial_price = 50000  # BTC starting price
    elif 'ETH' in symbol:
        initial_price = 3000   # ETH starting price
    else:
        initial_price = 100    # Default starting price
    
    # Generate realistic price movement - reduced volatility from 0.02 to 0.01
    n = len(dates)
    price_changes = np.random.normal(0, 0.01, n)  # Daily returns with 1% volatility
    
    # Add some trends and cycles - reduced trend from 0.5 to 0.2
    trend = np.linspace(0, 0.2, n)  # Moderate upward trend
    cycles = 0.05 * np.sin(np.linspace(0, 10, n)) + 0.03 * np.sin(np.linspace(0, 25, n))
    
    # Combine components - scale down the overall movement
    cumulative_returns = price_changes + trend/n + cycles
    cumulative_returns = np.cumsum(cumulative_returns)
    
    # Limit the maximum growth to a realistic range (max 100% growth)
    max_return = 1.0
    if cumulative_returns[-1] > max_return:
        cumulative_returns = cumulative_returns * (max_return / cumulative_returns[-1])
    
    # Calculate prices
    closes = initial_price * (1 + cumulative_returns)
    
    # Generate OHLC based on close prices - reduced volatility
    highs = closes * (1 + np.random.uniform(0, 0.015, n))
    lows = closes * (1 - np.random.uniform(0, 0.015, n))
    opens = closes.copy()
    opens[1:] = closes[:-1]
    opens[0] = opens[1] * (1 + np.random.normal(0, 0.005))
    
    # Generate volume
    volumes = initial_price * closes * np.random.uniform(0.5, 1.5, n) * 10
    
    # Create DataFrame
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    print(f"Generated synthetic {symbol} data on {timeframe} timeframe from {start_date} to {end or 'now'}")
    print(f"Initial price: ${initial_price:.2f}, Final price: ${closes[-1]:.2f}")
    print(f"Total rows: {len(df)}")
    
    return df 