#!/usr/bin/env python3
"""
Test script to check if DataFetcher works with different timeframes.
"""

import os
import sys
import pandas as pd

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import DataFetcher
from src.utils.logger import logger

def test_timeframes():
    """Test DataFetcher with different timeframes."""
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']
    results = {}
    
    for tf in timeframes:
        print(f"Testing timeframe: {tf}")
        fetcher = DataFetcher(use_testnet=True)
        data = fetcher.fetch_historical_data(days=7, timeframe=tf)
        fetcher.close()
        
        if data is not None and not data.empty:
            results[tf] = {
                'rows': len(data),
                'columns': list(data.columns),
                'start': data.index[0],
                'end': data.index[-1],
                'success': True
            }
            print(f"  ✓ Success: {len(data)} rows fetched")
        else:
            results[tf] = {
                'success': False
            }
            print(f"  ✗ Failed: No data or empty DataFrame")
    
    print("\nSummary:")
    for tf, result in results.items():
        status = "Success" if result['success'] else "Failed"
        print(f"{tf}: {status}")
        if result['success']:
            print(f"  - Rows: {result['rows']}")
            print(f"  - Date range: {result['start']} to {result['end']}")
    
    return results

if __name__ == "__main__":
    test_timeframes() 