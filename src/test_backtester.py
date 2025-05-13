#!/usr/bin/env python3
"""
Test script to check if the backtester works with different timeframes.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.fetcher import DataFetcher
from src.backtest.backtest import Backtester
from src.indicators.technical import apply_indicators
from src.utils.logger import logger

def test_backtest_timeframes():
    """Test backtesting with different timeframes."""
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h']
    results = {}
    
    for tf in timeframes:
        print(f"\n=== Testing backtester with timeframe: {tf} ===")
        
        try:
            # Step 1: Fetch data
            print(f"1. Fetching data for timeframe {tf}...")
            fetcher = DataFetcher(use_testnet=True)
            data = fetcher.fetch_historical_data(days=7, timeframe=tf)
            fetcher.close()
            
            if data is None or data.empty:
                print(f"  ❌ Failed: Could not fetch data for {tf}")
                results[tf] = {"success": False, "stage": "data_fetch"}
                continue
                
            print(f"  ✅ Data fetched: {len(data)} rows")
            
            # Step 2: Apply indicators
            print(f"2. Applying indicators for timeframe {tf}...")
            try:
                data_with_indicators = apply_indicators(data)
                print(f"  ✅ Indicators applied successfully")
                # Print sample indicator values
                print(f"  Sample indicator values:")
                if not data_with_indicators.empty:
                    sample = data_with_indicators.iloc[-1]
                    print(f"    - EMA Trend: {sample.get('ema_trend', 'N/A')}")
                    print(f"    - RSI: {sample.get('rsi', 'N/A')}")
                    print(f"    - Volume Spike: {sample.get('volume_spike', 'N/A')}")
            except Exception as e:
                print(f"  ❌ Failed to apply indicators: {str(e)}")
                results[tf] = {"success": False, "stage": "indicators", "error": str(e)}
                continue
            
            # Step 3: Run backtest
            print(f"3. Running backtest for timeframe {tf}...")
            try:
                backtester = Backtester(data=data_with_indicators, initial_balance=10000)
                backtest_results = backtester.run()
                
                # Check results
                if backtest_results:
                    print(f"  ✅ Backtest completed with {backtest_results.get('total_trades', 0)} trades")
                    results[tf] = {
                        "success": True,
                        "trades": backtest_results.get('total_trades', 0),
                        "return": backtest_results.get('total_return', 0),
                        "win_rate": backtest_results.get('win_rate', 0)
                    }
                else:
                    print(f"  ❌ Backtest returned no results")
                    results[tf] = {"success": False, "stage": "backtest_empty"}
            except Exception as e:
                print(f"  ❌ Failed to run backtest: {str(e)}")
                import traceback
                traceback.print_exc()
                results[tf] = {"success": False, "stage": "backtest", "error": str(e)}
                
        except Exception as e:
            print(f"  ❌ General failure: {str(e)}")
            results[tf] = {"success": False, "stage": "general", "error": str(e)}
    
    # Print summary
    print("\n=== SUMMARY ===")
    for tf, result in results.items():
        if result.get("success", False):
            print(f"{tf}: ✅ Success - {result.get('trades', 0)} trades, Return: {result.get('return', 0)*100:.2f}%, Win Rate: {result.get('win_rate', 0)*100:.2f}%")
        else:
            print(f"{tf}: ❌ Failed at stage: {result.get('stage')} - {result.get('error', 'Unknown error')}")
    
    return results

if __name__ == "__main__":
    test_backtest_timeframes() 