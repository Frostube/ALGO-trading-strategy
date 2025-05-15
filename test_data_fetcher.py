#!/usr/bin/env python3
"""
Test DataFetcher with Testnet

This script tests the DataFetcher class with Binance Testnet
to verify that it can connect and fetch data properly.
"""
import os
import sys
import asyncio
from src.data.fetcher import DataFetcher
from src.utils.logger import setup_logger

async def main():
    # Set up logging
    logger = setup_logger(debug=True)
    
    try:
        # Initialize DataFetcher with testnet enabled
        logger.info("Initializing DataFetcher with testnet enabled...")
        fetcher = DataFetcher(use_testnet=True)
        
        # Test fetching historical data
        logger.info("Fetching historical data from testnet...")
        df = fetcher.fetch_historical_data(days=7)  # Just get 7 days to keep it quick
        
        # Display information about the retrieved data
        print("\n=== Historical Data from Testnet ===")
        print(f"Symbol: {fetcher.symbol}")
        print(f"Timeframe: {fetcher.timeframe}")
        print(f"Rows retrieved: {len(df)}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        # Display the first 5 rows
        print("\nFirst 5 candles:")
        print(df.head())
        
        # Display stats
        print("\nPrice statistics:")
        print(df['close'].describe())
        
        # Clean up
        fetcher.close()
        
        print("\nSuccessfully tested DataFetcher with Binance Testnet!")
        return 0
        
    except Exception as e:
        logger.error(f"Error testing DataFetcher with testnet: {str(e)}")
        return 1
    
if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 