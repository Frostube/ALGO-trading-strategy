#!/usr/bin/env python3
"""
Runner script for paper trading with correct import paths.
"""
import os
import sys
import argparse

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BTC/USDT Intra-Day Scalper - Paper Trading Mode')
    parser.add_argument('--balance', type=float, default=10000, help='Initial account balance for paper trading')
    
    args = parser.parse_args()
    
    # Import and run the main function
    from src.main import main
    
    # Run the main function with paper trading mode (not live)
    import asyncio
    asyncio.run(main(args)) 