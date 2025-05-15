#!/usr/bin/env python3
"""
EMA Grid Search Runner

This script runs a comprehensive grid search for optimal EMA parameters
across multiple symbols. It finds the best combinations of fast/slow EMAs,
trend filters, and ATR multipliers with walk-forward validation.

Results are saved to the params/ directory for use by the trading strategies.
"""

import os
import sys
import argparse
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategy.ema_optimizer import optimize_ema_parameters, save_parameters
from src.utils.logger import logger

def main():
    """Main entry point for EMA grid search"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run EMA grid search optimization")
    parser.add_argument("--symbols", type=str, 
                        default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,ADA/USDT,AVAX/USDT",
                        help="Comma-separated list of symbols to optimize")
    parser.add_argument("--timeframe", type=str, default="2h", 
                        help="Timeframe for analysis (default: 2h)")
    parser.add_argument("--days", type=int, default=270, 
                        help="Historical days to analyze (default: 270)")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of parallel processes (default: 4)")
    parser.add_argument("--output", type=str, default="params", 
                        help="Output directory for parameter files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Parse symbols list
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Print configuration
    print(f"\n==== EMA Grid Search Configuration ====")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Historical data: {args.days} days")
    print(f"Output directory: {args.output}")
    print(f"Parallel workers: {args.workers}")
    print(f"======================================\n")
    
    # Optimization start time
    start_time = time.time()
    
    # Run optimization for each symbol
    all_results = []
    for i, symbol in enumerate(symbols):
        print(f"\n[{i+1}/{len(symbols)}] Optimizing {symbol} on {args.timeframe} timeframe...")
        
        # Run optimization
        symbol_start = time.time()
        results = optimize_ema_parameters(
            symbol=symbol,
            timeframe=args.timeframe,
            history_days=args.days,
            max_workers=args.workers
        )
        
        # Log results
        symbol_duration = (time.time() - symbol_start) / 60
        print(f"Optimization for {symbol} completed in {symbol_duration:.1f} minutes")
        print(f"Found {len(results)} valid parameter sets")
        
        # Add to all results
        all_results.extend(results)
    
    # Save all parameters
    save_parameters(all_results, args.output)
    
    # Calculate total duration
    total_duration = (time.time() - start_time) / 60
    
    # Print summary
    print(f"\n==== EMA Grid Search Summary ====")
    print(f"Total optimization time: {total_duration:.1f} minutes")
    print(f"Total symbols optimized: {len(symbols)}")
    print(f"Total valid parameter sets: {len(all_results)}")
    print(f"Parameters saved to: {args.output}")
    print(f"=================================\n")
    
    # Print next steps
    print("Next steps:")
    print(" 1. The strategies will now automatically use these optimized EMA parameters")
    print(" 2. For each symbol, the top parameters were saved as JSON files")
    print(" 3. You can view the parameters in the params/ directory")
    print(" 4. To update parameters in the future, run this script again")

if __name__ == "__main__":
    main() 