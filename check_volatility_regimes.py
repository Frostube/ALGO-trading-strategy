#!/usr/bin/env python3
"""
Volatility Regime Monitor CLI

This script checks the current volatility regime for specified symbols,
showing the current volatility level, regime classification, and risk adjustments.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk.vol_regime_switch import VolatilityRegimeMonitor
from src.data.fetcher import fetch_ohlcv
from src.utils.logger import logger

def main():
    """Main entry point for volatility regime monitor CLI"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Check volatility regimes for trading pairs")
    parser.add_argument("--symbols", type=str, 
                        default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT",
                        help="Comma-separated list of symbols to check")
    parser.add_argument("--timeframe", type=str, default="2h", 
                        help="Timeframe for analysis (default: 2h)")
    parser.add_argument("--days", type=int, default=45, 
                        help="Historical days to analyze (default: 45)")
    parser.add_argument("--plot", action="store_true", 
                        help="Generate volatility regime plots")
    parser.add_argument("--output", type=str, default="reports", 
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory if plotting is enabled
    if args.plot:
        os.makedirs(args.output, exist_ok=True)
    
    # Initialize volatility regime monitor
    monitor = VolatilityRegimeMonitor(lookback_days=30)
    
    # Parse symbols list
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Print header
    print("\n==== Volatility Regime Monitor ====")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback: 30 days, Timeframe: {args.timeframe}")
    print("====================================\n")
    
    # Create a table for results
    results = []
    
    # Process each symbol
    for symbol in symbols:
        print(f"Checking {symbol}...")
        
        # Fetch historical data
        df = fetch_ohlcv(symbol, args.timeframe, days=args.days)
        
        if df is None or len(df) < 100:
            print(f"  ERROR: Insufficient data for {symbol}")
            continue
            
        # Update regime
        regime_info = monitor.update_regime(symbol, df)
        
        # Get risk adjustment
        risk_adjustment = monitor.get_risk_adjustment(symbol)
        
        # Get strategy for regime
        strategy = monitor.get_strategy_for_regime(symbol)
        
        # Check pyramiding
        pyramiding = "Enabled" if monitor.should_enable_pyramiding(symbol) else "Disabled"
        
        # Add to results table
        results.append({
            'Symbol': symbol,
            'Volatility': f"{regime_info['volatility']:.2f}%",
            'Regime': regime_info['regime'],
            'Risk Adj': f"{risk_adjustment:.2f}×",
            'Strategy': strategy,
            'Pyramiding': pyramiding
        })
        
        # Generate and save plot if requested
        if args.plot:
            plot_path = os.path.join(args.output, f"{symbol.replace('/', '_')}_volatility.png")
            monitor.plot_volatility_history(symbol, save_path=plot_path)
            print(f"  Volatility plot saved to {plot_path}")
            
    # Print results table
    if results:
        print("\nVolatility Regime Summary:")
        print("---------------------------")
        
        # Calculate column widths
        col_widths = {}
        for col in results[0].keys():
            col_widths[col] = max(len(col), max(len(str(row[col])) for row in results))
            
        # Print header
        header = "  ".join(col.ljust(col_widths[col]) for col in results[0].keys())
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in results:
            print("  ".join(str(val).ljust(col_widths[col]) for col, val in row.items()))
            
    # Print recommendations
    print("\nRecommended Actions:")
    print("-------------------")
    for row in results:
        regime = row['Regime']
        symbol = row['Symbol']
        
        if regime == 'QUIET':
            print(f"{symbol}: Use MEAN REVERSION strategy with 50% position size")
        elif regime == 'EXPLOSIVE':
            print(f"{symbol}: Use BREAKOUT strategy with 100% position size, pyramiding enabled")
        else:
            print(f"{symbol}: Continue with EMA CROSSOVER strategy with 75% position size")
            
    print("\nDone!")

if __name__ == "__main__":
    main() 