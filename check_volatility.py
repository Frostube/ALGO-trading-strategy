#!/usr/bin/env python3
"""
Check Volatility Regimes

Command-line tool to check current volatility regimes for all trading symbols.
Used to determine market conditions and adjust strategy parameters accordingly.
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk.volatility_monitor import VolatilityMonitor
from src.utils.logger import logger

def main():
    """Main entry point for volatility regime check"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Check current volatility regimes")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT",
                        help="Comma-separated list of symbols to monitor")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe for volatility calculation")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback period in days")
    parser.add_argument("--plot", action="store_true", help="Generate volatility plots")
    parser.add_argument("--plot-days", type=int, default=90, help="Days to include in plot")
    parser.add_argument("--force-update", action="store_true", help="Force recalculation of volatility")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create volatility monitor
    symbols = args.symbols.split(",")
    monitor = VolatilityMonitor(symbols=symbols, timeframe=args.timeframe, lookback=args.lookback)
    
    # Update all symbols if forced or if no cached data
    if args.force_update:
        print("Forcing recalculation of volatility metrics...")
        monitor.update_all_symbols()
    
    # Get the current regimes
    regimes = monitor.get_all_regimes()
    
    # If no regime data, update now
    if not regimes:
        print("No cached data found. Calculating volatility metrics...")
        monitor.update_all_symbols()
        regimes = monitor.get_all_regimes()
    
    # Print results
    print(f"\nVolatility Regimes ({args.timeframe}, {args.lookback}-day lookback):")
    print("-" * 60)
    print(f"{'Symbol':<12} {'Volatility':<12} {'Regime':<10} {'Updated'}")
    print("-" * 60)
    
    for symbol in symbols:
        if symbol in regimes:
            data = regimes[symbol]
            vol_pct = data["volatility"] * 100
            regime = data["regime"].upper()
            updated = datetime.fromisoformat(data["updated_at"]).strftime("%Y-%m-%d %H:%M")
            print(f"{symbol:<12} {vol_pct:>6.2f}%      {regime:<10} {updated}")
        else:
            print(f"{symbol:<12} {'N/A':<12} {'N/A':<10} {'N/A'}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating volatility plots...")
        os.makedirs("reports/volatility", exist_ok=True)
        
        for symbol in symbols:
            save_path = f"reports/volatility/{symbol.replace('/', '_')}_vol_{args.timeframe}.png"
            monitor.plot_volatility_history(symbol, days=args.plot_days, save_path=save_path)
            print(f"  Plot saved: {save_path}")
    
    print("\nTrading Implications:")
    print("-" * 60)
    for symbol in symbols:
        if symbol in regimes:
            data = regimes[symbol]
            regime = data["regime"]
            vol_pct = data["volatility"] * 100
            
            if regime == "calm":
                print(f"{symbol:<12} CALM ({vol_pct:.2f}%) → halve position sizes, disable pyramiding")
            elif regime == "storm":
                print(f"{symbol:<12} STORM ({vol_pct:.2f}%) → full position sizes, enable pyramiding")
            else:
                print(f"{symbol:<12} NORMAL ({vol_pct:.2f}%) → standard position sizes")
        else:
            print(f"{symbol:<12} N/A → using default settings")

if __name__ == "__main__":
    main() 