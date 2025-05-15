#!/usr/bin/env python3
"""
Fee Impact Analysis

This script runs a backtest and analyzes the impact of trading fees
on strategy performance, generating visualizations showing the 
difference between gross and net returns.
"""
import argparse
import os
from pathlib import Path
from src.utils.visualization import plot_fee_impact, compare_fee_tiers, plot_equity_curves
from src.utils.logger import setup_logger

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze fee impact on trading strategy')
    parser.add_argument('--symbol', type=str, default="BTC/USDT",
                        help='Trading symbol to analyze')
    parser.add_argument('--timeframe', type=str, default="4h",
                        help='Trading timeframe')
    parser.add_argument('--days', type=int, default=90,
                        help='Number of days to backtest')
    parser.add_argument('--output', type=str, default="results",
                        help='Output directory for results (defaults to "results")')
    parser.add_argument('--compare-tiers', action='store_true',
                        help='Run fee tier comparison analysis')
    parser.add_argument('--equity-curves', action='store_true',
                        help='Generate equity curves showing gross vs net returns over time')
    parser.add_argument('--initial-balance', type=float, default=10000,
                        help='Initial account balance')
    parser.add_argument('--risk', type=float, default=0.01,
                        help='Risk per trade (as a decimal)')
    parser.add_argument('--test', action='store_true',
                        help='Use test data instead of running a real backtest')
    return parser.parse_args()

def main():
    """Run fee impact analysis"""
    args = parse_args()
    
    # Setup logging
    setup_logger(debug=True)
    
    # Create output directory (using os.makedirs which is more reliable cross-platform)
    os.makedirs(args.output, exist_ok=True)
    print(f"Output will be saved to: {os.path.abspath(args.output)}")
    
    # Run fee impact analysis with bar charts
    if not args.equity_curves:
        print(f"Analyzing fee impact for {args.symbol} over {args.days} days on {args.timeframe} timeframe...")
        
        results = plot_fee_impact(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            initial_balance=args.initial_balance,
            risk_per_trade=args.risk,
            save_path=args.output,
            use_test_data=args.test
        )
        
        if results:
            print("\nFee Impact Summary:")
            print(f"Gross P&L: ${results['gross_pnl']:.2f}")
            print(f"Net P&L: ${results['net_pnl']:.2f}")
            print(f"Fee Impact: {results['fee_impact_pct']:.2f}% of gross profits")
            print(f"Total Fees: ${results['total_fees']:.2f}")
            print(f"  - Commission: ${results['total_commission']:.2f}")
            print(f"  - Slippage: ${results['total_slippage']:.2f}")
            print(f"  - Funding: ${results['total_funding']:.2f}")
            print(f"Average Fee Impact Per Trade: {results['avg_fee_impact_pct']:.2f}%")
            
            # Check if fee impact is significant
            if results['fee_impact_pct'] > 25:
                print("\nWARNING: Transaction costs are consuming a significant portion of your gross profits.")
                print("Consider adjusting your strategy to reduce trading frequency or improve average return per trade.")
        else:
            print("Analysis failed - no trades generated in backtest.")
    
    # Run equity curves visualization
    if args.equity_curves:
        print(f"Generating equity curves for {args.symbol} over {args.days} days on {args.timeframe} timeframe...")
        
        curve_results = plot_equity_curves(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            initial_balance=args.initial_balance,
            risk_per_trade=args.risk,
            save_path=args.output,
            use_test_data=args.test
        )
        
        if curve_results:
            print("\nEquity Curves Summary:")
            print(f"Initial Balance: ${curve_results['initial_balance']:.2f}")
            print(f"Final Gross Equity: ${curve_results['final_gross_equity']:.2f} ({curve_results['gross_return_pct']:.2f}%)")
            print(f"Final Net Equity: ${curve_results['final_net_equity']:.2f} ({curve_results['net_return_pct']:.2f}%)")
            print(f"Fee Impact: {curve_results['fee_impact_pct']:.2f}% of gross return")
            print(f"Max Gross Drawdown: {curve_results['max_gross_drawdown']:.2f}%")
            print(f"Max Net Drawdown: {curve_results['max_net_drawdown']:.2f}%")
            print(f"Trade Count: {curve_results['trade_count']}")
            
            # Check if fee impact is significant
            if curve_results['fee_impact_pct'] > 25:
                print("\nWARNING: Transaction costs are consuming a significant portion of your gross profits.")
                print("Consider adjusting your strategy to reduce trading frequency or improve average return per trade.")
        else:
            print("Analysis failed - no trades generated in backtest.")
    
    # Run fee tier comparison if requested
    if args.compare_tiers:
        print("\nComparing different fee tiers...")
        tier_results = compare_fee_tiers(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            initial_balance=args.initial_balance,
            save_path=args.output,
            use_test_data=args.test
        )
        
        if tier_results:
            print("\nFee Tier Comparison Summary:")
            for tier_name, tier_data in sorted(tier_results.items(), 
                                            key=lambda x: x[1]['net_pnl'], 
                                            reverse=True):
                print(f"{tier_name}:")
                print(f"  Net P&L: ${tier_data['net_pnl']:.2f}")
                print(f"  Gross P&L: ${tier_data.get('gross_pnl', 0):.2f}")  
                print(f"  Total Fees: ${tier_data['total_fees']:.2f}")
                print(f"  ROI: {tier_data['roi']:.2f}%")
                print(f"  Total Trades: {tier_data['trades']}")
                print("")

if __name__ == "__main__":
    main() 