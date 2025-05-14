#!/usr/bin/env python3
import argparse
import traceback
import sys
from src.backtest.ema_backtest import run_ema_backtest

def main():
    parser = argparse.ArgumentParser(description='Run EMA Crossover backtest for BTC/USDT')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Trading timeframe')
    parser.add_argument('--days', type=int, default=30, help='Number of days of historical data to use')
    parser.add_argument('--balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--plot', action='store_true', help='Generate equity curve plot')
    parser.add_argument('--optimize', action='store_true', help='Find optimal parameters before backtesting')
    args = parser.parse_args()
    
    print(f"Running EMA Crossover backtest for {args.symbol} on {args.timeframe} timeframe")
    print(f"Days: {args.days}, Initial Balance: ${args.balance}")
    print(f"Optimize: {args.optimize}, Plot: {args.plot}")
    
    try:
        # Run the backtest
        results = run_ema_backtest(
            symbol=args.symbol,
            timeframe=args.timeframe,
            days=args.days,
            initial_balance=args.balance,
            plot=args.plot,
            optimize=args.optimize
        )
        
        # Print results
        print("\nBacktest Results Summary:")
        print("========================")
        print(f"Symbol: {args.symbol}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        print(f"Win Rate: {results['win_rate']*100:.2f}%")
        print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        # Print optimal parameters if using EMA strategy
        if args.optimize and 'strategy_params' in results:
            print(f"\nOptimal Parameters:")
            print(f"Fast EMA: {results['strategy_params'].get('fast_ema', 'N/A')}")
            print(f"Slow EMA: {results['strategy_params'].get('slow_ema', 'N/A')}")
        
        print("\nBacktest completed successfully")
        
        if args.plot:
            print(f"Plot saved to ema_backtest_{args.symbol.replace('/', '_')}_{args.timeframe}.png")
    
    except TypeError as e:
        if "unsupported operand type(s) for -: 'slice' and 'int'" in str(e):
            print(f"Error: Slicing operation failed. This might be due to incorrect handling of list or array indices.")
            print(f"Original error: {str(e)}")
            traceback.print_exc(file=sys.stdout)
        else:
            print(f"TypeError occurred: {str(e)}")
            traceback.print_exc(file=sys.stdout)
    except Exception as e:
        print(f"Error running backtest: {type(e).__name__}: {str(e)}")
        traceback.print_exc(file=sys.stdout)

if __name__ == '__main__':
    main() 