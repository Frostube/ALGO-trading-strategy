import argparse
import webbrowser
import os
from src.utils.walk_forward import run_wfo, analyze_market_regimes
from src.data.fetcher import fetch_ohlcv

def main():
    p = argparse.ArgumentParser(description="Run walk-forward optimization for trading strategies")
    p.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol (e.g. BTC/USDT)')
    p.add_argument('--timeframe', default='4h', help='Candlestick timeframe (e.g. 1h, 4h, 1d)')
    p.add_argument('--years', type=int, default=2, help='Number of years of historical data to fetch')
    p.add_argument('--train-days', type=int, default=90, help='Number of days in training window')
    p.add_argument('--test-days', type=int, default=30, help='Number of days in test window')
    p.add_argument('--exit-strategy', choices=['fixed', 'trailing', 'time', 'atr'], 
                  help='Exit strategy type (fixed, trailing, time-based, or ATR-based)')
    p.add_argument('--take-profit', type=float, help='Take profit percentage (for fixed exit)')
    p.add_argument('--stop-loss', type=float, help='Stop loss percentage (for fixed exit)')
    p.add_argument('--trail-pct', type=float, help='Trailing stop percentage (for trailing exit)')
    p.add_argument('--max-bars', type=int, help='Maximum number of bars to hold (for time exit)')
    p.add_argument('--analyze-regimes', action='store_true', help='Analyze market regimes before optimization')
    p.add_argument('--open-dashboard', action='store_true', help='Automatically open dashboard in browser')
    p.add_argument('--output-dir', default='wfo_results', help='Directory to save results and visualizations')
    p.add_argument('--parameter-grid', help='JSON string with parameter grid, e.g. \'{"fast_ema":[3,5,8],"slow_ema":[13,21,34]}\'')
    
    args = p.parse_args()

    print(f"WFO >> {args.symbol}@{args.timeframe}, {args.years}yr, "
          f"train {args.train_days}d/test {args.test_days}d")
    
    # Fetch data once to avoid duplication
    print(f"Fetching historical data for {args.symbol}...")
    df = fetch_ohlcv(symbol=args.symbol, tf=args.timeframe, days=args.years*365)
    if df.empty:
        print("No data fetched. Exiting.")
        return
    
    # Analyze market regimes if requested
    if args.analyze_regimes:
        print(f"Analyzing market regimes for {args.symbol}...")
        regime_df = analyze_market_regimes(df, args.symbol, output_dir=args.output_dir)
        print("Market regime analysis complete.")
    
    # Prepare exit strategy parameters
    exit_strategy = args.exit_strategy
    exit_params = None
    
    if exit_strategy:
        exit_params = {}
        if exit_strategy == 'fixed':
            if args.take_profit:
                exit_params['take_profit_pct'] = args.take_profit / 100  # Convert to decimal
            if args.stop_loss:
                exit_params['stop_loss_pct'] = args.stop_loss / 100  # Convert to decimal
        elif exit_strategy == 'trailing':
            if args.trail_pct:
                exit_params['trail_pct'] = args.trail_pct / 100  # Convert to decimal
        elif exit_strategy == 'time':
            if args.max_bars:
                exit_params['max_bars'] = args.max_bars
        
        print(f"Using {exit_strategy} exit strategy with parameters: {exit_params}")
    
    # Parse parameter grid if provided
    parameter_grid = None
    if args.parameter_grid:
        import json
        try:
            parameter_grid = json.loads(args.parameter_grid)
            print(f"Using custom parameter grid: {parameter_grid}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON for parameter grid. Using default grid.")
    
    # Run walk-forward optimization
    out_df, best_df = run_wfo(
        args.symbol, 
        args.timeframe, 
        args.years,
        args.train_days, 
        args.test_days,
        exit_strategy,
        exit_params,
        output_dir=args.output_dir,
        parameter_grid=parameter_grid
    )
    
    # Open dashboard in browser if requested
    if args.open_dashboard and best_df is not None:
        dashboard_files = [f for f in os.listdir(args.output_dir) if f.startswith('dashboard_')]
        if dashboard_files:
            newest_dashboard = sorted(dashboard_files)[-1]
            dashboard_path = os.path.join(args.output_dir, newest_dashboard)
            print(f"Opening dashboard: {dashboard_path}")
            try:
                webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
            except Exception as e:
                print(f"Could not open browser: {e}")

if __name__ == '__main__':
    main() 