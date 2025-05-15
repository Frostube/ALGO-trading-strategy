import os
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.fetcher import fetch_ohlcv
from src.strategy.trend_filtered_adaptive_ema import TrendFilteredAdaptiveEMA
from src.strategy.regime_adaptive_ema import RegimeAdaptiveEMAStrategy
from src.backtest.backtest import backtest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run backtest with trend-filtered adaptive strategy")
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol (e.g. BTC/USDT)')
    parser.add_argument('--timeframe', default='4h', help='Candlestick timeframe (e.g. 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to fetch')
    parser.add_argument('--params_file', default='regime_params.csv', help='Path to regime parameters CSV file')
    parser.add_argument('--daily_ema', type=int, default=200, help='Period for daily EMA trend filter')
    parser.add_argument('--no_trend_filter', action='store_true', help='Disable daily trend filter')
    parser.add_argument('--vol_percentile', type=int, default=80, help='Percentile threshold for volatility filter (0-100)')
    parser.add_argument('--save_results', action='store_true', help='Save results to CSV')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with baseline adaptive strategy')
    
    args = parser.parse_args()
    
    # Ensure wfo_results directory exists for output
    os.makedirs('wfo_results', exist_ok=True)
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    
    # Fetch historical data
    logger.info(f"Fetching {args.days} days of {args.timeframe} data for {args.symbol}")
    df = fetch_ohlcv(args.symbol, args.timeframe, args.days)
    
    if df.empty:
        logger.error(f"No data fetched for {args.symbol}")
        return 1
    
    logger.info(f"Loaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
    
    # Run trend-filtered adaptive strategy backtest
    logger.info(f"Running trend-filtered adaptive backtest with parameters from {args.params_file}")
    trend_filtered_strategy = TrendFilteredAdaptiveEMA(
        fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
        lookback_window=20, vol_window=14, use_trend_filter=True,
        regime_params_file=args.params_file,
        daily_ema_period=args.daily_ema,
        enforce_trend_alignment=not args.no_trend_filter,
        vol_threshold_percentile=args.vol_percentile
    )
    
    filtered_results = backtest(df, trend_filtered_strategy, symbol=args.symbol)
    
    # Print trend-filtered strategy performance
    logger.info(f"Trend-Filtered Adaptive Strategy Results:")
    logger.info(f"Total Return: {filtered_results.get('total_return', 0):.2%}")
    logger.info(f"Profit Factor: {filtered_results.get('profit_factor', 0):.2f}")
    logger.info(f"Win Rate: {filtered_results.get('win_rate', 0):.2%}")
    logger.info(f"Max Drawdown: {filtered_results.get('max_drawdown', 0):.2%}")
    logger.info(f"Number of Trades: {filtered_results.get('n_trades', 0)}")
    
    # Compare with baseline adaptive strategy if requested
    if args.compare_baseline:
        logger.info("Running baseline adaptive strategy for comparison")
        baseline_strategy = RegimeAdaptiveEMAStrategy(
            fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
            lookback_window=20, vol_window=14, use_trend_filter=True,
            regime_params_file=args.params_file
        )
        
        baseline_results = backtest(df, baseline_strategy)
        
        # Print baseline strategy performance
        logger.info(f"Baseline Adaptive Strategy Results:")
        logger.info(f"Total Return: {baseline_results.get('total_return', 0):.2%}")
        logger.info(f"Profit Factor: {baseline_results.get('profit_factor', 0):.2f}")
        logger.info(f"Win Rate: {baseline_results.get('win_rate', 0):.2%}")
        logger.info(f"Max Drawdown: {baseline_results.get('max_drawdown', 0):.2%}")
        logger.info(f"Number of Trades: {baseline_results.get('n_trades', 0)}")
        
        # Calculate improvement
        return_improvement = filtered_results.get('total_return', 0) - baseline_results.get('total_return', 0)
        pf_improvement = filtered_results.get('profit_factor', 0) - baseline_results.get('profit_factor', 0)
        trade_reduction = baseline_results.get('n_trades', 0) - filtered_results.get('n_trades', 0)
        trade_pct_reduction = (trade_reduction / baseline_results.get('n_trades', 1)) * 100 if baseline_results.get('n_trades', 0) > 0 else 0
        
        logger.info(f"Improvement with Trend-Filtered Strategy:")
        logger.info(f"Return Improvement: {return_improvement:.2%}")
        logger.info(f"Profit Factor Improvement: {pf_improvement:.2f}")
        logger.info(f"Trade Reduction: {trade_reduction} trades ({trade_pct_reduction:.1f}%)")
    
    # Save results if requested
    if args.save_results:
        # Extract strategy info
        strategy_info = trend_filtered_strategy.get_info()
        
        # Save strategy info
        info_dict = {k: [v] for k, v in strategy_info.items()}
        info_df = pd.DataFrame(info_dict)
        info_file = f"wfo_results/trend_filtered_info_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        info_df.to_csv(info_file, index=False)
        logger.info(f"Saved strategy info to {info_file}")
        
        # Save backtest results
        results_file = f"wfo_results/trend_filtered_results_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        df_with_signals = filtered_results.get('dataframe', pd.DataFrame())
        df_with_signals.to_csv(results_file)
        logger.info(f"Saved backtest results to {results_file}")
    
    # Plot results if requested
    if args.plot:
        df_with_signals = filtered_results.get('dataframe', pd.DataFrame())
        
        if not df_with_signals.empty:
            plt.figure(figsize=(12, 12))
            
            # Plot price with daily EMA
            plt.subplot(3, 1, 1)
            plt.plot(df_with_signals.index, df_with_signals['close'], label='Price', color='blue', alpha=0.7)
            
            # If we have daily EMA data, plot it
            if hasattr(trend_filtered_strategy, 'daily_ema') and trend_filtered_strategy.daily_ema is not None:
                plt.axhline(y=trend_filtered_strategy.daily_ema, color='purple', linestyle='-', 
                          label=f'Daily EMA({args.daily_ema})', alpha=0.7)
            
            # Mark filtered signals if we can identify them
            if 'filtered_signal' in df_with_signals.columns:
                filter_points = df_with_signals[df_with_signals['filtered_signal'] != 0]
                if not filter_points.empty:
                    plt.scatter(filter_points.index, filter_points['close'], 
                              marker='x', color='red', s=100, label='Filtered Signals')
            
            plt.title(f"Price with Daily EMA - {args.symbol}")
            plt.legend()
            
            # Plot volatility
            plt.subplot(3, 1, 2)
            if 'norm_atr' in df_with_signals.columns:
                plt.plot(df_with_signals.index, df_with_signals['norm_atr'], 
                       label='Normalized ATR', color='orange')
                
                if hasattr(trend_filtered_strategy, 'vol_threshold') and trend_filtered_strategy.vol_threshold is not None:
                    plt.axhline(y=trend_filtered_strategy.vol_threshold, color='red', linestyle='--', 
                              label=f'Volatility Threshold ({args.vol_percentile}%)', alpha=0.7)
            
            plt.title(f"Volatility - {args.symbol}")
            plt.legend()
            
            # Plot equity curves
            plt.subplot(3, 1, 3)
            
            # If we have comparison data, plot both curves
            if args.compare_baseline:
                baseline_df = baseline_results.get('dataframe', pd.DataFrame())
                
                if not baseline_df.empty and 'equity' in baseline_df.columns:
                    plt.plot(baseline_df.index, baseline_df['equity'], 
                           label='Baseline Adaptive Strategy', color='blue', alpha=0.7)
            
            # Plot trend-filtered strategy equity curve
            if 'equity' in df_with_signals.columns:
                plt.plot(df_with_signals.index, df_with_signals['equity'], 
                       label='Trend-Filtered Strategy', color='green')
            
            plt.title(f"Equity Curve - {args.symbol}")
            plt.legend()
            
            # Save plot
            plot_file = f"wfo_results/trend_filtered_plot_{args.symbol.replace('/', '_')}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_file)
            logger.info(f"Saved plot to {plot_file}")
            
            # Show plot in interactive mode
            plt.show()
    
    return 0

if __name__ == "__main__":
    main() 