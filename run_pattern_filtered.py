import os
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

from src.data.fetcher import fetch_ohlcv
from src.strategy.pattern_filtered_strategy import PatternFilteredStrategy
from src.strategy.trend_filtered_adaptive_ema import TrendFilteredAdaptiveEMA
from src.backtest.backtest import backtest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_candlestick(ax, df, title=None):
    """
    Plot candlestick chart with indicators.
    
    Args:
        ax: Matplotlib axis
        df: DataFrame with OHLCV data
        title: Chart title
    """
    # Determine the color for each candle
    colors = ['green' if close > open else 'red' for open, close in zip(df['open'], df['close'])]
    
    # Plot the candlesticks
    width = 0.6
    width2 = 0.05
    
    # Draw the wicks
    ax.vlines(range(len(df)), df['low'], df['high'], color='black', linewidth=1)
    
    # Draw the candles
    rect = ax.bar(range(len(df)), df['close'] - df['open'], width, 
                 bottom=np.minimum(df['open'], df['close']), 
                 color=colors, alpha=0.8)
    
    # Set the date labels
    ax.set_xticks(range(0, len(df), len(df) // 10))
    date_labels = [date.strftime('%Y-%m-%d') for date in df.index[::len(df) // 10]]
    ax.set_xticklabels(date_labels, rotation=45)
    
    if title:
        ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    
    return ax

def main():
    parser = argparse.ArgumentParser(description="Run backtest with pattern-filtered strategy")
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol (e.g. BTC/USDT)')
    parser.add_argument('--timeframe', default='4h', help='Candlestick timeframe (e.g. 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=180, help='Number of days of historical data to fetch')
    parser.add_argument('--params_file', default='regime_params.csv', help='Path to regime parameters CSV file')
    parser.add_argument('--daily_ema', type=int, default=200, help='Period for daily EMA trend filter')
    parser.add_argument('--no_pattern_filter', action='store_true', help='Disable candlestick pattern filter')
    parser.add_argument('--doji_threshold', type=float, default=0.1, help='Threshold for doji detection')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with trend-filtered strategy')
    parser.add_argument('--save_results', action='store_true', help='Save results to CSV')
    parser.add_argument('--plot', action='store_true', help='Plot results')
    
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
    
    # Run pattern-filtered strategy backtest
    logger.info(f"Running pattern-filtered strategy backtest")
    pattern_strategy = PatternFilteredStrategy(
        fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
        lookback_window=20, vol_window=14, use_trend_filter=True,
        regime_params_file=args.params_file,
        daily_ema_period=args.daily_ema,
        enforce_trend_alignment=True,
        vol_threshold_percentile=80,
        require_pattern_confirmation=not args.no_pattern_filter,
        doji_threshold=args.doji_threshold
    )
    
    pattern_results = backtest(df, pattern_strategy, symbol=args.symbol)
    
    # Print pattern-filtered strategy performance
    logger.info(f"Pattern-Filtered Strategy Results:")
    logger.info(f"Total Return: {pattern_results.get('total_return', 0):.2%}")
    logger.info(f"Profit Factor: {pattern_results.get('profit_factor', 0):.2f}")
    logger.info(f"Win Rate: {pattern_results.get('win_rate', 0):.2%}")
    logger.info(f"Max Drawdown: {pattern_results.get('max_drawdown', 0):.2%}")
    logger.info(f"Number of Trades: {pattern_results.get('n_trades', 0)}")
    
    # Compare with baseline trend-filtered strategy if requested
    if args.compare_baseline:
        logger.info("Running baseline trend-filtered strategy for comparison")
        baseline_strategy = TrendFilteredAdaptiveEMA(
            fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
            lookback_window=20, vol_window=14, use_trend_filter=True,
            regime_params_file=args.params_file,
            daily_ema_period=args.daily_ema,
            enforce_trend_alignment=True,
            vol_threshold_percentile=80
        )
        
        baseline_results = backtest(df, baseline_strategy, symbol=args.symbol)
        
        # Print baseline strategy performance
        logger.info(f"Baseline Trend-Filtered Strategy Results:")
        logger.info(f"Total Return: {baseline_results.get('total_return', 0):.2%}")
        logger.info(f"Profit Factor: {baseline_results.get('profit_factor', 0):.2f}")
        logger.info(f"Win Rate: {baseline_results.get('win_rate', 0):.2%}")
        logger.info(f"Max Drawdown: {baseline_results.get('max_drawdown', 0):.2%}")
        logger.info(f"Number of Trades: {baseline_results.get('n_trades', 0)}")
        
        # Calculate improvement
        return_improvement = pattern_results.get('total_return', 0) - baseline_results.get('total_return', 0)
        pf_improvement = pattern_results.get('profit_factor', 0) - baseline_results.get('profit_factor', 0)
        trade_reduction = baseline_results.get('n_trades', 0) - pattern_results.get('n_trades', 0)
        trade_pct_reduction = (trade_reduction / baseline_results.get('n_trades', 1)) * 100 if baseline_results.get('n_trades', 0) > 0 else 0
        
        logger.info(f"Improvement with Pattern-Filtered Strategy:")
        logger.info(f"Return Improvement: {return_improvement:.2%}")
        logger.info(f"Profit Factor Improvement: {pf_improvement:.2f}")
        logger.info(f"Trade Reduction: {trade_reduction} trades ({trade_pct_reduction:.1f}%)")
    
    # Save results if requested
    if args.save_results:
        # Extract strategy info and pattern stats
        strategy_info = pattern_strategy.get_info()
        
        # Save strategy info
        info_dict = {k: [v] for k, v in strategy_info.items() if k != 'pattern_stats'}
        info_df = pd.DataFrame(info_dict)
        info_file = f"wfo_results/pattern_filtered_info_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        info_df.to_csv(info_file, index=False)
        logger.info(f"Saved strategy info to {info_file}")
        
        # Save pattern stats
        pattern_stats = strategy_info.get('pattern_stats', {})
        stats_dict = {k: [v] for k, v in pattern_stats.items()}
        stats_df = pd.DataFrame(stats_dict)
        stats_file = f"wfo_results/pattern_stats_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved pattern stats to {stats_file}")
        
        # Save backtest results
        results_file = f"wfo_results/pattern_filtered_results_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        df_with_signals = pattern_results.get('dataframe', pd.DataFrame())
        df_with_signals.to_csv(results_file)
        logger.info(f"Saved backtest results to {results_file}")
    
    # Plot results if requested
    if args.plot:
        df_with_signals = pattern_results.get('dataframe', pd.DataFrame())
        
        if not df_with_signals.empty:
            plt.figure(figsize=(12, 16))
            
            # Plot candlestick chart with signals
            plt.subplot(4, 1, 1)
            ax = plt.gca()
            plot_candlestick(ax, df.iloc[-60:], title=f"Recent Price Action - {args.symbol}")
            
            # Mark signals on candlestick chart
            last_60_signals = df_with_signals.iloc[-60:]
            buy_signals = last_60_signals[last_60_signals['signal'] == 1]
            sell_signals = last_60_signals[last_60_signals['signal'] == -1]
            
            # Convert to positions relative to the subset
            buy_indices = [list(last_60_signals.index).index(idx) for idx in buy_signals.index]
            sell_indices = [list(last_60_signals.index).index(idx) for idx in sell_signals.index]
            
            # Plot buy and sell markers
            if buy_indices:
                plt.scatter(buy_indices, buy_signals['low'] * 0.995, marker='^', color='green', s=100, label='Buy Signal')
            if sell_indices:
                plt.scatter(sell_indices, sell_signals['high'] * 1.005, marker='v', color='red', s=100, label='Sell Signal')
                
            # Mark filtered signals if available
            if 'filtered_signal' in df_with_signals.columns:
                filtered_signals = last_60_signals[last_60_signals['filtered_signal'] != 0]
                filtered_indices = [list(last_60_signals.index).index(idx) for idx in filtered_signals.index]
                
                if filtered_indices:
                    plt.scatter(filtered_indices, filtered_signals['close'], marker='x', color='black', s=80, label='Filtered Signal')
            
            plt.legend()
            
            # Plot price with daily EMA
            plt.subplot(4, 1, 2)
            plt.plot(df_with_signals.index, df_with_signals['close'], label='Price', color='blue', alpha=0.7)
            
            # If we have daily EMA data, plot it
            if hasattr(pattern_strategy, 'daily_ema') and pattern_strategy.daily_ema is not None:
                plt.axhline(y=pattern_strategy.daily_ema, color='purple', linestyle='-', 
                          label=f'Daily EMA({args.daily_ema})', alpha=0.7)
            
            plt.title(f"Price with Daily EMA - {args.symbol}")
            plt.legend()
            
            # Plot pattern confirmation stats
            plt.subplot(4, 1, 3)
            pattern_stats = pattern_strategy.pattern_stats
            if pattern_stats['total_signals'] > 0:
                confirmed = pattern_stats['patterns_confirmed']
                rejected = pattern_stats['patterns_rejected']
                total = pattern_stats['total_signals']
                
                bars = plt.bar([0, 1], [confirmed, rejected], color=['green', 'red'])
                plt.xticks([0, 1], ['Confirmed', 'Rejected'])
                plt.bar_label(bars, fmt='%d')
                
                confirmed_pct = (confirmed / total) * 100
                rejected_pct = (rejected / total) * 100
                
                plt.title(f"Pattern Confirmation Stats: {confirmed}/{total} ({confirmed_pct:.1f}%) Confirmed")
                
                # Add pie chart inset
                ax_inset = plt.axes([0.65, 0.55, 0.2, 0.2])
                wedges, texts, autotexts = ax_inset.pie([confirmed, rejected], 
                                                     labels=None, 
                                                     autopct='%1.1f%%',
                                                     colors=['green', 'red'],
                                                     startangle=90)
                ax_inset.set_aspect('equal')
                
                # Create legend
                green_patch = mpatches.Patch(color='green', label='Confirmed')
                red_patch = mpatches.Patch(color='red', label='Rejected')
                ax_inset.legend(handles=[green_patch, red_patch], loc='upper left', fontsize=8)
            else:
                plt.text(0.5, 0.5, "No pattern statistics available", 
                       horizontalalignment='center', verticalalignment='center',
                       transform=plt.gca().transAxes)
                
            plt.tight_layout()
            
            # Plot equity curves
            plt.subplot(4, 1, 4)
            
            # If we have comparison data, plot both curves
            if args.compare_baseline:
                baseline_df = baseline_results.get('dataframe', pd.DataFrame())
                
                if not baseline_df.empty and 'equity' in baseline_df.columns:
                    plt.plot(baseline_df.index, baseline_df['equity'], 
                           label='Baseline Trend-Filtered Strategy', color='blue', alpha=0.7)
            
            # Plot pattern-filtered strategy equity curve
            if 'equity' in df_with_signals.columns:
                plt.plot(df_with_signals.index, df_with_signals['equity'], 
                       label='Pattern-Filtered Strategy', color='green')
            
            plt.title(f"Equity Curve - {args.symbol}")
            plt.legend()
            
            # Save plot
            plot_file = f"wfo_results/pattern_filtered_plot_{args.symbol.replace('/', '_')}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_file)
            logger.info(f"Saved plot to {plot_file}")
            
            # Show plot in interactive mode
            plt.show()
    
    return 0

if __name__ == "__main__":
    main() 