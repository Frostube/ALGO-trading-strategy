import os
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import numpy as np

from src.data.fetcher import fetch_ohlcv
from src.strategy.enhanced_strategy import EnhancedConfirmationStrategy
from src.strategy.pattern_filtered_strategy import PatternFilteredStrategy
from src.backtest.enhanced_backtest import backtest

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

def plot_confirmation_stats(ax, stats, title="Confirmation Statistics"):
    """
    Plot confirmation statistics as a bar chart.
    
    Args:
        ax: Matplotlib axis
        stats: Dictionary with confirmation statistics
        title: Chart title
    """
    if stats['total_signals'] == 0:
        ax.text(0.5, 0.5, "No confirmation statistics available", 
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes)
        return ax
    
    # Prepare data for plotting
    categories = []
    passed = []
    rejected = []
    
    if 'momentum_confirmed' in stats:
        categories.append('Momentum')
        passed.append(stats['momentum_confirmed'])
        rejected.append(stats['momentum_rejected'])
        
    if 'volume_confirmed' in stats:
        categories.append('Volume')
        passed.append(stats['volume_confirmed'])
        rejected.append(stats['volume_rejected'])
        
    if 'volatility_confirmed' in stats:
        categories.append('Volatility')
        passed.append(stats['volatility_confirmed'])
        rejected.append(stats['volatility_rejected'])
        
    if 'pivot_confirmed' in stats:
        categories.append('Pivot')
        passed.append(stats['pivot_confirmed'])
        rejected.append(stats['pivot_rejected'])
    
    categories.append('Pattern')
    if 'patterns_confirmed' in stats:
        passed.append(stats['patterns_confirmed'])
        rejected.append(stats['patterns_rejected'])
    else:
        passed.append(0)
        rejected.append(0)
    
    categories.append('Overall')
    passed.append(stats.get('signals_passed', 0))
    rejected.append(stats.get('signals_rejected', 0))
    
    # Set up positions for bars
    x = np.arange(len(categories))
    width = 0.35
    
    # Create bars
    ax.bar(x - width/2, passed, width, label='Passed', color='green', alpha=0.7)
    ax.bar(x + width/2, rejected, width, label='Rejected', color='red', alpha=0.7)
    
    # Add labels and title
    ax.set_ylabel('Number of Signals')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Add percentage labels
    for i, (p, r) in enumerate(zip(passed, rejected)):
        total = p + r
        if total > 0:
            pass_pct = p / total * 100
            ax.text(i - width/2, p + 1, f"{pass_pct:.0f}%", ha='center', va='bottom', fontsize=8)
    
    return ax

def main():
    parser = argparse.ArgumentParser(description="Run backtest with enhanced confirmation strategy")
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol (e.g. BTC/USDT)')
    parser.add_argument('--timeframe', default='4h', help='Candlestick timeframe (e.g. 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=180, help='Number of days of historical data to fetch')
    parser.add_argument('--params_file', default='regime_params.csv', help='Path to regime parameters CSV file')
    parser.add_argument('--daily_ema', type=int, default=200, help='Period for daily EMA trend filter')
    
    # Pattern filter options
    parser.add_argument('--no_pattern_filter', action='store_true', help='Disable candlestick pattern filter')
    parser.add_argument('--doji_threshold', type=float, default=0.1, help='Threshold for doji detection')
    
    # New confirmation filter options
    parser.add_argument('--no_momentum_filter', action='store_true', help='Disable momentum oscillator filters')
    parser.add_argument('--no_volume_filter', action='store_true', help='Disable volume-based filters')
    parser.add_argument('--no_volatility_filter', action='store_true', help='Disable volatility-based filters')
    parser.add_argument('--no_pivot_filter', action='store_true', help='Disable pivot point filters')
    
    parser.add_argument('--rsi_threshold', type=int, default=50, help='RSI threshold for trend alignment')
    parser.add_argument('--volume_factor', type=float, default=1.5, help='Factor for volume spike detection')
    parser.add_argument('--volume_lookback', type=int, default=20, help='Lookback period for volume average')
    parser.add_argument('--atr_percentile', type=int, default=80, help='Maximum ATR percentile for trading')
    parser.add_argument('--min_confirmations', type=int, default=3, help='Minimum number of confirmations required')
    
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with pattern-filtered strategy')
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
    
    # Run enhanced strategy backtest
    logger.info(f"Running enhanced confirmation strategy backtest")
    enhanced_strategy = EnhancedConfirmationStrategy(
        fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
        lookback_window=20, vol_window=14, use_trend_filter=True,
        regime_params_file=args.params_file,
        daily_ema_period=args.daily_ema,
        enforce_trend_alignment=True,
        vol_threshold_percentile=80,
        require_pattern_confirmation=not args.no_pattern_filter,
        doji_threshold=args.doji_threshold,
        use_momentum_filter=not args.no_momentum_filter,
        use_volume_filter=not args.no_volume_filter,
        use_volatility_filter=not args.no_volatility_filter,
        use_pivot_filter=not args.no_pivot_filter,
        rsi_threshold=args.rsi_threshold,
        volume_factor=args.volume_factor,
        atr_percentile=args.atr_percentile,
        min_confirmations=args.min_confirmations
    )
    
    enhanced_results = backtest(df, enhanced_strategy, symbol=args.symbol)
    
    # Print enhanced strategy performance
    logger.info(f"Enhanced Confirmation Strategy Results:")
    logger.info(f"Total Return: {enhanced_results.get('total_return', 0):.2%}")
    logger.info(f"Profit Factor: {enhanced_results.get('profit_factor', 0):.2f}")
    logger.info(f"Win Rate: {enhanced_results.get('win_rate', 0):.2%}")
    logger.info(f"Max Drawdown: {enhanced_results.get('max_drawdown', 0):.2%}")
    logger.info(f"Number of Trades: {enhanced_results.get('n_trades', 0)}")
    
    # Compare with baseline pattern-filtered strategy if requested
    if args.compare_baseline:
        logger.info("Running baseline pattern-filtered strategy for comparison")
        baseline_strategy = PatternFilteredStrategy(
            fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
            lookback_window=20, vol_window=14, use_trend_filter=True,
            regime_params_file=args.params_file,
            daily_ema_period=args.daily_ema,
            enforce_trend_alignment=True,
            vol_threshold_percentile=80,
            require_pattern_confirmation=not args.no_pattern_filter,
            doji_threshold=args.doji_threshold
        )
        
        baseline_results = backtest(df, baseline_strategy, symbol=args.symbol)
        
        # Print baseline strategy performance
        logger.info(f"Baseline Pattern-Filtered Strategy Results:")
        logger.info(f"Total Return: {baseline_results.get('total_return', 0):.2%}")
        logger.info(f"Profit Factor: {baseline_results.get('profit_factor', 0):.2f}")
        logger.info(f"Win Rate: {baseline_results.get('win_rate', 0):.2%}")
        logger.info(f"Max Drawdown: {baseline_results.get('max_drawdown', 0):.2%}")
        logger.info(f"Number of Trades: {baseline_results.get('n_trades', 0)}")
        
        # Calculate improvement
        return_improvement = enhanced_results.get('total_return', 0) - baseline_results.get('total_return', 0)
        pf_improvement = enhanced_results.get('profit_factor', 0) - baseline_results.get('profit_factor', 0)
        trade_reduction = baseline_results.get('n_trades', 0) - enhanced_results.get('n_trades', 0)
        trade_pct_reduction = (trade_reduction / baseline_results.get('n_trades', 1)) * 100 if baseline_results.get('n_trades', 0) > 0 else 0
        
        logger.info(f"Improvement with Enhanced Confirmation Strategy:")
        logger.info(f"Return Improvement: {return_improvement:.2%}")
        logger.info(f"Profit Factor Improvement: {pf_improvement:.2f}")
        logger.info(f"Trade Reduction: {trade_reduction} trades ({trade_pct_reduction:.1f}%)")
    
    # Save results if requested
    if args.save_results:
        # Extract strategy info
        strategy_info = enhanced_strategy.get_info()
        
        # Save strategy info
        info_dict = {k: [v] for k, v in strategy_info.items() if k != 'confirmation_stats' and k != 'pattern_stats'}
        info_df = pd.DataFrame(info_dict)
        info_file = f"wfo_results/enhanced_strategy_info_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        info_df.to_csv(info_file, index=False)
        logger.info(f"Saved strategy info to {info_file}")
        
        # Save confirmation stats
        confirmation_stats = strategy_info.get('confirmation_stats', {})
        pattern_stats = strategy_info.get('pattern_stats', {})
        
        # Combine stats
        all_stats = {**confirmation_stats, **pattern_stats}
        stats_dict = {k: [v] for k, v in all_stats.items()}
        stats_df = pd.DataFrame(stats_dict)
        stats_file = f"wfo_results/confirmation_stats_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info(f"Saved confirmation stats to {stats_file}")
        
        # Save backtest results
        results_file = f"wfo_results/enhanced_results_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        df_with_signals = enhanced_results.get('dataframe', pd.DataFrame())
        df_with_signals.to_csv(results_file)
        logger.info(f"Saved backtest results to {results_file}")
    
    # Plot results if requested
    if args.plot:
        df_with_signals = enhanced_results.get('dataframe', pd.DataFrame())
        
        if not df_with_signals.empty:
            plt.figure(figsize=(14, 20))
            
            # Plot candlestick chart with signals
            plt.subplot(5, 1, 1)
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
                
            # Mark filtered signals
            if 'enhanced_filtered' in df_with_signals.columns:
                enhanced_filtered = last_60_signals[last_60_signals['enhanced_filtered'] != 0]
                enhanced_indices = [list(last_60_signals.index).index(idx) for idx in enhanced_filtered.index]
                
                if enhanced_indices:
                    plt.scatter(enhanced_indices, enhanced_filtered['close'], marker='x', color='black', s=80, label='Enhanced Filtered')
                    
            if 'filtered_signal' in df_with_signals.columns:
                pattern_filtered = last_60_signals[last_60_signals['filtered_signal'] != 0]
                pattern_indices = [list(last_60_signals.index).index(idx) for idx in pattern_filtered.index]
                
                if pattern_indices:
                    plt.scatter(pattern_indices, pattern_filtered['close'], marker='o', color='blue', s=80, label='Pattern Filtered')
            
            plt.legend()
            
            # Plot momentum indicators
            plt.subplot(5, 1, 2)
            
            if 'rsi' in df_with_signals.columns:
                plt.plot(df_with_signals.index[-60:], df_with_signals['rsi'].iloc[-60:], label='RSI', color='purple')
                plt.axhline(y=args.rsi_threshold, color='black', linestyle='--', alpha=0.5)
                plt.axhline(y=30, color='green', linestyle='--', alpha=0.5)
                plt.axhline(y=70, color='red', linestyle='--', alpha=0.5)
                plt.ylim(0, 100)
                plt.title(f"RSI with Threshold {args.rsi_threshold}")
                plt.legend()
                
            # Plot volume analysis  
            plt.subplot(5, 1, 3)
            
            if 'volume' in df_with_signals.columns:
                # Calculate volume MA for comparison
                volume_ma = df_with_signals['volume'].rolling(args.volume_lookback).mean()
                
                # Plot volume and its moving average
                volume_bars = plt.bar(range(len(df_with_signals.index[-60:])), 
                                     df_with_signals['volume'].iloc[-60:], 
                                     alpha=0.5, label='Volume')
                
                # Color volume bars by price direction
                for i in range(60):
                    idx = -60 + i
                    if idx >= 0 and idx < len(df_with_signals):
                        if df_with_signals['close'].iloc[idx] >= df_with_signals['open'].iloc[idx]:
                            volume_bars[i].set_color('green')
                        else:
                            volume_bars[i].set_color('red')
                
                plt.plot(range(len(df_with_signals.index[-60:])), 
                       volume_ma.iloc[-60:], 
                       color='blue', label=f'Volume MA({args.volume_lookback})')
                
                plt.title(f"Volume Analysis with Spike Factor {args.volume_factor}")
                plt.legend()
                
            # Plot confirmation statistics
            plt.subplot(5, 1, 4)
            confirmation_stats = enhanced_strategy.get_info().get('confirmation_stats', {})
            pattern_stats = enhanced_strategy.get_info().get('pattern_stats', {})
            
            # Combine stats
            all_stats = {**confirmation_stats, **pattern_stats}
            ax = plot_confirmation_stats(plt.gca(), all_stats, title="Confirmation Filter Statistics")
            
            # Plot equity curves
            plt.subplot(5, 1, 5)
            
            # If we have comparison data, plot both curves
            if args.compare_baseline:
                baseline_df = baseline_results.get('dataframe', pd.DataFrame())
                
                if not baseline_df.empty and 'equity' in baseline_df.columns:
                    plt.plot(baseline_df.index, baseline_df['equity'], 
                           label='Baseline Pattern-Filtered Strategy', color='blue', alpha=0.7)
            
            # Plot enhanced strategy equity curve
            if 'equity' in df_with_signals.columns:
                plt.plot(df_with_signals.index, df_with_signals['equity'], 
                       label='Enhanced Confirmation Strategy', color='green')
            
            plt.title(f"Equity Curve - {args.symbol}")
            plt.legend()
            
            # Save plot
            plot_file = f"wfo_results/enhanced_strategy_plot_{args.symbol.replace('/', '_')}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_file)
            logger.info(f"Saved plot to {plot_file}")
            
            # Show plot in interactive mode
            plt.show()
    
    return 0

if __name__ == "__main__":
    main() 