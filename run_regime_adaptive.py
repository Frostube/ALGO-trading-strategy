import os
import pandas as pd
import argparse
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.fetcher import fetch_ohlcv
from src.strategy.regime_adaptive_ema import RegimeAdaptiveEMAStrategy
from src.backtest.backtest import backtest

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run backtest with regime-adaptive strategy")
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading pair symbol (e.g. BTC/USDT)')
    parser.add_argument('--timeframe', default='4h', help='Candlestick timeframe (e.g. 1h, 4h, 1d)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to fetch')
    parser.add_argument('--params_file', default='regime_params.csv', help='Path to regime parameters CSV file')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with baseline non-adaptive strategy')
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
    
    # Run regime-adaptive strategy backtest
    logger.info(f"Running regime-adaptive backtest with parameters from {args.params_file}")
    adaptive_strategy = RegimeAdaptiveEMAStrategy(
        fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
        lookback_window=20, vol_window=14, use_trend_filter=True,
        regime_params_file=args.params_file
    )
    
    adaptive_results = backtest(df, adaptive_strategy)
    
    # Print adaptive strategy performance
    logger.info(f"Regime-Adaptive Strategy Results:")
    logger.info(f"Total Return: {adaptive_results.get('total_return', 0):.2%}")
    logger.info(f"Profit Factor: {adaptive_results.get('profit_factor', 0):.2f}")
    logger.info(f"Win Rate: {adaptive_results.get('win_rate', 0):.2%}")
    logger.info(f"Max Drawdown: {adaptive_results.get('max_drawdown', 0):.2%}")
    logger.info(f"Number of Trades: {adaptive_results.get('n_trades', 0)}")
    
    # Compare with baseline non-adaptive strategy if requested
    if args.compare_baseline:
        from src.strategy.ema_crossover import EMACrossoverStrategy
        
        logger.info("Running baseline non-adaptive strategy for comparison")
        baseline_strategy = EMACrossoverStrategy(
            fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
            use_trend_filter=True
        )
        
        baseline_results = backtest(df, baseline_strategy)
        
        # Print baseline strategy performance
        logger.info(f"Baseline Strategy Results:")
        logger.info(f"Total Return: {baseline_results.get('total_return', 0):.2%}")
        logger.info(f"Profit Factor: {baseline_results.get('profit_factor', 0):.2f}")
        logger.info(f"Win Rate: {baseline_results.get('win_rate', 0):.2%}")
        logger.info(f"Max Drawdown: {baseline_results.get('max_drawdown', 0):.2%}")
        logger.info(f"Number of Trades: {baseline_results.get('n_trades', 0)}")
        
        # Calculate improvement
        return_improvement = adaptive_results.get('total_return', 0) - baseline_results.get('total_return', 0)
        pf_improvement = adaptive_results.get('profit_factor', 0) - baseline_results.get('profit_factor', 0)
        
        logger.info(f"Improvement with Regime-Adaptive Strategy:")
        logger.info(f"Return Improvement: {return_improvement:.2%}")
        logger.info(f"Profit Factor Improvement: {pf_improvement:.2f}")
    
    # Save results if requested
    if args.save_results:
        # Extract regime history
        regime_history = pd.DataFrame(adaptive_strategy.regime_history)
        
        # Save regime history
        regime_history_file = f"wfo_results/regime_history_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        regime_history.to_csv(regime_history_file, index=False)
        logger.info(f"Saved regime history to {regime_history_file}")
        
        # Save backtest results
        results_file = f"wfo_results/adaptive_results_{args.symbol.replace('/', '_')}_{timestamp}.csv"
        df_with_signals = adaptive_results.get('dataframe', pd.DataFrame())
        df_with_signals.to_csv(results_file)
        logger.info(f"Saved backtest results to {results_file}")
    
    # Plot results if requested
    if args.plot:
        df_with_signals = adaptive_results.get('dataframe', pd.DataFrame())
        
        if not df_with_signals.empty:
            # Plot price with regime changes
            plt.figure(figsize=(12, 8))
            
            # Plot price
            plt.subplot(2, 1, 1)
            plt.plot(df_with_signals.index, df_with_signals['close'], label='Price')
            
            # Extract regime changes
            regime_changes = pd.DataFrame(adaptive_strategy.regime_history)
            if not regime_changes.empty:
                # Convert timestamp to datetime if it's a string
                if isinstance(regime_changes['timestamp'].iloc[0], str):
                    regime_changes['timestamp'] = pd.to_datetime(regime_changes['timestamp'])
                
                # Create map of regimes to colors
                regime_colors = {
                    'bull_low_vol': 'green',
                    'bull_high_vol': 'darkgreen',
                    'bear_low_vol': 'red',
                    'bear_high_vol': 'darkred'
                }
                
                # Plot vertical lines at regime changes
                prev_regime = None
                for idx, row in regime_changes.iterrows():
                    if row['regime'] != prev_regime:
                        plt.axvline(x=row['timestamp'], color=regime_colors.get(row['regime'], 'gray'), 
                                   linestyle='--', alpha=0.7)
                        prev_regime = row['regime']
            
            plt.title(f"Price with Regime Changes - {args.symbol}")
            plt.legend()
            
            # Plot equity curve
            plt.subplot(2, 1, 2)
            
            # If we have comparison data, plot both curves
            if args.compare_baseline:
                baseline_df = baseline_results.get('dataframe', pd.DataFrame())
                
                if not baseline_df.empty and 'equity' in baseline_df.columns:
                    plt.plot(baseline_df.index, baseline_df['equity'], 
                             label='Baseline Strategy', color='blue', alpha=0.7)
            
            # Plot adaptive strategy equity curve
            if 'equity' in df_with_signals.columns:
                plt.plot(df_with_signals.index, df_with_signals['equity'], 
                         label='Regime-Adaptive Strategy', color='green')
            
            plt.title(f"Equity Curve - {args.symbol}")
            plt.legend()
            
            # Save plot
            plot_file = f"wfo_results/adaptive_plot_{args.symbol.replace('/', '_')}_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(plot_file)
            logger.info(f"Saved plot to {plot_file}")
            
            # Show plot in interactive mode
            plt.show()
    
    return 0

if __name__ == "__main__":
    main() 