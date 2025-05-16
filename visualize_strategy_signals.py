#!/usr/bin/env python3
"""
High Leverage Strategy Signal Visualizer

This script visualizes the HighLeverageStrategy signals and shows how different
filters affect the signal generation process. It uses all the filters from the 
original strategy, but with more permissive parameters to generate actual trades.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project components
from src.data.fetcher import DataFetcher
from src.strategy.high_leverage_strategy import HighLeverageStrategy
from src.simulation.market_conditions import MarketConditionDetector, MarketCondition

def create_strategy(params=None):
    """
    Create a HighLeverageStrategy instance with configurable parameters.
    
    Args:
        params: Dictionary of strategy parameters
        
    Returns:
        HighLeverageStrategy instance
    """
    # Default parameters - more permissive than original
    default_params = {
        "fast_ema": 8,
        "slow_ema": 21,
        "trend_ema": 50,
        "risk_per_trade": 0.02,
        
        # Multi-timeframe filter parameters
        "use_mtf_filter": True,
        "mtf_signal_mode": "any",  # 'any' is more permissive than 'both'
        "mtf_alignment_weight": 1.0,
        
        # Momentum filter parameters
        "use_momentum_filter": True,
        "momentum_period": 14,
        "momentum_threshold": 30,  # More permissive than 50
        "momentum_lookback": 3,
        
        # Volatility sizing parameters
        "use_volatility_sizing": True,
        "volatility_target": 0.01,
        "max_position_size": 0.05
    }
    
    # Update default parameters with provided ones
    if params:
        default_params.update(params)
    
    # Create and return strategy
    return HighLeverageStrategy(**default_params)

def analyze_signals(data, higher_tf_data, strategy, symbol, timeframe):
    """
    Analyze signal generation process and filter effects.
    
    Args:
        data: DataFrame with OHLCV data
        higher_tf_data: Dictionary with higher timeframe data
        strategy: HighLeverageStrategy instance
        symbol: Trading symbol
        timeframe: Timeframe string
        
    Returns:
        Dictionary with analysis results
    """
    # Generate signals
    signals = strategy.generate_signals(data.copy(), higher_tf_data)
    
    # Find raw signals (before filtering)
    raw_signals = signals[signals['signal'] != 0].copy()
    
    # Apply filters and track which ones pass/fail
    filtered_signals = []
    
    for idx in raw_signals.index:
        idx_pos = signals.index.get_loc(idx)
        signal_info = {
            'timestamp': idx,
            'price': signals['close'].iloc[idx_pos],
            'signal': signals['signal'].iloc[idx_pos],
            'raw_signal': True
        }
        
        # Check MTF filter
        mtf_pass = strategy.check_multi_timeframe_alignment(signals, idx_pos, higher_tf_data)
        signal_info['mtf_pass'] = mtf_pass
        
        # Check momentum filter
        momentum_pass = strategy.check_momentum_filter(signals, idx_pos)
        signal_info['momentum_pass'] = momentum_pass
        
        # Check volatility filter
        vol_regime = signals['vol_regime'].iloc[idx_pos] if 'vol_regime' in signals.columns else 'NORMAL'
        extreme_vol = (vol_regime == 'HIGH' and 
                      signals['annualized_vol'].iloc[idx_pos] > strategy.volatility_target * 2) if 'annualized_vol' in signals.columns else False
        volatility_pass = not extreme_vol
        signal_info['volatility_pass'] = volatility_pass
        
        # Overall signal passes all filters
        signal_info['final_pass'] = mtf_pass and momentum_pass and volatility_pass
        
        # Track RSI and volatility for debugging
        signal_info['rsi'] = signals['rsi'].iloc[idx_pos] if 'rsi' in signals.columns else None
        signal_info['volatility'] = signals['annualized_vol'].iloc[idx_pos] if 'annualized_vol' in signals.columns else None
        signal_info['vol_regime'] = vol_regime
        
        filtered_signals.append(signal_info)
    
    # Create result object
    result = {
        'symbol': symbol,
        'timeframe': timeframe,
        'total_bars': len(signals),
        'raw_signals': len(raw_signals),
        'filtered_signals': sum(1 for s in filtered_signals if s['final_pass']),
        'signals': filtered_signals,
        'filters': {
            'mtf': {
                'pass': sum(1 for s in filtered_signals if s['mtf_pass']),
                'fail': sum(1 for s in filtered_signals if not s['mtf_pass'])
            },
            'momentum': {
                'pass': sum(1 for s in filtered_signals if s['momentum_pass']),
                'fail': sum(1 for s in filtered_signals if not s['momentum_pass'])
            },
            'volatility': {
                'pass': sum(1 for s in filtered_signals if s['volatility_pass']),
                'fail': sum(1 for s in filtered_signals if not s['volatility_pass'])
            }
        }
    }
    
    return result, signals

def visualize_signals(signals_df, signal_analysis, strategy_params, symbol, timeframe):
    """
    Create visualization of signals with filter analysis.
    
    Args:
        signals_df: DataFrame with signal data
        signal_analysis: Analysis results from analyze_signals
        strategy_params: Dictionary of strategy parameters
        symbol: Trading symbol
        timeframe: Timeframe string
    """
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    
    # Setup grid
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    
    # Plot price and EMAs
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(signals_df.index, signals_df['close'], label='Price', color='black', alpha=0.5)
    
    # Get EMA column names - the high leverage strategy might use different names
    ema_cols = {
        'fast': next((col for col in signals_df.columns if 'fast' in col.lower() or 'ema_short' in col.lower()), None),
        'slow': next((col for col in signals_df.columns if 'slow' in col.lower() or 'ema_medium' in col.lower()), None),
        'trend': next((col for col in signals_df.columns if 'trend' in col.lower() or 'ema_long' in col.lower()), None)
    }
    
    # Plot EMAs if available
    if ema_cols['fast']:
        ax1.plot(signals_df.index, signals_df[ema_cols['fast']], 
                label=f"Fast EMA ({strategy_params['fast_ema']})", color='blue')
    if ema_cols['slow']:
        ax1.plot(signals_df.index, signals_df[ema_cols['slow']], 
                label=f"Slow EMA ({strategy_params['slow_ema']})", color='orange')
    if ema_cols['trend']:
        ax1.plot(signals_df.index, signals_df[ema_cols['trend']], 
                label=f"Trend EMA ({strategy_params['trend_ema']})", color='red')
    
    # Extract signals for plotting
    filtered_signals = signal_analysis['signals']
    
    # Plot raw signals (those that didn't pass all filters)
    raw_signals = [s for s in filtered_signals if not s['final_pass']]
    if raw_signals:
        timestamps = [s['timestamp'] for s in raw_signals]
        prices = [s['price'] for s in raw_signals]
        signal_types = [s['signal'] for s in raw_signals]
        
        # Plot buy signals that didn't pass filters
        buy_idx = [i for i, sig in enumerate(signal_types) if sig > 0]
        if buy_idx:
            ax1.scatter([timestamps[i] for i in buy_idx], 
                       [prices[i] for i in buy_idx], 
                       marker='^', color='lightgreen', s=100, alpha=0.5,
                       label='Raw Buy Signal (Filtered)')
        
        # Plot sell signals that didn't pass filters
        sell_idx = [i for i, sig in enumerate(signal_types) if sig < 0]
        if sell_idx:
            ax1.scatter([timestamps[i] for i in sell_idx], 
                       [prices[i] for i in sell_idx], 
                       marker='v', color='lightcoral', s=100, alpha=0.5,
                       label='Raw Sell Signal (Filtered)')
    
    # Plot final signals (those that passed all filters)
    final_signals = [s for s in filtered_signals if s['final_pass']]
    if final_signals:
        timestamps = [s['timestamp'] for s in final_signals]
        prices = [s['price'] for s in final_signals]
        signal_types = [s['signal'] for s in final_signals]
        
        # Plot buy signals
        buy_idx = [i for i, sig in enumerate(signal_types) if sig > 0]
        if buy_idx:
            ax1.scatter([timestamps[i] for i in buy_idx], 
                       [prices[i] for i in buy_idx], 
                       marker='^', color='green', s=120, 
                       label='Final Buy Signal')
        
        # Plot sell signals
        sell_idx = [i for i, sig in enumerate(signal_types) if sig < 0]
        if sell_idx:
            ax1.scatter([timestamps[i] for i in sell_idx], 
                       [prices[i] for i in sell_idx], 
                       marker='v', color='red', s=120, 
                       label='Final Sell Signal')
    
    # Configure price plot
    ax1.set_title(f'{symbol} {timeframe} - High Leverage Strategy Signals', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot RSI in second subplot if available
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if 'rsi' in signals_df.columns:
        ax2.plot(signals_df.index, signals_df['rsi'], color='purple', label='RSI')
        ax2.axhline(y=strategy_params['momentum_threshold'], color='green', linestyle='--', 
                   alpha=0.7, label=f"RSI Threshold ({strategy_params['momentum_threshold']})")
        ax2.axhline(y=100-strategy_params['momentum_threshold'], color='red', linestyle='--', 
                   alpha=0.7)
        
        # Add RSI values for filtered signals
        for signal in filtered_signals:
            if 'momentum_pass' in signal and not signal['momentum_pass'] and 'rsi' in signal and signal['rsi'] is not None:
                marker = '^' if signal['signal'] > 0 else 'v'
                color = 'red'  # RSI filter failed
                ax2.scatter(signal['timestamp'], signal['rsi'], marker=marker, color=color, s=80)
        
        # Configure RSI plot
        ax2.set_title('RSI Momentum Filter', fontsize=12)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
    else:
        ax2.text(0.5, 0.5, 'RSI data not available', 
                horizontalalignment='center', verticalalignment='center')
    
    # Plot volatility in third subplot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if 'annualized_vol' in signals_df.columns:
        ax3.plot(signals_df.index, signals_df['annualized_vol'], color='orange', label='Volatility')
        ax3.axhline(y=strategy_params.get('volatility_target', 0.01), color='blue', linestyle='--', 
                   alpha=0.7, label=f"Target ({strategy_params.get('volatility_target', 0.01):.2f})")
        
        # Mark volatility regimes
        if 'vol_regime' in signals_df.columns:
            regime_colors = {'LOW': 'green', 'NORMAL': 'gray', 'HIGH': 'red'}
            regimes = signals_df['vol_regime'].unique()
            
            for regime in regimes:
                regime_data = signals_df[signals_df['vol_regime'] == regime]
                ax3.scatter(regime_data.index, regime_data['annualized_vol'], 
                           marker='.', color=regime_colors.get(regime, 'blue'), 
                           alpha=0.5, s=10, label=f"{regime} Volatility")
        
        # Configure volatility plot
        ax3.set_title('Volatility Regimes', fontsize=12)
        ax3.set_ylabel('Annualized Volatility')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
    else:
        ax3.text(0.5, 0.5, 'Volatility data not available', 
                horizontalalignment='center', verticalalignment='center')
    
    # Configure layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    # Save figure
    filename = f'charts/{symbol.replace("/", "_")}_{timeframe}_highleverage_signals.png'
    plt.savefig(filename)
    print(f'Chart saved to {filename}')
    
    # Print signal statistics
    print("\nSignal Statistics:")
    print(f"Raw Signals: {signal_analysis['raw_signals']}")
    print(f"Final Signals (Passed All Filters): {signal_analysis['filtered_signals']}")
    
    # Print filter effectiveness
    print("\nFilter Effects:")
    for filter_name, stats in signal_analysis['filters'].items():
        pass_rate = (stats['pass'] / signal_analysis['raw_signals']) * 100 if signal_analysis['raw_signals'] > 0 else 0
        print(f"{filter_name.upper()} Filter: {stats['pass']} passed, {stats['fail']} failed ({pass_rate:.1f}% pass rate)")
    
    # Print column names for debugging
    print("\nAvailable Columns in Signals DataFrame:")
    for col in signals_df.columns:
        print(f"- {col}")

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Visualize High Leverage Strategy Signals")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="4h", help="Trading timeframe")
    parser.add_argument("--days", type=int, default=60, help="Days of historical data")
    parser.add_argument("--mtf-mode", type=str, default="any", choices=["any", "both", "weighted"],
                       help="Multi-timeframe filter mode")
    parser.add_argument("--momentum-threshold", type=int, default=30, 
                       help="RSI momentum threshold (lower is more permissive)")
    parser.add_argument("--disable-mtf", action="store_true", help="Disable multi-timeframe filter")
    parser.add_argument("--disable-momentum", action="store_true", help="Disable momentum filter")
    
    args = parser.parse_args()
    
    print(f"Visualizing High Leverage Strategy Signals for {args.symbol} ({args.timeframe})")
    
    # Create data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch historical data
    print(f"Fetching {args.days} days of {args.timeframe} data...")
    
    data = data_fetcher.fetch_historical_data(
        symbol=args.symbol, 
        days=args.days, 
        timeframe=args.timeframe
    )
    
    if data is None or data.empty:
        print(f"ERROR: Failed to fetch data for {args.symbol} ({args.timeframe})")
        return
    
    # Determine higher timeframe
    higher_tf = '1d' if args.timeframe in ['5m', '15m', '30m', '1h', '4h'] else '1w'
    
    # Fetch higher timeframe data
    higher_tf_data = data_fetcher.fetch_historical_data(
        symbol=args.symbol,
        days=args.days,
        timeframe=higher_tf
    )
    
    print(f"Received {len(data)} bars for {args.timeframe} and {len(higher_tf_data)} bars for {higher_tf}")
    
    # Create strategy parameters based on arguments
    strategy_params = {
        "fast_ema": 8,
        "slow_ema": 21,
        "trend_ema": 50,
        "risk_per_trade": 0.02,
        "use_mtf_filter": not args.disable_mtf,
        "mtf_signal_mode": args.mtf_mode,
        "use_momentum_filter": not args.disable_momentum,
        "momentum_threshold": args.momentum_threshold,
        "mtf_timeframes": [args.timeframe, higher_tf]
    }
    
    # Create strategy
    strategy = create_strategy(strategy_params)
    
    # Create MTF data dictionary
    mtf_data = {higher_tf: higher_tf_data}
    
    # Analyze signals
    signal_analysis, signals_df = analyze_signals(data, mtf_data, strategy, args.symbol, args.timeframe)
    
    # Visualize signals
    visualize_signals(signals_df, signal_analysis, strategy_params, args.symbol, args.timeframe)
    
    # Save analysis results
    os.makedirs('results', exist_ok=True)
    
    # Convert timestamps to strings for JSON serialization
    serializable_analysis = signal_analysis.copy()
    for signal in serializable_analysis['signals']:
        if isinstance(signal['timestamp'], pd.Timestamp):
            signal['timestamp'] = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    
    filename = f"results/{args.symbol.replace('/', '_')}_{args.timeframe}_signal_analysis.json"
    with open(filename, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
    
    print(f"Analysis saved to {filename}")
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Higher Timeframe: {higher_tf}")
    print(f"MTF Filter: {'Enabled' if not args.disable_mtf else 'Disabled'}")
    if not args.disable_mtf:
        print(f"  Mode: {args.mtf_mode}")
    print(f"Momentum Filter: {'Enabled' if not args.disable_momentum else 'Disabled'}")
    if not args.disable_momentum:
        print(f"  Threshold: {args.momentum_threshold}")

if __name__ == "__main__":
    main() 