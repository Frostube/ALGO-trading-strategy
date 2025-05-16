#!/usr/bin/env python3
"""
Test Simplified Strategy

A script to test the simplified strategy with visualization.
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
from src.strategy.simplified_strategy import SimplifiedStrategy
from src.simulation.market_conditions import MarketConditionDetector, MarketCondition

# Helper function to convert numpy types to Python types for JSON serialization
def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, (np.integer)):
        return int(obj)
    elif isinstance(obj, (np.floating)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def run_strategy_test(symbol, timeframe, days, params=None):
    """
    Run a test of the simplified strategy.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe to test
        days: Number of days to look back
        params: Optional strategy parameters
        
    Returns:
        Tuple of (signals, strategy, analysis)
    """
    print(f"Testing simplified strategy for {symbol} on {timeframe} timeframe")
    
    # Create data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch historical data
    print(f"Fetching {days} days of {timeframe} data...")
    
    data = data_fetcher.fetch_historical_data(
        symbol=symbol, 
        days=days, 
        timeframe=timeframe
    )
    
    if data is None or data.empty:
        print(f"ERROR: Failed to fetch data for {symbol} ({timeframe})")
        return None, None, None
    
    # Determine higher timeframe
    higher_tf = '1d' if timeframe in ['5m', '15m', '30m', '1h', '4h'] else '1w'
    
    # Fetch higher timeframe data
    higher_tf_data = data_fetcher.fetch_historical_data(
        symbol=symbol,
        days=days,
        timeframe=higher_tf
    )
    
    print(f"Received {len(data)} bars for {timeframe} and {len(higher_tf_data)} bars for {higher_tf}")
    
    # Default parameters (more permissive than original)
    default_params = {
        "fast_ema": 8,
        "slow_ema": 21,
        "trend_ema": 50,
        "risk_per_trade": 0.02,
        
        # Filter parameters
        "use_mtf_filter": params.get('use_mtf_filter', True) if params else True,
        "mtf_signal_mode": params.get('mtf_mode', 'any') if params else 'any',
        "mtf_timeframes": [timeframe, higher_tf],
        
        "use_momentum_filter": params.get('use_momentum_filter', True) if params else True,
        "momentum_threshold": params.get('momentum_threshold', 30) if params else 30,
        
        # Enhanced signal generation
        "force_signals": params.get('force_signals', True) if params else True,
        "signal_threshold": params.get('signal_threshold', 0.1) if params else 0.1
    }
    
    # Update with provided parameters
    if params:
        default_params.update(params)
    
    # Create strategy
    strategy = SimplifiedStrategy(**default_params)
    
    # Create MTF data dictionary
    mtf_data = {higher_tf: higher_tf_data}
    
    # Generate signals
    signals = strategy.generate_signals(data, mtf_data)
    
    # Analyze signal filtering
    analysis = analyze_signals(signals, mtf_data, strategy)
    
    return signals, strategy, analysis

def analyze_signals(signals, mtf_data, strategy):
    """
    Analyze the signals and filter effectiveness.
    
    Args:
        signals: DataFrame with signals
        mtf_data: Higher timeframe data
        strategy: Strategy instance
        
    Returns:
        Dictionary with analysis results
    """
    # Get raw signals (before filtering)
    raw_signals = signals[signals['signal'] != 0].copy()
    
    if len(raw_signals) == 0:
        print("No raw signals generated!")
        return {
            'raw_signals': 0,
            'filtered_signals': 0,
            'filtered_details': []
        }
    
    # Track filtered signals
    filtered_signals = []
    
    for idx in raw_signals.index:
        idx_pos = signals.index.get_loc(idx)
        
        # Create signal info
        signal_info = {
            'timestamp': idx,
            'price': float(signals['close'].iloc[idx_pos]),  # Convert to native Python float
            'signal': int(signals['signal'].iloc[idx_pos]),  # Convert to native Python int
        }
        
        # Check MTF filter
        mtf_pass = strategy.check_multi_timeframe_alignment(signals, idx_pos, mtf_data)
        signal_info['mtf_pass'] = bool(mtf_pass)  # Convert to native Python bool
        
        # Check momentum filter
        momentum_pass = strategy.check_momentum_filter(signals, idx_pos)
        signal_info['momentum_pass'] = bool(momentum_pass)  # Convert to native Python bool
        
        # Check volatility
        volatility_pass = True
        if strategy.use_volatility_sizing and 'vol_regime' in signals.columns:
            extreme_vol = (signals['vol_regime'].iloc[idx_pos] == 'HIGH' and 
                          signals['annualized_vol'].iloc[idx_pos] > strategy.volatility_target * 3)
            volatility_pass = not extreme_vol
        signal_info['volatility_pass'] = bool(volatility_pass)  # Convert to native Python bool
        
        # Overall decision
        signal_info['final_pass'] = bool(mtf_pass and momentum_pass and volatility_pass)
        
        # Additional info for diagnostics
        if 'rsi' in signals.columns:
            signal_info['rsi'] = float(signals['rsi'].iloc[idx_pos])  # Convert to native Python float
        if 'vol_regime' in signals.columns:
            signal_info['vol_regime'] = str(signals['vol_regime'].iloc[idx_pos])  # Convert to native Python str
        if 'annualized_vol' in signals.columns:
            signal_info['volatility'] = float(signals['annualized_vol'].iloc[idx_pos])  # Convert to native Python float
        
        filtered_signals.append(signal_info)
    
    # Create analysis result
    analysis = {
        'raw_signals': int(len(raw_signals)),  # Convert to native Python int
        'filtered_signals': int(sum(1 for s in filtered_signals if s['final_pass'])),  # Convert to native Python int
        'filtered_details': filtered_signals,
        'filters': {
            'mtf': {
                'pass': int(sum(1 for s in filtered_signals if s['mtf_pass'])),  # Convert to native Python int
                'fail': int(sum(1 for s in filtered_signals if not s['mtf_pass']))  # Convert to native Python int
            },
            'momentum': {
                'pass': int(sum(1 for s in filtered_signals if s['momentum_pass'])),  # Convert to native Python int
                'fail': int(sum(1 for s in filtered_signals if not s['momentum_pass']))  # Convert to native Python int
            },
            'volatility': {
                'pass': int(sum(1 for s in filtered_signals if s['volatility_pass'])),  # Convert to native Python int
                'fail': int(sum(1 for s in filtered_signals if not s['volatility_pass']))  # Convert to native Python int
            }
        }
    }
    
    return analysis

def visualize_results(signals, analysis, symbol, timeframe, strategy_params):
    """
    Create visualization of the strategy results.
    
    Args:
        signals: DataFrame with signals
        analysis: Analysis results
        symbol: Trading symbol
        timeframe: Timeframe string
        strategy_params: Dictionary of strategy parameters
    """
    if signals is None or analysis is None:
        print("Nothing to visualize")
        return
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    
    # Setup grid
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 1])
    
    # Plot price and EMAs
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(signals.index, signals['close'], label='Price', color='black', alpha=0.5)
    ax1.plot(signals.index, signals['ema_fast'], label=f"Fast EMA ({strategy_params['fast_ema']})", color='blue')
    ax1.plot(signals.index, signals['ema_slow'], label=f"Slow EMA ({strategy_params['slow_ema']})", color='orange')
    ax1.plot(signals.index, signals['ema_trend'], label=f"Trend EMA ({strategy_params['trend_ema']})", color='red')
    
    # Extract signals for plotting
    filtered_signals = analysis['filtered_details']
    
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
    ax1.set_title(f'{symbol} {timeframe} - Simplified Strategy Signals', fontsize=14)
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot RSI in second subplot
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if 'rsi' in signals.columns:
        ax2.plot(signals.index, signals['rsi'], color='purple', label='RSI')
        ax2.axhline(y=strategy_params['momentum_threshold'], color='green', linestyle='--', 
                   alpha=0.7, label=f"RSI Threshold ({strategy_params['momentum_threshold']})")
        ax2.axhline(y=100-strategy_params['momentum_threshold'], color='red', linestyle='--', 
                   alpha=0.7)
        
        # Add RSI values for filtered signals
        for signal in filtered_signals:
            if 'momentum_pass' in signal and not signal['momentum_pass'] and 'rsi' in signal:
                marker = '^' if signal['signal'] > 0 else 'v'
                color = 'red'  # RSI filter failed
                ax2.scatter(signal['timestamp'], signal['rsi'], marker=marker, color=color, s=80)
        
        # Configure RSI plot
        ax2.set_title('RSI Momentum Filter', fontsize=12)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
    
    # Plot volatility in third subplot
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if 'annualized_vol' in signals.columns:
        ax3.plot(signals.index, signals['annualized_vol'], color='orange', label='Volatility')
        ax3.axhline(y=strategy_params['volatility_target'], color='blue', linestyle='--', 
                   alpha=0.7, label=f"Target ({strategy_params['volatility_target']})")
        
        # Mark volatility regimes
        if 'vol_regime' in signals.columns:
            regime_colors = {'LOW': 'green', 'NORMAL': 'gray', 'HIGH': 'red'}
            regimes = signals['vol_regime'].unique()
            
            for regime in regimes:
                regime_data = signals[signals['vol_regime'] == regime]
                ax3.scatter(regime_data.index, regime_data['annualized_vol'], 
                           marker='.', color=regime_colors.get(regime, 'blue'), 
                           alpha=0.5, s=10, label=f"{regime} Volatility")
        
        # Configure volatility plot
        ax3.set_title('Volatility Regimes', fontsize=12)
        ax3.set_ylabel('Annualized Volatility')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='upper left')
    
    # Configure layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('charts', exist_ok=True)
    
    # Save figure
    filename = f'charts/{symbol.replace("/", "_")}_{timeframe}_simplified_strategy.png'
    plt.savefig(filename)
    print(f'Chart saved to {filename}')
    
    # Print signal statistics
    print("\nSignal Statistics:")
    print(f"Raw Signals: {analysis['raw_signals']}")
    print(f"Final Signals (Passed All Filters): {analysis['filtered_signals']}")
    
    # Print filter effectiveness
    print("\nFilter Effects:")
    for filter_name, stats in analysis['filters'].items():
        pass_rate = (stats['pass'] / analysis['raw_signals']) * 100 if analysis['raw_signals'] > 0 else 0
        print(f"{filter_name.upper()} Filter: {stats['pass']} passed, {stats['fail']} failed ({pass_rate:.1f}% pass rate)")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test Simplified Strategy")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading symbol")
    parser.add_argument("--timeframe", type=str, default="4h", help="Trading timeframe")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data")
    parser.add_argument("--force-signals", action="store_true", default=True, help="Force signal generation")
    parser.add_argument("--signal-threshold", type=float, default=0.1, help="Signal threshold")
    parser.add_argument("--momentum-threshold", type=int, default=30, help="RSI momentum threshold")
    parser.add_argument("--mtf-mode", type=str, default="any", choices=["any", "both", "weighted"], help="MTF mode")
    parser.add_argument("--disable-mtf", action="store_true", help="Disable MTF filter")
    parser.add_argument("--disable-momentum", action="store_true", help="Disable momentum filter")
    parser.add_argument("--disable-vol", action="store_true", help="Disable volatility filter")
    
    args = parser.parse_args()
    
    # Create strategy parameters
    params = {
        "force_signals": args.force_signals,
        "signal_threshold": args.signal_threshold,
        "momentum_threshold": args.momentum_threshold,
        "mtf_mode": args.mtf_mode,
        "use_mtf_filter": not args.disable_mtf,
        "use_momentum_filter": not args.disable_momentum,
        "use_volatility_sizing": not args.disable_vol
    }
    
    # Run strategy test
    signals, strategy, analysis = run_strategy_test(args.symbol, args.timeframe, args.days, params)
    
    # Visualize results
    visualize_results(signals, analysis, args.symbol, args.timeframe, 
                     {**strategy.__dict__, **params})
    
    # Save analysis to file
    if analysis:
        os.makedirs('results', exist_ok=True)
        
        # Convert analysis to JSON-serializable format
        serializable_analysis = convert_to_serializable(analysis)
        
        filename = f"results/{args.symbol.replace('/', '_')}_{args.timeframe}_simplified_analysis.json"
        with open(filename, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"Analysis saved to {filename}")
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Days: {args.days}")
    print(f"Force Signals: {args.force_signals}")
    print(f"Signal Threshold: {args.signal_threshold}")
    print(f"MTF Filter: {'Disabled' if args.disable_mtf else 'Enabled'}")
    if not args.disable_mtf:
        print(f"  Mode: {args.mtf_mode}")
    print(f"Momentum Filter: {'Disabled' if args.disable_momentum else 'Enabled'}")
    if not args.disable_momentum:
        print(f"  Threshold: {args.momentum_threshold}")
    print(f"Volatility Filter: {'Disabled' if args.disable_vol else 'Enabled'}")

if __name__ == "__main__":
    main() 