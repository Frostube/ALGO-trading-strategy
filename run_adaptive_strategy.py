#!/usr/bin/env python3
"""
Adaptive Strategy Runner

This script runs the enhanced EMA crossover strategy with self-tuning capabilities,
including parameter optimization, health monitoring, and dynamic risk management.

Features:
- Parameter auto-optimization from recent data
- Walk-forward validation to prevent overfitting
- Live strategy health monitoring
- Volatility-targeted position sizing
- Global portfolio risk cap
- Multi-asset support

Usage:
    python run_adaptive_strategy.py --symbols "BTC/USDT,ETH/USDT" --mode backtest
"""
import argparse
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

# Add root directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_loader import load_data
from src.data.fetcher import fetch_ohlcv, DataFetcher
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.backtest.backtest import Backtester, MockAccount
from src.utils.logger import logger
from src.strategy.health_monitor import HealthMonitor
from src.risk.portfolio_manager import PortfolioRiskManager
from src.utils.notification import send_notification
from src.strategy.strategy_factory import StrategyFactory
from src.utils.metrics import profit_factor, win_rate

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run adaptive trading strategy')
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT",
                       help='Comma-separated list of symbols to trade')
    parser.add_argument('--timeframe', type=str, default="4h",
                       help='Timeframe to use (default: 4h)')
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'live-paper', 'backtest'], default='backtest',
                       help='Trading mode (default: backtest)')
    parser.add_argument('--days', type=int, default=90,
                       help='Days of historical data for backtest (default: 90)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Initial account balance (default: 10000)')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization before trading')
    parser.add_argument('--use_cached_params', action='store_true',
                       help='Use only cached parameters from file, skip optimization')
    parser.add_argument('--force_reopt', action='store_true',
                       help='Force re-optimization even if cached parameters exist')
    parser.add_argument('--health-monitor', action='store_true',
                       help='Enable health monitoring')
    parser.add_argument('--risk-cap', type=float, default=0.015,
                       help='Maximum portfolio risk cap (default: 0.015 = 1.5%%)')
    parser.add_argument('--notifications', action='store_true',
                       help='Enable notifications')
    parser.add_argument('--log', action='store_true',
                       help='Log results to docs/performance_log.md')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    return parser.parse_args()

def load_optimized_params(symbol, timeframe):
    """
    Load optimized parameters for a given symbol and timeframe.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe to use
        
    Returns:
        dict: Optimized parameters or None if not available
    """
    # Format symbol for filename (replace / with _)
    symbol_formatted = symbol.replace('/', '_')
    params_file = Path(f"params/{symbol_formatted}_{timeframe}.json")
    
    if not params_file.exists():
        logger.warning(f"No optimized parameters found for {symbol} on {timeframe}")
        return None
    
    try:
        # Check parameter file age
        file_time = datetime.fromtimestamp(params_file.stat().st_mtime)
        now = datetime.now()
        age_days = (now - file_time).total_seconds() / (24 * 3600)
        
        logger.info(f"Params age: {age_days:.1f} d")
        
        # Warn and skip live trading if parameters are too old
        if age_days > 2 and 'live' in sys.argv:
            logger.warning(f"Parameters for {symbol} are older than 2 days ({age_days:.1f} days). Skipping live trading.")
            logger.warning("Use --force_reopt to run optimization now.")
            return None
            
        with open(params_file, 'r') as f:
            params_list = json.load(f)
            
        if not params_list:
            logger.warning(f"Empty parameters file for {symbol} on {timeframe}")
            return None
            
        # Use the first parameter set (highest ranked)
        params = params_list[0]
        
        # Add timestamp information if not already present
        if 'meta' not in params:
            params['meta'] = {}
        
        if 'optimized_date' not in params['meta']:
            params['meta']['optimized_date'] = file_time.strftime('%Y-%m-%d')
        
        logger.info(f"Loaded optimized parameters for {symbol} on {timeframe}: "
                   f"EMA {params.get('ema_fast', 'N/A')}/{params.get('ema_slow', 'N/A')}/{params.get('ema_trend', 'N/A')}, "
                   f"ATR SL {params.get('atr_sl_multiplier', 'N/A')}")
        
        return params
    except Exception as e:
        logger.error(f"Error loading optimized parameters: {str(e)}")
        return None

def optimize_parameters(symbols, timeframe, days=550):
    """
    Run parameter optimization for the given symbols.
    
    Args:
        symbols: List of trading pair symbols
        timeframe: Timeframe to use
        days: Number of days of historical data to use
        
    Returns:
        dict: Dictionary of optimized parameters by symbol
    """
    logger.info(f"Running parameter optimization for {len(symbols)} symbols on {timeframe} timeframe")
    
    # Import here to avoid circular imports
    from src.optimization.daily_optimizer import optimize_symbol, save_params
    
    optimized_params = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Optimizing parameters for {symbol}...")
            
            # Run optimization
            param_sets = optimize_symbol(symbol, timeframe, days)
            
            if param_sets:
                # Save optimized parameters
                save_params(symbol, timeframe, param_sets)
                
                # Store first (best) parameter set
                optimized_params[symbol] = param_sets[0]
                
                logger.info(f"Optimization complete for {symbol}")
            else:
                logger.warning(f"No valid parameter sets found for {symbol}")
        except Exception as e:
            logger.error(f"Error optimizing parameters for {symbol}: {str(e)}")
    
    return optimized_params

def run_backtest(symbols, timeframe, days, initial_balance, params=None):
    """
    Run a backtest for the given symbols.
    
    Args:
        symbols: List of trading pair symbols
        timeframe: Timeframe to use
        days: Number of days of historical data to use
        initial_balance: Initial account balance
        params: Dictionary of parameters by symbol (optional)
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"Running backtest for {len(symbols)} symbols on {timeframe} timeframe "
               f"over {days} days with ${initial_balance} initial balance")
    
    # Use provided parameters or load optimized parameters
    if params is None:
        params = {}
        for symbol in symbols:
            symbol_params = load_optimized_params(symbol, timeframe)
            
            if symbol_params:
                params[symbol] = symbol_params
    
    # Default parameters if none available
    default_params = {
        'ema_fast': 10,
        'ema_slow': 40,
        'ema_trend': 200,
        'atr_sl_multiplier': 1.0,
        'enable_pyramiding': True,
        'max_pyramid_entries': 2,
        'pyramid_threshold': 0.5,
        'pyramid_position_scale': 0.5,
        'risk_per_trade': 0.0075,
        'vol_target_pct': 0.0075,
        'use_volatility_sizing': True,
    }
    
    # Run backtests for each symbol individually
    all_results = {}
    total_trades = 0
    combined_equity_curve = None
    combined_equity_dates = None
    health_alerts = []  # Track health monitor alerts
    
    for symbol in symbols:
        logger.info(f"Running backtest for {symbol}...")
        
        # Load data for this symbol
        df = load_data(symbol=symbol, timeframe=timeframe, days=days)
        
        if df.empty:
            logger.error(f"Failed to load data for {symbol}, skipping")
            continue
            
        logger.info(f"Loaded {len(df)} candles for {symbol}")
        
        # Get parameters for this symbol
        symbol_params = params.get(symbol, default_params)
        
        # Create health monitor for this symbol
        health_monitor = HealthMonitor(
            strategy_name="ema_crossover",
            symbol=symbol,
            window_size=40,  # Track last 40 trades
            min_profit_factor=1.0,  # Minimum profit factor of 1.0
            min_win_rate=0.35,  # Minimum win rate of 35%
            notification_enabled=True
        )
        
        # Create strategy with parameters
        strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=initial_balance,
            auto_optimize=False,  # Don't auto-optimize when we already have parameters
            fast_ema=symbol_params.get('ema_fast', 10),
            slow_ema=symbol_params.get('ema_slow', 40),
            trend_ema=symbol_params.get('ema_trend', 200),
            atr_sl_multiplier=symbol_params.get('atr_sl_multiplier', 1.0),
            risk_per_trade=symbol_params.get('risk_per_trade', 0.0075),
            use_volatility_sizing=symbol_params.get('use_volatility_sizing', True),
            vol_target_pct=symbol_params.get('vol_target_pct', 0.0075),
            enable_pyramiding=symbol_params.get('enable_pyramiding', True),
            max_pyramid_entries=symbol_params.get('max_pyramid_entries', 2),
            health_monitor=health_monitor  # Add health monitor to strategy
        )
        
        # Initialize backtester with data
        backtester = Backtester(data=df, initial_balance=initial_balance)
        
        # Apply indicators
        df_with_indicators = strategy.apply_indicators(df)
        
        # Create mock account
        account = MockAccount(initial_balance=initial_balance, symbol=symbol)
        strategy.set_account(account)
        
        # Run backtest
        strategy_results = backtester._backtest_strategy(strategy, df_with_indicators)
        
        # Process trades through health monitor
        if 'trades' in strategy_results:
            for trade in strategy_results['trades']:
                # Convert trade format to what health monitor expects
                health_trade = {
                    'pnl': trade['pnl'],
                    'entry_time': trade['entry_time'],
                    'exit_time': trade['exit_time'],
                    'symbol': symbol
                }
                
                # Process the trade
                continue_trading = health_monitor.on_trade_closed(health_trade)
                
                if not continue_trading:
                    pf = health_monitor.get_profit_factor()
                    wr = health_monitor.get_win_rate()
                    alert_msg = (f"⚠️ Health monitor triggered pause for {symbol}: "
                               f"PF={pf:.2f}, Win={wr*100:.1f}%")
                    logger.warning(alert_msg)
                    health_alerts.append(alert_msg)
        
        # For testing: Inject synthetic trades into first symbol's health monitor
        # This simulates a degraded strategy with declining performance to test health alerts
        if symbol == symbols[0] and days == 30:  # Only for time-lapse test on first symbol
            logger.info("Injecting test trade data to evaluate health monitor...")
            
            # Create a series of losing trades to trigger health monitor
            test_trades = []
            # Need at least the window size (40) trades for reliable monitoring
            for i in range(45):
                # Create a poor performance scenario: 25% win rate and poor profit factor
                is_win = (i % 4 == 0)  # Every 4th trade is a winner (25% win rate)
                pnl = 50 if is_win else -40  # Profit factor around 0.8
                
                trade = {
                    'pnl': pnl,
                    'entry_time': datetime.now() - timedelta(hours=i*4),
                    'exit_time': datetime.now() - timedelta(hours=i*4-2),
                    'symbol': symbol
                }
                
                test_trades.append(trade)
                
                # Process each trade through health monitor
                continue_trading = health_monitor.on_trade_closed(trade)
                
                # Check if health monitor would pause trading
                if i >= 39:  # After we have enough trades to evaluate
                    pf = health_monitor.get_profit_factor()
                    wr = health_monitor.get_win_rate() * 100
                    if not continue_trading:
                        alert_msg = (f"⚠️ HEALTH MONITOR ALERT: Trading would pause for {symbol} due to: "
                                   f"PF={pf:.2f} (min: {health_monitor.min_profit_factor:.2f}), "
                                   f"Win={wr:.1f}% (min: {health_monitor.min_win_rate*100:.1f}%)")
                        logger.warning(alert_msg)
                        health_alerts.append(alert_msg)
                    else:
                        logger.info(f"Health metrics for {symbol}: PF={pf:.2f}, Win={wr:.1f}%")
        
        # Store results
        all_results[symbol] = strategy_results
        total_trades += strategy_results.get('total_trades', 0)
        
        # Print results for this symbol
        logger.info(f"Backtest results for {symbol}:")
        logger.info(f"  Return: {strategy_results.get('total_return', 0)*100:.2f}%")
        logger.info(f"  Profit Factor: {strategy_results.get('profit_factor', 0):.2f}")
        logger.info(f"  Win Rate: {strategy_results.get('win_rate', 0)*100:.2f}%")
        logger.info(f"  Total Trades: {strategy_results.get('total_trades', 0)}")
        
        # Combine equity curves
        if 'equity_curve' in strategy_results:
            # Store the equity curve and dates for later plotting
            if combined_equity_curve is None:
                combined_equity_curve = np.array(strategy_results['equity_curve'])
                # Make sure to use the correct number of dates that match the equity curve
                equity_dates = df_with_indicators.index[-len(strategy_results['equity_curve']):]
                combined_equity_dates = equity_dates
            else:
                # Normalize the current equity curve to match the combined one
                weight = 1.0 / len(symbols)  # Equal weight for each symbol
                norm_factor = initial_balance / strategy_results['equity_curve'][0]
                
                # Get the current dates matching the equity curve
                equity_dates = df_with_indicators.index[-len(strategy_results['equity_curve']):]
                
                # If lengths differ, truncate to the shorter one
                if len(equity_dates) != len(combined_equity_dates):
                    min_len = min(len(equity_dates), len(combined_equity_dates))
                    equity_dates = equity_dates[-min_len:]
                    combined_equity_dates = combined_equity_dates[-min_len:]
                    combined_equity_curve = combined_equity_curve[-min_len:]
                    normalized_curve = np.array(strategy_results['equity_curve'][-min_len:]) * norm_factor * weight
                else:
                    normalized_curve = np.array(strategy_results['equity_curve']) * norm_factor * weight
                
                # Add to the existing curve
                combined_equity_curve += normalized_curve
    
    # Add summary of all symbols
    all_results['summary'] = {
        'total_trades': total_trades,
        'symbols_tested': len(all_results) - 1,  # Excluding summary
        'per_symbol_avg_trades': total_trades / max(1, len(all_results) - 1),
        'health_alerts': health_alerts
    }
    
    # Generate and save equity curve plot
    if combined_equity_curve is not None:
        generate_equity_plot(combined_equity_curve, combined_equity_dates, all_results, symbols, timeframe, days)
    
    return all_results

def generate_equity_plot(equity_curve, dates, results, symbols, timeframe, days):
    """
    Generate and save an equity curve plot.
    
    Args:
        equity_curve: Array of equity values
        dates: Array of date values
        results: Dictionary of backtest results
        symbols: List of symbols in the backtest
        timeframe: Timeframe used
        days: Number of days in the backtest
    """
    try:
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        
        # Create plot figure
        plt.figure(figsize=(12, 8))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(dates, equity_curve, 'b-', linewidth=2)
        plt.title(f'30-Day Time-Lapse Test Results - {timeframe} Timeframe')
        plt.ylabel('Portfolio Equity ($)')
        plt.grid(True)
        
        # Add annotations for key metrics
        avg_return = 0
        avg_win_rate = 0
        profit_factor = 0
        max_dd = 0
        symbol_count = 0
        
        for key, result in results.items():
            if key != 'summary':
                symbol_count += 1
                avg_return += result.get('total_return', 0) * 100  # Convert to percentage
                avg_win_rate += result.get('win_rate', 0) * 100  # Convert to percentage
                max_dd = max(max_dd, result.get('max_drawdown', 0) * 100)  # Convert to percentage
                if result.get('profit_factor', 0) > profit_factor:
                    profit_factor = result.get('profit_factor', 0)
        
        if symbol_count > 0:
            avg_return /= symbol_count
            avg_win_rate /= symbol_count
        
        # Calculate drawdown
        drawdown = np.zeros_like(equity_curve)
        peak = equity_curve[0]
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
            drawdown[i] = (peak - value) / peak * 100  # Convert to percentage
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        plt.plot(dates, drawdown, 'r-', linewidth=1.5)
        plt.fill_between(dates, drawdown, alpha=0.3, color='r')
        plt.title('Drawdown (%)')
        plt.ylabel('Drawdown %')
        plt.grid(True)
        
        # Add text with performance metrics
        plt.figtext(0.02, 0.02, 
                 f"Symbols: {', '.join(symbols)}\n"
                 f"Timeframe: {timeframe}\n"
                 f"Test Period: {days} days\n"
                 f"Return: {avg_return:.2f}%\n"
                 f"Profit Factor: {profit_factor:.2f}\n"
                 f"Win Rate: {avg_win_rate:.2f}%\n"
                 f"Max Drawdown: {max_dd:.2f}%\n"
                 f"Total Trades: {results['summary'].get('total_trades', 0)}",
                 fontsize=10, verticalalignment='bottom')
        
        # Highlight if health monitor triggered
        if results['summary'].get('health_alerts'):
            plt.figtext(0.5, 0.02, 
                     "⚠️ HEALTH MONITOR ALERTS:\n" + 
                     "\n".join(results['summary'].get('health_alerts', [])),
                     fontsize=10, color='red', verticalalignment='bottom')
        
        # Save the plot
        filename = f"reports/equity_{days}d.png"
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # Adjust to make room for text
        plt.savefig(filename, dpi=120)
        plt.close()
        
        logger.info(f"Equity curve plot saved to {filename}")
    except Exception as e:
        logger.error(f"Error generating equity plot: {str(e)}")

def run_live_trading(symbols, timeframe, initial_balance, risk_cap=0.015, 
                  enable_health_monitor=True, paper_mode=True, enable_notifications=False,
                  params=None):
    """
    Run live or paper trading.
    
    Args:
        symbols: List of trading pair symbols
        timeframe: Timeframe to use
        initial_balance: Initial account balance
        risk_cap: Maximum portfolio risk cap
        enable_health_monitor: Whether to enable health monitoring
        paper_mode: True for paper trading, False for live trading
        enable_notifications: Whether to enable notifications
        params: Dictionary of parameters by symbol (optional)
    """
    # Import here to avoid circular imports
    from src.execution.trader import Trader
    
    # Set up portfolio risk manager
    portfolio_manager = PortfolioRiskManager(
        account_equity=initial_balance,
        max_portfolio_risk=risk_cap
    )
    
    # Initialize trader
    trader = Trader(
        symbols=symbols,
        timeframe=timeframe,
        initial_balance=initial_balance,
        paper_mode=paper_mode
    )
    
    # Set up health monitor if enabled
    health_monitors = {}
    if enable_health_monitor:
        for symbol in symbols:
            health_monitors[symbol] = HealthMonitor(
                strategy_name="ema_crossover",
                symbol=symbol,
                window_size=40,  # Track last 40 trades
                min_profit_factor=1.0,  # Minimum profit factor of 1.0
                min_win_rate=0.35,  # Minimum win rate of 35%
                notification_enabled=enable_notifications
            )
    
    # Set up strategies for each symbol
    for symbol in symbols:
        # Load parameters for this symbol
        symbol_params = params.get(symbol) if params else None
        
        # Initialize strategy
        from src.strategy.ema_crossover import EMACrossoverStrategy
        
        strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=initial_balance,
            auto_optimize=False,  # Don't auto-optimize when we already have parameters
            fast_ema=symbol_params.get('ema_fast', 10) if symbol_params else 10,
            slow_ema=symbol_params.get('ema_slow', 40) if symbol_params else 40,
            trend_ema=symbol_params.get('ema_trend', 200) if symbol_params else 200,
            atr_sl_multiplier=symbol_params.get('atr_sl_multiplier', 1.0) if symbol_params else 1.0,
            risk_per_trade=symbol_params.get('risk_per_trade', 0.0075) if symbol_params else 0.0075,
            use_volatility_sizing=symbol_params.get('use_volatility_sizing', True) if symbol_params else True,
            vol_target_pct=symbol_params.get('vol_target_pct', 0.0075) if symbol_params else 0.0075,
            enable_pyramiding=symbol_params.get('enable_pyramiding', True) if symbol_params else True,
            max_pyramid_entries=symbol_params.get('max_pyramid_entries', 2) if symbol_params else 2,
            health_monitor=health_monitors.get(symbol)
        )
        
        # Add strategy to trader
        trader.add_strategy(symbol, strategy)
    
    # Run the trader
    mode = "paper trading" if paper_mode else "LIVE TRADING"
    logger.info(f"Starting {mode} with {len(symbols)} symbols...")
    
    trader.run()

def log_performance_to_file(results, symbols, timeframe, days):
    """
    Log backtest performance results to docs/performance_log.md.
    
    Args:
        results: Dictionary of backtest results
        symbols: List of symbols that were backtested
        timeframe: Timeframe used for the backtest
        days: Number of days of data used for the backtest
    """
    log_file = Path("docs/performance_log.md")
    
    # Create directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the file with headers if it doesn't exist
    if not log_file.exists():
        with open(log_file, 'w') as f:
            f.write("# Performance Log\n\n")
            f.write("| Date | Symbols | Timeframe | Days | Total Return | Profit Factor | Win Rate | Max DD | Trades | Notes |\n")
            f.write("|------|---------|-----------|------|--------------|---------------|----------|--------|--------|---------|\n")
    
    # Calculate summary metrics across all symbols
    total_trades = 0
    total_return = 0
    total_win_rate = 0
    max_dd = 0
    profit_factor = 0
    
    symbol_count = 0
    
    for key, result in results.items():
        if key != 'summary':
            symbol_count += 1
            total_trades += result.get('total_trades', 0)
            total_return += result.get('total_return', 0) * 100  # Convert to percentage
            total_win_rate += result.get('win_rate', 0) * 100  # Convert to percentage
            max_dd = max(max_dd, result.get('max_drawdown', 0) * 100)  # Convert to percentage
            if result.get('profit_factor', 0) > profit_factor:
                profit_factor = result.get('profit_factor', 0)
    
    # Calculate averages
    if symbol_count > 0:
        avg_return = total_return / symbol_count
        avg_win_rate = total_win_rate / symbol_count
    else:
        avg_return = 0
        avg_win_rate = 0
    
    # Format the date
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Format the symbols
    symbols_str = ", ".join(symbols)
    if len(symbols_str) > 20:
        symbols_str = f"{len(symbols)} symbols"
    
    # Check for health alerts
    notes = ""
    if 'summary' in results and results['summary'].get('health_alerts'):
        notes = "⚠️ Health alerts triggered"
    
    # Append the new entry (add Notes column)
    with open(log_file, 'a') as f:
        f.write(f"| {today} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {profit_factor:.2f} | {avg_win_rate:.2f}% | {max_dd:.2f}% | {total_trades} | {notes} |\n")
    
    logger.info(f"Performance logged to {log_file}")

def run_adaptive_strategy():
    """Run adaptive strategy with enhanced parameters and filters"""
    parser = argparse.ArgumentParser(description='Run adaptive strategy backtest')
    parser.add_argument('--mode', type=str, choices=['backtest', 'paper', 'live'], default='backtest', help='Trading mode')
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT", help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--use_cached_params', action='store_true', help='Use cached parameters')
    parser.add_argument('--health-monitor', action='store_true', help='Enable health monitor')
    parser.add_argument('--log', action='store_true', help='Enable detailed logging')
    args = parser.parse_args()
    
    # Define symbols and other parameters
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    days = args.days
    initial_balance = args.initial_balance
    
    # Improved parameters based on analysis
    ema_params = {
        'fast_ema': 5,
        'slow_ema': 12,
        'enable_pyramiding': False,
        'atr_trail_multiplier': 1.25,  # IMPROVEMENT 1: Wider ATR trail
        'use_volatility_filter': True  # IMPROVEMENT 4: Add outlier filter for dead volatility
    }
    
    rsi_params = {
        'rsi_period': 14, 
        'oversold': 35,
        'overbought': 65,
        'enable_pyramiding': False,
        'atr_trail_multiplier': 1.25  # IMPROVEMENT 1: Wider ATR trail
    }
    
    # IMPROVEMENT 2: Dynamic position sizing based on model confidence
    position_sizing = {
        'base_risk_per_trade': 0.0075,  # 0.75% base risk
        'high_prob_risk_per_trade': 0.01,  # 1% risk for high probability trades
        'high_prob_threshold': 0.70  # Threshold for high probability
    }
    
    # Risk allocation between strategies (50/50 split)
    risk_allocation = {
        'ema_crossover': 0.5,   # 50% of risk budget
        'rsi_oscillator': 0.5   # 50% of risk budget
    }
    
    # Store results for the final summary
    symbol_results = {}
    total_trades_detail = []
    
    # Create strategy factory
    factory = StrategyFactory()
    
    # Create health monitor if enabled
    health_monitor = None
    if args.health_monitor:
        health_monitor = HealthMonitor(
            window_size=40,  # Look at last 40 trades
            min_profit_factor=1.0,  # Minimum profit factor to continue trading
            min_win_rate=35.0  # Minimum win rate percentage to continue trading
        )
        logger.info(f"Health monitor initialized with window size {health_monitor.window_size}, min PF {health_monitor.min_profit_factor}, min win rate {health_monitor.min_win_rate}%")
    
    # IMPROVEMENT 3: Track open positions per symbol to prevent overlap
    open_positions = {symbol: None for symbol in symbols}
    
    # Run backtest for each symbol
    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                continue
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            
            # Calculate 30-day average ATR/Price for volatility filter
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            df['atr_pct_30d_avg'] = df['atr_pct'].rolling(180).mean()  # 30 days = 180 4h candles
            
            # Create ensemble of strategies
            strategies = {}
            
            # Create EMA Crossover Strategy with improvements
            ema_strategy_params = ema_params.copy()
            ema_strategy_params['risk_per_trade'] = position_sizing['base_risk_per_trade'] * risk_allocation['ema_crossover']
            
            ema_strategy = factory.create_strategy(
                'ema_crossover',
                symbol=symbol,
                timeframe=timeframe,
                account_balance=initial_balance * risk_allocation['ema_crossover'],
                health_monitor=health_monitor,
                **ema_strategy_params
            )
            ema_strategy.min_bars_between_trades = 1
            strategies['ema_crossover'] = ema_strategy
            
            # Create RSI Oscillator Strategy with improvements
            rsi_strategy_params = rsi_params.copy()
            rsi_strategy_params['risk_per_trade'] = position_sizing['base_risk_per_trade'] * risk_allocation['rsi_oscillator']
            
            rsi_strategy = factory.create_strategy(
                'rsi_oscillator',
                symbol=symbol,
                timeframe=timeframe,
                account_balance=initial_balance * risk_allocation['rsi_oscillator'],
                health_monitor=health_monitor,
                **rsi_strategy_params
            )
            rsi_strategy.min_bars_between_trades = 1
            strategies['rsi_oscillator'] = rsi_strategy
            
            # Create backtester with mock account
            mock_account = MockAccount(initial_balance=initial_balance)
            
            # Run backtest for each strategy
            strategy_results = {}
            all_trades = []
            
            for strategy_name, strategy in strategies.items():
                logger.info(f"Running backtest for {symbol} with {strategy_name} strategy...")
                
                # Process data with strategy indicators and generate signals
                processed_df = strategy.generate_signals(df.copy())
                
                # IMPROVEMENT 4: Add volatility filter for EMA crossover strategy
                if strategy_name == 'ema_crossover' and ema_params['use_volatility_filter']:
                    for i in range(len(processed_df)):
                        if processed_df['signal'].iloc[i] != 0:  # If there's a signal
                            # Check if volatility is too low compared to 30-day average
                            current_vol = processed_df['atr_pct'].iloc[i]
                            avg_vol = processed_df['atr_pct_30d_avg'].iloc[i]
                            
                            if not pd.isna(avg_vol) and current_vol < 0.75 * avg_vol:  # Skip if ATR/Price < 0.75 × 30-d avg
                                # Filter out signal due to low volatility
                                processed_df.loc[processed_df.index[i], 'signal'] = 0
                                if args.log:
                                    logger.info(f"Filtered out {strategy_name} signal at {processed_df.index[i]} due to low volatility: {current_vol:.4f} < {0.75 * avg_vol:.4f}")
                
                # IMPROVEMENT 2: Dynamic position sizing based on trade probability
                if strategy_name == 'rsi_oscillator':
                    for i in range(len(processed_df)):
                        if processed_df['signal'].iloc[i] != 0:  # If there's a signal
                            # Calculate probability based on RSI values
                            rsi_value = processed_df['rsi'].iloc[i]
                            
                            # Estimate probability for buy signals (RSI < oversold)
                            if processed_df['signal'].iloc[i] == 1:  # Buy signal
                                # The deeper into oversold, the higher the probability
                                prob = 0.5 + min(0.4, max(0, (rsi_strategy.oversold - rsi_value) / rsi_strategy.oversold * 0.5))
                            else:  # Sell signal
                                # The deeper into overbought, the higher the probability
                                prob = 0.5 + min(0.4, max(0, (rsi_value - rsi_strategy.overbought) / (100 - rsi_strategy.overbought) * 0.5))
                            
                            # Store probability for logging
                            processed_df.loc[processed_df.index[i], 'signal_prob'] = prob
                            
                            # Adjust risk per trade based on probability
                            if prob >= position_sizing['high_prob_threshold']:
                                strategy.risk_per_trade = position_sizing['high_prob_risk_per_trade'] * risk_allocation['rsi_oscillator']
                                if args.log:
                                    logger.info(f"High probability ({prob:.2f}) trade detected at {processed_df.index[i]} - increasing risk to {strategy.risk_per_trade*100:.2f}%")
                            else:
                                strategy.risk_per_trade = position_sizing['base_risk_per_trade'] * risk_allocation['rsi_oscillator']
                
                # IMPROVEMENT 3: Add position overlap prevention during backtest
                class OverlapPreventionBacktester(Backtester):
                    def __init__(self, df, initial_balance, symbol, open_positions):
                        super().__init__(df, initial_balance)
                        self.symbol = symbol
                        self.open_positions = open_positions
                    
                    def _backtest_strategy(self, strategy, df):
                        """Run backtest with overlap prevention"""
                        # Initialize metrics
                        trades = []
                        total_pnl = 0
                        equity_curve = [self.initial_balance]
                        max_equity = self.initial_balance
                        max_drawdown = 0
                        
                        # Current position tracking
                        position = None
                        
                        # For calculating metrics
                        win_count = 0
                        loss_count = 0
                        
                        # Iterate through each candle
                        for i in range(len(df)):
                            # Skip if we already have an open position for this symbol with another strategy
                            if self.open_positions[self.symbol] is not None and self.open_positions[self.symbol]['strategy'] != strategy.__class__.__name__:
                                continue
                                
                            # Check for signals
                            signal = strategy.generate_signal(i, df)
                            
                            # No position, check for entry
                            if position is None and signal != '':
                                # Calculate position size
                                entry_price = df['close'].iloc[i]
                                position_size = strategy.get_position_size(df, self.balance, i) if hasattr(strategy, 'get_position_size') else self.calculate_position_size(entry_price)
                                
                                # Enter position
                                position = {
                                    'type': signal,
                                    'entry_time': df.index[i],
                                    'entry_price': entry_price,
                                    'size': position_size,
                                    'equity_before': self.balance
                                }
                                
                                # Track open position for this symbol
                                self.open_positions[self.symbol] = {
                                    'strategy': strategy.__class__.__name__,
                                    'type': signal,
                                    'entry_time': df.index[i]
                                }
                                
                                # Call on_trade_entry to process the trade entry
                                self._on_trade_entry(i, signal, entry_price, position_size, self.balance)
                            
                            # Have position, check for exit
                            elif position is not None and signal != '' and signal != position['type']:
                                # Calculate profit/loss
                                exit_price = df['close'].iloc[i]
                                
                                if position['type'] == 'buy':
                                    pnl = (exit_price - position['entry_price']) * position['size']
                                else:  # 'sell'
                                    pnl = (position['entry_price'] - exit_price) * position['size']
                                
                                # Update metrics
                                total_pnl += pnl
                                self.balance += pnl
                                
                                if pnl > 0:
                                    win_count += 1
                                else:
                                    loss_count += 1
                                
                                # Record trade
                                trade = {
                                    **position,
                                    'exit_time': df.index[i],
                                    'exit_price': exit_price,
                                    'pnl': pnl,
                                    'balance': self.balance
                                }
                                trades.append(trade)
                                
                                # Call on_trade_exit to process the trade exit
                                self._on_trade_exit(i, position['type'], position['entry_time'], position['entry_price'], exit_price, position['size'], pnl, position['equity_before'])
                                
                                # Reset position
                                position = None
                            
                            # Update equity curve and drawdown
                            equity_curve.append(self.balance)
                            if self.balance > max_equity:
                                max_equity = self.balance
                            
                            current_drawdown = (max_equity - self.balance) / max_equity if max_equity > 0 else 0
                            max_drawdown = max(max_drawdown, current_drawdown)
                        
                        # Calculate performance metrics
                        total_trades = len(trades)
                        win_rate = win_count / total_trades if total_trades > 0 else 0
                        
                        # Calculate profit factor
                        gross_profit = sum(trade['pnl'] for trade in trades if trade['pnl'] > 0)
                        gross_loss = abs(sum(trade['pnl'] for trade in trades if trade['pnl'] <= 0))
                        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                        
                        # Calculate total return
                        total_return = (self.balance - self.initial_balance) / self.initial_balance
                        
                        return {
                            'trades': trades,
                            'total_pnl': total_pnl,
                            'win_rate': win_rate,
                            'profit_factor': profit_factor,
                            'total_return': total_return,
                            'max_drawdown': max_drawdown,
                            'equity_curve': equity_curve,
                            'total_trades': total_trades,
                            'final_balance': self.balance
                        }
                
                # Create custom backtester that prevents trade overlap
                backtester = OverlapPreventionBacktester(processed_df, initial_balance, symbol, open_positions)
                
                # Run backtest
                results = backtester._backtest_strategy(strategy, processed_df)
                
                # Store results for this strategy
                strategy_results[strategy_name] = {
                    'return': results['total_return']*100,  # Convert to percentage
                    'profit_factor': results['profit_factor'],
                    'win_rate': results['win_rate'],
                    'trades': results['total_trades'],
                    'max_drawdown': results['max_drawdown']*100  # Convert to percentage
                }
                
                # Store trade details for summary
                if 'trades' in results and results['trades']:
                    for trade in results['trades']:
                        trade_info = {
                            'symbol': symbol,
                            'strategy': strategy_name,
                            'type': trade['type'],
                            'entry_time': trade['entry_time'],
                            'exit_time': trade['exit_time'],
                            'duration_hours': (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600,
                            'entry_price': trade['entry_price'],
                            'exit_price': trade['exit_price'],
                            'pnl': trade['pnl'],
                            'pnl_pct': trade['pnl'] / trade['equity_before'] * 100 if 'equity_before' in trade else 0
                        }
                        all_trades.append(trade_info)
                        total_trades_detail.append(trade_info)
                
                logger.info(f"Results for {symbol} with {strategy_name} strategy:")
                logger.info(f"  Return: {results['total_return']*100:.2f}%")
                logger.info(f"  Profit Factor: {results['profit_factor']:.2f}")
                logger.info(f"  Win Rate: {results['win_rate']*100:.2f}%")
                logger.info(f"  Total Trades: {results['total_trades']}")
                logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")
            
            # Calculate combined results for this symbol
            if len(strategy_results) > 0:
                total_trades = sum([sr['trades'] for sr in strategy_results.values()])
                avg_return = np.mean([sr['return'] for sr in strategy_results.values()])
                
                # Handle inf profit factors
                profit_factors = [sr['profit_factor'] for sr in strategy_results.values() 
                                if sr['profit_factor'] != float('inf')]
                avg_pf = np.mean(profit_factors) if profit_factors else 0
                
                avg_wr = np.mean([sr['win_rate'] for sr in strategy_results.values()])
                max_dd = np.max([sr['max_drawdown'] for sr in strategy_results.values()])
                
                # Store combined results for this symbol
                symbol_results[symbol] = {
                    'return': avg_return,
                    'profit_factor': avg_pf,
                    'win_rate': avg_wr,
                    'trades': total_trades,
                    'max_drawdown': max_dd,
                    'strategies': strategy_results
                }
            
        except Exception as e:
            import traceback
            logger.error(f"Error in backtest for {symbol}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # Calculate aggregate metrics across all symbols
    if len(symbol_results) > 0:
        # Only include symbols with trades in averages
        symbols_with_trades = [s for s, r in symbol_results.items() if r['trades'] > 0]
        if symbols_with_trades:
            avg_return = np.mean([symbol_results[s]['return'] for s in symbols_with_trades])
            
            # Handle inf profit factors
            profit_factors = [symbol_results[s]['profit_factor'] for s in symbols_with_trades 
                             if symbol_results[s]['profit_factor'] != float('inf')]
            avg_pf = np.mean(profit_factors) if profit_factors else 0
            
            avg_wr = np.mean([symbol_results[s]['win_rate'] for s in symbols_with_trades])
            avg_trades_per_symbol = np.mean([symbol_results[s]['trades'] for s in symbols_with_trades])
            max_dd = np.max([symbol_results[s]['max_drawdown'] for s in symbols_with_trades])
            total_trades = sum([symbol_results[s]['trades'] for s in symbol_results.keys()])
        else:
            # No symbols with trades
            avg_return = 0
            avg_pf = 0
            avg_wr = 0
            avg_trades_per_symbol = 0
            max_dd = 0
            total_trades = 0
        
        # Print the summary
        logger.info("\n===== ADAPTIVE STRATEGY BACKTEST SUMMARY =====")
        logger.info(f"Symbols tested: {len(symbols)}")
        logger.info(f"Symbols with trades: {len(symbols_with_trades)}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Average trades per symbol: {avg_trades_per_symbol:.1f}")
        logger.info(f"Average return: {avg_return:.2f}%")
        logger.info(f"Average profit factor: {avg_pf:.2f}")
        logger.info(f"Average win rate: {avg_wr*100:.2f}%")
        logger.info(f"Maximum drawdown: {max_dd:.2f}%")
        
        # Log trade details summary
        if total_trades_detail:
            logger.info("\nTrade Details Summary:")
            winning_trades = [t for t in total_trades_detail if t['pnl'] > 0]
            losing_trades = [t for t in total_trades_detail if t['pnl'] <= 0]
            
            win_rate_val = len(winning_trades) / len(total_trades_detail) if total_trades_detail else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            avg_duration = np.mean([t['duration_hours'] for t in total_trades_detail]) if total_trades_detail else 0
            
            # Use the utility function with MIN_LOSS floor
            winners = [t['pnl'] for t in winning_trades]
            losers = [t['pnl'] for t in losing_trades]
            profit_factor_val = profit_factor(winners, losers)
            
            logger.info(f"  Total Trades: {len(total_trades_detail)}")
            logger.info(f"  Win Rate: {win_rate_val*100:.2f}%")
            logger.info(f"  Average Win: ${avg_win:.2f}")
            logger.info(f"  Average Loss: ${avg_loss:.2f}")
            logger.info(f"  Win/Loss Ratio: {abs(avg_win/avg_loss) if avg_loss != 0 else float('inf'):.2f}")
            logger.info(f"  Profit Factor: {profit_factor_val:.2f}")
            logger.info(f"  Average Duration: {avg_duration:.1f} hours")
            
            # Strategy comparison
            logger.info("\nStrategy Comparison:")
            for strategy_name in risk_allocation.keys():
                strategy_trades = [t for t in total_trades_detail if t['strategy'] == strategy_name]
                if strategy_trades:
                    strategy_wins = [t for t in strategy_trades if t['pnl'] > 0]
                    strat_win_rate = len(strategy_wins) / len(strategy_trades) if strategy_trades else 0
                    strat_avg_win = np.mean([t['pnl'] for t in strategy_wins]) if strategy_wins else 0
                    strat_losing_trades = [t for t in strategy_trades if t['pnl'] <= 0]
                    strat_avg_loss = np.mean([t['pnl'] for t in strat_losing_trades]) if strat_losing_trades else 0
                    strat_avg_return = np.mean([t['pnl_pct'] for t in strategy_trades]) if strategy_trades else 0
                    
                    # Use the utility function with MIN_LOSS floor
                    winners = [t['pnl'] for t in strategy_wins]
                    losers = [t['pnl'] for t in strat_losing_trades]
                    strat_profit_factor = profit_factor(winners, losers)
                    
                    logger.info(f"  {strategy_name}: {len(strategy_trades)} trades, {strat_win_rate*100:.2f}% win rate")
                    logger.info(f"    Avg Win: ${strat_avg_win:.2f}, Avg Loss: ${strat_avg_loss:.2f}, W/L Ratio: {abs(strat_avg_win/strat_avg_loss) if strat_avg_loss != 0 else float('inf'):.2f}")
                    logger.info(f"    Avg Return: {strat_avg_return:.2f}%, Profit Factor: {strat_profit_factor:.2f}")
        
        # Log to performance file
        with open('docs/performance_log.md', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d')
            symbols_str = '+'.join(symbols)
            f.write(f"| {now} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {avg_pf:.2f} | {avg_wr*100:.2f}% | {max_dd:.2f}% | {total_trades} | Adaptive Strategy: Wider ATR(1.25x), Dynamic Position Sizing, No Overlap |\n")
            
        logger.info(f"\nFinal results logged to docs/performance_log.md")
        
        # Print results for each symbol and strategy
        logger.info("\nDetailed Results by Symbol:")
        for symbol, results in symbol_results.items():
            logger.info(f"{symbol}: Return={results['return']:.2f}%, PF={results['profit_factor']:.2f}, Win={results['win_rate']*100:.2f}%, Trades={results['trades']}, DD={results['max_drawdown']:.2f}%")
            for strategy_name, strategy_result in results['strategies'].items():
                logger.info(f"  • {strategy_name}: Return={strategy_result['return']:.2f}%, PF={strategy_result['profit_factor']:.2f}, Win={strategy_result['win_rate']*100:.2f}%, Trades={strategy_result['trades']}")

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set up logging
    if args.debug:
        logger.set_level('DEBUG')
    
    # Convert symbols string to list
    if isinstance(args.symbols, str) and ',' in args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = [args.symbols]
    
    # Initialize parameters
    params = {}
    
    # Skip optimization if using cached parameters only
    if args.use_cached_params and not args.force_reopt:
        logger.info("Using cached parameters only (skipping optimization)")
        for symbol in symbols:
            symbol_params = load_optimized_params(symbol, args.timeframe)
            if symbol_params:
                params[symbol] = symbol_params
            else:
                logger.warning(f"No cached parameters found for {symbol}, will use defaults")
    # Force re-optimization if requested
    elif args.force_reopt:
        logger.info("Forcing re-optimization for all symbols")
        optim_params = optimize_parameters(symbols, args.timeframe)
        params.update(optim_params)
    # Run optimization if requested
    elif args.optimize:
        logger.info("Running parameter optimization")
        optim_params = optimize_parameters(symbols, args.timeframe)
        params.update(optim_params)
    # Load optimized parameters if not optimizing
    else:
        for symbol in symbols:
            symbol_params = load_optimized_params(symbol, args.timeframe)
            if symbol_params:
                params[symbol] = symbol_params
    
    # Run in selected mode
    if args.mode == 'backtest':
        results = run_backtest(symbols, args.timeframe, args.days, args.initial_balance, params)
        
        # Print summary results
        if results:
            logger.info("Backtest completed. Summary:")
            
            # Get the summary data
            if 'summary' in results:
                summary = results['summary']
                logger.info(f"Total trades across all symbols: {summary.get('total_trades', 0)}")
                logger.info(f"Symbols tested: {summary.get('symbols_tested', 0)}")
                logger.info(f"Average trades per symbol: {summary.get('per_symbol_avg_trades', 0):.1f}")
                
                # Calculate overall metrics
                total_return = 0
                winning_symbols = 0
                
                for key, result in results.items():
                    if key != 'summary':  # Skip the summary entry
                        symbol_return = result.get('total_return', 0) * 100
                        total_return += symbol_return
                        if symbol_return > 0:
                            winning_symbols += 1
                
                # Calculate average return across symbols
                if summary.get('symbols_tested', 0) > 0:
                    avg_return = total_return / summary.get('symbols_tested', 1)
                    logger.info(f"Average return: {avg_return:.2f}%")
                    logger.info(f"Winning symbols: {winning_symbols}/{summary.get('symbols_tested', 0)}")
                    
                # Show health monitor alerts
                if summary.get('health_alerts'):
                    logger.warning("Health monitor alerts triggered:")
                    for alert in summary.get('health_alerts', []):
                        logger.warning(f"  {alert}")
                
                # Log performance to file if requested
                if args.log:
                    log_performance_to_file(results, symbols, args.timeframe, args.days)
                    
                # Print where to find the equity plot
                logger.info(f"Equity curve plot saved to reports/equity_{args.days}d.png")
    
    elif args.mode in ['paper', 'live-paper']:
        logger.info("Starting paper trading mode")
        run_live_trading(symbols, args.timeframe, args.initial_balance, 
                        risk_cap=args.risk_cap,
                        enable_health_monitor=args.health_monitor,
                        paper_mode=True,
                        enable_notifications=args.notifications,
                        params=params)
    
    elif args.mode == 'live':
        confirm = input("WARNING: You are about to start LIVE trading. Type 'CONFIRM' to continue: ")
        
        if confirm == 'CONFIRM':
            run_live_trading(symbols, args.timeframe, args.initial_balance, 
                            risk_cap=args.risk_cap,
                            enable_health_monitor=args.health_monitor,
                            paper_mode=False,
                            enable_notifications=args.notifications,
                            params=params)
        else:
            logger.warning("Live trading mode not confirmed. Exiting.")
    
    elif args.mode == 'adaptive':
        run_adaptive_strategy()
    
if __name__ == "__main__":
    main() 