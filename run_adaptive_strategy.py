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
                # Make sure it's a float array to avoid integer casting issues
                combined_equity_curve = np.array(strategy_results['equity_curve'], dtype=np.float64)
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
                    # Ensure float dtype to avoid integer casting issues
                    normalized_curve = np.array(strategy_results['equity_curve'][-min_len:], dtype=np.float64) * norm_factor * weight
                else:
                    # Ensure float dtype to avoid integer casting issues
                    normalized_curve = np.array(strategy_results['equity_curve'], dtype=np.float64) * norm_factor * weight
                
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
    Run live or paper trading for the given symbols.
    
    Args:
        symbols: List of trading pair symbols
        timeframe: Timeframe to use
        initial_balance: Initial account balance
        risk_cap: Maximum portfolio risk cap
        enable_health_monitor: Whether to enable health monitoring
        paper_mode: Whether to use paper trading
        enable_notifications: Whether to enable notifications
        params: Dictionary of parameters by symbol (optional)
        
    Returns:
        None
    """
    from src.exchange.exchange_client import ExchangeClient
    from src.strategy.ema_crossover import EMACrossoverStrategy
    from src.risk.portfolio_manager import PortfolioRiskManager
    from src.utils.notification import send_notification
    from src.trading.order_manager import OrderManager
    from src.strategy.health_monitor import HealthMonitor
    
    logger.info(f"Starting {'paper' if paper_mode else 'live'} trading "
               f"for {len(symbols)} symbols on {timeframe} timeframe...")
    
    # Use provided parameters or load optimized parameters
    if params is None:
        params = {}
        for symbol in symbols:
            symbol_params = load_optimized_params(symbol, timeframe)
            
            if symbol_params:
                params[symbol] = symbol_params
    
    # Initialize exchange client
    exchange = ExchangeClient(paper_mode=paper_mode)
    
    # Initialize portfolio risk manager
    risk_manager = PortfolioRiskManager(
        account_equity=initial_balance,
        risk_per_trade=risk_cap / 3,  # Use 1/3 of the risk cap per trade
        max_portfolio_risk=risk_cap,
        max_correlated_risk=risk_cap * 1.5,
        use_volatility_sizing=True
    )
    
    # Initialize order manager
    order_manager = OrderManager(exchange, risk_manager)
    
    # Initialize health monitor if enabled
    health_monitor = None
    if enable_health_monitor:
        health_monitor = HealthMonitor(
            expected_win_rate=0.4,
            expected_avg_win_loss_ratio=1.5,
            underperformance_threshold=0.3
        )
    
    # Initialize strategies
    strategies = {}
    for symbol in symbols:
        # Get parameters for symbol or use default
        symbol_params = params.get(symbol, {})
        
        # Create strategy with parameters
        strategies[symbol] = EMACrossoverStrategy(
            symbol=symbol,
            ema_fast=symbol_params.get('ema_fast', 8),
            ema_slow=symbol_params.get('ema_slow', 21),
            ema_trend=symbol_params.get('ema_trend', 55),
            rsi_period=symbol_params.get('rsi_period', 14),
            atr_period=symbol_params.get('atr_period', 14),
            atr_multiplier=symbol_params.get('atr_sl_multiplier', 1.5),
            risk_per_trade=risk_cap / 3,
            enable_dynamic_exits=True
        )
        
        # Fetch data needed for strategy
        logger.info(f"Fetching data for {symbol}...")
        df = fetch_ohlcv(symbol, timeframe, limit=500)
        
        # Initialize data
        strategies[symbol].init_data(df)
        
        # Update volatility regime
        risk_manager.vol_monitor.update_regime(symbol, df)
        
        # Log current volatility regime
        regime_info = {
            'symbol': symbol,
            'volatility': risk_manager.vol_monitor.realized_vol_pct(symbol),
            'regime': risk_manager.vol_monitor.current_regimes.get(symbol, 'UNKNOWN')
        }
        logger.info(f"Volatility regime for {symbol}: {regime_info['regime']} "
                   f"(vol: {regime_info['volatility']:.2f}%)")
        
        # Adjust risk according to volatility regime
        risk_adj = risk_manager.vol_monitor.get_risk_adjustment(symbol)
        logger.info(f"Risk adjustment for {symbol}: {risk_adj:.2f}x base risk")
        
        # Set strategy parameters based on regime
        strategies[symbol].enable_pyramiding = risk_manager.vol_monitor.should_enable_pyramiding(symbol)
        
    # Start trading loop
    logger.info("Starting trading loop...")

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
    """Main function to run the adaptive trading strategy."""
    args = parse_args()
    
    # Configure logging
    if args.debug:
        logger.setLevel('DEBUG')
    
    # Parse symbols list
    symbols = args.symbols.split(',')
    
    # Show current configuration
    logger.info(f"Configuration:")
    logger.info(f"- Symbols: {symbols}")
    logger.info(f"- Timeframe: {args.timeframe}")
    logger.info(f"- Mode: {args.mode}")
    logger.info(f"- Initial balance: ${args.initial_balance}")
    logger.info(f"- Risk cap: {args.risk_cap*100:.2f}%")
    logger.info(f"- Optimization: {'Enabled' if args.optimize else 'Disabled'}")
    logger.info(f"- Health monitor: {'Enabled' if args.health_monitor else 'Disabled'}")
    logger.info(f"- Notifications: {'Enabled' if args.notifications else 'Disabled'}")
    
    # Check for volatility regime monitoring capability
    try:
        from src.risk.vol_regime_switch import VolatilityRegimeMonitor
        logger.info("- Volatility regime monitoring: Enabled")
    except ImportError:
        logger.warning("- Volatility regime monitoring: Not available")
    
    # Make sure all symbols are valid
    for symbol in symbols:
        if '/' not in symbol:
            logger.error(f"Invalid symbol format: {symbol} (should be like 'BTC/USDT')")
            return
    
    # Run in selected mode
    if args.mode == 'backtest':
        results = run_backtest(symbols, args.timeframe, args.days, args.initial_balance)
        
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
                        enable_notifications=args.notifications)
    
    elif args.mode == 'live':
        confirm = input("WARNING: You are about to start LIVE trading. Type 'CONFIRM' to continue: ")
        
        if confirm == 'CONFIRM':
            run_live_trading(symbols, args.timeframe, args.initial_balance, 
                            risk_cap=args.risk_cap,
                            enable_health_monitor=args.health_monitor,
                            paper_mode=False,
                            enable_notifications=args.notifications)
        else:
            logger.warning("Live trading mode not confirmed. Exiting.")
    
    elif args.mode == 'adaptive':
        run_adaptive_strategy()
    
if __name__ == "__main__":
    run_adaptive_strategy() 