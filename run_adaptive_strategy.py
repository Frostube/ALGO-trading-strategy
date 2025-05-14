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
from datetime import datetime, timedelta
from pathlib import Path

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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run adaptive trading strategy')
    parser.add_argument('--symbols', type=str, default="BTC/USDT",
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
            max_pyramid_entries=symbol_params.get('max_pyramid_entries', 2)
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
        
        # Store results
        all_results[symbol] = strategy_results
        total_trades += strategy_results.get('total_trades', 0)
        
        # Print results for this symbol
        logger.info(f"Backtest results for {symbol}:")
        logger.info(f"  Return: {strategy_results.get('total_return', 0)*100:.2f}%")
        logger.info(f"  Profit Factor: {strategy_results.get('profit_factor', 0):.2f}")
        logger.info(f"  Win Rate: {strategy_results.get('win_rate', 0)*100:.2f}%")
        logger.info(f"  Total Trades: {strategy_results.get('total_trades', 0)}")
    
    # Add summary of all symbols
    all_results['summary'] = {
        'total_trades': total_trades,
        'symbols_tested': len(all_results) - 1,  # Excluding summary
        'per_symbol_avg_trades': total_trades / max(1, len(all_results) - 1)
    }
    
    return all_results

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
            f.write("| Date | Symbols | Timeframe | Days | Total Return | Profit Factor | Win Rate | Max DD | Trades |\n")
            f.write("|------|---------|-----------|------|--------------|---------------|----------|--------|--------|\n")
    
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
    
    # Append the new entry
    with open(log_file, 'a') as f:
        f.write(f"| {today} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {profit_factor:.2f} | {avg_win_rate:.2f}% | {max_dd:.2f}% | {total_trades} |\n")
    
    logger.info(f"Performance logged to {log_file}")

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
                    
                # Log performance to file if requested
                if args.log:
                    log_performance_to_file(results, symbols, args.timeframe, args.days)
    
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
    
if __name__ == "__main__":
    main() 