#!/usr/bin/env python3
"""
Adaptive Strategy Runner

This script runs the trading strategy with adaptive parameters based on market conditions.
The strategy parameters are adjusted based on volatility regimes.
"""
import argparse
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Local imports
from src.utils.logger import logger, setup_logger
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.risk.portfolio_manager import PortfolioRiskManager
from src.strategy.health_monitor import HealthMonitor
from src.utils.metrics import calculate_metrics, plot_equity_curve, plot_drawdown_curve
from src.backtest.backtest import Backtester
from src.data.fetcher import fetch_ohlcv
from src.utils.notification import send_notification
from src.strategy.strategy_factory import StrategyFactory

# Default parameters
DEFAULT_SYMBOLS = ["BTC/USDT"]
DEFAULT_TIMEFRAME = "4h"
DEFAULT_DAYS = 90
DEFAULT_RISK = 0.015  # 1.5% risk per trade
DEFAULT_INITIAL_BALANCE = 10000
DEFAULT_TARGET_VOL_USD = 200  # $200 daily volatility target

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run adaptive trading strategy')
    parser.add_argument('--mode', choices=['backtest', 'live', 'paper'], default='backtest',
                        help='Trading mode: backtest, live, or paper trading')
    parser.add_argument('--symbols', type=str, default=','.join(DEFAULT_SYMBOLS),
                        help='Comma-separated list of trading symbols')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME,
                        help='Trading timeframe')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS,
                        help='Number of days to backtest')
    parser.add_argument('--risk-cap', type=float, default=DEFAULT_RISK,
                        help='Risk cap per trade')
    parser.add_argument('--initial-balance', type=float, default=DEFAULT_INITIAL_BALANCE,
                        help='Initial account balance')
    parser.add_argument('--params-file', type=str, default=None,
                        help='JSON file containing strategy parameters')
    parser.add_argument('--target-vol-usd', type=float, default=DEFAULT_TARGET_VOL_USD,
                        help='Target daily volatility in USD for position sizing')
    parser.add_argument('--optimize', action='store_true',
                        help='Run parameter optimization before trading')
    parser.add_argument('--health-monitor', action='store_true',
                        help='Enable health monitoring')
    parser.add_argument('--notifications', action='store_true',
                        help='Enable notifications')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def optimize_parameters(symbol, timeframe="4h", days=90):
    """
    Run parameter optimization for a single symbol.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Timeframe to use
        days: Days of historical data to use
        
    Returns:
        dict: Optimized parameters for the symbol
    """
    from src.optimization.optimizer import GridOptimizer
    
    logger.info(f"Optimizing parameters for {symbol} on {timeframe} timeframe...")
    
    # Fetch historical data
    df = fetch_ohlcv(symbol, timeframe, days=days)
    
    if df is None or len(df) < 30:
        logger.error(f"Insufficient data for {symbol}")
        return None
    
    # Define parameter grid
    param_grid = {
        'ema_short': [5, 8, 13, 21],
        'ema_long': [21, 34, 55, 89],
        'atr_periods': [14, 21],
        'atr_multiplier': [1.5, 2.0, 2.5, 3.0],
        'rsi_periods': [14],
        'rsi_overbought': [70],
        'rsi_oversold': [30]
    }
    
    # Initialize optimizer
    optimizer = GridOptimizer(param_grid)
    
    # Run optimization
    optimized_params = optimizer.optimize(df, symbol, timeframe)
    
    logger.info(f"Optimized parameters for {symbol}: {optimized_params}")
    
    return optimized_params

def run_backtest(symbols, timeframe, days, initial_balance, risk_cap=0.015, params=None, 
                 target_vol_usd=DEFAULT_TARGET_VOL_USD, debug=False):
    """
    Run backtest for the given symbols.
    
    Args:
        symbols: List of trading pair symbols
        timeframe: Timeframe to use
        days: Number of days to backtest
        initial_balance: Initial account balance
        risk_cap: Maximum risk per trade
        params: Dictionary of parameters by symbol (optional)
        target_vol_usd: Target daily volatility in USD for position sizing
        debug: Whether to enable debug logging
        
    Returns:
        dict: Backtest results
    """
    # Setup logger
    setup_logger(debug=debug)
    
    logger.info(f"Running backtest for {len(symbols)} symbols on {timeframe} timeframe "
               f"over {days} days with ${initial_balance} initial balance "
               f"and ${target_vol_usd} volatility target")
    
    # Create output directory for results
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir(parents=True)
    
    # Create portfolio risk manager
    portfolio_manager = PortfolioRiskManager(
        account_equity=initial_balance,
        risk_per_trade=risk_cap,
        max_pos_pct=0.25,
        max_portfolio_risk=0.05,
        use_volatility_sizing=True,
        target_vol_usd=target_vol_usd
    )
    
    # Initialize backtester
    backtester = Backtester(portfolio_manager=portfolio_manager, initial_balance=initial_balance)
    
    # Use provided parameters or load optimized parameters
    if params is None:
        params = {}
        for symbol in symbols:
            # Try to load cached parameters
            param_file = Path(f"params/{symbol.replace('/', '_')}.json")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    params[symbol] = json.load(f)
                logger.info(f"Loaded cached parameters for {symbol}")
            else:
                # Use default parameters
                params[symbol] = {
                    'ema_short': 8,
                    'ema_long': 21,
                    'atr_periods': 14,
                    'atr_multiplier': 2.0,
                    'rsi_periods': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                }
                logger.info(f"Using default parameters for {symbol}")
    
    # Create health monitor
    health_monitor = HealthMonitor()
    
    # Run backtest for each symbol
    all_trades = []
    equity_curves = {}
    
    for symbol in symbols:
        logger.info(f"Backtesting {symbol}...")
        
        # Fetch historical data
        df = fetch_ohlcv(symbol, timeframe, days=days)
        
        if df is None or len(df) < 30:
            logger.error(f"Insufficient data for {symbol}")
            continue
        
        # Get parameters for this symbol
        symbol_params = params.get(symbol, {})
        
        # Create strategy instance
        strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            ema_short=symbol_params.get('ema_short', 8),
            ema_long=symbol_params.get('ema_long', 21),
            atr_periods=symbol_params.get('atr_periods', 14),
            atr_multiplier=symbol_params.get('atr_multiplier', 2.0),
            rsi_periods=symbol_params.get('rsi_periods', 14),
            rsi_overbought=symbol_params.get('rsi_overbought', 70),
            rsi_oversold=symbol_params.get('rsi_oversold', 30)
        )
        
        # Add indicators to data
        df_with_indicators = strategy.add_indicators(df)
        
        # Run backtest
        symbol_results = backtester.backtest_strategy(strategy, df_with_indicators)
        
        if symbol_results and 'trades' in symbol_results:
            # Process trades through health monitor
            for trade in symbol_results['trades']:
                health_monitor.add_trade(trade)
            
            # Add to overall results
            all_trades.extend(symbol_results['trades'])
            equity_curves[symbol] = symbol_results['equity_curve']
    
    # Calculate overall metrics
    if all_trades:
        metrics = calculate_metrics(all_trades, initial_balance)
        
        # Plot results
        plot_equity_curve(equity_curves, f"results/equity_curve_{datetime.now().strftime('%Y%m%d')}.png")
        
        # Return metrics
        return metrics
    else:
        logger.warning("No trades generated in backtest")
        return {
            'net_profit': 0,
            'net_profit_pct': 0,
            'max_drawdown': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'profit_factor': 0,
            'avg_r_multiple': 0
        }

def run_live_trading(symbols, timeframe, initial_balance, risk_cap=0.015, 
                  enable_health_monitor=True, paper_mode=True, enable_notifications=False,
                  target_vol_usd=DEFAULT_TARGET_VOL_USD, params=None):
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
        target_vol_usd: Target daily volatility in USD for position sizing
        params: Dictionary of parameters by symbol (optional)
        
    Returns:
        None
    """
    from src.live.trader import LiveTrader
    from src.exchange.exchange import Exchange
    
    # Setup logger
    setup_logger(debug=True)
    
    # Create exchange connection
    exchange = Exchange(paper_mode=paper_mode)
    
    # Create portfolio risk manager
    portfolio_manager = PortfolioRiskManager(
        account_equity=initial_balance,
        risk_per_trade=risk_cap,
        max_pos_pct=0.25,
        max_portfolio_risk=0.05,
        max_correlated_risk=risk_cap * 1.5,
        use_volatility_sizing=True,
        target_vol_usd=target_vol_usd
    )
    
    # Create health monitor if enabled
    health_monitor = HealthMonitor() if enable_health_monitor else None
    
    # Use provided parameters or load optimized parameters
    if params is None:
        params = {}
        for symbol in symbols:
            # Try to load cached parameters
            param_file = Path(f"params/{symbol.replace('/', '_')}.json")
            if param_file.exists():
                with open(param_file, 'r') as f:
                    params[symbol] = json.load(f)
                logger.info(f"Loaded cached parameters for {symbol}")
            else:
                # Use default parameters
                params[symbol] = {
                    'ema_short': 8,
                    'ema_long': 21,
                    'atr_periods': 14,
                    'atr_multiplier': 2.0,
                    'rsi_periods': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30
                }
                logger.info(f"Using default parameters for {symbol}")
    
    # Create strategy instances
    strategies = {}
    for symbol in symbols:
        symbol_params = params.get(symbol, {})
        
        # Create strategy
        strategy = EMACrossoverStrategy(
            symbol=symbol,
            timeframe=timeframe,
            ema_short=symbol_params.get('ema_short', 8),
            ema_long=symbol_params.get('ema_long', 21),
            atr_periods=symbol_params.get('atr_periods', 14),
            atr_multiplier=symbol_params.get('atr_multiplier', 2.0),
            rsi_periods=symbol_params.get('rsi_periods', 14),
            rsi_overbought=symbol_params.get('rsi_overbought', 70),
            rsi_oversold=symbol_params.get('rsi_oversold', 30)
        )
        
        strategies[symbol] = strategy
    
    # Create and run live trader
    trader = LiveTrader(
        exchange=exchange,
        portfolio_manager=portfolio_manager,
        strategies=strategies,
        timeframe=timeframe,
        health_monitor=health_monitor,
        enable_notifications=enable_notifications
    )
    
    # Run trading loop
    trader.run_trading_loop()

def run_adaptive_strategy():
    """Run the adaptive trading strategy"""
    args = parse_args()
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Load params from file if provided
    params = None
    if args.params_file:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    
    # Log configuration
    logger.info("Running adaptive trading strategy with configuration:")
    logger.info(f"- Mode: {args.mode}")
    logger.info(f"- Symbols: {args.symbols}")
    logger.info(f"- Timeframe: {args.timeframe}")
    logger.info(f"- Initial balance: ${args.initial_balance}")
    logger.info(f"- Risk cap: {args.risk_cap*100:.2f}%")
    logger.info(f"- Target volatility: ${args.target_vol_usd:.2f}")
    logger.info(f"- Health monitor: {'Enabled' if args.health_monitor else 'Disabled'}")
    
    # Run in selected mode
    if args.mode == 'backtest':
        results = run_backtest(symbols, args.timeframe, args.days, args.initial_balance, 
                              args.risk_cap, params, args.target_vol_usd, args.debug)
        
        # Print summary
        print("\nBacktest Results Summary:")
        print(f"Net Profit: ${results['net_profit']:.2f} ({results['net_profit_pct']:.2f}%)")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {results['sortino_ratio']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Average R-Multiple: {results.get('avg_r_multiple', 0):.2f}")
    
    elif args.mode == 'paper':
        run_live_trading(symbols, args.timeframe, args.initial_balance, args.risk_cap,
                      args.health_monitor, True, args.notifications, 
                      args.target_vol_usd, params)
    
    elif args.mode == 'live':
        # Confirm before live trading
        confirm = input("Are you sure you want to run live trading? This will use real funds. (y/n): ")
        if confirm.lower() == 'y':
            run_live_trading(symbols, args.timeframe, args.initial_balance, args.risk_cap,
                          args.health_monitor, False, args.notifications,
                          args.target_vol_usd, params)
        else:
            logger.warning("Live trading mode not confirmed. Exiting.")

if __name__ == "__main__":
    run_adaptive_strategy() 