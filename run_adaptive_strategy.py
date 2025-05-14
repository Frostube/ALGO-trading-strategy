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
from src.backtest.backtest import Backtester
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
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'backtest'], default='backtest',
                       help='Trading mode (default: backtest)')
    parser.add_argument('--days', type=int, default=90,
                       help='Days of historical data for backtest (default: 90)')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Initial account balance (default: 10000)')
    parser.add_argument('--optimize', action='store_true',
                       help='Run parameter optimization before trading')
    parser.add_argument('--health-monitor', action='store_true',
                       help='Enable health monitoring')
    parser.add_argument('--risk-cap', type=float, default=0.015,
                       help='Maximum portfolio risk cap (default: 0.015 = 1.5%%)')
    parser.add_argument('--notifications', action='store_true',
                       help='Enable notifications')
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
        with open(params_file, 'r') as f:
            params_list = json.load(f)
            
        if not params_list:
            logger.warning(f"Empty parameters file for {symbol} on {timeframe}")
            return None
            
        # Use the first parameter set (highest ranked)
        params = params_list[0]
        
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
    
    # Load data for the primary symbol (first in list)
    primary_symbol = symbols[0]
    df = load_data(symbol=primary_symbol, timeframe=timeframe, days=days)
    
    if df.empty:
        logger.error(f"Failed to load data for {primary_symbol}")
        return {}
    
    # Load data for all symbols
    all_symbol_data = {primary_symbol: df}
    
    for symbol in symbols[1:]:
        symbol_df = load_data(symbol=symbol, timeframe=timeframe, days=days)
        
        if symbol_df.empty:
            logger.warning(f"Failed to load data for {symbol}, excluding from backtest")
            continue
            
        all_symbol_data[symbol] = symbol_df
    
    # Use provided parameters or load optimized parameters
    if params is None:
        params = {}
        for symbol in symbols:
            symbol_params = load_optimized_params(symbol, timeframe)
            
            if symbol_params:
                params[symbol] = symbol_params
    
    # Create default parameters if none are available
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
    
    # Initialize backtester
    backtester = Backtester(
        data=df,
        initial_balance=initial_balance
    )
    
    # Add all symbol data
    backtester.all_symbol_data = all_symbol_data
    
    # Configure backtester params
    backtester.params = {
        'symbols': symbols,
        'params_by_symbol': params,
        'default_params': default_params,
        'enable_health_monitor': True,
        'enable_portfolio_risk_manager': True,
        'max_portfolio_risk': 0.015
    }
    
    # Run backtest
    results = backtester.run(symbols=symbols, strategies=['ema_crossover'])
    
    # Display results
    if 'portfolio' in results:
        portfolio = results['portfolio']
        
        logger.info("\n============================================================")
        logger.info("PORTFOLIO PERFORMANCE RESULTS:")
        logger.info(f"Total Return: {portfolio.get('total_return', 0) * 100:.2f}%")
        logger.info(f"Annual Return: {portfolio.get('annual_return', 0) * 100:.2f}%")
        logger.info(f"Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {portfolio.get('max_drawdown', 0) * 100:.2f}%")
        logger.info(f"Win Rate: {portfolio.get('win_rate', 0) * 100:.2f}%")
        logger.info(f"Profit Factor: {portfolio.get('profit_factor', 0):.2f}")
        logger.info(f"Total Trades: {portfolio.get('total_trades', 0)}")
    
    return results

def run_live_trading(symbols, timeframe, initial_balance, risk_cap=0.015, enable_health_monitor=True):
    """
    Run live trading for the given symbols.
    
    Args:
        symbols: List of trading pair symbols
        timeframe: Timeframe to use
        initial_balance: Initial account balance
        risk_cap: Maximum portfolio risk cap
        enable_health_monitor: Whether to enable health monitoring
    """
    logger.info(f"Starting live trading for {len(symbols)} symbols on {timeframe} timeframe "
               f"with ${initial_balance} initial balance")
    
    # Initialize portfolio risk manager
    risk_manager = PortfolioRiskManager(
        account_equity=initial_balance,
        max_portfolio_risk=risk_cap,
        risk_per_trade=0.0075,
        max_position_pct=0.20
    )
    
    # Initialize health monitors for each symbol
    health_monitors = {}
    if enable_health_monitor:
        for symbol in symbols:
            health_monitors[symbol] = HealthMonitor(
                strategy_name="ema_crossover",
                symbol=symbol,
                window_size=40,
                min_profit_factor=1.0,
                min_win_rate=0.35
            )
    
    # Initialize strategies for each symbol
    strategies = {}
    for symbol in symbols:
        # Load optimized parameters
        params = load_optimized_params(symbol, timeframe)
        
        # Use default parameters if none are available
        if params is None:
            params = {
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
        
        # Initialize strategy
        strategies[symbol] = EMACrossoverStrategy(
            config=params,
            symbol=symbol,
            timeframe=timeframe,
            account_balance=initial_balance
        )
        
        logger.info(f"Initialized strategy for {symbol} with "
                   f"EMA {params.get('ema_fast', 10)}/{params.get('ema_slow', 40)}/{params.get('ema_trend', 200)}")
    
    # Placeholder for live trading loop
    # In a real implementation, this would connect to an exchange API
    # and process real-time data
    
    logger.info("Live trading not fully implemented in this example")
    logger.info("To implement live trading, connect to exchange API and process real-time data")

def main():
    """Main function."""
    args = parse_args()
    
    # Parse symbols list
    symbols = args.symbols.split(',')
    
    if args.debug:
        logger.setLevel('DEBUG')
    
    # Create params directory if it doesn't exist
    Path("params").mkdir(parents=True, exist_ok=True)
    
    # Run parameter optimization if requested
    if args.optimize:
        optimize_parameters(symbols, args.timeframe)
    
    # Run in appropriate mode
    if args.mode == 'backtest':
        run_backtest(symbols, args.timeframe, args.days, args.initial_balance)
    elif args.mode in ['live', 'paper']:
        run_live_trading(
            symbols, 
            args.timeframe, 
            args.initial_balance,
            risk_cap=args.risk_cap,
            enable_health_monitor=args.health_monitor
        )

if __name__ == "__main__":
    main() 