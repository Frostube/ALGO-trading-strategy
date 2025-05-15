#!/usr/bin/env python
import pandas as pd
import numpy as np
import logging
from src.data.fetcher import DataFetcher, fetch_ohlcv
from src.strategy.strategy_factory import StrategyFactory
from src.backtest.backtest import Backtester
try:
    from src.market.regime_detector import MarketRegimeDetector
    HAS_REGIME_DETECTOR = True
except ImportError:
    HAS_REGIME_DETECTOR = False
    print("Market regime detector not available, disabling regime detection")
from src.strategy.base_strategy import BaseStrategy
from src.utils.logger import logger
from src.utils.performance_logger import append_result, log_performance_results
from src.utils.prompt_or_flag import should_log, is_result_sane
import matplotlib.pyplot as plt
import os
import sys
import argparse
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description='Test enhanced trading strategy')
    parser.add_argument('--live-feed', choices=['binance', 'mock'], default='mock',
                      help='Data source to use (binance or mock)')
    parser.add_argument('--days', type=int, default=90, help='Number of days to backtest')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Symbol to test')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe to use')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    parser.add_argument('--log', action='store_true', help='Automatically log performance results')
    parser.add_argument('--ema-fast', type=int, help='Fast EMA period')
    parser.add_argument('--ema-slow', type=int, help='Slow EMA period')
    parser.add_argument('--ema-trend', type=int, help='Trend EMA period')
    parser.add_argument('--rsi-period', type=int, help='RSI period')
    parser.add_argument('--rsi-long', type=int, help='RSI oversold threshold')
    parser.add_argument('--rsi-short', type=int, help='RSI overbought threshold')
    parser.add_argument('--risk', type=float, help='Risk percentage per trade')
    parser.add_argument('--vol-target', type=float, help='Volatility target percentage')
    parser.add_argument('--enable-pyramiding', action='store_true', help='Enable pyramiding')
    parser.add_argument('--aggressive', action='store_true', help='Use aggressive trading settings')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    return parser.parse_args()

def test_strategy(config=None, args=None):
    """
    Run a backtest of the enhanced trading strategy
    
    Args:
        config: Dictionary of strategy and backtest configuration
        args: Command line arguments or dict with parameters
        
    Returns:
        Dictionary of backtest results
    """
    # If args is None, parse from command line
    if args is None:
        args = parse_args()
    
    # If args is a dict, convert to object for easier access
    if isinstance(args, dict):
        # Extract values from dict with defaults
        symbol = args.get('symbol', 'BTC/USDT')
        timeframe = args.get('timeframe', '4h')
        days = args.get('days', 90)
        initial_balance = args.get('initial_balance', 10000)
        live_feed = args.get('live_feed', 'mock')
        enable_debug = args.get('debug', False)
        enable_log = args.get('log', False)
        enable_aggressive = args.get('aggressive', False)
        enable_pyramiding = args.get('enable_pyramiding', False)
        
        # Optional strategy parameters
        ema_fast = args.get('ema_fast', None)
        ema_slow = args.get('ema_slow', None)
        ema_trend = args.get('ema_trend', None)
        rsi_period = args.get('rsi_period', None)
        rsi_long = args.get('rsi_long', None)
        rsi_short = args.get('rsi_short', None)
        risk = args.get('risk', None)
        vol_target = args.get('vol_target', None)
    else:
        # Extract values from args object
        symbol = args.symbol
        timeframe = args.timeframe
        days = args.days
        initial_balance = args.initial_balance
        live_feed = args.live_feed if hasattr(args, 'live_feed') else 'mock'
        enable_debug = args.debug if hasattr(args, 'debug') else False
        enable_log = args.log if hasattr(args, 'log') else False
        enable_aggressive = args.aggressive if hasattr(args, 'aggressive') else False
        enable_pyramiding = args.enable_pyramiding if hasattr(args, 'enable_pyramiding') else False
        
        # Optional strategy parameters
        ema_fast = args.ema_fast if hasattr(args, 'ema_fast') else None
        ema_slow = args.ema_slow if hasattr(args, 'ema_slow') else None
        ema_trend = args.ema_trend if hasattr(args, 'ema_trend') else None
        rsi_period = args.rsi_period if hasattr(args, 'rsi_period') else None
        rsi_long = args.rsi_long if hasattr(args, 'rsi_long') else None
        rsi_short = args.rsi_short if hasattr(args, 'rsi_short') else None
        risk = args.risk if hasattr(args, 'risk') else None
        vol_target = args.vol_target if hasattr(args, 'vol_target') else None
    
    # Enable debug logging if requested
    if enable_debug:
        logger.setLevel(logging.DEBUG)
        
    # 4-hour timeframe optimized config defaults
    default_config = {
        # Strategy parameters - 4h timeframe settings
        'ema_fast': 10,  # Changed from 9 to 10
        'ema_slow': 40,  # Changed from 21 to 40
        'ema_trend': 200,  # Changed from 50 to 200 for long-term trend filter
        'rsi_period': 14,  # Changed from 7 to standard 14
        'rsi_long_threshold': 30,  # Changed from 35 to standard 30
        'rsi_short_threshold': 70,  # Changed from 65 to standard 70
        'atr_period': 14,
        'atr_sl_multiplier': 1.0,  # Changed from 1.5 to 1.0 (1× ATR)
        'atr_tp_multiplier': None,  # Changed from 3.0 to None (use trailing stops instead)
        
        # Portfolio parameters - more conservative for 4h timeframe
        'vol_target_pct': 0.0075,  # Targeting 0.75% volatility (reduced from 0.15)
        'max_allocation': 0.20,  # Reduced from 0.35 to 0.20
        'vol_lookback': 20,     # Days to look back for volatility calculation
        
        # Position sizing parameters
        'use_dynamic_sizing': True,
        'use_volatility_sizing': True,  # New volatility-targeted position sizing
        'risk_per_trade': 0.0075,  # Reduced from 0.03 to 0.0075 (0.75%)
        'max_position_pct': 0.20,  # Reduced from 0.30 to 0.20 (20%)
        
        # Pyramiding parameters
        'enable_pyramiding': True,
        'max_pyramid_entries': 2,  # Reduced from 3 to 2
        'pyramid_threshold': 0.5,  # 0.5 × ATR move before pyramiding
        'pyramid_position_scale': 0.5,  # 50% of initial position size
        
        # Market regime parameters
        'enable_regime_detection': HAS_REGIME_DETECTOR,  # Only enable if available
        'regime_lookback': 50,   # Increased from 30 to 50 days for 4h timeframe
        'ranging_strategies': ['rsi_momentum'],
        'trending_strategies': ['ema_crossover'],
        'normal_strategies': ['ema_crossover', 'rsi_momentum'],
        
        # Trail stop parameters
        'use_trailing_stop': True,
        'trail_activation_pct': 0.005,  # 0.5% move to activate trailing stop
        'trail_atr_multiplier': 1.0,    # 1× ATR for trailing stops
    }
    
    # Configuration parameters - override with args if provided
    strategy_config = config.copy() if config is not None else default_config.copy()
    
    # Override with any command line arguments if provided
    if ema_fast:
        strategy_config['ema_fast'] = ema_fast
    if ema_slow:
        strategy_config['ema_slow'] = ema_slow
    if ema_trend:
        strategy_config['ema_trend'] = ema_trend
    if rsi_period:
        strategy_config['rsi_period'] = rsi_period
    if rsi_long:
        strategy_config['rsi_long_threshold'] = rsi_long
    if rsi_short:
        strategy_config['rsi_short_threshold'] = rsi_short
    if risk:
        strategy_config['risk_per_trade'] = risk / 100  # Convert from percentage to decimal
    if vol_target:
        strategy_config['vol_target_pct'] = vol_target / 100  # Convert from percentage to decimal
    if enable_pyramiding:
        strategy_config['enable_pyramiding'] = True
    
    # Apply aggressive settings if requested
    if enable_aggressive:
        strategy_config['risk_per_trade'] = 0.01  # Risk 1% per trade (higher than default)
        strategy_config['max_position_pct'] = 0.30  # Allow 30% position size (larger than default)
        strategy_config['ema_fast'] = 8  # Faster EMA
        strategy_config['ema_slow'] = 30  # Faster slow EMA
        strategy_config['max_pyramid_entries'] = 3  # More pyramid entries
        strategy_config['pyramid_threshold'] = 0.4  # Pyramid after smaller moves (0.4 × ATR)

    # Fetch historical data
    if live_feed == 'binance':
        # Use real data from Binance
        use_mock = False
        logger.info(f"Fetching real data for {symbol} from Binance")
        df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
    else:
        # Use mock data
        use_mock = True
        logger.info(f"Using mock data")
        data_fetcher = DataFetcher(use_testnet=True, use_mock=True)
        df = data_fetcher.fetch_historical_data(symbol=symbol, 
                                              timeframe=timeframe, 
                                              days=days)
    
    if df.empty:
        logger.error("Failed to get data. Exiting.")
        return {}
    
    # Initialize strategy through factory
    strategy_factory = StrategyFactory(strategy_config)
    
    # Determine which strategy to use
    strategy_to_use = 'ema_crossover'  # Default to EMA crossover
    if hasattr(args, 'strategy') and args.strategy != 'both':
        strategy_to_use = args.strategy
    elif hasattr(args, 'strategy') and args.strategy == 'both':
        # Later we'll run both strategies
        pass
    
    strategy = strategy_factory.create_strategy(strategy_to_use)
    
    # Initialize backtester
    backtester = Backtester(data=df, initial_balance=initial_balance, params=strategy_config)
    
    # Determine the strategies to backtest
    strategies_to_run = ['ema_crossover']
    if hasattr(args, 'strategy'):
        if args.strategy == 'rsi_momentum':
            strategies_to_run = ['rsi_momentum']
        elif args.strategy == 'both':
            strategies_to_run = ['ema_crossover', 'rsi_momentum']
    
    # Run backtest
    try:
        logger.info(f"Running backtest for {symbol} with {timeframe} timeframe...")
        logger.info(f"Using strategies: {strategies_to_run}")
        results = backtester.run(strategies=strategies_to_run)
        
        # Print strategy-level results
        for key, result in results.items():
            if key != 'portfolio' and key != 'weights':
                logger.info("\n============================================================")
                logger.info(f"STRATEGY RESULTS: {key}")
                logger.info(f"Total Return: {result.get('total_return', 0) * 100:.2f}%")
                logger.info(f"Annual Return: {result.get('annual_return', 0) * 100:.2f}%")
                logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
                logger.info(f"Max Drawdown: {result.get('max_drawdown', 0) * 100:.2f}%")
                logger.info(f"Win Rate: {result.get('win_rate', 0) * 100:.2f}%")
                logger.info(f"Profit Factor: {result.get('profit_factor', 0):.2f}")
                logger.info(f"Total Trades: {result.get('total_trades', 0)}")
                
                # Print some sample trades
                trades = result.get('trades', [])
                if trades:
                    logger.info("\nSAMPLE TRADES:")
                    logger.info("Type\tEntry\t\tExit\t\tPrice In\tPrice Out\tPnL\tReturn")
                    for i, trade in enumerate(trades[:5]):  # Show first 5 trades
                        logger.info(f"{trade['type']}\t{trade['entry_time'].strftime('%Y-%m-%d')}\t{trade['exit_time'].strftime('%Y-%m-%d')}\t"
                              f"{trade['entry_price']:.2f}\t{trade['exit_price']:.2f}\t"
                              f"{trade['pnl']:.2f}\t{trade['pct_return']*100:.2f}%")
                    
                    # Also show last 5 trades
                    if len(trades) > 5:
                        logger.info("...")
                        for i, trade in enumerate(trades[-5:]):
                            logger.info(f"{trade['type']}\t{trade['entry_time'].strftime('%Y-%m-%d')}\t{trade['exit_time'].strftime('%Y-%m-%d')}\t"
                                  f"{trade['entry_price']:.2f}\t{trade['exit_price']:.2f}\t"
                                  f"{trade['pnl']:.2f}\t{trade['pct_return']*100:.2f}%")
        
        # Print portfolio-level results
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
            
            # Get core metrics for logging
            pf = portfolio.get('profit_factor', 0)
            win_rate = portfolio.get('win_rate', 0) * 100
            max_dd = portfolio.get('max_drawdown', 0) * 100
            net_return = portfolio.get('total_return', 0) * 100
            
            # Build strategy_name string
            strategies_used = strategies_to_run
            regime_str = " + Regime Detection" if strategy_config.get('enable_regime_detection', False) else ""
            pyramiding_str = " + Pyramiding" if strategy_config.get('enable_pyramiding', False) else ""
            trend_filter_str = " + Trend Filter" if strategy_config.get('ema_trend', 0) > 0 else ""
            strategy_name = f"{'/'.join(strategies_used)}{regime_str}{pyramiding_str}{trend_filter_str}"
            
            # Build dataset string
            dataset = f"{symbol} {timeframe} ({days}d)"
            
            # Build params string
            params = (f"ATR SL {strategy_config.get('atr_sl_multiplier', 1.0)}× / "
                     f"EMA {strategy_config.get('ema_fast', 10)}/{strategy_config.get('ema_slow', 40)}/{strategy_config.get('ema_trend', 200)} / "
                     f"Risk {strategy_config.get('risk_per_trade', 0.0075) * 100}%")
            
            # Prepare results for log_performance_results format
            formatted_results = {
                'strategy': strategy_name,
                'symbol': symbol,
                'timeframe': timeframe,
                'period': f"{days} days",
                'initial_balance': initial_balance,
                'symbol': args.symbol,
                'timeframe': args.timeframe,
                'period': f"{args.days} days",
                'initial_balance': args.initial_balance,
                'total_trades': portfolio.get('total_trades', 0),
                'win_count': int(portfolio.get('total_trades', 0) * portfolio.get('win_rate', 0)),
                'loss_count': int(portfolio.get('total_trades', 0) * (1 - portfolio.get('win_rate', 0))),
                'win_rate': win_rate,
                'total_profit_loss': portfolio.get('pnl', 0),
                'pnl_percentage': net_return,
                'best_trade_pct': portfolio.get('best_trade_pct', 0) * 100 if 'best_trade_pct' in portfolio else 0,
                'worst_trade_pct': portfolio.get('worst_trade_pct', 0) * 100 if 'worst_trade_pct' in portfolio else 0,
                'max_drawdown': max_dd,
                'profit_factor': pf,
                'parameters': {
                    'ema_fast': strategy_config.get('ema_fast', 10),
                    'ema_slow': strategy_config.get('ema_slow', 40),
                    'ema_trend': strategy_config.get('ema_trend', 200),
                    'rsi_period': strategy_config.get('rsi_period', 14),
                    'rsi_long': strategy_config.get('rsi_long_threshold', 30),
                    'rsi_short': strategy_config.get('rsi_short_threshold', 70),
                    'risk_per_trade': strategy_config.get('risk_per_trade', 0.0075),
                    'vol_target': strategy_config.get('vol_target_pct', 0.0075),
                    'enable_pyramiding': strategy_config.get('enable_pyramiding', True),
                    'aggressive': getattr(args, 'aggressive', False)
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Calculate win and loss amounts if available
            if 'win_amount' in portfolio and 'loss_amount' in portfolio:
                formatted_results['win_amount'] = portfolio['win_amount']
                formatted_results['loss_amount'] = portfolio['loss_amount']
            
            # Print core stats summary
            logger.info(f"\nCore Stats: PF {pf:.2f} | Win {win_rate:.1f}% | DD {max_dd:.1f}% | Net {net_return:.1f}%")
            
            # Return the formatted results
            return formatted_results
        
    except Exception as e:
        import traceback
        logger.error(f"Error running backtest: {e}")
        traceback.print_exc()
        return {}

if __name__ == "__main__":
    args = parse_args()
    test_strategy(args=args) 