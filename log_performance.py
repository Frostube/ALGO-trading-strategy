#!/usr/bin/env python
import argparse
import sys
import logging
from datetime import datetime, timedelta

from src.config import (
    SYMBOL, TIMEFRAME, EMA_FAST, EMA_SLOW, EMA_TREND,
    RSI_PERIOD, RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD,
    RISK_PER_TRADE, VOL_TARGET_PCT
)
from src.utils.performance_logger import log_performance_results
from src.utils.prompt_or_flag import check_prompt_or_flag
from src.utils.git_commit import auto_commit_log
from src.data.data_loader import load_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('log_performance')


def parse_args():
    parser = argparse.ArgumentParser(description='Run trading strategy backtest and log results')
    
    parser.add_argument('--symbol', type=str, default=SYMBOL, 
                        help=f'Trading pair symbol (default: {SYMBOL})')
    parser.add_argument('--symbols', type=str, 
                        help='Comma-separated list of trading pair symbols (e.g. "BTCUSDT,ETHUSDT,SOLUSDT")')
    parser.add_argument('--timeframe', type=str, default=TIMEFRAME,
                        help=f'Timeframe for analysis (default: {TIMEFRAME})')
    parser.add_argument('--days', type=int, default=90,
                        help='Number of days to backtest (default: 90)')
    parser.add_argument('--strategy', type=str, default='both',
                        choices=['ema_crossover', 'rsi_momentum', 'both'],
                        help='Strategy to backtest (default: both)')
    parser.add_argument('--initial_balance', type=float, default=10000.0,
                        help='Initial account balance (default: 10000.0)')
    parser.add_argument('--log', action='store_true',
                        help='Always log performance results')
    parser.add_argument('--no-log', action='store_true',
                        help='Never log performance results')
    parser.add_argument('--aggressive', action='store_true',
                        help='Use aggressive parameter settings')
    parser.add_argument('--ema_fast', type=int, default=None,
                        help=f'Fast EMA period (default: {EMA_FAST})')
    parser.add_argument('--ema_slow', type=int, default=None,
                        help=f'Slow EMA period (default: {EMA_SLOW})')
    parser.add_argument('--ema_trend', type=int, default=None,
                        help=f'Trend EMA period (default: {EMA_TREND})')
    parser.add_argument('--rsi_period', type=int, default=None,
                        help=f'RSI period (default: {RSI_PERIOD})')
    parser.add_argument('--rsi_long', type=int, default=None,
                        help=f'RSI long threshold (default: {RSI_LONG_THRESHOLD})')
    parser.add_argument('--rsi_short', type=int, default=None,
                        help=f'RSI short threshold (default: {RSI_SHORT_THRESHOLD})')
    parser.add_argument('--risk', type=float, default=None,
                        help=f'Risk per trade (default: {RISK_PER_TRADE*100}%)')
    parser.add_argument('--vol_target', type=float, default=None,
                        help=f'Volatility target (default: {VOL_TARGET_PCT*100}%)')
    parser.add_argument('--enable-pyramiding', '--enable_pyramiding', action='store_true',
                        help='Enable pyramiding (adding to winning positions)')
    parser.add_argument('--live_feed', type=str, default='mock',
                        choices=['binance', 'mock'],
                        help='Data feed source (default: mock)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--git', action='store_true',
                        help='Auto-commit logs to git')
    
    return parser.parse_args()


def run_strategy_backtest(args):
    """
    Run the backtest with the specified strategy and parameters.
    """
    from test_enhanced_strategy import test_strategy
    
    # Create dict with args for test_enhanced_strategy
    test_args = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'days': args.days,
        'initial_balance': args.initial_balance,
        'live_feed': 'mock',
        'log': args.log,
        'aggressive': args.aggressive
    }
    
    # Add optional strategy parameters if provided
    if args.ema_fast:
        test_args['ema_fast'] = args.ema_fast
    if args.ema_slow:
        test_args['ema_slow'] = args.ema_slow
    if args.ema_trend:
        test_args['ema_trend'] = args.ema_trend
    if args.rsi_period:
        test_args['rsi_period'] = args.rsi_period
    if args.rsi_long:
        test_args['rsi_long'] = args.rsi_long
    if args.rsi_short:
        test_args['rsi_short'] = args.rsi_short
    if args.risk:
        test_args['risk'] = args.risk
    if args.vol_target:
        test_args['vol_target'] = args.vol_target
    if hasattr(args, 'enable_pyramiding') and args.enable_pyramiding:
        test_args['enable_pyramiding'] = True
    if hasattr(args, 'debug') and args.debug:
        test_args['debug'] = True
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # First, check if we're in multi-asset mode
    if args.symbols:
        symbols = args.symbols.split(',')
        logger.info(f"Running backtest for multiple assets: {symbols}")
        
        # Track combined results
        total_trades = 0
        total_profit_loss = 0
        total_win_count = 0
        total_loss_count = 0
        total_pnl_pct = 0
        best_trade_pct = 0
        worst_trade_pct = 0
        max_drawdown = 0
        
        # Run backtest for each symbol
        for symbol in symbols:
            logger.info(f"Testing {symbol}...")
            test_args['symbol'] = symbol
            
            # Run the test for this symbol
            try:
                results = test_strategy(test_args)
                if results:
                    # Accumulate results
                    total_trades += results.get('total_trades', 0)
                    total_profit_loss += results.get('total_profit_loss', 0)
                    total_win_count += results.get('win_count', 0)
                    total_loss_count += results.get('loss_count', 0)
                    total_pnl_pct += results.get('pnl_percentage', 0)
                    
                    # Track best and worst trades
                    symbol_best = results.get('best_trade_pct', 0)
                    symbol_worst = results.get('worst_trade_pct', 0)
                    
                    if symbol_best > best_trade_pct:
                        best_trade_pct = symbol_best
                    
                    if symbol_worst < worst_trade_pct:
                        worst_trade_pct = symbol_worst
                    
                    # Track max drawdown (use max of all symbols)
                    symbol_dd = results.get('max_drawdown', 0)
                    if symbol_dd > max_drawdown:
                        max_drawdown = symbol_dd
            except Exception as e:
                logger.error(f"Error testing {symbol}: {str(e)}")
        
        # Calculate combined metrics
        win_rate = (total_win_count / total_trades) * 100 if total_trades > 0 else 0
        avg_pnl_pct = total_pnl_pct / len(symbols) if symbols else 0
        
        # Create combined results dict
        combined_results = {
            'strategy': f"{args.strategy} (multi-asset)",
            'symbols': args.symbols,
            'timeframe': args.timeframe,
            'period': f"{args.days} days",
            'initial_balance': args.initial_balance,
            'total_trades': total_trades,
            'win_count': total_win_count,
            'loss_count': total_loss_count,
            'win_rate': win_rate,
            'total_profit_loss': total_profit_loss,
            'pnl_percentage': avg_pnl_pct,
            'best_trade_pct': best_trade_pct,
            'worst_trade_pct': worst_trade_pct,
            'max_drawdown': max_drawdown,
            'parameters': {
                'ema_fast': args.ema_fast,
                'ema_slow': args.ema_slow,
                'ema_trend': args.ema_trend,
                'rsi_period': args.rsi_period,
                'rsi_long': args.rsi_long,
                'rsi_short': args.rsi_short,
                'risk_per_trade': args.risk,
                'vol_target': args.vol_target,
                'enable_pyramiding': args.enable_pyramiding,
                'aggressive': args.aggressive
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Log the combined results
        should_log = check_prompt_or_flag(
            args.log, args.no_log, 
            f"Log multi-asset backtest results? ({total_trades} trades, {total_profit_loss:.2f} USD, {avg_pnl_pct:.2f}% average return)"
        )
        
        if should_log:
            log_performance_results(combined_results)
            logger.info("Performance results logged to docs/performance_log.md")
            
            if args.git:
                auto_commit_log()
        
        return combined_results
    
    else:
        # Single asset mode
        logger.info(f"Running backtest for {args.symbol}")
        results = test_strategy(test_args)
        
        if results:
            # Only log results if we have permission
            should_log = check_prompt_or_flag(
                args.log, args.no_log, 
                f"Log backtest results? ({results.get('total_trades', 0)} trades, {results.get('total_profit_loss', 0):.2f} USD, {results.get('pnl_percentage', 0):.2f}% return)"
            )
            
            if should_log:
                log_performance_results(results)
                logger.info("Performance results logged to docs/performance_log.md")
                
                if args.git:
                    auto_commit_log()
        
        return results


def main():
    """
    Main function to run the performance logging.
    """
    args = parse_args()
    
    logger.info(f"Running {args.strategy} strategy backtest over {args.days} days")
    logger.info(f"Using {'aggressive' if args.aggressive else 'standard'} parameters")
    
    # Check data availability before running the test
    if args.symbols:
        # Check data for multiple symbols
        symbols = args.symbols.split(',')
        for symbol in symbols:
            try:
                # Convert symbol to format expected by data_loader
                formatted_symbol = symbol.replace('USDT', '/USDT')
                
                # Load some data to check availability
                end_date = datetime.now()
                start_date = end_date - timedelta(days=args.days)
                
                logger.info(f"Checking data availability for {symbol} from {start_date.date()} to {end_date.date()}")
                data = load_data(formatted_symbol, args.timeframe, start_date, end_date)
                
                if data is None or len(data) < 5:
                    logger.error(f"Insufficient data for {symbol} with timeframe {args.timeframe}")
                    return
                else:
                    logger.info(f"Successfully loaded {len(data)} candles for {symbol}")
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                return
    else:
        # Check data for single symbol
        try:
            # Convert symbol if needed
            formatted_symbol = args.symbol.replace('USDT', '/USDT') if 'USDT' in args.symbol and '/' not in args.symbol else args.symbol
            
            # Load some data to check availability
            end_date = datetime.now()
            start_date = end_date - timedelta(days=args.days)
            
            logger.info(f"Checking data availability for {args.symbol} from {start_date.date()} to {end_date.date()}")
            data = load_data(formatted_symbol, args.timeframe, start_date, end_date)
            
            if data is None or len(data) < 5:
                logger.error(f"Insufficient data for {args.symbol} with timeframe {args.timeframe}")
                return
            else:
                logger.info(f"Successfully loaded {len(data)} candles for {args.symbol}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return
    
    # Run the backtest
    results = run_strategy_backtest(args)
    
    if results:
        logger.info("Backtest completed successfully!")
        
        # Print a summary of the results
        if isinstance(results.get('symbols', ''), str) and ',' in results.get('symbols', ''):
            # Multi-asset summary
            logger.info(f"Multi-asset results for {results.get('symbols')}:")
        else:
            # Single asset summary
            logger.info(f"Results for {args.symbol}:")
        
        logger.info(f"Total trades: {results.get('total_trades', 0)}")
        logger.info(f"Win rate: {results.get('win_rate', 0):.2f}%")
        logger.info(f"Total P&L: ${results.get('total_profit_loss', 0):.2f} ({results.get('pnl_percentage', 0):.2f}%)")
        logger.info(f"Max drawdown: {results.get('max_drawdown', 0):.2f}%")
        
        # Calculate Profit Factor if we have win and loss data
        win_amount = results.get('win_amount', 0)
        loss_amount = abs(results.get('loss_amount', 0))
        
        if loss_amount > 0:
            profit_factor = win_amount / loss_amount
            logger.info(f"Profit Factor: {profit_factor:.2f}")


if __name__ == "__main__":
    main() 