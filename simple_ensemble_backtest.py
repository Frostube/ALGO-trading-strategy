#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import argparse
import matplotlib.pyplot as plt

from src.data.fetcher import fetch_ohlcv
from src.backtest.backtest import Backtester, MockAccount
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.strategy.rsi_strategy import RSIOscillatorStrategy
from src.utils.logger import logger

def run_simple_ensemble_backtest():
    """Run backtest with an ensemble of strategies - simplified version"""
    parser = argparse.ArgumentParser(description='Run ensemble strategy backtest')
    parser.add_argument('--symbols', type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT", help='Comma-separated list of symbols')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=120, help='Number of days to backtest')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial account balance')
    args = parser.parse_args()
    
    # Define symbols and other parameters
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    days = args.days
    initial_balance = args.initial_balance
    
    # Risk allocation between strategies (50/50 split)
    risk_allocation = {
        'ema_crossover': 0.5,   # 50% of risk budget
        'rsi_oscillator': 0.5   # 50% of risk budget
    }
    
    # Store results for the final summary
    symbol_results = {}
    total_trades_detail = []
    
    # Run backtest for each symbol
    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=days)
            if df is None or len(df) == 0:
                logger.error(f"No data found for {symbol}")
                continue
            
            logger.info(f"Loaded {len(df)} candles for {symbol}")
            
            # Create EMA Crossover strategy with optimized parameters
            ema_strategy = EMACrossoverStrategy(
                symbol=symbol,
                timeframe=timeframe,
                fast_ema=5,
                slow_ema=12,
                enable_pyramiding=False,
                risk_per_trade=0.0075 * risk_allocation['ema_crossover'],
                use_volatility_sizing=True,
                vol_target_pct=0.0075
            )
            # Set additional parameters
            ema_strategy.min_bars_between_trades = 1
            
            # Create RSI Oscillator strategy with optimized parameters
            rsi_strategy = RSIOscillatorStrategy(
                symbol=symbol,
                timeframe=timeframe,
                rsi_period=14,
                oversold=35,
                overbought=65,
                enable_pyramiding=False,
                risk_per_trade=0.0075 * risk_allocation['rsi_oscillator'],
                use_volatility_sizing=True,
                vol_target_pct=0.0075
            )
            # Set additional parameters
            rsi_strategy.min_bars_between_trades = 1
            
            # Store strategies in a dictionary
            strategies = {
                'ema_crossover': ema_strategy,
                'rsi_oscillator': rsi_strategy
            }
            
            # Run backtest for each strategy
            strategy_results = {}
            all_trades = []
            
            for strategy_name, strategy in strategies.items():
                logger.info(f"Running backtest for {symbol} with {strategy_name} strategy...")
                
                try:
                    # Process data with strategy indicators
                    if hasattr(strategy, 'generate_signals'):
                        logger.info(f"Calling generate_signals for {strategy_name}")
                        processed_df = strategy.generate_signals(df.copy())
                    else:
                        # For strategies without generate_signals method
                        logger.info(f"Calling apply_indicators for {strategy_name}")
                        processed_df = strategy.apply_indicators(df.copy())
                    
                    # Create backtester with this data and strategy
                    backtester = Backtester(processed_df, initial_balance)
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
                except Exception as e:
                    import traceback
                    logger.error(f"Error in backtest for {symbol} with {strategy_name}: {str(e)}")
                    logger.error(traceback.format_exc())
                
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
            logger.error(f"Error in backtest for {symbol}: {str(e)}")
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
        logger.info("\n===== ENSEMBLE BACKTEST SUMMARY =====")
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
            
            win_rate = len(winning_trades) / len(total_trades_detail) if total_trades_detail else 0
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            avg_duration = np.mean([t['duration_hours'] for t in total_trades_detail]) if total_trades_detail else 0
            
            logger.info(f"  Total Trades: {len(total_trades_detail)}")
            logger.info(f"  Win Rate: {win_rate*100:.2f}%")
            logger.info(f"  Average Win: ${avg_win:.2f}")
            logger.info(f"  Average Loss: ${avg_loss:.2f}")
            logger.info(f"  Average Duration: {avg_duration:.1f} hours")
            
            # Strategy comparison
            logger.info("\nStrategy Comparison:")
            for strategy_name in ['ema_crossover', 'rsi_oscillator']:
                strategy_trades = [t for t in total_trades_detail if t['strategy'] == strategy_name]
                if strategy_trades:
                    strategy_wins = [t for t in strategy_trades if t['pnl'] > 0]
                    strat_win_rate = len(strategy_wins) / len(strategy_trades) if strategy_trades else 0
                    strat_avg_return = np.mean([t['pnl_pct'] for t in strategy_trades]) if strategy_trades else 0
                    logger.info(f"  {strategy_name}: {len(strategy_trades)} trades, {strat_win_rate*100:.2f}% win rate, {strat_avg_return:.2f}% avg return per trade")
        
        # Log to performance file
        with open('docs/performance_log.md', 'a') as f:
            now = datetime.now().strftime('%Y-%m-%d')
            symbols_str = '+'.join(symbols)
            f.write(f"| {now} | {symbols_str} | {timeframe} | {days} | {avg_return:.2f}% | {avg_pf:.2f} | {avg_wr*100:.2f}% | {max_dd:.2f}% | {total_trades} | Ensemble: EMA(5/12) + RSI(35/65), MinBars=1 |\n")
            
        logger.info(f"\nFinal results logged to docs/performance_log.md")
        
        # Print results for each symbol and strategy
        logger.info("\nDetailed Results by Symbol:")
        for symbol, results in symbol_results.items():
            logger.info(f"{symbol}: Return={results['return']:.2f}%, PF={results['profit_factor']:.2f}, Win={results['win_rate']*100:.2f}%, Trades={results['trades']}, DD={results['max_drawdown']:.2f}%")
            for strategy_name, strategy_result in results['strategies'].items():
                logger.info(f"  • {strategy_name}: Return={strategy_result['return']:.2f}%, PF={strategy_result['profit_factor']:.2f}, Win={strategy_result['win_rate']*100:.2f}%, Trades={strategy_result['trades']}")

if __name__ == "__main__":
    run_simple_ensemble_backtest() 