#!/usr/bin/env python3
"""
EMA Crossover Strategy Backtester

This module runs backtests specifically for the EMA Crossover strategy,
with optional parameter optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import ccxt

from src.utils.logger import logger
from src.strategy.ema_optimizer import fetch_historical_data, find_best_ema_pair
from src.strategy.ema_crossover import EMACrossoverStrategy

def run_ema_backtest(symbol='BTC/USDT', timeframe='1h', days=30, initial_balance=10000.0, 
                    plot=False, optimize=True):
    """
    Run a backtest for the EMA Crossover strategy.
    
    Args:
        symbol (str): Trading symbol
        timeframe (str): Timeframe for the backtest
        days (int): Number of days of historical data to use
        initial_balance (float): Initial account balance
        plot (bool): Whether to generate equity curve plot
        optimize (bool): Whether to find optimal EMA parameters
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"Running EMA Crossover backtest for {symbol} on {timeframe} timeframe")
    
    # Fetch historical data
    logger.info(f"Fetching {days} days of historical data...")
    df = fetch_historical_data(symbol, timeframe, days=days)
    
    if df.empty:
        logger.error(f"No data available for {symbol} {timeframe}")
        return {
            'total_return': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0,
            'strategy_params': {'fast_ema': None, 'slow_ema': None}
        }
    
    logger.info(f"Fetched {len(df)} bars of data from {df.index[0]} to {df.index[-1]}")
    
    # Find optimal EMA parameters if requested
    fast_ema, slow_ema = 9, 21  # Default values
    if optimize:
        logger.info("Finding optimal EMA parameters...")
        fast_ema, slow_ema, _ = find_best_ema_pair(
            symbol=symbol,
            timeframe=timeframe,
            history_days=max(days, 365)  # Use at least a year of data for optimization
        )
        logger.info(f"Optimal EMA parameters: fast={fast_ema}, slow={slow_ema}")
    
    # Create strategy instance
    strategy = EMACrossoverStrategy(
        symbol=symbol,
        timeframe=timeframe,
        account_balance=initial_balance,
        auto_optimize=False  # We already optimized if needed
    )
    
    # Set the EMA parameters
    strategy.fast_ema = fast_ema
    strategy.slow_ema = slow_ema
    
    # Apply indicators to data
    df_with_indicators = strategy.apply_indicators(df)
    
    # Run backtest
    logger.info("Running backtest simulation...")
    equity_curve, trades = _simulate_trades(strategy, df_with_indicators, initial_balance)
    
    # Calculate performance metrics
    metrics = _calculate_performance_metrics(equity_curve, trades, initial_balance)
    
    # Add strategy parameters to results
    metrics['strategy_params'] = {
        'fast_ema': fast_ema,
        'slow_ema': slow_ema
    }
    
    # Generate plot if requested
    if plot:
        _generate_equity_plot(df_with_indicators, equity_curve, trades, metrics, symbol, timeframe)
    
    return metrics

def _simulate_trades(strategy, df, initial_balance):
    """
    Simulate trades on historical data.
    
    Args:
        strategy: Strategy instance
        df: DataFrame with indicators
        initial_balance: Initial account balance
        
    Returns:
        tuple: (equity_curve, trades list)
    """
    equity = initial_balance
    equity_curve = [equity]
    trades = []
    active_trade = None
    
    # Reset strategy state
    strategy.account_balance = initial_balance
    strategy.active_trade = None
    
    # Iterate through data
    for i in range(1, len(df)):
        current_data = df.iloc[:i+1]
        signal = strategy.get_signal(current_data)
        
        # Handle trade entry
        if not active_trade and signal['signal'] in ['buy', 'sell']:
            row = df.iloc[i]
            current_price = row['close']
            
            # Calculate stop loss and take profit
            side = signal['signal']
            atr = row.get('atr', current_price * 0.01)  # Default to 1% if ATR not available
            
            if side == 'buy':
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 3.0)
            else:  # sell
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 3.0)
            
            # Calculate position size (risk 1% of equity)
            risk_amount = equity * 0.01
            risk_per_unit = abs(current_price - stop_loss)
            position_size = risk_amount / risk_per_unit
            
            # Open trade
            active_trade = {
                'entry_time': row.name,
                'entry_price': current_price,
                'side': side,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'amount': position_size,
                'rsi': row.get('rsi', None),
                'fast_ema': strategy.fast_ema,
                'slow_ema': strategy.slow_ema
            }
        
        # Handle trade exit
        elif active_trade:
            row = df.iloc[i]
            current_price = row['close']
            
            # Check if stop loss or take profit has been hit
            exit_reason = None
            
            if active_trade['side'] == 'buy':
                if current_price <= active_trade['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= active_trade['take_profit']:
                    exit_reason = 'take_profit'
                # Also exit on bearish crossover
                elif row['ema_trend'] < 0 and df.iloc[i-1]['ema_trend'] > 0:
                    exit_reason = 'crossover'
            else:  # sell
                if current_price >= active_trade['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= active_trade['take_profit']:
                    exit_reason = 'take_profit'
                # Also exit on bullish crossover
                elif row['ema_trend'] > 0 and df.iloc[i-1]['ema_trend'] < 0:
                    exit_reason = 'crossover'
            
            if exit_reason:
                # Calculate PnL
                if active_trade['side'] == 'buy':
                    pnl = (current_price - active_trade['entry_price']) * active_trade['amount']
                    pnl_pct = (current_price - active_trade['entry_price']) / active_trade['entry_price']
                else:  # sell
                    pnl = (active_trade['entry_price'] - current_price) * active_trade['amount']
                    pnl_pct = (active_trade['entry_price'] - current_price) / active_trade['entry_price']
                
                # Update equity
                equity += pnl
                
                # Record completed trade
                completed_trade = {
                    **active_trade,
                    'exit_time': row.name,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                }
                trades.append(completed_trade)
                
                # Reset active trade
                active_trade = None
        
        # Record equity at each step
        equity_curve.append(equity)
    
    # Close any active trade at the end of the simulation
    if active_trade:
        last_row = df.iloc[-1]
        current_price = last_row['close']
        
        # Calculate PnL
        if active_trade['side'] == 'buy':
            pnl = (current_price - active_trade['entry_price']) * active_trade['amount']
            pnl_pct = (current_price - active_trade['entry_price']) / active_trade['entry_price']
        else:  # sell
            pnl = (active_trade['entry_price'] - current_price) * active_trade['amount']
            pnl_pct = (active_trade['entry_price'] - current_price) / active_trade['entry_price']
        
        # Update equity
        equity += pnl
        equity_curve[-1] = equity
        
        # Record completed trade
        completed_trade = {
            **active_trade,
            'exit_time': last_row.name,
            'exit_price': current_price,
            'exit_reason': 'end_of_data',
            'pnl': pnl,
            'pnl_pct': pnl_pct
        }
        trades.append(completed_trade)
    
    return equity_curve, trades

def _calculate_performance_metrics(equity_curve, trades, initial_balance):
    """
    Calculate performance metrics from backtest results.
    
    Args:
        equity_curve: List of equity values
        trades: List of completed trades
        initial_balance: Initial account balance
        
    Returns:
        dict: Performance metrics
    """
    if not equity_curve or len(equity_curve) < 2:
        return {
            'total_return': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'total_trades': 0
        }
    
    # Calculate returns
    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_balance) / initial_balance
    
    # Calculate daily returns for Sharpe ratio
    daily_returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    
    # Calculate max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    # Calculate trade metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Calculate profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Calculate average trade metrics
    avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / winning_trades if winning_trades > 0 else 0
    avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / (total_trades - winning_trades) if (total_trades - winning_trades) > 0 else 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'equity_curve': equity_curve
    }

def _generate_equity_plot(df, equity_curve, trades, metrics, symbol, timeframe):
    """
    Generate equity curve and trade plots.
    
    Args:
        df: DataFrame with price data
        equity_curve: List of equity values
        trades: List of completed trades
        metrics: Performance metrics
        symbol: Trading symbol
        timeframe: Trading timeframe
    """
    try:
        plt.figure(figsize=(12, 10))
        
        # Plot equity curve
        plt.subplot(3, 1, 1)
        plt.plot(equity_curve)
        plt.title(f'Equity Curve - {symbol} {timeframe}')
        plt.grid(True)
        
        # Plot price with entry/exit points
        plt.subplot(3, 1, 2)
        plt.plot(df.index, df['close'])
        
        # Plot EMAs
        fast_ema = f"ema_{metrics['strategy_params']['fast_ema']}"
        slow_ema = f"ema_{metrics['strategy_params']['slow_ema']}"
        
        if fast_ema in df.columns:
            plt.plot(df.index, df[fast_ema], 'g--', label=f'Fast EMA ({metrics["strategy_params"]["fast_ema"]})')
        if slow_ema in df.columns:
            plt.plot(df.index, df[slow_ema], 'r--', label=f'Slow EMA ({metrics["strategy_params"]["slow_ema"]})')
        
        # Mark trade entries and exits
        for trade in trades:
            if trade['side'] == 'buy':
                plt.plot(trade['entry_time'], trade['entry_price'], '^', color='g', markersize=8)
                plt.plot(trade['exit_time'], trade['exit_price'], 'v', 
                        color='r' if trade['exit_reason'] == 'stop_loss' else 'g', markersize=8)
            else:
                plt.plot(trade['entry_time'], trade['entry_price'], 'v', color='r', markersize=8)
                plt.plot(trade['exit_time'], trade['exit_price'], '^', 
                        color='r' if trade['exit_reason'] == 'stop_loss' else 'g', markersize=8)
        
        plt.title(f'Price Chart with Trades - {symbol} {timeframe}')
        plt.grid(True)
        plt.legend()
        
        # Plot trade P&L
        plt.subplot(3, 1, 3)
        pnls = [t['pnl'] for t in trades]
        plt.bar(range(len(pnls)), pnls, color=['g' if p > 0 else 'r' for p in pnls])
        plt.axhline(y=0, color='black', linestyle='-')
        plt.title('Trade P&L')
        plt.grid(True)
        
        # Add metrics text
        plt.figtext(0.01, 0.01, 
                  f"Total Return: {metrics['total_return']*100:.2f}%\n"
                  f"Win Rate: {metrics['win_rate']*100:.2f}%\n"
                  f"Profit Factor: {metrics['profit_factor']:.2f}\n"
                  f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
                  f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%\n"
                  f"Total Trades: {metrics['total_trades']}\n"
                  f"Fast EMA: {metrics['strategy_params']['fast_ema']}\n"
                  f"Slow EMA: {metrics['strategy_params']['slow_ema']}",
                  fontsize=10, verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(f"ema_backtest_{symbol.replace('/', '_')}_{timeframe}.png")
        plt.close()
        
        logger.info(f"Saved backtest plot to ema_backtest_{symbol.replace('/', '_')}_{timeframe}.png")
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")

if __name__ == "__main__":
    # Example direct usage
    results = run_ema_backtest(
        symbol='BTC/USDT',
        timeframe='1h',
        days=60,
        plot=True,
        optimize=True
    )
    
    print("\nBacktest Results:")
    print(f"Total Return: {results['total_return']*100:.2f}%")
    print(f"Win Rate: {results['win_rate']*100:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Fast EMA: {results['strategy_params']['fast_ema']}")
    print(f"Slow EMA: {results['strategy_params']['slow_ema']}") 