#!/usr/bin/env python3
"""
EMA Crossover Optimizer Module

This module contains functions to find the optimal EMA pair for a given trading symbol
and timeframe, inspired by the BEC (Bot EMA Cross) project.

It uses backtesting to evaluate various EMA combinations and returns the pair with the
highest profit factor or return.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt

class _EMACrossStrategy:
    """
    Simple EMA Crossover strategy for backtesting.
    """
    def __init__(self, fast_ema, slow_ema):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.position = None
        self.trades = []

    def run(self, df):
        """Run backtest on the provided dataframe"""
        # Calculate EMAs
        df[f'ema_{self.fast_ema}'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df[f'ema_{self.slow_ema}'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        
        # Initialize results
        equity = 1000.0  # Starting equity
        equity_curve = [equity]
        trade_results = []
        
        # Run through each candle
        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            
            # Check for crossover (entry signal)
            fast_ema_current = row[f'ema_{self.fast_ema}']
            fast_ema_prev = prev_row[f'ema_{self.fast_ema}']
            slow_ema_current = row[f'ema_{self.slow_ema}']
            slow_ema_prev = prev_row[f'ema_{self.slow_ema}']
            
            # Long entry: Fast EMA crosses above Slow EMA
            if fast_ema_prev <= slow_ema_prev and fast_ema_current > slow_ema_current and not self.position:
                self.position = {
                    'entry_price': row['close'],
                    'entry_time': row.name,
                    'direction': 'long',
                }
            
            # Exit: Fast EMA crosses below Slow EMA
            elif fast_ema_prev >= slow_ema_prev and fast_ema_current < slow_ema_current and self.position and self.position['direction'] == 'long':
                # Calculate trade results
                exit_price = row['close']
                entry_price = self.position['entry_price']
                pnl_pct = (exit_price - entry_price) / entry_price
                
                # Update equity
                equity *= (1 + pnl_pct)
                
                # Record trade
                trade = {
                    'entry_time': self.position['entry_time'],
                    'entry_price': entry_price,
                    'exit_time': row.name,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'direction': 'long'
                }
                trade_results.append(trade)
                
                # Reset position
                self.position = None
            
            # Update equity curve
            equity_curve.append(equity)
        
        # Close any open position at the end
        if self.position:
            last_row = df.iloc[-1]
            exit_price = last_row['close']
            entry_price = self.position['entry_price']
            pnl_pct = (exit_price - entry_price) / entry_price if self.position['direction'] == 'long' else 0
            
            # Update equity
            equity *= (1 + pnl_pct)
            
            # Record trade
            trade = {
                'entry_time': self.position['entry_time'],
                'entry_price': entry_price,
                'exit_time': last_row.name,
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'direction': 'long'
            }
            trade_results.append(trade)
            
            # Reset position
            self.position = None
            
            # Update equity curve
            equity_curve[-1] = equity
        
        # Calculate performance metrics
        total_return = (equity - 1000) / 1000
        win_trades = sum(1 for t in trade_results if t['pnl_pct'] > 0)
        loss_trades = sum(1 for t in trade_results if t['pnl_pct'] <= 0)
        win_rate = win_trades / len(trade_results) if trade_results else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl_pct'] for t in trade_results if t['pnl_pct'] > 0)
        gross_loss = abs(sum(t['pnl_pct'] for t in trade_results if t['pnl_pct'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'equity_curve': equity_curve,
            'trades': trade_results,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trade_results)
        }


def fetch_historical_data(symbol, timeframe, days=365, exchange=None):
    """
    Fetch historical OHLCV data for the given symbol and timeframe.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Candle timeframe (e.g., '1h', '1d')
        days (int): Number of days of historical data to fetch
        exchange (ccxt.Exchange, optional): CCXT exchange instance
    
    Returns:
        pd.DataFrame: DataFrame with OHLCV data
    """
    # Create exchange instance if not provided
    if exchange is None:
        exchange = ccxt.binance({
            'enableRateLimit': True,
        })
    
    # Calculate start time
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    
    # Fetch OHLCV data
    ohlcv = []
    while True:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if len(data) == 0:
                break
            
            ohlcv.extend(data)
            since = data[-1][0] + 1
            
            if since >= exchange.milliseconds():
                break
                
        except Exception as e:
            print(f"Error fetching data: {e}")
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    return df


def find_best_ema_pair(symbol='BTC/USDT', timeframe='1h', history_days=365, 
                      fast_range=range(5, 51, 5), slow_range=range(20, 201, 20)):
    """
    Find the best EMA pair for the given symbol and timeframe.
    
    Args:
        symbol (str): Trading pair symbol (e.g., 'BTC/USDT')
        timeframe (str): Candle timeframe (e.g., '1h', '1d')
        history_days (int): Number of days of historical data to use
        fast_range (range): Range of fast EMA periods to test
        slow_range (range): Range of slow EMA periods to test
    
    Returns:
        tuple: (best_fast_ema, best_slow_ema, backtest_stats)
    """
    # Fetch historical data
    df = fetch_historical_data(symbol, timeframe, days=history_days)
    
    if df.empty:
        print(f"No data available for {symbol} {timeframe}")
        return (9, 21, None)  # Default values
    
    # Test all combinations
    best_pf = -float('inf')
    best_pair = (None, None)
    best_stats = None
    
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue  # Skip invalid combinations
            
            # Run backtest
            strategy = _EMACrossStrategy(fast, slow)
            results = strategy.run(df.copy())
            
            # Check if this is the best so far
            # We prioritize profit factor for robustness
            if results['profit_factor'] > best_pf and results['total_trades'] >= 5:
                best_pf = results['profit_factor']
                best_pair = (fast, slow)
                best_stats = results
    
    if best_pair == (None, None):
        # If no valid combination was found, return default values
        return (9, 21, None)
    
    return (*best_pair, best_stats)


if __name__ == "__main__":
    # Example usage
    fast, slow, stats = find_best_ema_pair(symbol='BTC/USDT', timeframe='1h', history_days=365)
    print(f"Best EMA pair: fast={fast}, slow={slow}")
    if stats:
        print(f"Total Return: {stats['total_return']*100:.2f}%")
        print(f"Profit Factor: {stats['profit_factor']:.2f}")
        print(f"Win Rate: {stats['win_rate']*100:.2f}%")
        print(f"Total Trades: {stats['total_trades']}") 