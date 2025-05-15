import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def backtest(df, strategy, symbol=None):
    """
    Run a backtest on the given strategy.
    
    Args:
        df (DataFrame): DataFrame with OHLCV data
        strategy: Strategy instance with on_new_candle and backtest methods
        symbol (str): Trading symbol for fetching daily data
        
    Returns:
        dict: Dictionary with backtest results
    """
    # Use the strategy's backtest method to generate signals
    df_with_signals = strategy.backtest(df, symbol)
    
    if df_with_signals is None or df_with_signals.empty:
        logger.error("Strategy backtest returned no data")
        return {
            'total_return': 0,
            'profit_factor': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'n_trades': 0,
            'dataframe': pd.DataFrame()
        }
    
    # Make a copy to avoid modifying the original
    backtest_df = df_with_signals.copy()
    
    # Initialize columns for backtest
    backtest_df['position'] = 0
    backtest_df['equity'] = 1.0  # Starting equity (normalized)
    backtest_df['trade_active'] = False
    backtest_df['trade_return'] = 0.0
    backtest_df['entry_price'] = 0.0
    
    # Keep track of trades
    trades = []
    current_trade = None
    
    # Run through each candle to simulate trading
    for i in range(1, len(backtest_df)):
        prev_idx = backtest_df.index[i-1]
        curr_idx = backtest_df.index[i]
        
        # Get signal for current candle
        signal = backtest_df.loc[curr_idx, 'signal']
        
        # Update position based on signal
        if signal == 1:  # Buy signal
            if backtest_df.loc[prev_idx, 'position'] <= 0:  # Not in a long position
                # Close any existing short position
                if backtest_df.loc[prev_idx, 'position'] < 0:
                    # Calculate return for closing short
                    entry_price = backtest_df.loc[prev_idx, 'entry_price']
                    exit_price = backtest_df.loc[curr_idx, 'open']
                    trade_return = (entry_price - exit_price) / entry_price  # Short profit calculation
                    
                    # Record the completed trade
                    if current_trade:
                        current_trade['exit_time'] = curr_idx
                        current_trade['exit_price'] = exit_price
                        current_trade['return'] = trade_return
                        trades.append(current_trade)
                    
                # Enter new long position
                backtest_df.loc[curr_idx, 'position'] = 1
                backtest_df.loc[curr_idx, 'trade_active'] = True
                backtest_df.loc[curr_idx, 'entry_price'] = backtest_df.loc[curr_idx, 'open']
                
                # Start new trade record
                current_trade = {
                    'entry_time': curr_idx,
                    'entry_price': backtest_df.loc[curr_idx, 'open'],
                    'direction': 'long',
                }
                
        elif signal == -1:  # Sell signal
            if backtest_df.loc[prev_idx, 'position'] >= 0:  # Not in a short position
                # Close any existing long position
                if backtest_df.loc[prev_idx, 'position'] > 0:
                    # Calculate return for closing long
                    entry_price = backtest_df.loc[prev_idx, 'entry_price']
                    exit_price = backtest_df.loc[curr_idx, 'open']
                    trade_return = (exit_price - entry_price) / entry_price  # Long profit calculation
                    
                    # Record the completed trade
                    if current_trade:
                        current_trade['exit_time'] = curr_idx
                        current_trade['exit_price'] = exit_price
                        current_trade['return'] = trade_return
                        trades.append(current_trade)
                
                # Enter new short position
                backtest_df.loc[curr_idx, 'position'] = -1
                backtest_df.loc[curr_idx, 'trade_active'] = True
                backtest_df.loc[curr_idx, 'entry_price'] = backtest_df.loc[curr_idx, 'open']
                
                # Start new trade record
                current_trade = {
                    'entry_time': curr_idx,
                    'entry_price': backtest_df.loc[curr_idx, 'open'],
                    'direction': 'short',
                }
                
        else:  # No signal or exit signal
            # Carry forward the previous position
            backtest_df.loc[curr_idx, 'position'] = backtest_df.loc[prev_idx, 'position']
            backtest_df.loc[curr_idx, 'trade_active'] = backtest_df.loc[prev_idx, 'trade_active']
            backtest_df.loc[curr_idx, 'entry_price'] = backtest_df.loc[prev_idx, 'entry_price']
        
        # Calculate equity for the current position
        if backtest_df.loc[curr_idx, 'trade_active']:
            entry_price = backtest_df.loc[curr_idx, 'entry_price']
            current_price = backtest_df.loc[curr_idx, 'close']
            
            # Calculate return based on position direction
            if backtest_df.loc[curr_idx, 'position'] > 0:
                # Long position
                trade_return = (current_price - entry_price) / entry_price
            else:
                # Short position
                trade_return = (entry_price - current_price) / entry_price
                
            backtest_df.loc[curr_idx, 'trade_return'] = trade_return
        
        # Update equity
        prev_equity = backtest_df.loc[prev_idx, 'equity']
        position_return = backtest_df.loc[curr_idx, 'trade_return'] if backtest_df.loc[curr_idx, 'trade_active'] else 0
        backtest_df.loc[curr_idx, 'equity'] = prev_equity * (1 + position_return)
    
    # Close any open trade at the end
    if current_trade and 'exit_time' not in current_trade:
        last_idx = backtest_df.index[-1]
        current_trade['exit_time'] = last_idx
        current_trade['exit_price'] = backtest_df.loc[last_idx, 'close']
        
        if current_trade['direction'] == 'long':
            current_trade['return'] = (current_trade['exit_price'] - current_trade['entry_price']) / current_trade['entry_price']
        else:
            current_trade['return'] = (current_trade['entry_price'] - current_trade['exit_price']) / current_trade['entry_price']
            
        trades.append(current_trade)
    
    # Calculate backtest metrics
    # 1. Total return
    total_return = backtest_df['equity'].iloc[-1] - backtest_df['equity'].iloc[0]
    
    # 2. Max drawdown
    rolling_max = backtest_df['equity'].cummax()
    drawdown = (backtest_df['equity'] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 3. Trade statistics
    n_trades = len(trades)
    
    if n_trades > 0:
        # Calculate win rate
        winning_trades = [t for t in trades if t['return'] > 0]
        win_rate = len(winning_trades) / n_trades
        
        # Calculate profit factor
        total_gain = sum([t['return'] for t in winning_trades]) if winning_trades else 0
        losing_trades = [t for t in trades if t['return'] <= 0]
        total_loss = abs(sum([t['return'] for t in losing_trades])) if losing_trades else 0
        profit_factor = total_gain / total_loss if total_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    # Return results
    return {
        'total_return': total_return,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'trades': trades,
        'dataframe': backtest_df
    } 