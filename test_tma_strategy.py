#!/usr/bin/env python3
"""
Test script for the TMA Overlay strategy
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath('.'))

try:
    print("Importing TMAOverlayStrategy...")
    from winning_strategies.tma_overlay_btc_strategy.tma_overlay_strategy import TMAOverlayStrategy
    print("Import successful")
    
    print("Creating strategy instance...")
    strategy = TMAOverlayStrategy()
    print("Strategy created successfully")
    print(f"Strategy configuration: {strategy.config}")
    
    print("Running example test...")
    try:
        from src.backtest.data_loader import DataLoader
        print("Data loader imported")
        
        # Create the strategy
        strategy = TMAOverlayStrategy()
        print("Strategy instance created")
        
        # Load sample data - using recent past dates instead of future dates
        loader = DataLoader()
        print("Loading data...")
        data = loader.load_data("BTC/USDT", "1h", start_date="2023-01-01", end_date="2023-02-01")
        print(f"Data loaded, shape: {data.shape if hasattr(data, 'shape') else 'unknown'}")
        
        # Add indicators to the data and check for signals
        print("Adding indicators to data...")
        data_with_indicators = strategy.populate_indicators(data)
        print("Generating signals...")
        data_with_signals = strategy.generate_signals(data_with_indicators)
        
        # Debug: Count the number of buy and sell signals
        buy_signals = data_with_signals['buy_signal'].sum()
        sell_signals = data_with_signals['sell_signal'].sum()
        print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
        # Show some example signals if they exist
        if buy_signals > 0 or sell_signals > 0:
            signal_rows = data_with_signals[(data_with_signals['buy_signal'] > 0) | (data_with_signals['sell_signal'] > 0)]
            print("\nExample Signals:")
            for idx, row in signal_rows.head(3).iterrows():
                signal_type = "BUY" if row['buy_signal'] > 0 else "SELL"
                print(f"{idx}: {signal_type} at price {row['close']:.2f}")
                
                # Show indicator values for this signal
                if 'ema_fast' in row and 'ema_slow' in row:
                    print(f"  EMA Fast: {row['ema_fast']:.2f}, EMA Slow: {row['ema_slow']:.2f}, EMA Signal: {row.get('ema_signal', 'N/A')}")
                if 'tma_mid' in row:
                    print(f"  TMA Mid: {row['tma_mid']:.2f}, Upper: {row.get('tma_upper', 'N/A'):.2f}, Lower: {row.get('tma_lower', 'N/A'):.2f}")
                if 'engulf_upper' in row or 'engulf_lower' in row:
                    print(f"  Engulf Upper: {row.get('engulf_upper', False)}, Engulf Lower: {row.get('engulf_lower', False)}")
                print("")
        
        # Run a simplified manual backtest
        print("Running simplified manual backtest...")
        
        # Initialize tracking variables for manual backtest
        balance = 10000.0  # Starting balance
        position = 0.0     # Current position size (0 = no position)
        entry_price = 0.0  # Entry price for current position
        position_type = None  # 'long' or 'short'
        trades = []        # List to track trades
        
        # Process each candle in sequence
        for i in range(1, len(data_with_signals)):
            current_row = data_with_signals.iloc[i]
            current_price = current_row['close']
            
            # Check for buy/sell signals
            if current_row['buy_signal'] > 0 and position == 0:
                # Buy signal and no position
                entry_price = current_price
                position_type = 'long'
                
                # Calculate position size (2% risk)
                stop_loss = strategy.calculate_stop_loss(data_with_signals, i, position_type, entry_price)
                risk_per_trade = 0.02  # 2% risk
                risk_amount = balance * risk_per_trade
                position = risk_amount / abs(entry_price - stop_loss)
                
                print(f"BUY at {entry_price:.2f}, Position: {position:.4f}, Stop Loss: {stop_loss:.2f}")
                
                trades.append({
                    'entry_time': data_with_signals.index[i],
                    'entry_price': entry_price,
                    'position_type': position_type,
                    'position_size': position,
                    'stop_loss': stop_loss
                })
                
            elif current_row['sell_signal'] > 0 and position == 0:
                # Sell signal and no position
                entry_price = current_price
                position_type = 'short'
                
                # Calculate position size (2% risk)
                stop_loss = strategy.calculate_stop_loss(data_with_signals, i, position_type, entry_price)
                risk_per_trade = 0.02  # 2% risk
                risk_amount = balance * risk_per_trade
                position = risk_amount / abs(entry_price - stop_loss)
                
                print(f"SELL at {entry_price:.2f}, Position: {position:.4f}, Stop Loss: {stop_loss:.2f}")
                
                trades.append({
                    'entry_time': data_with_signals.index[i],
                    'entry_price': entry_price,
                    'position_type': position_type,
                    'position_size': position,
                    'stop_loss': stop_loss
                })
                
            # Check for exit signals if we have a position
            elif position > 0:
                # Check for exit signal or stop loss hit
                exit_price = None
                exit_reason = None
                
                if (position_type == 'long' and current_row.get('exit_long', 0) > 0) or \
                   (position_type == 'short' and current_row.get('exit_short', 0) > 0):
                    exit_price = current_price
                    exit_reason = 'Exit signal'
                    
                # Check for take profit (simple implementation)
                take_profit = strategy.calculate_take_profit(data_with_signals, i, position_type, entry_price)
                if position_type == 'long' and current_price >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'Take profit'
                elif position_type == 'short' and current_price <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'Take profit'
                    
                # Check for stop loss hit
                stop_loss = trades[-1]['stop_loss']
                if position_type == 'long' and current_price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop loss'
                elif position_type == 'short' and current_price >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'Stop loss'
                    
                # Execute the exit if needed
                if exit_price is not None:
                    # Calculate profit/loss
                    if position_type == 'long':
                        pnl = (exit_price - entry_price) * position
                    else:  # short
                        pnl = (entry_price - exit_price) * position
                        
                    # Update balance
                    balance += pnl
                    
                    # Add exit details to trade
                    trades[-1].update({
                        'exit_time': data_with_signals.index[i],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'pnl_pct': (pnl / balance) * 100
                    })
                    
                    print(f"EXIT at {exit_price:.2f}, Reason: {exit_reason}, PnL: ${pnl:.2f} ({(pnl / balance) * 100:.2f}%)")
                    
                    # Reset position
                    position = 0
                    entry_price = 0
                    position_type = None
        
        # Calculate overall performance
        total_trades = len([t for t in trades if 'exit_price' in t])
        winning_trades = len([t for t in trades if 'pnl' in t and t['pnl'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum([t.get('pnl', 0) for t in trades])
        roi = (total_profit / 10000) * 100  # ROI percentage
        
        # Print results
        print("\n--- MANUAL BACKTEST RESULTS ---")
        print(f"Strategy: {strategy.name}")
        print(f"Initial Balance: $10,000.00")
        print(f"Final Balance: ${balance:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Trades: {total_trades}")
        
        # Compare with estimated performance
        estimated_metrics = strategy.get_performance_metrics()
        print("\nEstimated Performance:")
        print(f"ROI: {estimated_metrics['estimated_roi']}%")
        print(f"Win Rate: {estimated_metrics['estimated_win_rate']}%")
        print(f"Profit Factor: {estimated_metrics['estimated_profit_factor']}")
        print(f"Sharpe Ratio: {estimated_metrics['estimated_sharpe_ratio']}")
        print(f"Max Drawdown: {estimated_metrics['estimated_max_drawdown']}%")
        
    except Exception as e:
        print(f"Error during example execution: {e}")
        print(traceback.format_exc())
    
except Exception as e:
    print(f"Error: {e}")
    print(traceback.format_exc()) 