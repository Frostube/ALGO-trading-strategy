#!/usr/bin/env python3
"""
Quick test script for the BTC/USDT scalping strategy logic using sample data.
This helps verify that the core strategy logic works before deploying to TradingView.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Sample data generation
def generate_sample_data(periods=500, volatility=0.002):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducibility
    
    # Start with a price of 50,000
    close_prices = [50000]
    
    # Generate random price movements
    for _ in range(periods-1):
        price_change = np.random.normal(0, volatility) * close_prices[-1]
        close_prices.append(close_prices[-1] + price_change)
    
    # Generate timestamps (1-minute intervals)
    start_time = datetime.now() - timedelta(minutes=periods)
    timestamps = [start_time + timedelta(minutes=i) for i in range(periods)]
    
    # Generate open, high, low based on close
    data = {
        'timestamp': timestamps,
        'open': [close_prices[i-1] if i > 0 else close_prices[0] for i in range(periods)],
        'high': [close + abs(np.random.normal(0, volatility)) * close for close in close_prices],
        'low': [close - abs(np.random.normal(0, volatility)) * close for close in close_prices],
        'close': close_prices,
        'volume': [abs(np.random.normal(100, 30)) for _ in range(periods)]
    }
    
    # Simulate volume spikes
    for i in range(periods):
        if np.random.random() < 0.05:  # 5% chance of volume spike
            data['volume'][i] *= 3
    
    return pd.DataFrame(data).set_index('timestamp')

# Calculate indicators
def calculate_indicators(df, ema_fast=12, ema_slow=26, ema_trend=200, rsi_period=5, volume_period=20, volume_threshold=1.5, atr_period=14):
    """Calculate all technical indicators."""
    # Calculate EMAs
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=ema_trend, adjust=False).mean()
    
    # Calculate trend signals
    df['ema_crossover'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    df['market_trend'] = np.where(df['close'] > df['ema_trend'], 1, -1)
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate ATR (Average True Range)
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    
    # Calculate volume indicators
    df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_spike'] = df['volume_ratio'] > volume_threshold
    
    # Clean up temporary columns
    df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
    
    return df

# Generate signals
def generate_signals(df, rsi_oversold=30, rsi_overbought=70):
    """Generate trading signals."""
    # Long signal: Market trend up, EMA crossover up, RSI oversold, volume spike
    df['long_signal'] = (
        (df['market_trend'] > 0) & 
        (df['ema_crossover'] > 0) & 
        (df['rsi'] < rsi_oversold) & 
        (df['volume_spike'])
    )
    
    # Short signal: Market trend down, EMA crossover down, RSI overbought, volume spike
    df['short_signal'] = (
        (df['market_trend'] < 0) & 
        (df['ema_crossover'] < 0) & 
        (df['rsi'] > rsi_overbought) & 
        (df['volume_spike'])
    )
    
    return df

# Calculate stop loss and take profit levels
def calculate_stops(row, use_atr=True, atr_sl_multiplier=1.5, atr_tp_multiplier=3.0, sl_pct=0.0015, tp_pct=0.0030):
    """Calculate stop loss and take profit levels based on ATR or fixed percentage."""
    if use_atr and not pd.isna(row['atr']):
        # ATR-based stops
        long_sl = row['close'] - (row['atr'] * atr_sl_multiplier)
        long_tp = row['close'] + (row['atr'] * atr_tp_multiplier)
        short_sl = row['close'] + (row['atr'] * atr_sl_multiplier)
        short_tp = row['close'] - (row['atr'] * atr_tp_multiplier)
    else:
        # Fixed percentage stops
        long_sl = row['close'] * (1 - sl_pct)
        long_tp = row['close'] * (1 + tp_pct)
        short_sl = row['close'] * (1 + sl_pct)
        short_tp = row['close'] * (1 - tp_pct)
    
    return long_sl, long_tp, short_sl, short_tp

# Backtest strategy
def backtest_strategy(df, use_atr_stops=True, initial_balance=10000):
    """Simple backtest to validate the strategy logic."""
    # Initialize variables
    balance = initial_balance
    in_position = False
    position_type = None
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trades = []
    
    # Fixed position size as percentage of balance (1%)
    risk_per_trade = 0.01
    
    # Loop through data points (skip the first 200 bars for indicator warmup)
    for i in range(200, len(df)):
        current_row = df.iloc[i]
        current_close = current_row['close']
        
        # If not in a position, check for entry signals
        if not in_position:
            # Long signal
            if current_row['long_signal']:
                # Calculate stop loss and take profit
                long_sl, long_tp, _, _ = calculate_stops(current_row, use_atr=use_atr_stops)
                
                # Calculate position size based on risk
                risk_amount = balance * risk_per_trade
                risk_per_unit = abs(current_close - long_sl)
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                
                if position_size > 0:
                    # Enter long position
                    position_type = 'long'
                    entry_price = current_close
                    stop_loss = long_sl
                    take_profit = long_tp
                    in_position = True
                    
                    print(f"LONG Entry at {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
                    print(f"Market Trend: {'Up' if current_row['market_trend'] > 0 else 'Down'}, RSI: {current_row['rsi']:.2f}, ATR: {current_row['atr']:.2f}")
            
            # Short signal
            elif current_row['short_signal']:
                # Calculate stop loss and take profit
                _, _, short_sl, short_tp = calculate_stops(current_row, use_atr=use_atr_stops)
                
                # Calculate position size based on risk
                risk_amount = balance * risk_per_trade
                risk_per_unit = abs(current_close - short_sl)
                position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                
                if position_size > 0:
                    # Enter short position
                    position_type = 'short'
                    entry_price = current_close
                    stop_loss = short_sl
                    take_profit = short_tp
                    in_position = True
                    
                    print(f"SHORT Entry at {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
                    print(f"Market Trend: {'Up' if current_row['market_trend'] > 0 else 'Down'}, RSI: {current_row['rsi']:.2f}, ATR: {current_row['atr']:.2f}")
        
        # If in a position, check for exit conditions
        elif in_position:
            # Long position exit conditions
            if position_type == 'long':
                # Stop loss hit
                if current_close <= stop_loss:
                    pnl = (current_close - entry_price) * position_size
                    balance += pnl
                    trades.append({
                        'type': position_type,
                        'entry': entry_price,
                        'exit': current_close,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'market_trend': current_row['market_trend'],
                        'rsi': current_row['rsi'],
                        'atr': current_row['atr']
                    })
                    
                    print(f"LONG Exit at {current_close:.2f} (Stop Loss), PnL: ${pnl:.2f}, Balance: ${balance:.2f}")
                    in_position = False
                
                # Take profit hit
                elif current_close >= take_profit:
                    pnl = (current_close - entry_price) * position_size
                    balance += pnl
                    trades.append({
                        'type': position_type,
                        'entry': entry_price,
                        'exit': current_close,
                        'pnl': pnl,
                        'exit_reason': 'take_profit',
                        'market_trend': current_row['market_trend'],
                        'rsi': current_row['rsi'],
                        'atr': current_row['atr']
                    })
                    
                    print(f"LONG Exit at {current_close:.2f} (Take Profit), PnL: ${pnl:.2f}, Balance: ${balance:.2f}")
                    in_position = False
            
            # Short position exit conditions
            elif position_type == 'short':
                # Stop loss hit
                if current_close >= stop_loss:
                    pnl = (entry_price - current_close) * position_size
                    balance += pnl
                    trades.append({
                        'type': position_type,
                        'entry': entry_price,
                        'exit': current_close,
                        'pnl': pnl,
                        'exit_reason': 'stop_loss',
                        'market_trend': current_row['market_trend'],
                        'rsi': current_row['rsi'],
                        'atr': current_row['atr']
                    })
                    
                    print(f"SHORT Exit at {current_close:.2f} (Stop Loss), PnL: ${pnl:.2f}, Balance: ${balance:.2f}")
                    in_position = False
                
                # Take profit hit
                elif current_close <= take_profit:
                    pnl = (entry_price - current_close) * position_size
                    balance += pnl
                    trades.append({
                        'type': position_type,
                        'entry': entry_price,
                        'exit': current_close,
                        'pnl': pnl,
                        'exit_reason': 'take_profit',
                        'market_trend': current_row['market_trend'],
                        'rsi': current_row['rsi'],
                        'atr': current_row['atr']
                    })
                    
                    print(f"SHORT Exit at {current_close:.2f} (Take Profit), PnL: ${pnl:.2f}, Balance: ${balance:.2f}")
                    in_position = False
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    
    # Calculate profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = sum(abs(t['pnl']) for t in trades if t['pnl'] < 0)
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate drawdown
    balance_curve = [initial_balance]
    for trade in trades:
        balance_curve.append(balance_curve[-1] + trade['pnl'])
    
    max_balance = initial_balance
    max_drawdown = 0
    for balance_point in balance_curve:
        max_balance = max(max_balance, balance_point)
        drawdown = (max_balance - balance_point) / max_balance * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    performance = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'trades': trades
    }
    
    return performance

def main():
    """Main function to run the quick test."""
    print("Generating sample data...")
    data = generate_sample_data(periods=1000)
    
    print("Calculating indicators...")
    data = calculate_indicators(data)
    
    print("Generating signals...")
    data = generate_signals(data)
    
    print("\nRunning backtest with ATR-based stops...")
    performance = backtest_strategy(data, use_atr_stops=True)
    
    # Display performance summary
    print("\nPerformance Summary:")
    print(f"Initial Balance: ${performance['initial_balance']:.2f}")
    print(f"Final Balance: ${performance['final_balance']:.2f}")
    print(f"Total Return: {performance['total_return']:.2f}%")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Win Rate: {performance['win_rate'] * 100:.2f}%")
    print(f"Profit Factor: {performance['profit_factor']:.2f}")
    print(f"Maximum Drawdown: {performance['max_drawdown']:.2f}%")
    
    # Plot the price chart with signals
    plt.figure(figsize=(12, 10))
    
    # Price and EMAs
    plt.subplot(4, 1, 1)
    plt.plot(data.index, data['close'], label='Price', alpha=0.7)
    plt.plot(data.index, data['ema_fast'], label=f'EMA(12)', color='blue')
    plt.plot(data.index, data['ema_slow'], label=f'EMA(26)', color='red')
    plt.plot(data.index, data['ema_trend'], label=f'EMA(200)', color='yellow')
    
    # Plot buy/sell signals
    buy_signals = data[data['long_signal']]
    sell_signals = data[data['short_signal']]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    # Color background based on market trend
    for i in range(1, len(data)):
        if data['market_trend'].iloc[i] > 0:
            plt.axvspan(data.index[i-1], data.index[i], facecolor='green', alpha=0.1)
        else:
            plt.axvspan(data.index[i-1], data.index[i], facecolor='red', alpha=0.1)
    
    plt.title('BTC/USDT Scalping Strategy Test')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # RSI subplot
    plt.subplot(4, 1, 2)
    plt.plot(data.index, data['rsi'], color='purple')
    plt.axhline(y=30, color='green', linestyle='--', label='Oversold')
    plt.axhline(y=70, color='red', linestyle='--', label='Overbought')
    plt.title('RSI(5)')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # ATR subplot
    plt.subplot(4, 1, 3)
    plt.plot(data.index, data['atr'], color='orange', label='ATR(14)')
    plt.title('ATR')
    plt.ylabel('ATR')
    plt.legend()
    plt.grid(True)
    
    # Volume subplot
    plt.subplot(4, 1, 4)
    plt.bar(data.index, data['volume'], color='blue', alpha=0.5)
    plt.plot(data.index, data['volume_ma'], color='orange', label='Volume MA')
    volume_spikes = data[data['volume_spike']]
    plt.scatter(volume_spikes.index, volume_spikes['volume'], color='red', s=30, label='Volume Spike')
    plt.title('Volume')
    plt.ylabel('Volume')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('strategy_test_results.png')
    plt.show()
    
    print(f"\nChart saved as 'strategy_test_results.png'")
    
    # Save trade list to CSV
    trades_df = pd.DataFrame(performance['trades'])
    if not trades_df.empty:
        trades_df.to_csv('strategy_test_trades.csv', index=False)
        print(f"Trade list saved as 'strategy_test_trades.csv'")

if __name__ == "__main__":
    main() 