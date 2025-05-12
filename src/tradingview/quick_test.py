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
def calculate_indicators(df, ema_fast=9, ema_slow=21, rsi_period=2, volume_period=20, volume_threshold=1.5):
    """Calculate all technical indicators."""
    # Calculate EMAs
    df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
    df['ema_trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    
    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate volume indicators
    df['volume_ma'] = df['volume'].rolling(window=volume_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    df['volume_spike'] = df['volume_ratio'] > volume_threshold
    
    return df

# Generate signals
def generate_signals(df, rsi_oversold=10, rsi_overbought=90):
    """Generate trading signals."""
    # Long signal: EMA trend up, RSI oversold, volume spike
    df['long_signal'] = (df['ema_trend'] > 0) & (df['rsi'] < rsi_oversold) & (df['volume_spike'])
    
    # Short signal: EMA trend down, RSI overbought, volume spike
    df['short_signal'] = (df['ema_trend'] < 0) & (df['rsi'] > rsi_overbought) & (df['volume_spike'])
    
    return df

# Backtest strategy
def backtest_strategy(df, stop_loss_pct=0.0015, take_profit_pct=0.0030, initial_balance=10000):
    """Simple backtest to validate the strategy logic."""
    # Initialize variables
    balance = initial_balance
    in_position = False
    position_type = None
    entry_price = 0
    trades = []
    
    # Fixed position size in BTC instead of using risk-based sizing
    fixed_btc_amount = 0.01
    
    # Loop through data points (skip the first volume_period bars for indicator warmup)
    for i in range(100, len(df)):
        current_close = df['close'].iloc[i]
        
        # If not in a position, check for entry signals
        if not in_position:
            # Long signal
            if df['long_signal'].iloc[i]:
                # Enter long position
                position_type = 'long'
                entry_price = current_close
                position_size = fixed_btc_amount  # Fixed BTC amount instead of risk-based calculation
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                in_position = True
                
                print(f"LONG Entry at {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
            
            # Short signal
            elif df['short_signal'].iloc[i]:
                # Enter short position
                position_type = 'short'
                entry_price = current_close
                position_size = fixed_btc_amount  # Fixed BTC amount instead of risk-based calculation
                stop_loss = entry_price * (1 + stop_loss_pct)
                take_profit = entry_price * (1 - take_profit_pct)
                in_position = True
                
                print(f"SHORT Entry at {entry_price:.2f}, Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")
        
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
                        'exit_reason': 'stop_loss'
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
                        'exit_reason': 'take_profit'
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
                        'exit_reason': 'stop_loss'
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
                        'exit_reason': 'take_profit'
                    })
                    
                    print(f"SHORT Exit at {current_close:.2f} (Take Profit), PnL: ${pnl:.2f}, Balance: ${balance:.2f}")
                    in_position = False
    
    # Calculate performance metrics
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    total_pnl = sum(t['pnl'] for t in trades)
    
    performance = {
        'initial_balance': initial_balance,
        'final_balance': balance,
        'total_return': (balance - initial_balance) / initial_balance * 100,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_pnl': total_pnl,
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
    
    print("\nRunning backtest...")
    performance = backtest_strategy(data)
    
    # Display performance summary
    print("\nPerformance Summary:")
    print(f"Initial Balance: ${performance['initial_balance']:.2f}")
    print(f"Final Balance: ${performance['final_balance']:.2f}")
    print(f"Total Return: {performance['total_return']:.2f}%")
    print(f"Total Trades: {performance['total_trades']}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Win Rate: {performance['win_rate'] * 100:.2f}%")
    
    # Plot the price chart with signals
    plt.figure(figsize=(12, 8))
    
    # Price and EMAs
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'], label='Price', alpha=0.7)
    plt.plot(data.index, data['ema_fast'], label=f'EMA(9)', color='blue')
    plt.plot(data.index, data['ema_slow'], label=f'EMA(21)', color='red')
    
    # Plot buy/sell signals
    buy_signals = data[data['long_signal']]
    sell_signals = data[data['short_signal']]
    plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy Signal')
    plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell Signal')
    
    plt.title('BTC/USDT Scalping Strategy Test')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # RSI subplot
    plt.subplot(3, 1, 2)
    plt.plot(data.index, data['rsi'], color='purple')
    plt.axhline(y=10, color='green', linestyle='--', label='Oversold')
    plt.axhline(y=90, color='red', linestyle='--', label='Overbought')
    plt.title('RSI(2)')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    
    # Volume subplot
    plt.subplot(3, 1, 3)
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