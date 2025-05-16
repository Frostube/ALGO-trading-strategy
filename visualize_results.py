import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our utils
from src.utils.logger import logger

def main():
    """
    Visualize the latest backtest results
    """
    # Find the most recent backtest results file
    results_dir = "results"
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    if not json_files:
        logger.error("No backtest results found in the results directory.")
        return
    
    # Sort by creation time (newest first)
    json_files.sort(key=lambda x: os.path.getctime(os.path.join(results_dir, x)), reverse=True)
    latest_file = os.path.join(results_dir, json_files[0])
    
    logger.info(f"Visualizing results from {latest_file}")
    
    # Load the backtest results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    # Extract the symbol and timeframe from the filename
    filename = os.path.basename(latest_file)
    parts = filename.split('_')
    symbol = parts[1]
    timeframe = parts[2]
    
    # Set up the plot
    plt.figure(figsize=(14, 10))
    
    # Plot equity curve
    plt.subplot(3, 1, 1)
    
    if 'equity_curve' in results:
        # Get equity curve data
        equity_curve = results['equity_curve']
        
        # Calculate log-scale for better visualization
        log_equity = np.log10(np.array(equity_curve))
        
        # Plot the equity curve on log scale
        plt.plot(log_equity)
        plt.title(f'Account Balance Over Time (Log10 Scale) - {symbol} {timeframe}')
        plt.ylabel('Log10(Balance)')
        plt.grid(True)
    
    # Plot drawdown
    plt.subplot(3, 1, 2)
    
    if 'drawdown_curve' in results:
        # Get drawdown data and convert to percentage
        drawdown_curve = [d * 100 for d in results['drawdown_curve']]
        
        plt.plot(drawdown_curve)
        plt.title('Drawdown Percentage')
        plt.ylabel('Drawdown %')
        plt.grid(True)
    
    # Plot trade P&L
    plt.subplot(3, 1, 3)
    
    if 'trades' in results and results['trades']:
        # Extract trade P&L
        trades = results['trades']
        pnl_values = [trade.get('pnl', 0) for trade in trades]
        
        # For percentage calculation, need entry price and size
        pnl_percent = []
        for trade in trades:
            entry_price = trade.get('entry_price', 0)
            size = trade.get('size', 0)
            pnl = trade.get('pnl', 0)
            
            # Calculate percentage P&L
            if entry_price > 0 and size > 0:
                pct = (pnl / (entry_price * size)) * 100
            else:
                pct = 0
            
            pnl_percent.append(pct)
        
        # Plot P&L for each trade
        colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
        plt.bar(range(len(pnl_percent)), pnl_percent, color=colors)
        plt.title('Trade P&L Percentage')
        plt.ylabel('P&L %')
        plt.xlabel('Trade #')
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = f"results/visualization_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file)
    plt.close()
    
    logger.info(f"Visualization saved to {output_file}")
    
    # Print summary statistics
    print("\n============================================================")
    print(f"Backtest Summary for {symbol} ({timeframe})")
    print("============================================================")
    print(f"Initial Balance: ${results['initial_balance']:.2f}")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Net Profit: ${results['net_profit']:.2f} ({results.get('roi', 0):.2f}%)")
    print(f"Win Rate: {results['win_rate'] * 100:.1f}% ({results['winning_trades']}/{results['total_trades']})")
    print(f"Max Drawdown: {results['max_drawdown'] * 100:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print("============================================================")
    
    # Display sample trades
    print("\nSample Trades:")
    print("------------------------------------------------------------")
    for i, trade in enumerate(trades[:5]):
        # Calculate percentage P&L
        entry_price = trade.get('entry_price', 0)
        size = trade.get('size', 0)
        pnl = trade.get('pnl', 0)
        
        if entry_price > 0 and size > 0:
            pct = (pnl / (entry_price * size)) * 100
        else:
            pct = 0
        
        direction = "LONG" if trade.get('type', '') == 'long' else "SHORT"
        emoji = "✅" if pnl > 0 else "❌"
        
        print(f"{i+1}. {direction} {emoji} " +
              f"Entry: ${entry_price:.2f}, Exit: ${trade.get('exit_price', 0):.2f}, " +
              f"P/L: ${pnl:.2f} ({pct:.2f}%)")
        
        if 'exit_reason' in trade:
            print(f"   Exit Reason: {trade['exit_reason']}")
    
    print("============================================================")

if __name__ == "__main__":
    main() 