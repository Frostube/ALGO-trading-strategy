import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from src.data.fetcher import DataFetcher
from src.indicators.technical import apply_indicators
from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, COMMISSION,
    SLIPPAGE, BACKTEST_TRAIN_SPLIT
)
from src.utils.logger import logger

class Backtester:
    """Backtesting engine for the scalping strategy."""
    
    def __init__(self, data=None, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
    def load_data(self, days=30):
        """Load historical data for backtesting."""
        logger.info(f"Loading {days} days of historical data for backtesting")
        data_fetcher = DataFetcher()
        self.data = data_fetcher.fetch_historical_data(days)
        data_fetcher.close()
        return self.data
    
    def run(self, train_test_split=None):
        """
        Run the backtest on the loaded data.
        
        Args:
            train_test_split: If provided, split data into train/test sets
                              and return separate results for each
        
        Returns:
            dict: Backtest results
        """
        if self.data is None or self.data.empty:
            logger.error("No data loaded for backtesting")
            return None
        
        # Apply indicators to the data
        data_with_indicators = apply_indicators(self.data)
        
        # Split data if requested
        if train_test_split:
            split_idx = int(len(data_with_indicators) * train_test_split)
            train_data = data_with_indicators.iloc[:split_idx]
            test_data = data_with_indicators.iloc[split_idx:]
            
            train_results = self._run_on_dataset(train_data, "Training")
            
            # Reset state for test run
            self.balance = self.initial_balance
            self.trades = []
            self.equity_curve = []
            
            test_results = self._run_on_dataset(test_data, "Testing")
            
            return {
                "train": train_results,
                "test": test_results
            }
        else:
            return self._run_on_dataset(data_with_indicators, "Full")
    
    def _run_on_dataset(self, data, dataset_name):
        """Run backtest on a specific dataset."""
        logger.info(f"Running backtest on {dataset_name} dataset with {len(data)} bars")
        
        # Initialize tracking variables
        position = None
        entry_price = 0
        entry_time = None
        stop_loss = 0
        take_profit = 0
        position_size = 0
        consecutive_sl = 0
        
        # Track equity at each bar
        self.equity_curve = [{"timestamp": data.index[0], "equity": self.balance}]
        
        # Iterate through each bar
        for idx, row in data.iterrows():
            current_price = row['close']
            
            # Update equity curve
            if position:
                # Calculate unrealized PnL
                if position == 'long':
                    unrealized_pnl = (current_price - entry_price) * position_size
                else:
                    unrealized_pnl = (entry_price - current_price) * position_size
                    
                current_equity = self.balance + unrealized_pnl
            else:
                current_equity = self.balance
                
            self.equity_curve.append({
                "timestamp": idx,
                "equity": current_equity
            })
            
            # Check if we need to close position
            if position:
                if position == 'long':
                    # Check if stop loss hit
                    if row['low'] <= stop_loss:
                        exit_price = stop_loss * (1 - SLIPPAGE)  # Apply slippage
                        self._close_position(exit_price, idx, 'stop_loss')
                        consecutive_sl += 1
                        position = None
                    # Check if take profit hit
                    elif row['high'] >= take_profit:
                        exit_price = take_profit * (1 - SLIPPAGE)  # Apply slippage
                        self._close_position(exit_price, idx, 'take_profit')
                        consecutive_sl = 0
                        position = None
                else:  # short position
                    # Check if stop loss hit
                    if row['high'] >= stop_loss:
                        exit_price = stop_loss * (1 + SLIPPAGE)  # Apply slippage
                        self._close_position(exit_price, idx, 'stop_loss')
                        consecutive_sl += 1
                        position = None
                    # Check if take profit hit
                    elif row['low'] <= take_profit:
                        exit_price = take_profit * (1 + SLIPPAGE)  # Apply slippage
                        self._close_position(exit_price, idx, 'take_profit')
                        consecutive_sl = 0
                        position = None
            
            # Check for new entry signals if not in a position
            if not position:
                # Apply the strategy rules
                
                # Buy signal: EMA trend up, RSI oversold, volume spike
                if (row['ema_trend'] > 0 and 
                    row['rsi'] < 10 and 
                    row['volume_spike']):
                    
                    position = 'long'
                    entry_price = current_price * (1 + SLIPPAGE)  # Apply slippage
                    entry_time = idx
                    stop_loss = entry_price * (1 - STOP_LOSS_PCT)
                    take_profit = entry_price * (1 + TAKE_PROFIT_PCT)
                    
                    # Calculate position size (1% risk)
                    risk_amount = self.balance * 0.01
                    risk_per_unit = entry_price - stop_loss
                    position_size = risk_amount / risk_per_unit
                    
                    logger.debug(f"BUY signal at {idx}: Price={entry_price}, SL={stop_loss}, TP={take_profit}")
                
                # Sell signal: EMA trend down, RSI overbought, volume spike
                elif (row['ema_trend'] < 0 and 
                      row['rsi'] > 90 and 
                      row['volume_spike']):
                    
                    position = 'short'
                    entry_price = current_price * (1 - SLIPPAGE)  # Apply slippage
                    entry_time = idx
                    stop_loss = entry_price * (1 + STOP_LOSS_PCT)
                    take_profit = entry_price * (1 - TAKE_PROFIT_PCT)
                    
                    # Calculate position size (1% risk)
                    risk_amount = self.balance * 0.01
                    risk_per_unit = stop_loss - entry_price
                    position_size = risk_amount / risk_per_unit
                    
                    logger.debug(f"SELL signal at {idx}: Price={entry_price}, SL={stop_loss}, TP={take_profit}")
        
        # Close any remaining position at the end of the backtest
        if position:
            self._close_position(current_price, data.index[-1], 'end_of_test')
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics()
        results['dataset'] = dataset_name
        
        logger.info(f"{dataset_name} backtest results: " + 
                   f"Return={results['total_return']*100:.2f}%, " +
                   f"Sharpe={results['sharpe_ratio']:.2f}, " +
                   f"Win Rate={results['win_rate']*100:.2f}%")
        
        return results
    
    def _close_position(self, exit_price, exit_time, reason):
        """Close an open position and record the trade."""
        if not hasattr(self, 'position') or not self.position:
            return
        
        position = self.position
        entry_price = self.entry_price
        entry_time = self.entry_time
        position_size = self.position_size
        
        # Calculate P&L
        if position == 'long':
            pnl = (exit_price - entry_price) * position_size
        else:
            pnl = (entry_price - exit_price) * position_size
        
        # Apply commission
        commission = (entry_price + exit_price) * position_size * COMMISSION
        pnl -= commission
        
        # Update balance
        self.balance += pnl
        
        # Record trade
        self.trades.append({
            'entry_time': entry_time,
            'exit_time': exit_time,
            'position': position,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'return': pnl / (entry_price * position_size),
            'reason': reason
        })
        
        logger.debug(f"Closed {position} position: Entry={entry_price}, Exit={exit_price}, PnL={pnl:.2f}")
        
        # Reset position
        self.position = None
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics from the backtest results."""
        if not self.trades:
            return {
                'total_return': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_trade_return': 0,
                'total_trades': 0
            }
        
        # Calculate returns
        initial_balance = self.initial_balance
        final_balance = self.balance
        total_return = (final_balance - initial_balance) / initial_balance
        
        # Calculate win rate
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        total_trades = len(self.trades)
        win_rate = winning_trades / total_trades
        
        # Calculate average trade return
        avg_trade_return = sum(t['return'] for t in self.trades) / total_trades
        
        # Calculate Sharpe ratio (assuming daily returns and risk-free rate of 0)
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.set_index('timestamp', inplace=True)
            
            equity_df['daily_return'] = equity_df['equity'].pct_change()
            sharpe_ratio = (
                equity_df['daily_return'].mean() / equity_df['daily_return'].std() * 
                (252 ** 0.5)  # Annualize (252 trading days)
            ) if equity_df['daily_return'].std() > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        if self.equity_curve:
            equity_series = pd.DataFrame(self.equity_curve)['equity']
            rolling_max = equity_series.cummax()
            drawdown = (equity_series - rolling_max) / rolling_max
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_return': avg_trade_return,
            'total_trades': total_trades
        }
    
    def plot_results(self, save_path=None):
        """Plot equity curve and trade distribution."""
        if not self.equity_curve:
            logger.warning("No equity curve data to plot")
            return
        
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        axes[0].plot(equity_df.index, equity_df['equity'])
        axes[0].set_title('Equity Curve')
        axes[0].set_ylabel('Equity ($)')
        axes[0].grid(True)
        
        # Plot trade returns
        if self.trades:
            trade_df = pd.DataFrame(self.trades)
            
            # Get colors based on PnL
            colors = ['green' if t > 0 else 'red' for t in trade_df['pnl']]
            
            axes[1].bar(range(len(trade_df)), trade_df['return'] * 100, color=colors)
            axes[1].set_title('Trade Returns (%)')
            axes[1].set_xlabel('Trade #')
            axes[1].set_ylabel('Return (%)')
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved backtest plot to {save_path}")
        else:
            plt.show()
        
        plt.close()

def run_backtest(days=30, initial_balance=10000, plot=True):
    """Run a backtest with the specified parameters."""
    logger.info(f"Running backtest with {days} days of data")
    
    backtester = Backtester(initial_balance=initial_balance)
    backtester.load_data(days=days)
    
    # Run the backtest with train/test split
    results = backtester.run(train_test_split=BACKTEST_TRAIN_SPLIT)
    
    if plot:
        backtester.plot_results(save_path="reports/backtest_results.png")
    
    return results

if __name__ == '__main__':
    run_backtest(days=30, plot=True) 