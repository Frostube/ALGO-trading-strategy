import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from src.data.fetcher import DataFetcher
from src.indicators.technical import apply_indicators, get_signal
from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, COMMISSION,
    SLIPPAGE, BACKTEST_TRAIN_SPLIT, RISK_PER_TRADE, USE_ATR_STOPS, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    HIGHER_TIMEFRAME, TIMEFRAME, EMA_TREND, EMA_SLOW, EMA_FAST,
    TRAIL_ACTIVATION_PCT, TRAIL_ATR_MULTIPLIER, USE_ML_FILTER, ML_MIN_TRADES_FOR_TRAINING
)
from src.utils.logger import logger

# Import ML filter if enabled
if USE_ML_FILTER:
    try:
        from src.ml.signal_filter import MLSignalFilter
        ml_filter = MLSignalFilter()
    except ImportError:
        logger.warning("ML filter module not found, disabling ML filtering")
        USE_ML_FILTER = False
    except Exception as e:
        logger.error(f"Error initializing ML filter: {e}")
        USE_ML_FILTER = False

class Backtester:
    """Backtesting engine for the scalping strategy."""
    
    def __init__(self, data=None, initial_balance=10000, higher_tf_data=None):
        self.data = data
        self.higher_tf_data = higher_tf_data  # Higher timeframe data for confirmation
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []
        self.equity_curve = []
        
    def load_data(self, days=30):
        """Load historical data for backtesting."""
        logger.info(f"Loading {days} days of historical data for backtesting")
        data_fetcher = DataFetcher(use_testnet=True)  # Use testnet for backtesting
        
        # Fetch primary timeframe data
        self.data = data_fetcher.fetch_historical_data(days)
        
        # Fetch higher timeframe data for multi-timeframe analysis
        if HIGHER_TIMEFRAME:
            logger.info(f"Fetching higher timeframe data ({HIGHER_TIMEFRAME}) for confirmation")
            self.higher_tf_data = data_fetcher.fetch_historical_data(days, timeframe=HIGHER_TIMEFRAME)
        
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
        
        # Apply indicators to the primary timeframe data
        data_with_indicators = apply_indicators(self.data)
        
        # Process higher timeframe data for confirmation if available
        if self.higher_tf_data is not None and not self.higher_tf_data.empty:
            logger.info("Processing higher timeframe data for confirmation")
            higher_tf_with_indicators = apply_indicators(self.higher_tf_data)
            
            # Add higher timeframe trend information to primary timeframe
            self._align_timeframes(data_with_indicators, higher_tf_with_indicators)
        
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
    
    def _align_timeframes(self, primary_df, higher_tf_df):
        """
        Align timeframes and add higher timeframe indicators to primary timeframe.
        
        Args:
            primary_df: Primary (lower) timeframe DataFrame
            higher_tf_df: Higher timeframe DataFrame with indicators
        """
        # Extract key indicators from higher timeframe
        if higher_tf_df.empty:
            return
            
        # Create forward-fill resampled version of higher timeframe data
        # to align with primary timeframe
        higher_tf_indicators = higher_tf_df[['ema_trend', 'rsi', 'market_trend']].copy()
        
        # Rename columns to distinguish from primary timeframe
        higher_tf_indicators = higher_tf_indicators.rename(columns={
            'ema_trend': 'higher_tf_ema_trend',
            'rsi': 'higher_tf_rsi',
            'market_trend': 'higher_tf_market_trend'
        })
        
        # Resample to primary timeframe with forward fill
        for idx in primary_df.index:
            # Find the most recent higher timeframe data point
            mask = higher_tf_df.index <= idx
            if mask.any():
                latest_idx = higher_tf_df.index[mask][-1]
                
                # Add higher timeframe indicators to current row
                for col in higher_tf_indicators.columns:
                    if col in higher_tf_indicators.columns:
                        primary_df.loc[idx, col] = higher_tf_indicators.loc[latest_idx, col]
        
        # Add multi-timeframe confirmation flags
        primary_df['multi_tf_bull'] = (primary_df['ema_trend'] > 0) & (primary_df.get('higher_tf_ema_trend', 0) > 0)
        primary_df['multi_tf_bear'] = (primary_df['ema_trend'] < 0) & (primary_df.get('higher_tf_ema_trend', 0) < 0)
    
    def _run_on_dataset(self, data, dataset_name=""):
        """Run backtest on a specific dataset."""
        # Check if data is empty
        if data.empty:
            print(f"Error: No data available for backtest in {dataset_name} dataset")
            return {
                "total_return": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "trades": []
            }
        
        # Initialize equity curve with starting balance
        self.equity_curve = [{"timestamp": data.index[0], "equity": self.balance}]
        
        # Initialize tracking variables
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        self.position_size = 0
        self.max_trade_duration = 60  # Maximum trade duration in bars (minutes)
        consecutive_sl = 0
        
        # Trailing stop variables
        self.trailing_active = False
        self.trailing_stop = 0
        self.max_favorable_excursion = 0
        
        # Market regime variables
        volatility_window = 20  # Window for volatility calculation
        trend_window = 50  # Window for trend strength calculation
        
        # Calculate market regime indicators
        # 1. Volatility regime - rolling ATR
        if 'atr' in data.columns:
            data['volatility_regime'] = data['atr'].rolling(window=volatility_window).mean() / data['close']
            data['high_volatility'] = data['volatility_regime'] > data['volatility_regime'].rolling(window=100).mean() * 1.2
        else:
            # If ATR not available, use standard deviation of returns
            data['returns'] = data['close'].pct_change()
            data['volatility_regime'] = data['returns'].rolling(window=volatility_window).std()
            data['high_volatility'] = data['volatility_regime'] > data['volatility_regime'].rolling(window=100).mean() * 1.2
        
        # 2. Trend strength indicator
        if 'ema_trend' in data.columns:
            # Use exponential moving average crossovers for trend
            data['trend_strength'] = (data['close'] - data[f'ema_{EMA_TREND}']).abs() / data[f'ema_{EMA_TREND}']
            # Direction of trend (1 for up, -1 for down, 0 for sideways)
            data['trend_direction'] = np.where(data['close'] > data[f'ema_{EMA_TREND}'], 1,
                                        np.where(data['close'] < data[f'ema_{EMA_TREND}'], -1, 0))
        else:
            # Simple moving average if EMA not available
            data['sma50'] = data['close'].rolling(window=50).mean()
            data['trend_strength'] = (data['close'] - data['sma50']).abs() / data['sma50']
            data['trend_direction'] = np.where(data['close'] > data['sma50'], 1,
                                      np.where(data['close'] < data['sma50'], -1, 0))
        
        # 3. Define favorable market conditions
        # Strong trend, normal volatility is favorable
        data['favorable_long'] = (data['trend_direction'] > 0) & (~data['high_volatility'])
        data['favorable_short'] = (data['trend_direction'] < 0) & (~data['high_volatility'])
        
        # Iterate through each bar
        for idx, row in data.iterrows():
            current_price = row['close']
            
            # Update equity curve
            if len(self.equity_curve) > 0:
                self.equity_curve.append({
                    "timestamp": idx,
                    "equity": self.balance
                })
            
            # Get current ATR for volatility-based position sizing and stops
            current_atr = row['atr'] if 'atr' in row else None
            current_volatility = row['volatility_regime'] if 'volatility_regime' in row else 0.001
            
            # Check for exit if in a position
            if self.position is not None:
                # Initialize exit trigger flags
                stop_loss_hit = False
                take_profit_hit = False
                trailing_stop_hit = False
                exit_reason = None
                
                # Calculate position PnL
                if self.position == "long":
                    unrealized_pnl = (current_price - self.entry_price) / self.entry_price
                    # For long positions, price below stop loss triggers exit
                    stop_loss_hit = current_price <= self.stop_loss
                    # Price above take profit triggers exit
                    take_profit_hit = current_price >= self.take_profit
                    
                    # Dynamic trailing stop based on ATR and current volatility
                    if current_atr is not None:
                        # If position is favorable and trailing not yet activated
                        if not self.trailing_active and current_price >= self.entry_price * (1 + TRAIL_ACTIVATION_PCT):
                            # Activate trailing stop at a percentage of ATR (tighter for high volatility)
                            trail_factor = TRAIL_ATR_MULTIPLIER * (1 - min(0.5, current_volatility * 10))
                            self.trailing_stop = max(self.stop_loss, current_price * (1 - trail_factor * current_atr / current_price))
                            self.trailing_active = True
                        
                        # If trailing is active, update trailing stop as price moves higher
                        elif self.trailing_active:
                            # Calculate new potential trailing stop
                            # In higher volatility conditions, allow more room before trailing
                            trail_factor = TRAIL_ATR_MULTIPLIER * (1 - min(0.5, current_volatility * 10))
                            new_trailing = current_price * (1 - trail_factor * current_atr / current_price)
                            
                            # Only raise trailing stop, never lower it
                            if new_trailing > self.trailing_stop:
                                self.trailing_stop = new_trailing
                            
                            # Check if price has fallen below trailing stop
                            trailing_stop_hit = current_price <= self.trailing_stop
                    
                else:  # Short position
                    unrealized_pnl = (self.entry_price - current_price) / self.entry_price
                    # For short positions, price above stop loss triggers exit
                    stop_loss_hit = current_price >= self.stop_loss
                    # Price below take profit triggers exit
                    take_profit_hit = current_price <= self.take_profit
                    
                    # Dynamic trailing stop based on ATR and current volatility
                    if current_atr is not None:
                        # If position is favorable and trailing not yet activated
                        if not self.trailing_active and current_price <= self.entry_price * (1 - TRAIL_ACTIVATION_PCT):
                            # Activate trailing stop at a percentage of ATR (tighter for high volatility)
                            trail_factor = TRAIL_ATR_MULTIPLIER * (1 - min(0.5, current_volatility * 10))
                            self.trailing_stop = min(self.stop_loss, current_price * (1 + trail_factor * current_atr / current_price))
                            self.trailing_active = True
                        
                        # If trailing is active, update trailing stop as price moves lower
                        elif self.trailing_active:
                            # Calculate new potential trailing stop
                            # In higher volatility conditions, allow more room before trailing
                            trail_factor = TRAIL_ATR_MULTIPLIER * (1 - min(0.5, current_volatility * 10))
                            new_trailing = current_price * (1 + trail_factor * current_atr / current_price)
                            
                            # Only lower trailing stop for shorts, never raise it
                            if new_trailing < self.trailing_stop:
                                self.trailing_stop = new_trailing
                            
                            # Check if price has risen above trailing stop
                            trailing_stop_hit = current_price >= self.trailing_stop
                
                # Track maximum favorable excursion for the trade
                current_pnl = unrealized_pnl
                if current_pnl > self.max_favorable_excursion:
                    self.max_favorable_excursion = current_pnl
                
                # Check if maximum trade duration exceeded
                trade_duration = (data.index.get_loc(idx) - data.index.get_loc(self.entry_time)) if self.entry_time in data.index else 0
                max_duration_exceeded = trade_duration >= self.max_trade_duration
                
                # Exit conditions
                if stop_loss_hit:
                    exit_reason = "stop_loss"
                    exit_price = self.stop_loss
                    self._close_position(exit_price, idx, exit_reason)
                    # Track consecutive stop-losses
                    consecutive_sl += 1
                elif trailing_stop_hit:
                    exit_reason = "trailing_stop"
                    exit_price = self.trailing_stop
                    self._close_position(exit_price, idx, exit_reason)
                    # Reset consecutive stop-losses as trailing stop means the trade was profitable for a time
                    consecutive_sl = 0
                elif take_profit_hit:
                    exit_reason = "take_profit"
                    exit_price = self.take_profit
                    self._close_position(exit_price, idx, exit_reason)
                    # Reset consecutive stop-losses as we hit take profit
                    consecutive_sl = 0
                elif max_duration_exceeded:
                    exit_reason = "time_exit"
                    exit_price = current_price
                    self._close_position(exit_price, idx, exit_reason)
                    # Don't update consecutive SL for time exits
                
                # Additional dynamic exit logic based on market conditions
                elif consecutive_sl >= 3:
                    # After 3 consecutive stop losses, use more conservative exits
                    # Exit if price returns to entry (break-even) after hitting 50% of take profit
                    if ((self.position == "long" and 
                         current_price <= self.entry_price and 
                         self.max_favorable_excursion > (self.take_profit - self.entry_price) / self.entry_price / 2) or
                        (self.position == "short" and 
                         current_price >= self.entry_price and 
                         self.max_favorable_excursion > (self.entry_price - self.take_profit) / self.entry_price / 2)):
                        exit_reason = "break_even_after_favorable"
                        self._close_position(current_price, idx, exit_reason)
            
            # Get trading signal from indicators
            signal_data = get_signal(data, idx, self.entry_time if self.position else None)
            
            # Add debugging for signal generation
            if signal_data['signal'] != 'neutral':
                logger.debug(f"Signal generated: {signal_data['signal']} at {idx} - RSI: {signal_data['rsi']:.2f}, Strategy: {signal_data.get('strategy', 'unknown')}")
            
            # Only take a new position if not already in one
            if self.position is None and signal_data['signal'] in ['buy', 'sell']:
                # Skip trade if in high volatility and we've had consecutive losses
                if consecutive_sl >= 2 and row.get('high_volatility', False):
                    logger.debug(f"Skipping {signal_data['signal']} signal due to consecutive stops in high volatility")
                    continue
                    
                # Implement trade
                self.position = "long" if signal_data['signal'] == 'buy' else "short"
                self.entry_price = current_price
                self.entry_time = idx
                
                # Log entry for debugging
                logger.info(f"Entering {self.position} position at {current_price:.2f} using {signal_data.get('strategy', 'unknown')} strategy")
                
                # Determine position size based on risk and volatility
                # Use ATR for dynamic position sizing if available
                if USE_ATR_STOPS and 'atr' in row and current_atr > 0:
                    # Calculate volatility-adjusted position size
                    # For higher volatility, reduce position size
                    volatility_factor = 1.0
                    if 'volatility_regime' in row:
                        volatility_factor = 1.0 / (1.0 + max(0, (row['volatility_regime'] - 0.001) * 10))
                        volatility_factor = max(0.5, min(1.0, volatility_factor))  # Clamp between 0.5 and 1.0
                    
                    # Calculate risk amount
                    risk_amount = self.balance * RISK_PER_TRADE * volatility_factor
                    
                    # Use ATR to calculate stop loss and position size
                    if self.position == "long":
                        stop_pct = ATR_SL_MULTIPLIER * current_atr / current_price
                        self.stop_loss = current_price * (1 - stop_pct)
                        take_profit_pct = ATR_TP_MULTIPLIER * current_atr / current_price
                        self.take_profit = current_price * (1 + take_profit_pct)
                    else:  # short
                        stop_pct = ATR_SL_MULTIPLIER * current_atr / current_price
                        self.stop_loss = current_price * (1 + stop_pct)
                        take_profit_pct = ATR_TP_MULTIPLIER * current_atr / current_price
                        self.take_profit = current_price * (1 - take_profit_pct)
                    
                    # Calculate position size to risk specified amount
                    risk_per_unit = abs(self.entry_price - self.stop_loss)
                    self.position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                else:
                    # Fixed stop loss and take profit percentages
                    if self.position == "long":
                        self.stop_loss = current_price * (1 - STOP_LOSS_PCT)
                        self.take_profit = current_price * (1 + TAKE_PROFIT_PCT)
                    else:  # short
                        self.stop_loss = current_price * (1 + STOP_LOSS_PCT)
                        self.take_profit = current_price * (1 - TAKE_PROFIT_PCT)
                    
                    # Calculate position size based on fixed risk percentage
                    risk_amount = self.balance * RISK_PER_TRADE
                    risk_per_unit = abs(self.entry_price - self.stop_loss)
                    self.position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
                
                # Initialize trailing stop variables
                self.trailing_active = False
                self.trailing_stop = 0
                self.max_favorable_excursion = 0
                
                # Record trade entry
                self.trades.append({
                    "side": self.position,
                    "entry_time": idx,
                    "entry_price": self.entry_price,
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                    "position_size": self.position_size,
                    "strategy": signal_data.get('strategy', 'unknown'),
                    "market_trend": signal_data.get('market_trend', 0),
                    "rsi": signal_data.get('rsi', 0),
                    "atr": signal_data.get('atr', 0),
                    "volume_spike": signal_data.get('volume_spike', False),
                    "price_pattern": next((pattern for pattern in [
                        'bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star', 
                        'morning_star', 'evening_star', 'doji'] 
                        if signal_data.get(pattern, False)), None)
                })
        
        # Close any remaining position at the end of the backtest
        if self.position:
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
        """
        Close the current position and record trade results.
        
        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            reason: Reason for exit
        """
        # Find the trade to update
        for trade in reversed(self.trades):
            if trade['entry_time'] == self.entry_time:
                # Calculate PnL
                if self.position == "long":
                    pnl = (exit_price - self.entry_price) * self.position_size
                    pnl_pct = (exit_price - self.entry_price) / self.entry_price
                else:  # short
                    pnl = (self.entry_price - exit_price) * self.position_size
                    pnl_pct = (self.entry_price - exit_price) / self.entry_price
                
                # Apply commission
                pnl = pnl - (self.entry_price + exit_price) * self.position_size * COMMISSION
                
                # Update trade record
                trade['exit_time'] = exit_time
                trade['exit_price'] = exit_price
                trade['exit_reason'] = reason
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct
                trade['duration'] = (exit_time - self.entry_time) if hasattr(exit_time, 'timestamp') else 0
                
                # Update account balance
                self.balance += pnl
                
                # Log trade details
                logger.debug(f"{self.position.upper()} position closed: Entry={self.entry_price:.2f}, " + 
                            f"Exit={exit_price:.2f}, PnL=${pnl:.2f}, Reason={reason}")
                break
        
        # Reset position state
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        self.trailing_active = False
        self.trailing_stop = 0
    
    def _calculate_performance_metrics(self):
        """Calculate performance metrics for the backtest."""
        if not self.trades:
            return {
                "total_return": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0,
                "total_trades": 0,
                "trades": []
            }
        
        # Calculate total return
        total_return = (self.balance / self.initial_balance) - 1
        
        # Calculate win rate
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t['pnl'] > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate average trade return
        avg_trade_return = sum(t['pnl_pct'] for t in self.trades) / total_trades
        
        # Calculate Sharpe ratio (assuming daily returns and risk-free rate of 0)
        # First, calculate daily returns from equity curve
        daily_returns = []
        if len(self.equity_curve) > 1:
            equity_values = [entry['equity'] for entry in self.equity_curve]
            daily_returns = pd.Series(equity_values).pct_change().dropna().values
        
        sharpe_ratio = 0
        if len(daily_returns) > 0 and np.std(daily_returns) > 0:
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = self.initial_balance
        
        for entry in self.equity_curve:
            equity = entry['equity']
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate drawdowns
        drawdowns = []
        in_drawdown = False
        start_idx = 0
        peak_equity = self.initial_balance
        peak_idx = 0
        
        for i, entry in enumerate(self.equity_curve):
            equity = entry['equity']
            
            if equity > peak_equity:
                peak_equity = equity
                peak_idx = i
                
                if in_drawdown:
                    # End of drawdown
                    end_idx = i - 1
                    recovery_time = end_idx - start_idx
                    drawdowns.append({
                        "start": self.equity_curve[start_idx]["timestamp"],
                        "end": self.equity_curve[end_idx]["timestamp"],
                        "depth": (peak_equity - self.equity_curve[end_idx]["equity"]) / peak_equity,
                        "recovery_time": recovery_time
                    })
                    in_drawdown = False
            elif equity < peak_equity and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = peak_idx
        
        # If still in drawdown at the end
        if in_drawdown:
            end_idx = len(self.equity_curve) - 1
            recovery_time = end_idx - start_idx
            drawdowns.append({
                "start": self.equity_curve[start_idx]["timestamp"],
                "end": self.equity_curve[end_idx]["timestamp"],
                "depth": (peak_equity - self.equity_curve[end_idx]["equity"]) / peak_equity,
                "recovery_time": recovery_time
            })
        
        # Train ML model if enough trades and ML filter is enabled
        if USE_ML_FILTER and len(self.trades) >= ML_MIN_TRADES_FOR_TRAINING:
            try:
                logger.info(f"Training ML filter with {len(self.trades)} historical trades")
                # Convert trades list to DataFrame
                trades_df = pd.DataFrame(self.trades)
                # Train the model
                metrics = ml_filter.train(trades_df, self.data)
                if metrics:
                    logger.info(f"ML filter trained successfully. Accuracy: {metrics['accuracy']:.2f}, Precision: {metrics['precision']:.2f}")
            except Exception as e:
                logger.error(f"Error training ML filter: {e}")
        
        return {
            "total_return": total_return,
            "win_rate": win_rate,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": total_trades,
            "avg_trade_pnl": sum(t['pnl'] for t in self.trades) / total_trades if total_trades > 0 else 0,
            "avg_trade_return": avg_trade_return,
            "profit_factor": sum(t['pnl'] for t in self.trades if t['pnl'] > 0) / abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0)) if sum(t['pnl'] for t in self.trades if t['pnl'] < 0) != 0 else 0,
            "avg_trade_duration": sum((t['exit_time'] - t['entry_time']).total_seconds() / 60 for t in self.trades if isinstance(t['exit_time'], pd.Timestamp) and isinstance(t['entry_time'], pd.Timestamp)) / total_trades if total_trades > 0 else 0,
            "trades": self.trades,
            "daily_returns": daily_returns,
            "drawdowns": drawdowns
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
            
            axes[1].bar(range(len(trade_df)), trade_df['pnl'] * 100, color=colors)
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
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch primary timeframe data
    primary_data = data_fetcher.fetch_historical_data(days=days, timeframe=TIMEFRAME)
    
    # Fetch higher timeframe data for multi-timeframe analysis
    higher_tf_data = None
    if HIGHER_TIMEFRAME:
        logger.info(f"Fetching higher timeframe data ({HIGHER_TIMEFRAME}) for confirmation")
        higher_tf_data = data_fetcher.fetch_historical_data(days=days, timeframe=HIGHER_TIMEFRAME)
    
    # Clean up resources
    data_fetcher.close()
    
    # Initialize backtester with both timeframes
    backtester = Backtester(data=primary_data, initial_balance=initial_balance, higher_tf_data=higher_tf_data)
    
    # Run the backtest with train/test split
    results = backtester.run(train_test_split=BACKTEST_TRAIN_SPLIT)
    
    if plot:
        backtester.plot_results(save_path="reports/backtest_results.png")
    
    return results

if __name__ == '__main__':
    run_backtest(days=30, plot=True) 