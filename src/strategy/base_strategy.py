#!/usr/bin/env python3
"""
Base strategy class implementing core trading functionality.
"""

import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class BaseStrategy:
    """
    Base class for trading strategies.
    Implements common functionality like position management and risk controls.
    """
    
    def __init__(self, config=None):
        """
        Initialize the strategy.
        
        Args:
            config (dict): Strategy configuration parameters
        """
        self.name = self.__class__.__name__
        self.config = config or {}
        
        # Position tracking
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        self.trail_stop = 0
        self.position_count = 0  # For pyramiding
        
        # Risk parameters
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.01)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.02)
        self.max_position_pct = self.config.get('max_position_pct', 0.1)
        
        # Pyramiding
        self.enable_pyramiding = self.config.get('enable_pyramiding', False)
        self.max_pyramid_units = self.config.get('max_pyramid_units', 3)
        self.pyramid_threshold = self.config.get('pyramid_threshold', 0.5)  # In ATR units
        
        # ATR parameters
        self.atr_period = self.config.get('atr_period', 14)
        self.atr_sl_multiplier = self.config.get('atr_sl_multiplier', 2.0)
        self.atr_tp_multiplier = self.config.get('atr_tp_multiplier', 3.0)
        self.atr_value = 0  # Current ATR value
        
        # Volatility targeting
        self.use_vol_targeting = self.config.get('use_vol_targeting', False)
        self.vol_target = self.config.get('vol_target', 0.1)
        self.vol_lookback = self.config.get('vol_lookback', 20)
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.returns = []
        self.account = None
        self.allocator = None  # For portfolio allocation
        
        # State variables
        self.is_live = False
        self.last_update_time = None
        self.indicators = {}
        self.ready = False
    
    def initialize(self, account):
        """
        Initialize the strategy with an account.
        
        Args:
            account: Trading account object
        """
        self.account = account
        self.ready = True
    
    def calculate_position_size(self, price, atr=None):
        """
        Calculate position size based on risk parameters.
        
        Args:
            price (float): Current price
            atr (float): Average True Range value for volatility targeting
            
        Returns:
            float: Position size in units of the base currency
        """
        if not self.account:
            return 0
            
        equity = self.account.equity
        
        # If allocator is provided, use it for position sizing
        if self.allocator and hasattr(self.allocator, 'get_position_size'):
            vol = atr / price if atr else 0.01  # Default 1% vol if ATR not provided
            return self.allocator.get_position_size(self.name, self.account.symbol, equity, vol)
        
        # Volatility targeting approach
        if self.use_vol_targeting and atr:
            # K-factor formula: position size = (target vol / asset vol) * account equity
            volatility = atr / price  # Normalized volatility
            k_factor = self.vol_target / volatility
            position_size = equity * k_factor
            
            # Cap at max size
            max_size = equity * self.max_position_pct
            return min(position_size, max_size)
        
        # Default fixed percentage approach
        return equity * self.max_position_pct
    
    def enter_position(self, price, position_type, reason="", size=None):
        """
        Enter a new position.
        
        Args:
            price (float): Entry price
            position_type (str): "long" or "short"
            reason (str): Reason for entry
            size (float): Optional custom position size
            
        Returns:
            dict: Position information
        """
        if self.position != 0:
            return {"status": "error", "message": "Position already open"}
        
        # Calculate position size if not provided
        if size is None:
            size = self.calculate_position_size(price, self.atr_value)
        
        # Set position based on type
        self.position = size if position_type == "long" else -size
        self.entry_price = price
        self.entry_time = datetime.now()
        self.position_count = 1
        
        # Set stop loss and take profit levels
        if self.atr_value and self.atr_value > 0:
            # ATR-based stops
            if position_type == "long":
                self.stop_loss = price - (self.atr_value * self.atr_sl_multiplier)
                self.take_profit = price + (self.atr_value * self.atr_tp_multiplier)
                self.trail_stop = self.stop_loss
            else:
                self.stop_loss = price + (self.atr_value * self.atr_sl_multiplier)
                self.take_profit = price - (self.atr_value * self.atr_tp_multiplier)
                self.trail_stop = self.stop_loss
        else:
            # Percentage-based stops
            if position_type == "long":
                self.stop_loss = price * (1 - self.stop_loss_pct)
                self.take_profit = price * (1 + self.take_profit_pct)
                self.trail_stop = self.stop_loss
            else:
                self.stop_loss = price * (1 + self.stop_loss_pct)
                self.take_profit = price * (1 - self.take_profit_pct)
                self.trail_stop = self.stop_loss
        
        # Record the trade entry
        entry = {
            "type": position_type,
            "entry_time": self.entry_time,
            "entry_price": price,
            "size": abs(self.position),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "reason": reason
        }
        
        return {"status": "success", "entry": entry}
    
    def exit_position(self, price, reason=""):
        """
        Exit the current position.
        
        Args:
            price (float): Exit price
            reason (str): Reason for exit
            
        Returns:
            dict: Trade information
        """
        if self.position == 0:
            return {"status": "error", "message": "No position open"}
        
        # Calculate profit/loss
        if self.position > 0:  # Long position
            pnl = (price - self.entry_price) / self.entry_price * 100  # Percentage
            pnl_amount = abs(self.position) * (price - self.entry_price)  # Dollar amount
        else:  # Short position
            pnl = (self.entry_price - price) / self.entry_price * 100  # Percentage
            pnl_amount = abs(self.position) * (self.entry_price - price)  # Dollar amount
        
        # Record the trade
        exit_time = datetime.now()
        duration = exit_time - self.entry_time
        
        trade = {
            "type": "long" if self.position > 0 else "short",
            "entry_time": self.entry_time,
            "exit_time": exit_time,
            "entry_price": self.entry_price,
            "exit_price": price,
            "size": abs(self.position),
            "pnl_pct": pnl,
            "pnl": pnl_amount,
            "duration": duration,
            "reason": reason,
            "pyramid_units": self.position_count
        }
        
        # Add to trades history
        self.trades.append(trade)
        
        # Record return
        self.returns.append(pnl / 100)  # Convert percentage to decimal
        
        # Reset position tracking
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.take_profit = 0
        self.trail_stop = 0
        self.position_count = 0
        
        return {"status": "success", "trade": trade}
    
    def add_to_position(self, price, position_type, reason="pyramiding", size=None):
        """
        Add to an existing position (pyramiding).
        
        Args:
            price (float): Entry price for additional position
            position_type (str): "long" or "short" (must match current position)
            reason (str): Reason for pyramiding
            size (float): Optional custom additional position size
            
        Returns:
            dict: Position information or error
        """
        # Check if pyramiding is enabled
        if not self.enable_pyramiding:
            return {"status": "error", "message": "Pyramiding not enabled"}
        
        # Check if we have reached the maximum pyramid units
        if self.position_count >= self.max_pyramid_units:
            return {"status": "error", "message": "Maximum pyramid units reached"}
        
        # Validate position type matches current position
        current_type = "long" if self.position > 0 else "short"
        if current_type != position_type:
            return {"status": "error", "message": f"Cannot add {position_type} to {current_type} position"}
        
        # Verify we're in profit by threshold amount
        if position_type == "long":
            min_price = self.entry_price + (self.atr_value * self.pyramid_threshold)
            if price < min_price:
                return {"status": "error", "message": "Price not high enough for pyramiding"}
        else:  # short
            max_price = self.entry_price - (self.atr_value * self.pyramid_threshold)
            if price > max_price:
                return {"status": "error", "message": "Price not low enough for pyramiding"}
        
        # Calculate additional position size (usually smaller than initial)
        if size is None:
            # Calculate based on current ATR, but use smaller size than initial
            pyramid_factor = max(0.2, 1.0 / self.position_count)  # Size decreases with each unit
            size = self.calculate_position_size(price, self.atr_value) * pyramid_factor
        
        # Add to position
        if position_type == "long":
            self.position += size
        else:
            self.position -= size
        
        # Update position count
        self.position_count += 1
        
        # Move stop loss to breakeven after second unit (if in sufficient profit)
        if self.position_count >= 2:
            if position_type == "long" and self.stop_loss < self.entry_price:
                self.stop_loss = self.entry_price
            elif position_type == "short" and self.stop_loss > self.entry_price:
                self.stop_loss = self.entry_price
        
        # Update the average entry price
        # This is a simplification; in reality, the average entry price calculation would be more complex
        # and would consider the actual sizes of each entry
        
        pyramid_info = {
            "type": position_type,
            "pyramid_unit": self.position_count,
            "entry_time": datetime.now(),
            "entry_price": price,
            "size": size,
            "total_position": abs(self.position),
            "stop_loss": self.stop_loss,
            "reason": reason
        }
        
        return {"status": "success", "pyramid": pyramid_info}
    
    def update_trailing_stop(self, price):
        """
        Update the trailing stop based on current price.
        
        Args:
            price (float): Current price
            
        Returns:
            float: New trailing stop level
        """
        if self.position == 0:
            return 0
            
        if self.position > 0:  # Long position
            # Calculate new potential trailing stop
            new_stop = price - (self.atr_value * self.atr_sl_multiplier)
            
            # Only move the stop up, never down
            if new_stop > self.trail_stop:
                self.trail_stop = new_stop
        else:  # Short position
            # Calculate new potential trailing stop
            new_stop = price + (self.atr_value * self.atr_sl_multiplier)
            
            # Only move the stop down, never up
            if self.trail_stop == 0 or new_stop < self.trail_stop:
                self.trail_stop = new_stop
        
        return self.trail_stop
    
    def manage_open_positions(self, candle):
        """
        Manage open positions, including stop loss and take profit.
        
        Args:
            candle (dict): Current price candle
            
        Returns:
            dict: Trade result if position was closed, None otherwise
        """
        if self.position == 0:
            return None
            
        current_price = candle['close']
        
        # Update trailing stop if applicable
        self.update_trailing_stop(current_price)
        
        # Check if price hit stop loss or take profit
        if self.position > 0:  # Long position
            # Check if price hit trailing stop
            if current_price < self.trail_stop:
                return self.exit_position(current_price, "Trailing stop hit")
                
            # Check if take profit hit
            if self.take_profit > 0 and current_price >= self.take_profit:
                return self.exit_position(current_price, "Take profit hit")
                
            # Check for additional pyramiding opportunity
            if (self.enable_pyramiding and 
                self.position_count < self.max_pyramid_units and 
                current_price > self.entry_price + (self.atr_value * self.pyramid_threshold)):
                
                # Only pyramid if we haven't pyramided recently
                if not self.entry_time or (datetime.now() - self.entry_time) > timedelta(hours=1):
                    self.add_to_position(current_price, "long")
                
        else:  # Short position
            # Check if price hit trailing stop
            if current_price > self.trail_stop:
                return self.exit_position(current_price, "Trailing stop hit")
                
            # Check if take profit hit
            if self.take_profit > 0 and current_price <= self.take_profit:
                return self.exit_position(current_price, "Take profit hit")
                
            # Check for additional pyramiding opportunity
            if (self.enable_pyramiding and 
                self.position_count < self.max_pyramid_units and 
                current_price < self.entry_price - (self.atr_value * self.pyramid_threshold)):
                
                # Only pyramid if we haven't pyramided recently
                if not self.entry_time or (datetime.now() - self.entry_time) > timedelta(hours=1):
                    self.add_to_position(current_price, "short")
        
        return None
    
    def calculate_atr(self, data, period=14):
        """
        Calculate Average True Range (ATR).
        
        Args:
            data (pd.DataFrame): Price data with high, low, close columns
            period (int): ATR period
            
        Returns:
            float: Current ATR value
        """
        if len(data) < period + 1:
            return 0
            
        # Calculate true range
        data = data.copy()
        data['prev_close'] = data['close'].shift(1)
        data['tr1'] = abs(data['high'] - data['low'])
        data['tr2'] = abs(data['high'] - data['prev_close'])
        data['tr3'] = abs(data['low'] - data['prev_close'])
        data['tr'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        data['atr'] = data['tr'].rolling(period).mean()
        
        # Return current ATR
        return data['atr'].iloc[-1]
    
    def update_indicators(self, data):
        """
        Update strategy indicators.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            dict: Updated indicators
        """
        # Calculate ATR
        self.atr_value = self.calculate_atr(data, self.atr_period)
        self.indicators['atr'] = self.atr_value
        
        return self.indicators
    
    def get_signal(self, data):
        """
        Get trading signal from the strategy.
        This method should be overridden by specific strategy implementations.
        
        Args:
            data (pd.DataFrame): Price data
            
        Returns:
            str: Signal - "buy", "sell", or ""
        """
        # Base implementation - no signal
        return ""
    
    def process_candle(self, candle, data=None):
        """
        Process a new price candle.
        
        Args:
            candle (dict): Current price candle
            data (pd.DataFrame): Historical price data including this candle
            
        Returns:
            dict: Processing result
        """
        if data is not None:
            # Update indicators
            self.update_indicators(data)
            
            # Get signal
            signal = self.get_signal(data)
            
            # Manage existing position
            position_result = self.manage_open_positions(candle)
            
            # If no position and we have a signal, enter new position
            if self.position == 0 and signal in ["buy", "sell"]:
                position_type = "long" if signal == "buy" else "short"
                entry_result = self.enter_position(candle['close'], position_type, reason=f"{signal} signal")
                return {"signal": signal, "action": "enter", "result": entry_result}
            
            # If position was closed, return that result
            if position_result:
                return {"signal": signal, "action": "exit", "result": position_result}
                
            # No action taken
            return {"signal": signal, "action": "none"}
        
        # No data provided
        return {"signal": "", "action": "none"}
    
    def get_performance_stats(self):
        """
        Get performance statistics for the strategy.
        
        Returns:
            dict: Performance statistics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "avg_trade_pnl": 0,
                "total_return": 0
            }
        
        # Calculate stats
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        total_profit = sum(t['pnl'] for t in winning_trades)
        total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] <= 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        avg_trade_pnl = sum(t['pnl'] for t in self.trades) / total_trades if total_trades > 0 else 0
        total_return = sum(self.returns)
        
        # Calculate Sharpe ratio
        if len(self.returns) > 0:
            returns_array = np.array(self.returns)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) or 1e-10  # Avoid division by zero
            sharpe_ratio = mean_return / std_return * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        if len(self.returns) > 0:
            cumulative = np.cumprod(1 + np.array(self.returns))
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / peak
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        else:
            max_drawdown = 0
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "avg_trade_pnl": avg_trade_pnl,
            "total_return": total_return,
            "returns": self.returns
        } 