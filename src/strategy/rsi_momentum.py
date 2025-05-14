#!/usr/bin/env python3
"""
RSI Momentum Strategy

This strategy uses RSI (Relative Strength Index) to identify overbought and oversold conditions
and enter trades based on momentum shifts in trending markets.
"""

import pandas as pd
import numpy as np
from src.strategy.base_strategy import BaseStrategy
from src.utils.logger import logger

class RSIMomentumStrategy(BaseStrategy):
    """
    RSI Momentum Strategy
    
    Entry Criteria:
    - Long: RSI crosses above oversold threshold (default 30)
    - Short: RSI crosses below overbought threshold (default 70)
    
    Exit Criteria:
    - Long: RSI crosses above overbought level or trailing stop hit
    - Short: RSI crosses below oversold level or trailing stop hit
    
    Features:
    - Dynamic RSI thresholds based on market volatility
    - Optional confirmation using volume and/or price action
    - Compatible with pyramiding and dynamic position sizing
    """
    
    def __init__(self, config=None):
        """Initialize the RSI Momentum strategy with config parameters."""
        super().__init__(config=config)
        
        # Set strategy name explicitly
        self.name = "rsi_momentum"
        
        # RSI parameters - more aggressive thresholds
        self.rsi_period = config.get('rsi_period', 7)  # Shorter period, more signals (was 14)
        self.rsi_long_threshold = config.get('rsi_long_threshold', 35)  # Higher threshold for longs (was 30)
        self.rsi_short_threshold = config.get('rsi_short_threshold', 65)  # Lower threshold for shorts (was 70)
        self.rsi_exit_long_threshold = config.get('rsi_exit_long_threshold', 65)  # Lower exit threshold (was 70)
        self.rsi_exit_short_threshold = config.get('rsi_exit_short_threshold', 35)  # Higher exit threshold (was 30)
        
        # Trend filter parameters
        self.use_trend_filter = config.get('use_trend_filter', True)  # Enable by default
        self.ema_trend_period = config.get('ema_trend_period', 50)  # Shorter trend filter (was 100)
        
        # Volume confirmation
        self.use_volume_filter = config.get('use_volume_filter', False)
        self.volume_period = config.get('volume_period', 20)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        
        # Adaptive parameters
        self.use_adaptive_thresholds = config.get('use_adaptive_thresholds', False)
        self.volatility_lookback = config.get('volatility_lookback', 20)
        
        # Additional attributes for tracking
        self.previous_rsi = None
        self.position_size = 0
        self.current_pyramid_entries = 0
        self.entry_prices = []
        self.trailing_stop_price = 0
        self.max_price = 0
        self.min_price = float('inf')
        self.trade_history = []
        
        logger.info(f"Initialized RSI Momentum Strategy with period {self.rsi_period}, "
                   f"thresholds {self.rsi_long_threshold}/{self.rsi_short_threshold}")
    
    def apply_indicators(self, data):
        """
        Apply RSI and other indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()
        
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate ATR if using ATR-based stops
        if self.use_atr_stops or self.use_trailing_stop:
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift(1)).abs()
            low_close = (df['low'] - df['close'].shift(1)).abs()
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=self.atr_period).mean()
        
        # Add trend filter if enabled
        if self.use_trend_filter:
            df['ema_trend'] = df['close'].ewm(span=self.ema_trend_period, adjust=False).mean()
        
        # Add volume filter if enabled
        if self.use_volume_filter:
            df['volume_ma'] = df['volume'].rolling(window=self.volume_period).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate adaptive thresholds if enabled
        if self.use_adaptive_thresholds:
            # Calculate historical RSI volatility
            rsi_std = df['rsi'].rolling(window=self.volatility_lookback).std()
            
            # Adjust thresholds based on volatility
            # Higher volatility -> wider thresholds
            volatility_factor = 1 + (rsi_std / 100)
            
            # Apply adaptive thresholds with reasonable limits
            df['rsi_long_threshold'] = np.maximum(20, self.rsi_long_threshold * volatility_factor)
            df['rsi_short_threshold'] = np.minimum(80, self.rsi_short_threshold * volatility_factor)
        
        return df
    
    def get_signal(self, data):
        """
        Generate trading signals based on RSI values.
        
        Args:
            data: DataFrame with price data and indicators
            
        Returns:
            str: "buy", "sell", or "" (no action)
        """
        if len(data) < self.rsi_period + 2:
            return ""  # Not enough data
        
        # Get the current and previous candle
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Store current RSI for state tracking
        current_rsi = current['rsi']
        previous_rsi = previous['rsi']
        
        # Default signal is no action
        signal = ""
        
        # Check if we have an open position
        has_position = self.position != 0
        
        # Determine adaptive thresholds if enabled
        if self.use_adaptive_thresholds and 'rsi_long_threshold' in current:
            long_threshold = current['rsi_long_threshold']
            short_threshold = current['rsi_short_threshold']
        else:
            long_threshold = self.rsi_long_threshold
            short_threshold = self.rsi_short_threshold
        
        # Check for entry conditions (no position or pyramiding enabled)
        if not has_position or (self.use_pyramiding and self.current_pyramid_entries < self.max_pyramid_entries):
            # Long entry signals - More entry conditions added
            
            # 1. Classic RSI oversold cross
            if previous_rsi < long_threshold and current_rsi >= long_threshold:
                signal = "buy"
            
            # 2. RSI bullish divergence: price makes lower low but RSI makes higher low
            elif len(data) >= 4:
                # Look at last few candles for divergence
                candle_3 = data.iloc[-4]
                candle_2 = data.iloc[-3]
                candle_1 = previous
                candle_0 = current
                
                # Check for bullish divergence
                if (candle_2['low'] > candle_3['low'] and  # Lower low in price
                    candle_0['low'] < candle_2['low'] and
                    candle_0['rsi'] > candle_2['rsi'] and  # Higher low in RSI
                    current_rsi < 45):  # RSI still relatively low
                    
                    # Check trend filter if enabled
                    trend_ok = True
                    if self.use_trend_filter:
                        trend_ok = current['close'] > current['ema_trend']
                        
                    if trend_ok:
                        signal = "buy"
            
            # 3. RSI momentum shift (RSI bouncing off extreme lows)
            elif current_rsi < 30 and current_rsi > previous_rsi + 5:  # Sharp upward momentum from extreme lows
                signal = "buy"
                    
            # Short entry signals
            
            # 1. Classic RSI overbought cross
            if previous_rsi > short_threshold and current_rsi <= short_threshold:
                signal = "sell"
                
            # 2. RSI bearish divergence: price makes higher high but RSI makes lower high
            elif len(data) >= 4:
                # Look at last few candles for divergence
                candle_3 = data.iloc[-4]
                candle_2 = data.iloc[-3]
                candle_1 = previous
                candle_0 = current
                
                # Check for bearish divergence
                if (candle_2['high'] < candle_3['high'] and  # Higher high in price
                    candle_0['high'] > candle_2['high'] and
                    candle_0['rsi'] < candle_2['rsi'] and  # Lower high in RSI
                    current_rsi > 55):  # RSI still relatively high
                    
                    # Check trend filter if enabled
                    trend_ok = True
                    if self.use_trend_filter:
                        trend_ok = current['close'] < current['ema_trend']
                        
                    if trend_ok:
                        signal = "sell"
            
            # 3. RSI momentum shift (RSI dropping from extreme highs)
            elif current_rsi > 70 and current_rsi < previous_rsi - 5:  # Sharp downward momentum from extreme highs
                signal = "sell"
                
            # Apply filters to signal if we have one
            if signal:
                # Volume filter - only apply if signal exists
                volume_ok = True
                if self.use_volume_filter:
                    volume_ok = current['volume_ratio'] > self.volume_threshold
                
                if not volume_ok:
                    signal = ""  # Cancel signal if volume is insufficient
        
        # Check exit conditions if we have a position
        elif has_position:
            if self.position > 0:  # Long position
                # Exit long if RSI crosses above overbought level
                if previous_rsi < self.rsi_exit_long_threshold and current_rsi >= self.rsi_exit_long_threshold:
                    signal = "sell"  # Exit long
            else:  # Short position
                # Exit short if RSI crosses below oversold level
                if previous_rsi > self.rsi_exit_short_threshold and current_rsi <= self.rsi_exit_short_threshold:
                    signal = "buy"  # Exit short
        
        return signal
    
    def check_exit_conditions(self, bar):
        """
        Check if any exit conditions are met.
        
        Args:
            bar: Current price bar with indicators
            
        Returns:
            str: Exit reason or None if no exit
        """
        if not self.has_position():
            return None
        
        current_price = bar['close']
        
        # Update trailing stops and price extremes
        if self.position > 0:  # Long position
            # Update maximum price seen
            if current_price > self.max_price:
                self.max_price = current_price
                
                # Update trailing stop if activated
                activation_pct = self.config.get('trail_activation_pct', 0.01)
                if current_price >= self.entry_price * (1 + activation_pct):
                    atr_multiplier = self.config.get('trail_atr_multiplier', 1.5)
                    atr_stop = current_price - (bar['atr'] * atr_multiplier)
                    self.trailing_stop_price = max(self.trailing_stop_price, atr_stop)
            
            # Check if trailing stop was hit
            if self.trailing_stop_price > 0 and current_price < self.trailing_stop_price:
                return "trailing_stop"
                
        else:  # Short position
            # Update minimum price seen
            if current_price < self.min_price:
                self.min_price = current_price
                
                # Update trailing stop if activated
                activation_pct = self.config.get('trail_activation_pct', 0.01)
                if current_price <= self.entry_price * (1 - activation_pct):
                    atr_multiplier = self.config.get('trail_atr_multiplier', 1.5)
                    atr_stop = current_price + (bar['atr'] * atr_multiplier)
                    # For shorts, we want the lowest trailing stop
                    if self.trailing_stop_price == 0 or atr_stop < self.trailing_stop_price:
                        self.trailing_stop_price = atr_stop
            
            # Check if trailing stop was hit
            if self.trailing_stop_price > 0 and current_price > self.trailing_stop_price:
                return "trailing_stop"
        
        # Check stop loss
        if self.stop_loss > 0:
            if self.position > 0 and current_price < self.stop_loss:  # Long position
                return "stop_loss"
            elif self.position < 0 and current_price > self.stop_loss:  # Short position
                return "stop_loss"
        
        # Check take profit
        if self.take_profit > 0:
            if self.position > 0 and current_price > self.take_profit:  # Long position
                return "take_profit"
            elif self.position < 0 and current_price < self.take_profit:  # Short position
                return "take_profit"
        
        return None
    
    def calculate_position_size(self, price):
        """
        Calculate position size based on current equity and risk parameters.
        
        Args:
            price (float): Current price
            
        Returns:
            float: Position size in base units
        """
        if not self.account:
            return 0
        
        # Basic position sizing - risk a percentage of equity per trade
        equity = self.account.equity
        risk_amount = equity * self.risk_per_trade
        
        # Use ATR for more precise risk calculation if available
        if self.use_atr_stops and hasattr(self, 'last_atr') and self.last_atr:
            # Risk is based on distance to stop loss in dollar terms
            price_risk = self.last_atr * self.atr_sl_multiplier
            position_size = risk_amount / price_risk
        else:
            # Fallback to percentage-based sizing
            position_size = risk_amount / price
        
        # Apply pyramiding factor if this is a pyramid entry
        if self.use_pyramiding and self.current_pyramid_entries > 0:
            position_size *= (self.pyramid_factor ** self.current_pyramid_entries)
        
        # Cap position size by max allocation percentage
        max_position_size = (equity * self.max_position_pct) / price
        position_size = min(position_size, max_position_size)
        
        return position_size
    
    def record_trade(self, trade_type, price, qty):
        """
        Record trade details for tracking.
        
        Args:
            trade_type (str): "entry" or "exit"
            price (float): Trade price
            qty (float): Trade quantity
        """
        if trade_type == "entry":
            self.entry_price = price
            self.entry_time = pd.Timestamp.now()
            
            # Initialize tracking variables for trailing stop
            if qty > 0:  # Long
                self.max_price = price
            else:  # Short
                self.min_price = price
            
            self.trailing_stop_price = 0
            
            # Increment pyramid counter if pyramiding is enabled
            if self.use_pyramiding:
                self.current_pyramid_entries += 1
                self.entry_prices.append(price)
        
        else:  # Exit
            # Reset tracking variables
            self.trailing_stop_price = 0
            self.max_price = 0
            self.min_price = float('inf')
            
            # Reset pyramid counter
            if self.use_pyramiding:
                self.current_pyramid_entries = 0
                self.entry_prices = []
    
    def update_trade_history(self, trade):
        """
        Update strategy's trade history with a completed trade.
        
        Args:
            trade (dict): Completed trade information
        """
        self.trade_history.append(trade)
        
        # Recalculate win rate and other metrics
        if len(self.trade_history) > 0:
            wins = sum(1 for t in self.trade_history if t['pnl'] > 0)
            self.win_rate = wins / len(self.trade_history)
    
    def reset(self):
        """Reset strategy state for a new backtest."""
        super().reset()
        self.previous_rsi = None
        self.trailing_stop_price = 0
        self.max_price = 0
        self.min_price = float('inf')
        self.current_pyramid_entries = 0
        self.entry_prices = []
        self.trade_history = [] 