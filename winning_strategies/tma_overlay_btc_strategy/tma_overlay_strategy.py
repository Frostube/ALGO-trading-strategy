#!/usr/bin/env python3
"""
TMA Overlay Strategy

A strategy based on the Triangular Moving Average (TMA) Overlay indicator
with dynamic bands and momentum engulfing signals. This strategy combines
noise-filtering TMA bands with EMA confirmation for improved signal quality.

Inspired by the TradingView indicator "TMA Overlay" by ArtyFXC.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytz

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from src.strategy.enhanced_strategy import EnhancedStrategy

class TMAOverlayStrategy(EnhancedStrategy):
    """
    TMA Overlay Strategy combining Triangular Moving Average bands with EMA confirmation.
    
    This strategy uses TMA bands for noise-filtering and dynamic support/resistance levels,
    along with EMA crossover confirmation and engulfing candle patterns for entry signals.
    """
    
    def __init__(self, custom_config=None):
        """
        Initialize the strategy with the TMA Overlay parameters
        
        Args:
            custom_config (dict, optional): Any custom configuration to override defaults
        """
        # Load the configuration
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            default_config = json.load(f)
        
        # Create initial configuration
        config = {
            # Base strategy settings
            'name': default_config["strategy_name"],
            'symbol': default_config["traded_symbol"],
            'timeframe': default_config["timeframe"],
            
            # TMA parameters
            'tma_period': default_config["tma_parameters"]["tma_period"],
            'atr_multiplier': default_config["tma_parameters"]["atr_multiplier"],
            'use_std_dev': default_config["tma_parameters"]["use_std_dev"],
            'std_dev_multiplier': default_config["tma_parameters"]["std_dev_multiplier"],
            
            # EMA parameters
            'fast_period': default_config["ema_parameters"]["fast_period"],
            'slow_period': default_config["ema_parameters"]["slow_period"],
            
            # Signal parameters
            'use_tma_direction_filter': default_config["signal_parameters"]["use_tma_direction_filter"],
            'use_engulfing_filter': default_config["signal_parameters"]["use_engulfing_filter"],
            'use_ema_confirmation': default_config["signal_parameters"]["use_ema_confirmation"],
            
            # Exit parameters
            'use_band_targets': default_config["exit_parameters"]["use_band_targets"],
            'use_dynamic_trailing_stop': default_config["exit_parameters"]["use_dynamic_trailing_stop"],
            'use_multi_tier_exits': default_config["exit_parameters"]["use_multi_tier_exits"],
            'profit_tiers': default_config["exit_parameters"]["profit_tiers"],
            'position_scale_out': default_config["exit_parameters"]["position_scale_out"],
            
            # Risk parameters
            'stop_loss_pct': default_config["risk_parameters"]["stop_loss_pct"],
            'take_profit_pct': default_config["risk_parameters"]["take_profit_pct"],
            'risk_per_trade': default_config["risk_parameters"]["risk_per_trade"],
            'position_size_method': default_config["risk_parameters"]["position_size_method"],
            'atr_period': default_config["risk_parameters"]["atr_period"],
            'atr_stop_multiplier': default_config["risk_parameters"]["atr_stop_multiplier"],
            'atr_target_multiplier': default_config["risk_parameters"]["atr_target_multiplier"],
            
            # Enhanced filters
            'use_htf_filter': default_config["enhanced_filters"]["use_htf_filter"],
            'htf_period': default_config["enhanced_filters"]["htf_period"],
            'use_candle_confirmation': default_config["enhanced_filters"]["use_candle_confirmation"],
            'candle_body_threshold': default_config["enhanced_filters"]["candle_body_threshold"],
            
            # Session parameters
            'timezone': default_config["session_parameters"]["timezone"],
            'session_start_hour': default_config["session_parameters"]["session_start_hour"],
            'session_start_minute': default_config["session_parameters"]["session_start_minute"],
            'session_end_hour': default_config["session_parameters"]["session_end_hour"],
            'session_end_minute': default_config["session_parameters"]["session_end_minute"],
            'use_session_filter': default_config["session_parameters"]["use_session_filter"]
        }
        
        # Override with any custom configuration
        if custom_config:
            config.update(custom_config)
            
        # Initialize parent class
        super().__init__(config)
        
        self.name = config['name']
        
    def populate_indicators(self, df):
        """
        Calculate required technical indicators for the strategy
        
        Args:
            df (pd.DataFrame): Price data
            
        Returns:
            pd.DataFrame: DataFrame with indicators added
        """
        # Make a copy of the dataframe to avoid warnings
        df = df.copy()
        
        # Calculate EMA indicators for trend direction
        df['ema_fast'] = df['close'].ewm(span=self.config['fast_period'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.config['slow_period'], adjust=False).mean()
        df['ema_signal'] = df['ema_fast'] - df['ema_slow']
        
        # Calculate Triangular Moving Average (TMA)
        # TMA is essentially an SMA of an SMA
        sma1 = df['close'].rolling(window=self.config['tma_period']).mean()
        df['tma_mid'] = sma1.rolling(window=self.config['tma_period']).mean()
        
        # Calculate TMA bands
        if self.config['use_std_dev']:
            # Use standard deviation for bands
            df['tma_band_width'] = df['close'].rolling(window=self.config['tma_period']).std() * self.config['std_dev_multiplier']
        else:
            # Use ATR for bands
            df['atr'] = self.calculate_atr(df, self.config['atr_period'])
            df['tma_band_width'] = df['atr'] * self.config['atr_multiplier']
        
        # Calculate upper and lower bands
        df['tma_upper'] = df['tma_mid'] + df['tma_band_width']
        df['tma_lower'] = df['tma_mid'] - df['tma_band_width']
        
        # Fill NaN values in tma columns with close price to avoid signal issues
        # This allows early signals before TMA stabilizes but will be more noise-prone
        df['tma_mid'] = df['tma_mid'].fillna(df['close'])
        df['tma_upper'] = df['tma_upper'].fillna(df['close'] * 1.01)  # 1% above close
        df['tma_lower'] = df['tma_lower'].fillna(df['close'] * 0.99)  # 1% below close
        
        # Calculate engulfing patterns on the bands
        df['engulf_upper'] = (df['close'] > df['tma_upper']) & (df['open'] < df['tma_upper'])
        df['engulf_lower'] = (df['close'] < df['tma_lower']) & (df['open'] > df['tma_lower'])
        
        # Calculate tma band touches 
        df['touch_upper'] = (df['high'] >= df['tma_upper']) & (df['close'] < df['tma_upper'])
        df['touch_lower'] = (df['low'] <= df['tma_lower']) & (df['close'] > df['tma_lower'])
        
        # Session filter
        if self.config['use_session_filter']:
            df['in_session'] = self.is_in_session(df.index)
        else:
            df['in_session'] = True
            
        return df
    
    def is_in_session(self, index):
        """
        Check if the time is within the configured trading session
        
        Args:
            index: DataFrame index with datetime values
            
        Returns:
            pd.Series: Boolean series indicating if time is in session
        """
        tz = pytz.timezone(self.config['timezone'])
        session_start = time(self.config['session_start_hour'], self.config['session_start_minute'])
        session_end = time(self.config['session_end_hour'], self.config['session_end_minute'])
        
        # Convert index to session timezone
        local_time = pd.Series([dt.astimezone(tz).time() for dt in index], index=index)
        
        # Check if in session
        in_session = (local_time >= session_start) & (local_time <= session_end)
        return in_session
    
    def calculate_atr(self, df, period=14):
        """
        Calculate Average True Range
        
        Args:
            df (pd.DataFrame): Price data with high, low, close
            period (int): ATR period
            
        Returns:
            pd.Series: ATR values
        """
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        # True Range calculation
        tr1 = high - low  # Current high - current low
        tr2 = (high - close).abs()  # Current high - previous close
        tr3 = (low - close).abs()  # Current low - previous close
        
        # True Range is the maximum of the three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of TR
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def generate_signals(self, df):
        """
        Generate buy and sell signals based on TMA Overlay and EMA indicators
        
        Args:
            df (pd.DataFrame): DataFrame with indicators
            
        Returns:
            pd.DataFrame: DataFrame with buy and sell signals
        """
        # Make a copy of the dataframe to avoid warnings
        df = df.copy()
        
        # Initialize signal columns
        df['buy_signal'] = 0
        df['sell_signal'] = 0
        df['exit_long'] = 0
        df['exit_short'] = 0
        
        # EMA crossover signals if enabled
        if self.config['use_ema_confirmation']:
            df['ema_buy'] = (df['ema_signal'] > 0) & (df['ema_signal'].shift(1) <= 0)
            df['ema_sell'] = (df['ema_signal'] < 0) & (df['ema_signal'].shift(1) >= 0)
        else:
            df['ema_buy'] = True
            df['ema_sell'] = True
        
        # Generate signals combining indicators
        for i in range(1, len(df)):
            # Skip processing if not in session and session filter is enabled
            if self.config['use_session_filter'] and not df['in_session'].iloc[i]:
                continue
                
            # Buy signal conditions
            buy_condition = True
            
            # TMA direction filter
            if self.config['use_tma_direction_filter']:
                # Price should be above the TMA midline for a buy
                buy_condition = buy_condition and (df['close'].iloc[i] > df['tma_mid'].iloc[i])
            
            # Engulfing filter
            if self.config['use_engulfing_filter']:
                # Check for engulfing at the lower band
                buy_condition = buy_condition and df['engulf_lower'].iloc[i]
            else:
                # If not using engulfing filter, just check for band touch or EMA confirmation
                buy_condition = buy_condition and (df['ema_buy'].iloc[i] or df['touch_lower'].iloc[i])
            
            # Set buy signal
            if buy_condition:
                df.loc[df.index[i], 'buy_signal'] = 1
                
            # Sell signal conditions
            sell_condition = True
            
            # TMA direction filter
            if self.config['use_tma_direction_filter']:
                # Price should be below the TMA midline for a sell
                sell_condition = sell_condition and (df['close'].iloc[i] < df['tma_mid'].iloc[i])
            
            # Engulfing filter
            if self.config['use_engulfing_filter']:
                # Check for engulfing at the upper band
                sell_condition = sell_condition and df['engulf_upper'].iloc[i]
            else:
                # If not using engulfing filter, just check for band touch or EMA confirmation
                sell_condition = sell_condition and (df['ema_sell'].iloc[i] or df['touch_upper'].iloc[i])
            
            # Set sell signal
            if sell_condition:
                df.loc[df.index[i], 'sell_signal'] = 1
                
            # Exit signals - exit when price crosses mid band in opposite direction
            # or when there's an engulfing pattern on the opposite band
            if pd.notna(df['tma_mid'].iloc[i]) and pd.notna(df['tma_mid'].iloc[i-1]):
                df.loc[df.index[i], 'exit_long'] = int(
                    (df['close'].iloc[i] < df['tma_mid'].iloc[i] and df['close'].iloc[i-1] > df['tma_mid'].iloc[i-1]) or
                    df['engulf_upper'].iloc[i]
                )
                df.loc[df.index[i], 'exit_short'] = int(
                    (df['close'].iloc[i] > df['tma_mid'].iloc[i] and df['close'].iloc[i-1] < df['tma_mid'].iloc[i-1]) or
                    df['engulf_lower'].iloc[i]
                )
        
        # Ensure we don't have buy and sell signals on the same candle
        conflicting_signals = (df['buy_signal'] == 1) & (df['sell_signal'] == 1)
        if conflicting_signals.any():
            print(f"Warning: {conflicting_signals.sum()} conflicting buy/sell signals detected and resolved")
            df.loc[conflicting_signals, 'buy_signal'] = 0
        
        return df
    
    def calculate_stop_loss(self, df, index, direction, entry_price):
        """
        Calculate dynamic stop loss based on TMA bands
        
        Args:
            df (pd.DataFrame): DataFrame with indicators
            index (int): Current row index
            direction (str): Trade direction ('long' or 'short')
            entry_price (float): Entry price
            
        Returns:
            float: Stop loss price
        """
        if not self.config['use_dynamic_trailing_stop']:
            # Simple percentage-based stop loss
            if direction == 'long':
                return entry_price * (1 - self.config['stop_loss_pct'])
            else:
                return entry_price * (1 + self.config['stop_loss_pct'])
        
        # Get ATR value
        atr = df['atr'].iloc[index] if not pd.isna(df['atr'].iloc[index]) else entry_price * 0.01
        
        # Calculate stop loss based on ATR and direction
        if direction == 'long':
            # For long positions, stop at lower band or ATR-based stop, whichever is higher (less risk)
            atr_stop = entry_price - (atr * self.config['atr_stop_multiplier'])
            band_stop = df['tma_lower'].iloc[index] if not pd.isna(df['tma_lower'].iloc[index]) else entry_price * 0.98
            
            # Ensure stop loss is not too far from entry (max 5%)
            min_stop = entry_price * 0.95
            return max(min(atr_stop, band_stop), min_stop)
        else:
            # For short positions, stop at upper band or ATR-based stop, whichever is lower (less risk)
            atr_stop = entry_price + (atr * self.config['atr_stop_multiplier'])
            band_stop = df['tma_upper'].iloc[index] if not pd.isna(df['tma_upper'].iloc[index]) else entry_price * 1.02
            
            # Ensure stop loss is not too far from entry (max 5%)
            max_stop = entry_price * 1.05
            return min(max(atr_stop, band_stop), max_stop)
    
    def calculate_take_profit(self, df, index, direction, entry_price):
        """
        Calculate dynamic take profit based on TMA bands
        
        Args:
            df (pd.DataFrame): DataFrame with indicators
            index (int): Current row index
            direction (str): Trade direction ('long' or 'short')
            entry_price (float): Entry price
            
        Returns:
            float: Take profit price
        """
        if not self.config['use_band_targets']:
            # Simple percentage-based take profit
            if direction == 'long':
                return entry_price * (1 + self.config['take_profit_pct'])
            else:
                return entry_price * (1 - self.config['take_profit_pct'])
        
        # If TMA values are not available, use percentage-based take profit
        if pd.isna(df['tma_mid'].iloc[index]) or pd.isna(df['tma_upper'].iloc[index]) or pd.isna(df['tma_lower'].iloc[index]):
            if direction == 'long':
                return entry_price * (1 + self.config['take_profit_pct'])
            else:
                return entry_price * (1 - self.config['take_profit_pct'])
        
        # Calculate take profit based on direction and TMA bands
        if direction == 'long':
            # For long positions, target is the upper band or midline + ATR multiple
            if df['close'].iloc[index] < df['tma_mid'].iloc[index]:
                # If entering below midline, target the midline
                return max(df['tma_mid'].iloc[index], entry_price * (1 + 0.01))
            else:
                # If already above midline, target the upper band
                return max(df['tma_upper'].iloc[index], entry_price * (1 + 0.02))
        else:
            # For short positions, target is the lower band or midline - ATR multiple
            if df['close'].iloc[index] > df['tma_mid'].iloc[index]:
                # If entering above midline, target the midline
                return min(df['tma_mid'].iloc[index], entry_price * (1 - 0.01))
            else:
                # If already below midline, target the lower band
                return min(df['tma_lower'].iloc[index], entry_price * (1 - 0.02))
    
    def calculate_position_size(self, df, index, direction, balance):
        """
        Calculate position size based on risk parameters and volatility
        
        Args:
            df (pd.DataFrame): DataFrame with indicators
            index (int): Current index in the dataframe
            direction (str): 'long' or 'short'
            balance (float): Current account balance
            
        Returns:
            float: Position size
        """
        if self.config['position_size_method'] != 'volatility_adjusted':
            # Use fixed percentage risk method
            entry_price = df['close'].iloc[index]
            stop_loss = self.calculate_stop_loss(df, index, direction, entry_price)
            
            # Calculate risk amount based on direction
            if direction == 'long':
                risk_amount = entry_price - stop_loss
            else:
                risk_amount = stop_loss - entry_price
            
            # Ensure risk amount is not too small
            if abs(risk_amount) < entry_price * 0.001:
                risk_amount = entry_price * 0.01  # Default to 1% risk
            
            # Calculate risk amount in account currency
            risk_in_currency = balance * self.config['risk_per_trade']
            
            # Position size in units
            position_size = risk_in_currency / abs(risk_amount)
            
            return position_size
        
        # Get ATR for volatility-based sizing
        atr = df['atr'].iloc[index] if not pd.isna(df['atr'].iloc[index]) else df['close'].iloc[index] * 0.01
        entry_price = df['close'].iloc[index]
        
        # Calculate stop loss based on our dynamic stop loss calculation
        stop_price = self.calculate_stop_loss(df, index, direction, entry_price)
        
        # Calculate risk amount based on direction
        if direction == 'long':
            risk_amount = entry_price - stop_price
        else:
            risk_amount = stop_price - entry_price
        
        # Ensure risk amount is not too small or NaN
        if pd.isna(risk_amount) or abs(risk_amount) < entry_price * 0.001:
            risk_amount = entry_price * 0.01  # Default to 1% risk
        
        # Calculate risk amount in account currency
        risk_in_currency = balance * self.config['risk_per_trade']
        
        # Calculate position size
        position_size = risk_in_currency / abs(risk_amount)
        
        # Normalize for the asset price to convert to units
        position_size = position_size / entry_price
        
        # Cap position size to a reasonable amount (max 20% of balance in value)
        max_position = (balance * 0.2) / entry_price
        position_size = min(position_size, max_position)
        
        return position_size
    
    @staticmethod
    def get_performance_metrics():
        """
        Get estimated performance metrics
        
        Returns:
            dict: Performance metrics
        """
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config["performance_metrics"]
        
    def process_candle(self, candle, data=None):
        """
        Process a new price candle and generate trade signals
        
        Args:
            candle (dict): Current price candle with OHLCV data
            data (pd.DataFrame): Historical data including this candle
            
        Returns:
            dict: Action result containing any trades executed
        """
        if data is None or data.empty:
            return {"action": "none", "reason": "No data available"}
            
        # Get the last row which should contain our signals
        last_row = data.iloc[-1]
        
        # Check if we have buy or sell signals
        if 'buy_signal' in last_row and last_row['buy_signal'] > 0:
            # Generate a buy signal
            entry_price = candle['close']
            
            # Execute buy trade
            result = self.enter_position(
                price=entry_price,
                position_type="long",
                reason="TMA Overlay buy signal"
            )
            
            return {
                "action": "enter",
                "direction": "long",
                "price": entry_price,
                "time": candle.get('timestamp', None),
                "result": result
            }
            
        elif 'sell_signal' in last_row and last_row['sell_signal'] > 0:
            # Generate a sell signal
            entry_price = candle['close']
            
            # Execute sell trade
            result = self.enter_position(
                price=entry_price,
                position_type="short",
                reason="TMA Overlay sell signal"
            )
            
            return {
                "action": "enter",
                "direction": "short",
                "price": entry_price,
                "time": candle.get('timestamp', None),
                "result": result
            }
            
        # Check for exit signals if we have an open position
        if hasattr(self, 'position') and self.position != 0:
            # Get exit signals
            exit_long = last_row.get('exit_long', 0) > 0
            exit_short = last_row.get('exit_short', 0) > 0
            
            # Check if we should exit
            if (self.position > 0 and exit_long) or (self.position < 0 and exit_short):
                result = self.exit_position(
                    price=candle['close'],
                    reason="TMA Overlay exit signal"
                )
                
                return {
                    "action": "exit",
                    "price": candle['close'],
                    "time": candle.get('timestamp', None),
                    "result": result
                }
                
        # If we have an open position, manage it
        if hasattr(self, 'manage_open_positions'):
            exit_result = self.manage_open_positions(candle)
            if exit_result:
                return exit_result
        
        # No action
        return {"action": "none", "reason": "No signal"}

def run_example():
    """
    Example showing how to use this strategy
    """
    from src.backtest.data_loader import DataLoader
    from src.backtest.backtest_enhanced import run_backtest
    
    # Create the strategy
    strategy = TMAOverlayStrategy()
    
    # Load sample data
    loader = DataLoader()
    data = loader.load_data("BTC/USDT", "1h", start_date="2025-04-17", end_date="2025-05-17")
    
    # Run a backtest
    results = run_backtest(data, strategy)
    
    # Print results
    print(f"Strategy: {strategy.name}")
    print(f"ROI: {results['roi']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    
    # Compare with estimated performance
    estimated_metrics = strategy.get_performance_metrics()
    print("\nEstimated Performance:")
    print(f"ROI: {estimated_metrics['estimated_roi']}%")
    print(f"Win Rate: {estimated_metrics['estimated_win_rate']}%")
    print(f"Profit Factor: {estimated_metrics['estimated_profit_factor']}")
    print(f"Sharpe Ratio: {estimated_metrics['estimated_sharpe_ratio']}")
    print(f"Max Drawdown: {estimated_metrics['estimated_max_drawdown']}%")
    
if __name__ == "__main__":
    run_example() 