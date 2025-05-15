#!/usr/bin/env python3
"""
EMA Crossover Strategy Module

Implementation of the EMA Crossover strategy inspired by the BEC (Bot EMA Cross) project.
This strategy:
1. Finds optimal EMA pairs for a given symbol and timeframe
2. Generates buy/sell signals based on EMA crossovers
3. Implements position management with stop-loss and take-profit
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, RISK_PER_TRADE,
    USE_ATR_STOPS, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    VOL_TARGET_PCT, MAX_POSITION_PCT, TRAIL_ACTIVATION_PCT,
    VOL_RATIO_MIN, RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT
)
from src.utils.logger import logger, log_trade
from src.strategy.ema_optimizer import find_best_ema_pair, fetch_historical_data
from src.db.models import Trade
from src.strategy.base_strategy import BaseStrategy
from src.utils.metrics import profit_factor

# ATR trailing stop parameters
ATR_TRAIL_START = 1.25   # Normal trailing stop multiplier
ATR_TRAIL_LOSER = 1.00   # Tighter trailing stop for losing trades
R_TRIGGER = 0.5          # Adverse move in R-multiples to trigger tighter trail

class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover strategy that automatically finds optimal parameters.
    """
    
    def __init__(self, symbol=SYMBOL, timeframe='4h', db_session=None, account_balance=1000.0, 
                history_days=365, auto_optimize=False, config=None,
                fast_ema=5, slow_ema=13, trend_ema=50, atr_sl_multiplier=1.0,
                risk_per_trade=0.0075, use_volatility_sizing=True, vol_target_pct=0.0075,
                enable_pyramiding=False, max_pyramid_entries=2, health_monitor=None,
                atr_trail_multiplier=1.25, atr_tp_multiplier=None, breakeven_trigger_r=0.5,
                pyramid_threshold=0.5, pyramid_position_scale=0.5, min_hold_bars=0,
                **kwargs):  # Add **kwargs to accept any additional parameters
        # Initialize the base class first
        super().__init__(config)
        
        # Handle parameter aliases from grid search
        # If rsi_period is in kwargs but not explicitly provided, use it
        if 'rsi_period' in kwargs and 'rsi_length' not in kwargs:
            kwargs['rsi_length'] = kwargs.pop('rsi_period')
        if 'rsi_oversold' in kwargs and 'rsi_lower' not in kwargs:
            kwargs['rsi_lower'] = kwargs.pop('rsi_oversold')
        if 'rsi_overbought' in kwargs and 'rsi_upper' not in kwargs:
            kwargs['rsi_upper'] = kwargs.pop('rsi_overbought')
            
        # Log any unexpected kwargs for debugging
        if kwargs:
            logger.debug(f"Additional parameters received: {kwargs}")
        
        # Set strategy-specific attributes
        self.symbol = symbol
        self.timeframe = timeframe  # Default changed to 4h
        self.db_session = db_session
        self.account_balance = account_balance
        self.active_trade = None
        self.consecutive_sl_count = 0
        self.last_signal_time = None
        self.history_days = history_days
        self.health_monitor = health_monitor
        self.min_bars_between_trades = 2  # Default 2 bars between trades
        
        # Get EMA parameters - either from config or explicitly provided
        if config:
            self.fast_ema = config.get('ema_fast', fast_ema)
            self.slow_ema = config.get('ema_slow', slow_ema)
            self.trend_ema = config.get('ema_trend', trend_ema)
            self.atr_sl_multiplier = config.get('atr_sl_multiplier', atr_sl_multiplier)
            self.atr_tp_multiplier = config.get('atr_tp_multiplier', atr_tp_multiplier)
            self.atr_trail_multiplier = config.get('atr_trail_multiplier', atr_trail_multiplier)
            self.risk_per_trade = config.get('risk_per_trade', risk_per_trade)
            self.use_volatility_sizing = config.get('use_volatility_sizing', use_volatility_sizing)
            self.vol_target_pct = config.get('vol_target_pct', vol_target_pct)
            self.enable_pyramiding = config.get('enable_pyramiding', enable_pyramiding)
            self.max_pyramid_entries = config.get('max_pyramid_entries', max_pyramid_entries)
            self.pyramid_threshold = config.get('pyramid_threshold', pyramid_threshold)
            self.pyramid_position_scale = config.get('pyramid_position_scale', pyramid_position_scale)
            self.breakeven_trigger_r = config.get('breakeven_trigger_r', breakeven_trigger_r)
            self.min_hold_bars = config.get('min_hold_bars', min_hold_bars)
            
            # RSI filter parameters
            self.use_rsi_filter = config.get('use_rsi_filter', True)
            self.rsi_period = config.get('rsi_period', RSI_PERIOD)
            self.rsi_oversold = config.get('rsi_oversold', RSI_OVERSOLD)
            self.rsi_overbought = config.get('rsi_overbought', RSI_OVERBOUGHT)
            
            # Volume filter parameters
            self.use_volume_filter = config.get('use_volume_filter', True)
            self.volume_threshold = config.get('volume_threshold', VOL_RATIO_MIN)
            
            # Dynamic trend filter parameters
            self.use_dynamic_trend_filter = config.get('use_dynamic_trend_filter', True)
            self.atr_low_threshold = config.get('atr_low_threshold', 0.015)  # 1.5% of price
            
            # IMPROVEMENT 4: Outlier volatility filter
            self.use_volatility_filter = config.get('use_volatility_filter', True)
            self.volatility_threshold_factor = config.get('volatility_threshold_factor', 0.75)
            
            # Volatility regime thresholds
            self.high_vol_threshold = config.get('high_vol_threshold', 0.010)  # 1.0% ATR/price
            self.medium_vol_threshold = config.get('medium_vol_threshold', 0.005)  # 0.5% ATR/price
            self.current_regime = "medium_vol"
            self.atr_pct = 0
            
            # Parameter adjustments for different regimes
            self.regime_adjustments = {
                "low_vol": {
                    "atr_sl_multiplier": 1.5,    # Tighter stops in low vol
                    "atr_trail_multiplier": 4.0,  # Wider trailing stops to catch trends
                    "breakeven_trigger_r": 0.2,   # More room before breakeven
                },
                "medium_vol": {
                    "atr_sl_multiplier": self.atr_sl_multiplier,  # Use default
                    "atr_trail_multiplier": self.atr_trail_multiplier,  # Use default
                    "breakeven_trigger_r": self.breakeven_trigger_r,  # Use default
                },
                "high_vol": {
                    "atr_sl_multiplier": 3.0,     # Wider stops in high vol
                    "atr_trail_multiplier": 2.0,   # Tighter trailing to lock in profits
                    "breakeven_trigger_r": 0.05,   # Move to breakeven faster
                }
            }
        else:
            # Use explicitly provided parameters
            self.fast_ema = fast_ema
            self.slow_ema = slow_ema
            self.trend_ema = trend_ema
            self.atr_sl_multiplier = atr_sl_multiplier
            self.atr_tp_multiplier = atr_tp_multiplier  # Can be None for trailing stop only
            self.atr_trail_multiplier = atr_trail_multiplier  # IMPROVEMENT 1: Wider trail (1.25x default)
            self.risk_per_trade = risk_per_trade
            self.use_volatility_sizing = use_volatility_sizing
            self.vol_target_pct = vol_target_pct
            self.enable_pyramiding = enable_pyramiding
            self.max_pyramid_entries = max_pyramid_entries
            self.pyramid_threshold = pyramid_threshold
            self.pyramid_position_scale = pyramid_position_scale
            self.breakeven_trigger_r = breakeven_trigger_r
            self.min_hold_bars = min_hold_bars
            
            # RSI filter parameters
            self.use_rsi_filter = True
            self.rsi_period = RSI_PERIOD
            self.rsi_oversold = RSI_OVERSOLD
            self.rsi_overbought = RSI_OVERBOUGHT
            
            # Volume filter parameters
            self.use_volume_filter = True
            self.volume_threshold = VOL_RATIO_MIN
            
            # Dynamic trend filter parameters
            self.use_dynamic_trend_filter = True
            self.atr_low_threshold = 0.015  # 1.5% of price
            
            # IMPROVEMENT 4: Outlier volatility filter
            self.use_volatility_filter = True
            self.volatility_threshold_factor = 0.75  # Filter if volatility < 75% of 30-day average
            
            # Volatility regime thresholds
            self.high_vol_threshold = 0.010  # 1.0% ATR/price
            self.medium_vol_threshold = 0.005  # 0.5% ATR/price
            self.current_regime = "medium_vol"
            self.atr_pct = 0
            
            # Parameter adjustments for different regimes
            self.regime_adjustments = {
                "low_vol": {
                    "atr_sl_multiplier": 1.5,    # Tighter stops in low vol
                    "atr_trail_multiplier": 4.0,  # Wider trailing stops to catch trends
                    "breakeven_trigger_r": 0.2,   # More room before breakeven
                },
                "medium_vol": {
                    "atr_sl_multiplier": self.atr_sl_multiplier,  # Use default
                    "atr_trail_multiplier": self.atr_trail_multiplier,  # Use default
                    "breakeven_trigger_r": self.breakeven_trigger_r,  # Use default
                },
                "high_vol": {
                    "atr_sl_multiplier": 3.0,     # Wider stops in high vol
                    "atr_trail_multiplier": 2.0,   # Tighter trailing to lock in profits
                    "breakeven_trigger_r": 0.05,   # Move to breakeven faster
                }
            }
        
        # Automatically find optimal EMA pair if requested
        if auto_optimize:
            self.optimize_ema_parameters()
            
        logger.info(f"Initialized EMA Crossover Strategy with EMA{self.fast_ema}/{self.slow_ema} on {self.timeframe} timeframe")
    
    def optimize_ema_parameters(self):
        """Find the optimal EMA parameters through backtesting"""
        logger.info(f"Finding optimal EMA parameters for {self.symbol} on {self.timeframe} timeframe...")
        
        try:
            # Updated to use the new grid search-based optimizer
            from src.strategy.ema_optimizer import find_best_ema_pair
            
            fast, slow, stats = find_best_ema_pair(
                symbol=self.symbol,
                timeframe=self.timeframe,
                history_days=self.history_days
            )
            
            if stats is not None:
                logger.info(f"Found optimal EMA parameters: EMA{fast}/{slow}")
                logger.info(f"Backtest results: Profit Factor: {stats['profit_factor']:.2f}, "
                           f"Win Rate: {stats['win_rate']*100:.2f}%, "
                           f"Return: {stats['total_return']*100:.2f}%, "
                           f"Trades: {stats['total_trades']}")
                
                # Update strategy parameters
                self.fast_ema = fast
                self.slow_ema = slow
                
                # Get trend EMA and ATR multiplier if available
                if isinstance(stats, dict) and 'trend_ema' in stats:
                    if stats['trend_ema'] is not None:
                        self.trend_ema = stats['trend_ema']
                        logger.info(f"Using trend EMA: {self.trend_ema}")
                
                if isinstance(stats, dict) and 'atr_mult' in stats:
                    self.atr_sl_multiplier = stats['atr_mult']
                    logger.info(f"Using optimized ATR multiplier: {self.atr_sl_multiplier}")
            else:
                logger.warning(f"No optimal EMA parameters found. Using defaults: EMA{self.fast_ema}/{self.slow_ema}")
        except Exception as e:
            logger.error(f"Error optimizing EMA parameters: {str(e)}")
            logger.warning(f"Using default EMA parameters: EMA{self.fast_ema}/{self.slow_ema}")
    
    def update(self, df, higher_tf_df=None):
        """
        Update the strategy with new data and generate signals.
        
        Args:
            df: DataFrame with OHLCV data
            higher_tf_df: Optional DataFrame with higher timeframe data
            
        Returns:
            dict: Current strategy state and signals
        """
        # Apply EMA indicators to the data
        df_with_indicators = self.apply_indicators(df)
        
        # Apply indicators to higher timeframe data if provided
        higher_tf_with_indicators = None
        if higher_tf_df is not None:
            higher_tf_with_indicators = self.apply_indicators(higher_tf_df)
        
        # Get the latest signal
        signal = self.get_signal(df_with_indicators, higher_tf_df=higher_tf_with_indicators)
        
        # Update active trade if exists
        if self.active_trade:
            self._update_active_trade(df_with_indicators.iloc[-1])
            
        # Check for new trade signals if no active trade
        if not self.active_trade and signal != '':
            self._open_trade(signal, df_with_indicators.iloc[-1])
            
            # Update last signal time
            if self.active_trade:  # If a trade was actually opened
                self.last_signal_time = df_with_indicators.iloc[-1].name
        
        return {
            'signal': signal,
            'active_trade': self.active_trade,
            'account_balance': self.account_balance
        }
    
    def apply_indicators(self, df):
        """
        Apply technical indicators to the dataframe.
        
        Args:
            df (DataFrame): OHLCV dataframe
            
        Returns:
            DataFrame: OHLCV dataframe with indicators added
        """
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()
        
        # Calculate crossover signals
        df['ema_crossover'] = 0
        for i in range(1, len(df)):
            if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
                df.loc[df.index[i], 'ema_crossover'] = 1  # Bullish crossover
            elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
                df.loc[df.index[i], 'ema_crossover'] = -1  # Bearish crossover
        
        # Calculate ATR for stop loss placement
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_period = getattr(self, 'atr_period', 14)  # Default to 14 if not defined
        df['atr'] = true_range.rolling(window=atr_period).mean()
        
        # Calculate ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        # Calculate RSI
        rsi_period = getattr(self, 'rsi_period', 14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
        avg_loss = loss.ewm(com=rsi_period-1, min_periods=rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def _calculate_rsi(self, df, period=14):
        """Calculate RSI indicator"""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Get gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gains and losses with exponential moving average
        avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
        
        # Calculate relative strength
        rs = avg_gain / avg_loss.replace(0, 1e-10)  # Avoid division by zero
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range for volatility-based stops"""
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate percentage ATR (ATR relative to price)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Clean up temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df
    
    def _calculate_volume_indicators(self, df, period=20):
        """Calculate volume indicators for confirmation"""
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=period).mean()
        
        # Calculate volume ratio
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume spike indicator (True when volume > 1.5 * MA)
        df['volume_spike'] = df['volume_ratio'] > 1.5
        
        return df
    
    def get_signal(self, df, index=-1, higher_tf_df=None):
        """
        Generate trading signal based on the latest indicators.
        
        Args:
            df: DataFrame with OHLCV data
            index: Index to get signal for (-1 for latest)
            higher_tf_df: Optional DataFrame with higher timeframe indicators
            
        Returns:
            str: Signal - "buy", "sell", or ""
        """
        # Apply indicators if they don't exist yet
        if 'ema_trend' not in df.columns:
            df = self.apply_indicators(df)
        
        # Get the row for the specified index
        if isinstance(index, int):
            row = df.iloc[index]
        else:
            row = df.loc[index]
        
        # Initialize signal
        signal_type = ""
        signal_reasons = []
        
        # Avoid trading with insufficient data (need at least 50 bars for reliable indicators)
        if len(df) < 50:
            return ""
        
        # EMA crossover signals without trend filter
        if 'ema_crossover' in df.columns:
            crossover_value = row['ema_crossover']
            
            # Check for buy signal - bullish crossover (fast EMA crosses above slow EMA)
            if crossover_value == 1:
                signal_type = "buy"
                signal_reasons.append(f"EMA Crossover: Fast({self.fast_ema}) > Slow({self.slow_ema})")
                
                # 200-EMA trend filter disabled
                # Keeping original volatility check logic intact
                if self.use_dynamic_trend_filter:
                    atr_pct = row['atr_pct']
                    if atr_pct > self.atr_low_threshold:
                        # In higher volatility - 200 EMA filter disabled
                        signal_reasons.append(f"200-EMA filter disabled for more signals")
                    else:
                        # In low volatility, ignore trend filter
                        signal_reasons.append(f"ATR({atr_pct:.4f}) < {self.atr_low_threshold:.4f} - Low volatility, trend filter bypassed")
                
                # Apply additional filters if enabled
                if signal_type and self.use_rsi_filter:
                    rsi_value = row['rsi']
                    if rsi_value < self.rsi_overbought:  # Check RSI is not overbought
                        signal_reasons.append(f"RSI({rsi_value:.1f}) < {self.rsi_overbought}")
                    else:
                        signal_type = ""  # RSI filter failed
                        signal_reasons.append(f"RSI too high: {rsi_value:.1f} >= {self.rsi_overbought}")
                
                if signal_type and self.use_volume_filter:
                    volume_ratio = row['volume_ratio']
                    if volume_ratio > self.volume_threshold:  # Check for increased volume
                        signal_reasons.append(f"Volume({volume_ratio:.2f}x) > {self.volume_threshold}")
                    else:
                        signal_type = ""  # Volume filter failed
                        signal_reasons.append(f"Volume too low: {volume_ratio:.2f}x < {self.volume_threshold}")
                
                # Apply dual-timeframe confirmation if provided
                if signal_type and higher_tf_df is not None:
                    if self.tf_confirm(signal_type, higher_tf_df):
                        signal_reasons.append(f"Higher TF confirms uptrend")
                    else:
                        signal_type = ""  # Higher TF filter failed
                        signal_reasons.append(f"Higher TF contradicts - no uptrend confirmation")
            
            # Check for sell signal - bearish crossover (fast EMA crosses below slow EMA)
            elif crossover_value == -1:
                signal_type = "sell"
                signal_reasons.append(f"EMA Crossover: Fast({self.fast_ema}) < Slow({self.slow_ema})")
                
                # 200-EMA trend filter disabled
                # Keeping original volatility check logic intact
                if self.use_dynamic_trend_filter:
                    atr_pct = row['atr_pct']
                    if atr_pct > self.atr_low_threshold:
                        # In higher volatility - 200 EMA filter disabled
                        signal_reasons.append(f"200-EMA filter disabled for more signals")
                    else:
                        # In low volatility, ignore trend filter
                        signal_reasons.append(f"ATR({atr_pct:.4f}) < {self.atr_low_threshold:.4f} - Low volatility, trend filter bypassed")
                
                # Apply additional filters if enabled
                if signal_type and self.use_rsi_filter:
                    rsi_value = row['rsi']
                    if rsi_value > self.rsi_oversold:  # Check RSI is not oversold
                        signal_reasons.append(f"RSI({rsi_value:.1f}) > {self.rsi_oversold}")
                    else:
                        signal_type = ""  # RSI filter failed
                        signal_reasons.append(f"RSI too low: {rsi_value:.1f} <= {self.rsi_oversold}")
                
                if signal_type and self.use_volume_filter:
                    volume_ratio = row['volume_ratio']
                    if volume_ratio > self.volume_threshold:  # Check for increased volume
                        signal_reasons.append(f"Volume({volume_ratio:.2f}x) > {self.volume_threshold}")
                    else:
                        signal_type = ""  # Volume filter failed
                        signal_reasons.append(f"Volume too low: {volume_ratio:.2f}x < {self.volume_threshold}")
                
                # Apply dual-timeframe confirmation if provided
                if signal_type and higher_tf_df is not None:
                    if self.tf_confirm(signal_type, higher_tf_df):
                        signal_reasons.append(f"Higher TF confirms downtrend")
                    else:
                        signal_type = ""  # Higher TF filter failed
                        signal_reasons.append(f"Higher TF contradicts - no downtrend confirmation")
        
        # Check if there was a recent trade (reduce overtrading)
        if signal_type and self.last_signal_time is not None:
            current_time = row.name if hasattr(row, 'name') else df.index[index]
            
            # Skip if too soon after last trade
            if (current_time - self.last_signal_time).total_seconds() < self.min_bars_between_trades * 14400:  # 4 hours = 14400 seconds
                signal_type = ""
                signal_reasons.append(f"Too soon after last trade at {self.last_signal_time}")
        
        # Log full reasoning if a signal was generated
        if signal_type:
            reason_str = " + ".join(signal_reasons)
            logger.info(f"Generated {signal_type.upper()} signal for {self.symbol}: {reason_str}")
        elif signal_reasons:
            # Log why a potential signal was rejected
            reason_str = " | ".join(signal_reasons)
            logger.debug(f"Rejected signal for {self.symbol}: {reason_str}")
        
        return signal_type
    
    def _open_trade(self, signal, bar_data):
        """
        Open a new trade with the current signal and market data.
        
        Args:
            signal: 'buy' or 'sell'
            bar_data: Dictionary with current OHLCV data
            
        Returns:
            dict: New trade object or None if no trade opened
        """
        if self.active_trade is not None:
            logger.warning("Attempted to open trade when one is already active")
            return None
        
        if not signal or signal == 'none':
            return None
        
        # Store last signal time to prevent overtrading
        self.last_signal_time = bar_data.get('time', datetime.now())
        current_price = bar_data.get('close', 0)
        
        # Get ATR-adjusted parameters based on current regime
        sl_multiplier = self._get_adjusted_param('atr_sl_multiplier')
        trail_multiplier = self._get_adjusted_param('atr_trail_multiplier')
        breakeven_trigger = self._get_adjusted_param('breakeven_trigger_r')
        
        # Add volatility regime to the bar data for better logging
        bar_data['vol_regime'] = self.current_regime
        bar_data['atr_pct'] = self.atr_pct
        
        # Check if ATR is available
        atr = bar_data.get('atr')
        if atr is not None and USE_ATR_STOPS:
            # Calculate stop loss based on ATR
            if signal == 'buy':
                stop_loss = current_price - (atr * sl_multiplier)
            else:  # sell/short signal
                stop_loss = current_price + (atr * sl_multiplier)
            
            # Calculate take profit based on ATR if specified
            take_profit = None
            if self.atr_tp_multiplier:
                if signal == 'buy':
                    take_profit = current_price + (atr * self.atr_tp_multiplier)
                else:
                    take_profit = current_price - (atr * self.atr_tp_multiplier)
                    
            logger.info(f"Using ATR-based stops: ATR={atr:.2f}, SL Mult={sl_multiplier}x, Trail Mult={trail_multiplier}x, Regime: {self.current_regime}")
        else:
            # Fallback to percentage-based stops
            if signal == 'buy':
                stop_loss = current_price * (1 - STOP_LOSS_PCT)
                take_profit = current_price * (1 + TAKE_PROFIT_PCT) if TAKE_PROFIT_PCT else None
            else:
                stop_loss = current_price * (1 + STOP_LOSS_PCT)
                take_profit = current_price * (1 - TAKE_PROFIT_PCT) if TAKE_PROFIT_PCT else None
                
            logger.info(f"Using percentage-based stops: SL={STOP_LOSS_PCT:.1%}, TP={TAKE_PROFIT_PCT:.1%}")
        
        # Calculate position size
        stop_distance = abs(current_price - stop_loss)
        risk_amount = self.account_balance * self.risk_per_trade
        
        # IMPROVEMENT 4: Volatility-based position sizing
        if self.use_volatility_sizing and atr is not None:
            # Target a specific dollar volatility
            target_vol_usd = self.account_balance * self.vol_target_pct
            
            # Daily volatility (ATR) in dollars
            daily_vol_usd = current_price * self.atr_pct
            
            # Position size to achieve target volatility (but respect risk limits)
            if daily_vol_usd > 0:
                vol_size = target_vol_usd / daily_vol_usd
                
                # Calculate risk-based size as a backup
                risk_size = risk_amount / stop_distance if stop_distance > 0 else 0
                
                # Use the smaller of the two sizes for safety
                position_size = min(vol_size, risk_size)
                
                logger.info(f"Using volatility-based position sizing: Vol={daily_vol_usd:.2f}, Target=${target_vol_usd:.2f}")
            else:
                position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                logger.info(f"Using fixed risk position sizing: {self.risk_per_trade:.2%}")
        else:
            # Fixed risk sizing
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            logger.info(f"Using fixed risk position sizing: {self.risk_per_trade:.2%}")
            
        # Create trade dictionary
        self.active_trade = {
            'symbol': self.symbol,
            'side': signal,
            'entry_price': current_price,
            'entry_time': bar_data.get('time', datetime.now()),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'size': position_size,
            'bars_held': 0,
            'high_since_entry': current_price if signal == 'buy' else 0,
            'low_since_entry': 0 if signal == 'buy' else current_price,
            'last_price': current_price,
            'atr': atr,
            'trailing_stop': None,  # Will be activated later
            'r_multiple': 0,        # Will calculate as trade progresses
            'breakeven': False,     # Track if moved to breakeven
            'regime': self.current_regime,  # Track the regime during entry
            'pyramid_entries': 0,   # Track number of scale-in entries
        }
        
        # For pyramiding tracking
        self.last_pyramid_price = current_price
        
        # For Improving IMPROVEMENT 1: Add an R-multiple tracker
        if atr is not None and stop_distance > 0:
            self.active_trade['r_value'] = stop_distance
            logger.info(f"Trade R value set to {stop_distance:.2f}")
            
        logger.info(f"Opened {signal} trade at {current_price} with stop loss at {stop_loss}")
        logger.info(f"Position size: {position_size:.8f} ({position_size * current_price:.2f} USD)")
        
        # Emit signal to health monitor if available
        if self.health_monitor:
            self.health_monitor.record_trade_entry(self.active_trade)
            
        return self.active_trade
    
    def _update_active_trade(self, bar_data):
        """
        Update the active trade with current market data.
        Check for stop loss/take profit triggers.
        
        Args:
            bar_data: Dictionary with current OHLCV data
            
        Returns:
            dict: Updated trade object
        """
        if not self.active_trade:
            return None
            
        # Extract current price info
        current_price = bar_data.get('close', 0)
        high_price = bar_data.get('high', current_price)
        low_price = bar_data.get('low', current_price)
        
        # Update the volatility regime if needed
        self._determine_regime(bar_data)
        
        # Update trade with current price and tracked highs/lows
        self.active_trade['last_price'] = current_price
        self.active_trade['bars_held'] += 1
        
        # Track price extremes for trailing logic
        if self.active_trade['side'] == 'buy':
            # For long positions, track the highest price
            if high_price > self.active_trade.get('high_since_entry', 0):
                self.active_trade['high_since_entry'] = high_price
        else:
            # For short positions, track the lowest price
            if (self.active_trade.get('low_since_entry', float('inf')) == 0 or
                low_price < self.active_trade.get('low_since_entry', float('inf'))):
                self.active_trade['low_since_entry'] = low_price
        
        # Check if minimum hold period has elapsed
        min_hold_met = self.active_trade['bars_held'] >= self.min_hold_bars
        
        # Calculate current R-multiple (profit in terms of initial risk)
        entry_price = self.active_trade['entry_price']
        r_value = self.active_trade.get('r_value', 0)
        if r_value > 0:
            if self.active_trade['side'] == 'buy':
                r_multiple = (current_price - entry_price) / r_value
            else:
                r_multiple = (entry_price - current_price) / r_value
            self.active_trade['r_multiple'] = r_multiple
            
            # Move to breakeven if profit exceeds threshold
            breakeven_trigger = self._get_adjusted_param('breakeven_trigger_r')
            if (not self.active_trade.get('breakeven', False) and 
                r_multiple >= breakeven_trigger and min_hold_met):
                self.active_trade['breakeven'] = True
                self.active_trade['stop_loss'] = entry_price
                logger.info(f"Moving to breakeven after {r_multiple:.2f}R profit")
                
                # Activate tighter trail on profit
                self.active_trade['trail_tightened'] = True
                
        # ====== Alternative Exit Strategies ======
        
        # Check for time-based exit if configured
        if hasattr(self, 'exit_strategy') and self.exit_strategy == 'time':
            max_bars = getattr(self, 'max_bars', 10)
            if self.active_trade['bars_held'] >= max_bars:
                self._close_trade(current_price, 'time_exit')
                return None
                
        # Check for fixed take-profit/stop-loss if configured
        if hasattr(self, 'exit_strategy') and self.exit_strategy == 'fixed':
            # Calculate fixed take-profit and stop-loss levels if not already set
            if 'fixed_tp' not in self.active_trade:
                tp_pct = getattr(self, 'take_profit_pct', 0.05)
                sl_pct = getattr(self, 'stop_loss_pct', 0.03)
                
                if self.active_trade['side'] == 'buy':
                    self.active_trade['fixed_tp'] = entry_price * (1 + tp_pct)
                    self.active_trade['fixed_sl'] = entry_price * (1 - sl_pct)
                else:
                    self.active_trade['fixed_tp'] = entry_price * (1 - tp_pct)
                    self.active_trade['fixed_sl'] = entry_price * (1 + sl_pct)
                    
            # Check if take-profit hit
            if self.active_trade['side'] == 'buy':
                if high_price >= self.active_trade['fixed_tp']:
                    self._close_trade(self.active_trade['fixed_tp'], 'take_profit')
                    return None
            else:
                if low_price <= self.active_trade['fixed_tp']:
                    self._close_trade(self.active_trade['fixed_tp'], 'take_profit')
                    return None
                    
            # Check if stop-loss hit
            if self.active_trade['side'] == 'buy':
                if low_price <= self.active_trade['fixed_sl']:
                    self._close_trade(self.active_trade['fixed_sl'], 'stop_loss')
                    return None
            else:
                if high_price >= self.active_trade['fixed_sl']:
                    self._close_trade(self.active_trade['fixed_sl'], 'stop_loss')
                    return None
                    
        # Check for percentage-based trailing stop if configured
        if hasattr(self, 'exit_strategy') and self.exit_strategy == 'trailing':
            trail_pct = getattr(self, 'trail_pct', 0.02)
            
            # Initialize trailing stop if not set
            if 'trailing_stop_pct' not in self.active_trade:
                if self.active_trade['side'] == 'buy':
                    self.active_trade['trailing_stop_pct'] = entry_price * (1 - trail_pct)
                else:
                    self.active_trade['trailing_stop_pct'] = entry_price * (1 + trail_pct)
                    
            # Update trailing stop
            if self.active_trade['side'] == 'buy':
                new_stop = self.active_trade['high_since_entry'] * (1 - trail_pct)
                if new_stop > self.active_trade['trailing_stop_pct']:
                    self.active_trade['trailing_stop_pct'] = new_stop
                    
                # Check if trailing stop hit
                if low_price <= self.active_trade['trailing_stop_pct'] and min_hold_met:
                    self._close_trade(self.active_trade['trailing_stop_pct'], 'trailing_stop')
                    return None
            else:
                new_stop = self.active_trade['low_since_entry'] * (1 + trail_pct)
                if (self.active_trade['trailing_stop_pct'] == 0 or 
                    new_stop < self.active_trade['trailing_stop_pct']):
                    self.active_trade['trailing_stop_pct'] = new_stop
                    
                # Check if trailing stop hit
                if high_price >= self.active_trade['trailing_stop_pct'] and min_hold_met:
                    self._close_trade(self.active_trade['trailing_stop_pct'], 'trailing_stop')
                    return None
                
        # ====== Default ATR-based Exit Logic ======
        
        # Initialize trail stop if not set
        if self.active_trade['trailing_stop'] is None:
            atr = self.active_trade.get('atr', 0)
            if atr > 0:
                # Initialize the trailing stop with ATR
                trail_multiplier = self._get_adjusted_param('atr_trail_multiplier')
                if self.active_trade['side'] == 'buy':
                    self.active_trade['trailing_stop'] = entry_price - (atr * trail_multiplier)
                else:
                    self.active_trade['trailing_stop'] = entry_price + (atr * trail_multiplier)
                logger.info(f"Initialized trailing stop at {self.active_trade['trailing_stop']:.2f}")
                
        # Regular trailing stop adjustment
        if self.active_trade['trailing_stop'] is not None:
            if self.active_trade['side'] == 'buy':
                # For long positions, move trailing stop up
                price_ref = high_price if min_hold_met else current_price
                trail_distance = price_ref * 0.01  # 1% fallback if ATR not available
                
                if self.active_trade.get('atr', 0) > 0:
                    # Use regime-adjusted trail multiplier
                    trail_multiplier = self._get_adjusted_param('atr_trail_multiplier')
                    # Use tighter trail if already tightened
                    if self.active_trade.get('trail_tightened', False):
                        trail_multiplier *= 0.75
                        
                    trail_distance = self.active_trade['atr'] * trail_multiplier
                
                new_stop = price_ref - trail_distance
                if new_stop > self.active_trade['trailing_stop']:
                    prev_stop = self.active_trade['trailing_stop']
                    self.active_trade['trailing_stop'] = new_stop
                    logger.info(f"Trail up: {prev_stop:.2f} -> {new_stop:.2f} ({new_stop - prev_stop:.2f})")
            else:
                # For short positions, move trailing stop down
                price_ref = low_price if min_hold_met else current_price
                trail_distance = price_ref * 0.01  # 1% fallback
                
                if self.active_trade.get('atr', 0) > 0:
                    # Use regime-adjusted trail multiplier
                    trail_multiplier = self._get_adjusted_param('atr_trail_multiplier')
                    # Use tighter trail if already tightened
                    if self.active_trade.get('trail_tightened', False):
                        trail_multiplier *= 0.75
                        
                    trail_distance = self.active_trade['atr'] * trail_multiplier
                
                new_stop = price_ref + trail_distance
                if (self.active_trade['trailing_stop'] == 0 or
                    new_stop < self.active_trade['trailing_stop']):
                    prev_stop = self.active_trade['trailing_stop']
                    self.active_trade['trailing_stop'] = new_stop
                    logger.info(f"Trail down: {prev_stop:.2f} -> {new_stop:.2f} ({prev_stop - new_stop:.2f})")
                    
        # Check for take profit hit
        take_profit = self.active_trade.get('take_profit')
        if take_profit and min_hold_met:
            if (self.active_trade['side'] == 'buy' and high_price >= take_profit) or \
               (self.active_trade['side'] == 'sell' and low_price <= take_profit):
                self._close_trade(current_price, 'take_profit')
                return None
                
        # Check for stop loss hit
        stop_loss = self.active_trade.get('stop_loss')
        if stop_loss:
            if (self.active_trade['side'] == 'buy' and low_price <= stop_loss) or \
               (self.active_trade['side'] == 'sell' and high_price >= stop_loss):
                self._close_trade(current_price, 'stop_loss')
                return None
                
        # Check for trailing stop hit (only if minimum hold period met)
        trailing_stop = self.active_trade.get('trailing_stop')
        if trailing_stop and min_hold_met:
            if (self.active_trade['side'] == 'buy' and low_price <= trailing_stop) or \
               (self.active_trade['side'] == 'sell' and high_price >= trailing_stop):
                self._close_trade(current_price, 'trailing_stop')
                return None
                
        return self.active_trade
    
    def _close_trade(self, current_price, reason):
        """
        Close the active trade.
        
        Args:
            current_price: Current price
            reason: Reason for closing ('stop_loss', 'take_profit', 'manual', etc.)
        """
        if not self.active_trade:
            logger.warning("Attempted to close trade when none is active")
            return
        
        # Calculate P&L
        if self.active_trade['side'] == 'buy':
            pnl = (current_price - self.active_trade['entry_price']) * self.active_trade['size']
            pnl_percent = (current_price - self.active_trade['entry_price']) / self.active_trade['entry_price']
        else:
            pnl = (self.active_trade['entry_price'] - current_price) * self.active_trade['size']
            pnl_percent = (self.active_trade['entry_price'] - current_price) / self.active_trade['entry_price']
        
        # Update account balance
        self.account_balance += pnl
        
        # Record trade result
        closed_trade = {
            **self.active_trade,
            'exit_time': datetime.now(),
            'exit_price': current_price,
            'exit_reason': reason,
            'pnl': pnl,
            'pnl_percent': pnl_percent
        }
        
        # Calculate R-multiple (risk unit)
        if 'stop_loss' in self.active_trade and self.active_trade['stop_loss'] is not None:
            risk = abs(self.active_trade['entry_price'] - self.active_trade['stop_loss']) * self.active_trade['size']
            if risk > 0:
                closed_trade['r_multiple'] = pnl / risk
            else:
                closed_trade['r_multiple'] = 0
        else:
            closed_trade['r_multiple'] = 0
            
        # Calculate trade duration
        trade_duration = (closed_trade['exit_time'] - closed_trade['entry_time']).total_seconds() / 60  # minutes
        
        # Log trade closing
        logger.info(f"Closed {self.active_trade['side']} trade at {current_price} for {reason}, "
                   f"PnL: ${pnl:.2f}, PnL%: {pnl_percent*100:.2f}%, "
                   f"Duration: {trade_duration:.1f} minutes")
        
        # Track consecutive stop losses
        if reason == 'stop_loss':
            self.consecutive_sl_count += 1
        else:
            self.consecutive_sl_count = 0
        
        # Update in database if available
        if self.db_session and 'id' in self.active_trade:
            try:
                trade = self.db_session.query(Trade).filter_by(id=self.active_trade['id']).first()
                if trade:
                    trade.exit_time = closed_trade['exit_time']
                    trade.exit_price = current_price
                    trade.exit_reason = reason
                    trade.pnl = pnl
                    trade.pnl_percent = pnl_percent
                    trade.duration_minutes = trade_duration
                    self.db_session.commit()
            except Exception as e:
                logger.error(f"Error updating trade in database: {str(e)}")
                if self.db_session:
                    self.db_session.rollback()
        
        # Reset active trade
        self.active_trade = None
        
        return closed_trade
    
    def fetch_historical(self, symbol=None, timeframe=None, days=None):
        """Fetch historical data for the symbol/timeframe"""
        if symbol is None:
            symbol = self.symbol
        if timeframe is None:
            timeframe = self.timeframe
        if days is None:
            days = self.history_days
            
        return fetch_historical_data(symbol, timeframe, days)
    
    def get_performance_summary(self):
        """Get performance summary for the strategy"""
        if not self.db_session:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_return': 0
            }
            
        try:
            # Query completed trades for this strategy
            trades = self.db_session.query(Trade).filter(
                Trade.strategy_type == 'ema_crossover',
                Trade.exit_time.isnot(None)
            ).all()
            
            total_trades = len(trades)
            if total_trades == 0:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_return': 0
                }
            
            winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
            losing_trades = sum(1 for t in trades if t.pnl and t.pnl <= 0)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(t.pnl for t in trades if t.pnl and t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in trades if t.pnl and t.pnl <= 0))
            
            # Use the utility function with MIN_LOSS floor
            winners = [t.pnl for t in trades if t.pnl and t.pnl > 0]
            losers = [t.pnl for t in trades if t.pnl and t.pnl <= 0]
            profit_factor_val = profit_factor(winners, losers)
            
            # Calculate average return percentage
            avg_return = sum(t.pnl_percent for t in trades if t.pnl_percent is not None) / total_trades
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor_val,
                'avg_return': avg_return
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_return': 0
            }
    
    def set_account(self, account):
        """
        Set the trading account for the strategy.
        
        Args:
            account: Trading account object
        """
        self.account = account

    def set_allocator(self, allocator):
        """
        Set the portfolio allocator.
        
        Args:
            allocator: Portfolio allocator instance
        """
        self.allocator = allocator

    def backtest(self, df):
        """
        Backtest the strategy on historical data.
        
        Args:
            df (DataFrame): OHLCV dataframe
            
        Returns:
            DataFrame: OHLCV dataframe with signals
        """
        if df.empty:
            logger.warning("Empty dataframe provided for backtesting")
            return df
            
        df = df.copy()
        
        # Apply indicators
        df = self.apply_indicators(df)
        
        # Generate signals from crossover
        df['signal'] = 0
        
        # Check if trend filtering is enabled
        use_trend_filter = hasattr(self, 'use_dynamic_trend_filter') and self.use_dynamic_trend_filter
        
        for i in range(1, len(df)):
            # Buy signal - fast crosses above slow
            if (df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and 
                df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]):
                
                # Check trend condition if filtering is enabled
                trend_condition = True
                if use_trend_filter:
                    # Apply the trend filter based on trend EMA
                    trend_condition = df['close'].iloc[i] > df['ema_trend'].iloc[i]
                
                if trend_condition:
                    df.loc[df.index[i], 'signal'] = 1
                
            # Sell signal - fast crosses below slow
            elif (df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and 
                  df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]):
                df.loc[df.index[i], 'signal'] = -1
        
        return df
    
    def generate_signal(self, i, df):
        """
        Generate trading signal for a specific candle.
        This method is called by the backtester for each candle.
        
        Args:
            i: Index of the current candle
            df: DataFrame with indicators and previous signals
            
        Returns:
            Signal type: 'buy', 'sell', or '' (empty string for no signal)
        """
        # Skip if not enough bars between trades
        current_time = df.index[i]
        if self.last_signal_time is not None and (current_time - self.last_signal_time).total_seconds() / 3600 < self.min_bars_between_trades * 4:
            return ''
            
        # Get the current signal from the processed DataFrame
        current_signal = df['signal'].iloc[i]
        
        if current_signal == 1:
            signal_reason = df['signal_reason'].iloc[i] if 'signal_reason' in df.columns else "EMA Crossover: Buy signal"
            logger.info(f"Generated BUY signal for {self.symbol}: {signal_reason}")
            self.last_signal_time = current_time
            return 'buy'
        elif current_signal == -1:
            signal_reason = df['signal_reason'].iloc[i] if 'signal_reason' in df.columns else "EMA Crossover: Sell signal"
            logger.info(f"Generated SELL signal for {self.symbol}: {signal_reason}")
            self.last_signal_time = current_time
            return 'sell'
        
        return ''
    
    def tf_confirm(self, signal_type, higher_tf_df):
        """
        Check if higher timeframe confirms the signal direction.
        Simplified to only check slow EMA trend alignment.
        
        Args:
            signal_type: The proposed signal ('buy' or 'sell')
            higher_tf_df: DataFrame with higher timeframe indicators
            
        Returns:
            bool: True if higher timeframe confirms the signal, False otherwise
        """
        if higher_tf_df is None or len(higher_tf_df) < 2:
            # No higher timeframe data available, bypass confirmation
            return True
            
        # Get the latest higher timeframe data
        latest_row = higher_tf_df.iloc[-1]
        prev_row = higher_tf_df.iloc[-2]
        
        # For buy signals: check if higher timeframe slow EMA is rising
        if signal_type == 'buy':
            return latest_row['ema_slow'] > prev_row['ema_slow']
        
        # For sell signals: check if higher timeframe slow EMA is falling
        elif signal_type == 'sell':
            return latest_row['ema_slow'] < prev_row['ema_slow']
        
        return False

    def calculate_indicators(self, df):
        """
        Calculate technical indicators used by the strategy.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Calculate EMAs
        data[f'ema_{self.fast_ema}'] = data['close'].ewm(span=self.fast_ema).mean()
        data[f'ema_{self.slow_ema}'] = data['close'].ewm(span=self.slow_ema).mean()
        
        # For trend filter
        data['ema_200'] = data['close'].ewm(span=200).mean()
        
        # Calculate RSI
        data['rsi'] = self._calculate_rsi(data)['rsi']
        
        # Calculate ATR
        data['atr'] = self._calculate_atr(data)['atr']
        
        # Calculate ATR as percentage of price
        data['atr_pct'] = data['atr'] / data['close']
        
        # Determine volatility regime
        data['vol_regime'] = 'medium_vol'  # Default
        data.loc[data['atr_pct'] > 0.010, 'vol_regime'] = 'high_vol'
        data.loc[data['atr_pct'] < 0.005, 'vol_regime'] = 'low_vol'
        
        return data

    def _determine_regime(self, row):
        """
        Determine market regime based on price and indicators.
        
        Args:
            row: Row of data with OHLCV and indicators
            
        Returns:
            str: Market regime type ('bull', 'bear', 'neutral')
        """
        # Default regime
        regime = "neutral"
        
        # Simple trend determination based on EMAs
        if hasattr(self, 'ema_trend') and 'ema_trend' in row:
            if row['close'] > row['ema_trend']:
                regime = "bull"
            else:
                regime = "bear"
        
        return regime
        
    def _get_adjusted_param(self, param_name):
        """
        Get parameter adjusted for current volatility regime.
        
        Args:
            param_name: Name of the parameter to adjust
            
        Returns:
            Adjusted parameter value
        """
        # Use regime-specific value if available, otherwise use config default
        if param_name in self.regime_adjustments.get(self.current_regime, {}):
            return self.regime_adjustments[self.current_regime][param_name]
        
        # If not in regime adjustments, return the instance variable with the same name
        if param_name == 'atr_sl_multiplier':
            return self.atr_sl_multiplier
        elif param_name == 'atr_trail_multiplier':
            return self.atr_trail_multiplier
        elif param_name == 'breakeven_trigger_r':
            return self.breakeven_trigger_r
        elif param_name == 'atr_tp_multiplier':
            return self.atr_tp_multiplier
        
        # Default fallback
        return getattr(self, param_name, None)

    def get_stop_loss_price(self, side, candles):
        """
        Calculate a stop loss price based on ATR or percentage.
        
        Args:
            side (str): 'long' or 'short'
            candles (list): List of OHLCV candles
            
        Returns:
            float: Stop loss price or None if cannot be calculated
        """
        try:
            # Convert candles to dataframe for indicator calculation
            df = pd.DataFrame([{
                'open': c['open'],
                'high': c['high'],
                'low': c['low'],
                'close': c['close'],
                'volume': c['volume']
            } for c in candles])
            
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift())
            low_close = abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.atr_period).mean().iloc[-1]
            
            # Get current price
            current_price = df['close'].iloc[-1]
            
            # Calculate ATR-based stop loss
            if not pd.isna(atr) and atr > 0:
                if side == 'long':
                    stop_loss = current_price - (atr * self.atr_multiplier)
                else:  # short
                    stop_loss = current_price + (atr * self.atr_multiplier)
                return stop_loss
            else:
                # Fallback to percentage-based stop loss (2% default)
                stop_loss_pct = 0.02
                if side == 'long':
                    stop_loss = current_price * (1 - stop_loss_pct)
                else:
                    stop_loss = current_price * (1 + stop_loss_pct)
                return stop_loss
                
        except Exception as e:
            print(f"Error calculating stop loss: {e}")
            return None

    def on_new_candle(self, candles):
        """
        Process new candle data and generate trading signals.
        
        Args:
            candles (list): List of OHLCV candles with keys: timestamp, open, high, low, close, volume
            
        Returns:
            str: Signal type - 'buy', 'sell', or 'none'
        """
        try:
            # Convert candles to dataframe
            df = pd.DataFrame([{
                'open': c['open'],
                'high': c['high'],
                'low': c['low'],
                'close': c['close'],
                'volume': c['volume'],
                'timestamp': pd.to_datetime(c['timestamp'], unit='ms')
            } for c in candles])
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Calculate indicators
            df = self.apply_indicators(df)
            
            # Check for crossover signals
            signal = 'none'
            
            # Need at least 2 candles to detect crossover
            if len(df) >= 2:
                # Get the last 2 rows for crossover detection
                last_two = df.tail(2)
                
                # Buy signal: Fast EMA crosses above Slow EMA
                if (last_two['ema_fast'].iloc[0] <= last_two['ema_slow'].iloc[0] and 
                    last_two['ema_fast'].iloc[1] > last_two['ema_slow'].iloc[1]):
                    signal = 'buy'
                    
                # Sell signal: Fast EMA crosses below Slow EMA
                elif (last_two['ema_fast'].iloc[0] >= last_two['ema_slow'].iloc[0] and 
                      last_two['ema_fast'].iloc[1] < last_two['ema_slow'].iloc[1]):
                    signal = 'sell'
            
            return signal
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return 'none'
            
    def set_exit_strategy(self, strategy_type, params=None):
        """
        Set alternative exit strategy for trades.
        
        Args:
            strategy_type (str): Type of exit strategy ('fixed', 'trailing', 'time', 'atr')
            params (dict): Parameters for the exit strategy
                - fixed: {'take_profit_pct': float, 'stop_loss_pct': float}
                - trailing: {'trail_pct': float}
                - time: {'max_bars': int}
                - atr: {'multiplier': float}
        """
        if not params:
            params = {}
            
        logger.info(f"Setting exit strategy to {strategy_type} with params: {params}")
        
        self.exit_strategy = strategy_type
        
        if strategy_type == 'fixed':
            # Fixed take-profit and stop-loss levels
            self.take_profit_pct = params.get('take_profit_pct', 0.05)  # Default 5%
            self.stop_loss_pct = params.get('stop_loss_pct', 0.03)  # Default 3%
            logger.info(f"Fixed exit strategy: TP={self.take_profit_pct:.1%}, SL={self.stop_loss_pct:.1%}")
            
        elif strategy_type == 'trailing':
            # Trailing stop based on fixed percentage
            self.trail_pct = params.get('trail_pct', 0.02)  # Default 2%
            logger.info(f"Trailing stop strategy: {self.trail_pct:.1%}")
            
        elif strategy_type == 'time':
            # Time-based exit after N bars
            self.max_bars = params.get('max_bars', 10)  # Default 10 bars
            logger.info(f"Time-based exit strategy: max {self.max_bars} bars")
            
        elif strategy_type == 'atr':
            # ATR-based trailing stop
            self.atr_sl_multiplier = params.get('multiplier', 2.0)  # Default 2x ATR
            logger.info(f"ATR-based exit strategy: {self.atr_sl_multiplier}x ATR")
            
        else:
            logger.warning(f"Unknown exit strategy: {strategy_type}. Using default.")
            self.exit_strategy = None 