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
                atr_trail_multiplier=1.25):
        # Initialize the base class first
        super().__init__(config)
        
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
            self.atr_tp_multiplier = config.get('atr_tp_multiplier', None)
            self.atr_trail_multiplier = config.get('atr_trail_multiplier', atr_trail_multiplier)
            self.risk_per_trade = config.get('risk_per_trade', risk_per_trade)
            self.use_volatility_sizing = config.get('use_volatility_sizing', use_volatility_sizing)
            self.vol_target_pct = config.get('vol_target_pct', vol_target_pct)
            self.enable_pyramiding = config.get('enable_pyramiding', enable_pyramiding)
            self.max_pyramid_entries = config.get('max_pyramid_entries', max_pyramid_entries)
            self.pyramid_threshold = config.get('pyramid_threshold', 0.5)
            self.pyramid_position_scale = config.get('pyramid_position_scale', 0.5)
            
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
        else:
            # Use explicitly provided parameters
            self.fast_ema = fast_ema
            self.slow_ema = slow_ema
            self.trend_ema = trend_ema
            self.atr_sl_multiplier = atr_sl_multiplier
            self.atr_tp_multiplier = None  # Removed fixed TP
            self.atr_trail_multiplier = atr_trail_multiplier  # IMPROVEMENT 1: Wider trail (1.25x default)
            self.risk_per_trade = risk_per_trade
            self.use_volatility_sizing = use_volatility_sizing
            self.vol_target_pct = vol_target_pct
            self.enable_pyramiding = enable_pyramiding
            self.max_pyramid_entries = max_pyramid_entries
            self.pyramid_threshold = 0.5  # 0.5 × ATR
            self.pyramid_position_scale = 0.5  # 50% of initial size
            
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
        Apply all indicators needed for strategy execution.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=self.fast_ema).mean()
        df['ema_slow'] = df['close'].ewm(span=self.slow_ema).mean()
        df['ema_trend'] = df['close'].ewm(span=self.trend_ema).mean()
        
        # Add crossover signals
        df['ema_crossover'] = 0
        # When fast EMA crosses above slow EMA, set to 1
        df.loc[(df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)), 'ema_crossover'] = 1
        # When fast EMA crosses below slow EMA, set to -1
        df.loc[(df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)), 'ema_crossover'] = -1
        
        # Calculate ATR for volatility sizing and dynamic trend filtering
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()
        
        # Calculate ATR as percentage of price for dynamic trend filtering
        df['atr_pct'] = df['atr'] / df['close']
        
        # Calculate RSI
        df = self._calculate_rsi(df, period=self.rsi_period)
        
        # Calculate volume indicators
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['volume_spike'] = df['volume_ratio'] > 2.0  # True when volume is >2x the average
        
        # Save the latest ATR value for position sizing
        if len(df) > 0:
            self.last_atr_value = df['atr'].iloc[-1]
        
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
        Open a new trade based on the signal.
        
        Args:
            signal: Signal dictionary
            bar_data: Current bar data with indicators
        """
        if self.active_trade:
            logger.warning("Attempted to open trade while another is active")
            return
        
        side = signal  # 'buy' or 'sell'
        current_price = bar_data['close']
        
        # Calculate stop loss and take profit levels using ATR
        if 'atr' in bar_data and not pd.isna(bar_data['atr']):
            # Use ATR-based stops
            atr_value = bar_data['atr']
            
            if side == 'buy':
                stop_loss_price = current_price - (atr_value * self.atr_sl_multiplier)
                # Only use take profit if specified, otherwise rely on trailing stop
                take_profit_price = current_price + (atr_value * self.atr_tp_multiplier) if self.atr_tp_multiplier else None
            else:  # sell
                stop_loss_price = current_price + (atr_value * self.atr_sl_multiplier)
                take_profit_price = current_price - (atr_value * self.atr_tp_multiplier) if self.atr_tp_multiplier else None
                
            logger.info(f"Using ATR-based stops: ATR={atr_value:.2f}, SL={self.atr_sl_multiplier}×ATR")
            
            # Store ATR value for trailing stop calculations
            self.last_atr_value = atr_value
        else:
            # Fallback to fixed percentage stops if ATR not available
            stop_loss_price = (
                current_price * (1 - STOP_LOSS_PCT) if side == 'buy' 
                else current_price * (1 + STOP_LOSS_PCT)
            )
            take_profit_price = (
                current_price * (1 + TAKE_PROFIT_PCT) if side == 'buy' 
                else current_price * (1 - TAKE_PROFIT_PCT)
            ) if TAKE_PROFIT_PCT else None
            
            logger.info(f"Using fixed percentage stops: SL={STOP_LOSS_PCT*100:.3f}%")
            
            self.last_atr_value = None
        
        # Calculate position size based on volatility targeting
        try:
            # Get recent profit factor from health monitor if available
            pf_recent = None
            if hasattr(self, 'health_monitor') and self.health_monitor:
                pf_recent = self.health_monitor.get_profit_factor_last_n(40)
                
            # Use portfolio manager if available with edge-weighted sizing
            if hasattr(self, 'allocator') and self.allocator:
                position_size, _, _ = self.allocator.calculate_position_size(
                    symbol=self.symbol,
                    current_price=current_price,
                    atr_value=atr_value if self.last_atr_value else current_price * 0.02,  # Fallback to 2% of price
                    strat_name="ema_crossover",
                    pf_recent=pf_recent,
                    side=side
                )
                
                # Check if pyramiding should be enabled based on volatility regime
                if hasattr(self, 'allocator') and hasattr(self.allocator, 'should_enable_pyramiding'):
                    self.enable_pyramiding = self.allocator.should_enable_pyramiding(self.symbol)
                    if self.enable_pyramiding:
                        logger.info(f"Pyramiding ENABLED for {self.symbol} based on volatility regime")
                    else:
                        logger.info(f"Pyramiding DISABLED for {self.symbol} based on volatility regime")
            elif 'atr' in bar_data and not pd.isna(bar_data['atr']) and hasattr(self, 'use_volatility_sizing') and self.use_volatility_sizing:
                # Volatility-targeted position sizing
                dollar_risk = self.account_balance * self.vol_target_pct  # Target volatility (0.75%)
                position_size = dollar_risk / atr_value  # Size inversely proportional to volatility
                
                logger.info(f"Using volatility-targeted position sizing: {self.vol_target_pct*100:.2f}% risk / {atr_value:.2f} ATR")
            else:
                # Traditional risk-based position sizing
                risk_amount = self.account_balance * self.risk_per_trade
                risk_per_unit = abs(current_price - stop_loss_price)  # Risk in price units per coin
                position_size = risk_amount / risk_per_unit
                
                logger.info(f"Using fixed risk position sizing: {self.risk_per_trade*100:.2f}% account risk")
            
            # Cap position size at maximum percentage of account
            max_position_size = (self.account_balance * MAX_POSITION_PCT) / current_price
            position_size = min(position_size, max_position_size)
            
            logger.info(f"Position size: {position_size:.6f} coins (${position_size * current_price:.2f})")
        except Exception as e:
            # Fallback to simple position sizing in case of error
            risk_amount = self.account_balance * self.risk_per_trade
            position_size = risk_amount / current_price
            logger.error(f"Error in position sizing calculation: {str(e)}. Using fallback method.")
        
        # Calculate notional value
        notional_value = position_size * current_price
        
        # Record the trade
        self.active_trade = {
            'symbol': self.symbol,
            'side': side,
            'entry_time': datetime.now(),
            'entry_price': current_price,
            'amount': position_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'notional_value': notional_value,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'rsi': bar_data.get('rsi', None),
            'atr': bar_data.get('atr', None),
            'last_price': current_price,  # Track last price for updates
            'trailing_stop': None,  # Initialize trailing stop as None
            'max_price': current_price if side == 'buy' else float('inf'),  # Track max price for trailing stop (long)
            'min_price': current_price if side == 'sell' else 0,  # Track min price for trailing stop (short)
            'pyramid_count': 0,
            'pyramid_levels': [],
            'trail_tightened': False,  # Flag to track if the trailing stop has been tightened
            'regime': getattr(self.allocator, 'current_regime', lambda x: 'normal')(self.symbol) if hasattr(self, 'allocator') and self.allocator is not None else 'normal'  # Track market regime
        }
        
        # Log the trade
        log_trade(self.active_trade)
        
        # Store in database if available
        if self.db_session:
            try:
                db_trade = Trade(
                    symbol=self.symbol,
                    side=side,
                    entry_time=self.active_trade['entry_time'],
                    entry_price=current_price,
                    amount=position_size,
                    stop_loss=stop_loss_price,
                    take_profit=take_profit_price,
                    rsi_value=bar_data.get('rsi', None),
                    atr_value=bar_data.get('atr', None),
                    strategy_type='ema_crossover',
                    strategy_params=f"fast={self.fast_ema},slow={self.slow_ema},trend={self.trend_ema}"  # Added trend EMA
                )
                self.db_session.add(db_trade)
                self.db_session.commit()
                
                # Update trade with ID from database
                self.active_trade['id'] = db_trade.id
            except Exception as e:
                logger.error(f"Error storing trade in database: {str(e)}")
                if self.db_session:
                    self.db_session.rollback()
        
        logger.info(f"Opened {side} trade at {current_price} with SL: {stop_loss_price}" + 
                   (f", TP: {take_profit_price}" if take_profit_price else ", using trailing stop"))
        logger.info(f"Strategy parameters: EMA{self.fast_ema}/{self.slow_ema}/{self.trend_ema}, " +
                   f"ATR: {bar_data.get('atr', None):.2f}, Regime: {self.active_trade['regime']}")
        
        return self.active_trade
    
    def _update_active_trade(self, latest_bar):
        """
        Update the active trade status with the latest price data.
        
        Args:
            latest_bar: Latest price bar data
        """
        if not self.active_trade:
            return
        
        current_price = latest_bar['close']
        trade = self.active_trade
        
        # Update last price
        trade['last_price'] = current_price
        
        # Check for pyramiding opportunity
        if self.enable_pyramiding and self.last_atr_value:
            self._check_pyramid_opportunity(current_price, latest_bar)
        
        # Calculate profit in R multiples (risk units)
        if trade['side'] == 'buy':
            entry_risk = abs(trade['entry_price'] - trade['stop_loss'])
            current_profit = current_price - trade['entry_price']
            r_multiple = current_profit / entry_risk if entry_risk > 0 else 0
        else:  # short position
            entry_risk = abs(trade['stop_loss'] - trade['entry_price'])
            current_profit = trade['entry_price'] - current_price
            r_multiple = current_profit / entry_risk if entry_risk > 0 else 0
        
        # Store R multiple in trade data
        trade['current_r'] = r_multiple
        
        # Move to breakeven after 0.5R profit
        trade.setdefault('moved_to_breakeven', False)
        if r_multiple >= 0.5 and not trade['moved_to_breakeven']:
            if trade['side'] == 'buy':
                # Add small buffer (10% of ATR)
                buffer = self.last_atr_value * 0.1 if self.last_atr_value else current_price * 0.001
                new_stop = trade['entry_price'] + buffer
                
                # Only update if it would raise the stop
                if trade['stop_loss'] < new_stop:
                    trade['stop_loss'] = new_stop
                    trade['moved_to_breakeven'] = True
                    logger.info(f"Moved stop loss to breakeven+ at {new_stop:.2f} after reaching {r_multiple:.2f}R profit")
            else:  # sell/short position
                # Add small buffer (10% of ATR)
                buffer = self.last_atr_value * 0.1 if self.last_atr_value else current_price * 0.001
                new_stop = trade['entry_price'] - buffer
                
                # Only update if it would lower the stop
                if trade['stop_loss'] > new_stop:
                    trade['stop_loss'] = new_stop
                    trade['moved_to_breakeven'] = True
                    logger.info(f"Moved stop loss to breakeven+ at {new_stop:.2f} after reaching {r_multiple:.2f}R profit")
                    
        # Calculate R-based profit relative to ATR (for trailing stop tightening)
        if 'atr' in trade:
            atr_entry = trade['atr']
            pnl_r = (current_price - trade['entry_price']) / atr_entry
            if trade['side'] == 'sell':
                pnl_r = -pnl_r  # Invert for short trades
                
            # Check if trade is losing more than R_TRIGGER and needs tighter trail
            if (not trade.get('trail_tightened')) and pnl_r < -R_TRIGGER:
                if trade['side'] == 'buy':
                    # For long trades, move trail closer up to current price
                    new_trail = current_price - (ATR_TRAIL_LOSER * atr_entry)
                    if 'trailing_stop' not in trade or new_trail > trade['trailing_stop']:
                        trade['trailing_stop'] = new_trail
                else:  # short position
                    # For short trades, move trail closer down to current price
                    new_trail = current_price + (ATR_TRAIL_LOSER * atr_entry)
                    if 'trailing_stop' not in trade or new_trail < trade['trailing_stop']:
                        trade['trailing_stop'] = new_trail
                        
                # Mark that we've tightened the trail to avoid doing it again
                trade['trail_tightened'] = True
                logger.info(f"Tightened trailing stop to {trade['trailing_stop']:.2f} after {pnl_r:.2f}R adverse move")
        
        # Progressive trailing stop based on profit
        # More aggressive trailing as profit increases
        if r_multiple >= 2.0:
            # When profit >= 2R, use even tighter trailing (0.5 × ATR)
            trail_multiplier = 0.5
        elif r_multiple >= 1.5:
            # When profit >= 1.5R, use tighter trailing (0.75 × ATR)
            trail_multiplier = 0.75
        elif r_multiple >= 1.0:
            # When profit >= 1R, use normal trailing (1.0 × ATR)
            trail_multiplier = 1.0
        else:
            # Default trailing stop multiplier - wider to let trades breathe
            trail_multiplier = self.atr_trail_multiplier  # Default is 1.25
        
        # Update max/min price seen for trailing stop calculations
        if trade['side'] == 'buy':
            # For long positions, track the highest price seen
            if current_price > trade['max_price']:
                trade['max_price'] = current_price
                
                # Update trailing stop if applicable and ATR value is available
                if self.last_atr_value:
                    # Calculate new trailing stop using ATR and the adjusted multiplier
                    new_trailing_stop = trade['max_price'] - (self.last_atr_value * trail_multiplier)
                    
                    # Only update if it would move the stop higher (for long positions)
                    if trade['trailing_stop'] is None or new_trailing_stop > trade['trailing_stop']:
                        trade['trailing_stop'] = new_trailing_stop
                        logger.info(f"Updated trailing stop to {new_trailing_stop:.2f} for {trade['side']} trade (R={r_multiple:.2f}, multiplier={trail_multiplier})")
                        
                        # Once we're at 0.5R profit, make sure trailing stop is at least at breakeven
                        if r_multiple >= 0.5 and new_trailing_stop < trade['entry_price']:
                            trade['trailing_stop'] = trade['entry_price']
                            logger.info(f"Moved trailing stop to breakeven at {trade['entry_price']:.2f}")
        else:  # short position
            # For short positions, track the lowest price seen
            if current_price < trade['min_price']:
                trade['min_price'] = current_price
                
                # Update trailing stop if applicable and ATR value is available
                if self.last_atr_value:
                    # Calculate new trailing stop using ATR and the adjusted multiplier
                    new_trailing_stop = trade['min_price'] + (self.last_atr_value * trail_multiplier)
                    
                    # Only update if it would move the stop lower (for short positions)
                    if trade['trailing_stop'] is None or new_trailing_stop < trade['trailing_stop']:
                        trade['trailing_stop'] = new_trailing_stop
                        logger.info(f"Updated trailing stop to {new_trailing_stop:.2f} for {trade['side']} trade (R={r_multiple:.2f}, multiplier={trail_multiplier})")
                        
                        # Once we're at 0.5R profit, make sure trailing stop is at least at breakeven
                        if r_multiple >= 0.5 and new_trailing_stop > trade['entry_price']:
                            trade['trailing_stop'] = trade['entry_price']
                            logger.info(f"Moved trailing stop to breakeven at {trade['entry_price']:.2f}")
        
        # Check if stop loss or take profit has been hit
        if trade['side'] == 'buy':
            # For long positions
            if trade['trailing_stop'] and current_price <= trade['trailing_stop']:
                self._close_trade(current_price, 'trailing_stop')
            elif current_price <= trade['stop_loss']:
                self._close_trade(current_price, 'stop_loss')
            elif trade['take_profit'] and current_price >= trade['take_profit']:
                self._close_trade(current_price, 'take_profit')
        else:
            # For short positions
            if trade['trailing_stop'] and current_price >= trade['trailing_stop']:
                self._close_trade(current_price, 'trailing_stop')
            elif current_price >= trade['stop_loss']:
                self._close_trade(current_price, 'stop_loss')
            elif trade['take_profit'] and current_price <= trade['take_profit']:
                self._close_trade(current_price, 'take_profit')
    
    def _check_pyramid_opportunity(self, current_price, latest_bar):
        """
        Check if we can add to a winning position (pyramid).
        
        Args:
            current_price: Current market price
            latest_bar: Latest price bar data
        """
        trade = self.active_trade
        
        # Don't pyramid if we've reached the maximum number of entries
        if 'pyramid_count' not in trade:
            trade['pyramid_count'] = 0
            trade['pyramid_levels'] = []
        
        if trade['pyramid_count'] >= self.max_pyramid_entries:
            return
        
        # Check if pyramiding is disabled based on volatility regime
        if hasattr(self, 'allocator') and hasattr(self.allocator, 'should_enable_pyramiding'):
            if not self.allocator.should_enable_pyramiding(self.symbol):
                logger.info(f"Skipping pyramid opportunity - pyramiding disabled in current regime ({trade.get('regime', 'normal')})")
                return
        elif not self.enable_pyramiding:
            return
        
        # Calculate profit threshold for adding to position
        if self.last_atr_value:
            profit_threshold = self.last_atr_value * self.pyramid_threshold
        else:
            profit_threshold = current_price * TRAIL_ACTIVATION_PCT
        
        # Check if trade has moved enough in our favor for pyramiding
        if trade['side'] == 'buy':
            profit_in_points = current_price - trade['entry_price']
            has_moved_enough = profit_in_points > profit_threshold
            
            # For long trades, add when price increases by threshold
            if has_moved_enough:
                self._add_to_position(current_price, latest_bar)
                
                # Move stop loss to breakeven + small buffer after first pyramid
                if trade['pyramid_count'] == 1 and 'initial_stop_loss' not in trade:
                    trade['initial_stop_loss'] = trade['stop_loss']  # Save initial stop for reference
                    
                    # Move stop to breakeven + small buffer (0.1 × ATR or 0.1%)
                    buffer = self.last_atr_value * 0.1 if self.last_atr_value else current_price * 0.001
                    new_stop = trade['entry_price'] + buffer
                    
                    # Only move stop if it would be higher than the current stop
                    if new_stop > trade['stop_loss']:
                        trade['stop_loss'] = new_stop
                        logger.info(f"Moved stop loss to breakeven+ at {new_stop:.2f} after first pyramid")
        
        else:  # short position
            profit_in_points = trade['entry_price'] - current_price
            has_moved_enough = profit_in_points > profit_threshold
            
            # For short trades, add when price decreases by threshold
            if has_moved_enough:
                self._add_to_position(current_price, latest_bar)
                
                # Move stop loss to breakeven + small buffer after first pyramid
                if trade['pyramid_count'] == 1 and 'initial_stop_loss' not in trade:
                    trade['initial_stop_loss'] = trade['stop_loss']  # Save initial stop for reference
                    
                    # Move stop to breakeven + small buffer (0.1 × ATR or 0.1%)
                    buffer = self.last_atr_value * 0.1 if self.last_atr_value else current_price * 0.001
                    new_stop = trade['entry_price'] - buffer
                    
                    # Only move stop if it would be lower than the current stop
                    if new_stop < trade['stop_loss']:
                        trade['stop_loss'] = new_stop
                        logger.info(f"Moved stop loss to breakeven+ at {new_stop:.2f} after first pyramid")
    
    def _add_to_position(self, current_price, latest_bar):
        """
        Add to an existing position (pyramid).
        
        Args:
            current_price: Current market price
            latest_bar: Latest price bar data
        """
        trade = self.active_trade
        
        # Calculate additional position size (50% of initial position)
        additional_size = trade['amount'] * self.pyramid_position_scale
        
        # Update the trade
        trade['pyramid_count'] += 1
        trade['pyramid_levels'].append({
            'price': current_price,
            'time': datetime.now(),
            'amount': additional_size
        })
        
        # Update the overall position
        initial_notional = trade['amount'] * trade['entry_price']
        additional_notional = additional_size * current_price
        total_notional = initial_notional + sum(level['amount'] * level['price'] for level in trade['pyramid_levels'])
        total_size = trade['amount'] + sum(level['amount'] for level in trade['pyramid_levels'])
        
        # Calculate new average entry price
        trade['avg_entry_price'] = total_notional / total_size
        
        # Update total position size
        trade['amount'] = total_size
        trade['notional_value'] = total_notional
        
        logger.info(f"Added to {trade['side']} position: {additional_size:.6f} coins at {current_price:.2f}")
        logger.info(f"New position size: {total_size:.6f} coins, Avg entry: {trade['avg_entry_price']:.2f}")
        
        # Update database if available
        if self.db_session and 'id' in trade:
            try:
                # Get the trade from the database
                db_trade = self.db_session.query(Trade).filter(Trade.id == trade['id']).first()
                if db_trade:
                    # Update the position size and add pyramid info
                    db_trade.amount = total_size
                    db_trade.avg_entry_price = trade['avg_entry_price']
                    db_trade.pyramid_info = json.dumps(trade['pyramid_levels'])
                    self.db_session.commit()
            except Exception as e:
                logger.error(f"Error updating trade in database: {str(e)}")
                if self.db_session:
                    self.db_session.rollback()
    
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
            pnl = (current_price - self.active_trade['entry_price']) * self.active_trade['amount']
            pnl_percent = (current_price - self.active_trade['entry_price']) / self.active_trade['entry_price']
        else:
            pnl = (self.active_trade['entry_price'] - current_price) * self.active_trade['amount']
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

    def generate_signals(self, df):
        """
        Generate trading signals based on EMA crossovers and filters.
        This method is used by the backtest system.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals added
        """
        # Apply indicators first (EMAs, ATR, RSI, etc.)
        df = self.apply_indicators(df)
        
        # Add a signal column
        df['signal'] = 0
        
        # Generate signals based on crossover events
        for i in range(1, len(df)):
            # Check for crossover events
            if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
                # Fast EMA crosses above slow EMA - potential buy signal
                if self.use_dynamic_trend_filter:
                    # Skip trend filter if volatility is low
                    if df['atr_pct'].iloc[i] < self.atr_low_threshold:
                        trend_filter_passed = True
                        trend_note = f"ATR({df['atr_pct'].iloc[i]:.4f}) < {self.atr_low_threshold} - Low volatility, trend filter bypassed"
                    else:
                        # Apply trend filter - price should be above the trend EMA
                        trend_filter_passed = df['close'].iloc[i] > df['ema_trend'].iloc[i]
                        trend_note = "200-EMA filter disabled for more signals"
                else:
                    trend_filter_passed = True
                    trend_note = "200-EMA filter disabled for more signals"
                
                # Apply RSI filter - don't buy if RSI is already overbought
                rsi_filter_passed = not self.use_rsi_filter or df['rsi'].iloc[i] < self.rsi_overbought
                rsi_note = f"RSI({df['rsi'].iloc[i]:.1f}) < {self.rsi_overbought}"
                
                # Apply volume filter - higher than average volume
                volume_filter_passed = not self.use_volume_filter or df['volume_ratio'].iloc[i] > self.volume_threshold
                volume_note = f"Volume({df['volume_ratio'].iloc[i]:.2f}x) > {self.volume_threshold}"
                
                # Set buy signal if all filters pass
                if trend_filter_passed and rsi_filter_passed and volume_filter_passed:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_reason'] = f"EMA Crossover: Fast({self.fast_ema}) > Slow({self.slow_ema}) + {trend_note} + {rsi_note} + {volume_note}"
                
            elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
                # Fast EMA crosses below slow EMA - potential sell signal
                if self.use_dynamic_trend_filter:
                    # Skip trend filter if volatility is low
                    if df['atr_pct'].iloc[i] < self.atr_low_threshold:
                        trend_filter_passed = True
                        trend_note = f"ATR({df['atr_pct'].iloc[i]:.4f}) < {self.atr_low_threshold} - Low volatility, trend filter bypassed"
                    else:
                        # Apply trend filter - price should be below the trend EMA
                        trend_filter_passed = df['close'].iloc[i] < df['ema_trend'].iloc[i]
                        trend_note = "200-EMA filter disabled for more signals"
                else:
                    trend_filter_passed = True
                    trend_note = "200-EMA filter disabled for more signals"
                
                # Apply RSI filter - don't sell if RSI is already oversold
                rsi_filter_passed = not self.use_rsi_filter or df['rsi'].iloc[i] > self.rsi_oversold
                rsi_note = f"RSI({df['rsi'].iloc[i]:.1f}) > {self.rsi_oversold}"
                
                # Apply volume filter - higher than average volume
                volume_filter_passed = not self.use_volume_filter or df['volume_ratio'].iloc[i] > self.volume_threshold
                volume_note = f"Volume({df['volume_ratio'].iloc[i]:.2f}x) > {self.volume_threshold}"
                
                # Set sell signal if all filters pass
                if trend_filter_passed and rsi_filter_passed and volume_filter_passed:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_reason'] = f"EMA Crossover: Fast({self.fast_ema}) < Slow({self.slow_ema}) + {trend_note} + {rsi_note} + {volume_note}"
        
        # Add signal type for logging
        df['signal_type'] = 'none'
        df.loc[df['signal'] == 1, 'signal_type'] = 'buy'
        df.loc[df['signal'] == -1, 'signal_type'] = 'sell'
        
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