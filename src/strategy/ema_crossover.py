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
    VOL_TARGET_PCT, MAX_POSITION_PCT, TRAIL_ACTIVATION_PCT
)
from src.utils.logger import logger, log_trade
from src.strategy.ema_optimizer import find_best_ema_pair, fetch_historical_data
from src.db.models import Trade
from src.strategy.base_strategy import BaseStrategy

class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover strategy that automatically finds optimal parameters.
    """
    
    def __init__(self, symbol=SYMBOL, timeframe='4h', db_session=None, account_balance=1000.0, 
                history_days=365, auto_optimize=False, config=None,
                fast_ema=10, slow_ema=40, trend_ema=200, atr_sl_multiplier=1.0,
                risk_per_trade=0.0075, use_volatility_sizing=True, vol_target_pct=0.0075,
                enable_pyramiding=True, max_pyramid_entries=2, health_monitor=None):
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
        
        # Get EMA parameters - either from config or explicitly provided
        if config:
            self.fast_ema = config.get('ema_fast', fast_ema)
            self.slow_ema = config.get('ema_slow', slow_ema)
            self.trend_ema = config.get('ema_trend', trend_ema)
            self.atr_sl_multiplier = config.get('atr_sl_multiplier', atr_sl_multiplier)
            self.atr_tp_multiplier = config.get('atr_tp_multiplier', None)
            self.atr_trail_multiplier = config.get('atr_trail_multiplier', atr_sl_multiplier)
            self.risk_per_trade = config.get('risk_per_trade', risk_per_trade)
            self.use_volatility_sizing = config.get('use_volatility_sizing', use_volatility_sizing)
            self.vol_target_pct = config.get('vol_target_pct', vol_target_pct)
            self.enable_pyramiding = config.get('enable_pyramiding', enable_pyramiding)
            self.max_pyramid_entries = config.get('max_pyramid_entries', max_pyramid_entries)
            self.pyramid_threshold = config.get('pyramid_threshold', 0.5)
            self.pyramid_position_scale = config.get('pyramid_position_scale', 0.5)
        else:
            # Use explicitly provided parameters
            self.fast_ema = fast_ema
            self.slow_ema = slow_ema
            self.trend_ema = trend_ema
            self.atr_sl_multiplier = atr_sl_multiplier
            self.atr_tp_multiplier = None  # Removed fixed TP
            self.atr_trail_multiplier = atr_sl_multiplier  # Default trail = SL
            self.risk_per_trade = risk_per_trade
            self.use_volatility_sizing = use_volatility_sizing
            self.vol_target_pct = vol_target_pct
            self.enable_pyramiding = enable_pyramiding
            self.max_pyramid_entries = max_pyramid_entries
            self.pyramid_threshold = 0.5  # 0.5 × ATR
            self.pyramid_position_scale = 0.5  # 50% of initial size
        
        # Automatically find optimal EMA pair if requested
        if auto_optimize:
            self.optimize_ema_parameters()
            
        logger.info(f"Initialized EMA Crossover Strategy with EMA{self.fast_ema}/{self.slow_ema} on {self.timeframe} timeframe")
    
    def optimize_ema_parameters(self):
        """Find the optimal EMA parameters through backtesting"""
        logger.info(f"Finding optimal EMA parameters for {self.symbol} on {self.timeframe} timeframe...")
        
        try:
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
                
                self.fast_ema = fast
                self.slow_ema = slow
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
        
        # Get the latest signal
        signal = self.get_signal(df_with_indicators)
        
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
        Apply EMA indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check if dataframe is empty or has sufficient data
        if df.empty:
            logger.warning("Cannot apply indicators to empty dataframe")
            return df
        
        try:
            # Calculate EMAs
            df[f'ema_{self.fast_ema}'] = df['close'].ewm(span=self.fast_ema, adjust=False).mean()
            df[f'ema_{self.slow_ema}'] = df['close'].ewm(span=self.slow_ema, adjust=False).mean()
            df[f'ema_{self.trend_ema}'] = df['close'].ewm(span=self.trend_ema, adjust=False).mean()  # Added trend EMA
            
            # Calculate EMA direction (1 for bullish, -1 for bearish, 0 for neutral)
            df['ema_trend'] = 0
            df.loc[df[f'ema_{self.fast_ema}'] > df[f'ema_{self.slow_ema}'], 'ema_trend'] = 1
            df.loc[df[f'ema_{self.fast_ema}'] < df[f'ema_{self.slow_ema}'], 'ema_trend'] = -1
            
            # Calculate crossover points (1 for bullish crossover, -1 for bearish crossover)
            df['ema_crossover'] = df['ema_trend'].diff().fillna(0)
            
            # Calculate long-term trend direction
            df['long_term_trend'] = 0
            df.loc[df['close'] > df[f'ema_{self.trend_ema}'], 'long_term_trend'] = 1
            df.loc[df['close'] < df[f'ema_{self.trend_ema}'], 'long_term_trend'] = -1
            
            # Add RSI for confirmation
            df = self._calculate_rsi(df, period=2)
            
            # Add ATR for volatility-based stops
            df = self._calculate_atr(df, period=14)
            
            # Add volume indicators
            df = self._calculate_volume_indicators(df)
            
        except Exception as e:
            logger.error(f"Error applying indicators: {str(e)}")
            # If we encounter an error, return the original dataframe with minimal required columns
            if 'ema_trend' not in df.columns:
                df['ema_trend'] = 0  # Neutral
            if 'long_term_trend' not in df.columns:
                df['long_term_trend'] = 0  # Neutral
            if 'rsi' not in df.columns:
                df['rsi'] = 50  # Neutral
            if 'volume_spike' not in df.columns:
                df['volume_spike'] = False  # No spike
            
        return df
    
    def _calculate_rsi(self, df, period=2):
        """Calculate RSI indicator"""
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        
        # Calculate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
        
        # Calculate average gains and losses
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        
        # Handle division by zero
        df['avg_loss'] = df['avg_loss'].replace(0, 0.001)
        
        # Calculate relative strength
        df['rs'] = df['avg_gain'] / df['avg_loss']
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Clean up temporary columns
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
        
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
    
    def get_signal(self, df, index=-1):
        """
        Generate trading signal based on the latest indicators.
        
        Args:
            df: DataFrame with OHLCV data
            index: Index to get signal for (-1 for latest)
            
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
        
        # EMA crossover signals with trend filter
        if 'ema_crossover' in row and row['ema_crossover'] != 0:
            # Long signal: Fast EMA crossed above slow EMA AND price > EMA200
            if row['ema_crossover'] > 0 and row['long_term_trend'] > 0:
                signal_type = "buy"
            # Short signal: Fast EMA crossed below slow EMA AND price < EMA200
            elif row['ema_crossover'] < 0 and row['long_term_trend'] < 0:
                signal_type = "sell"
        
        # Additional conditions for signal validity
        if signal_type:
            # Avoid trading with insufficient data (need at least 200 bars for reliable trend)
            if len(df) < 200:
                signal_type = ""
                
            # Check if there was a recent trade (reduce overtrading)
            if self.last_signal_time is not None:
                min_bars_between_trades = 2  # Reduced from 5 to 2 for 4h timeframe
                current_time = row.name if hasattr(row, 'name') else df.index[index]
                
                # Skip if too soon after last trade
                if (current_time - self.last_signal_time).total_seconds() < min_bars_between_trades * 14400:  # 4 hours = 14400 seconds
                    signal_type = ""
        
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
            if 'atr' in bar_data and not pd.isna(bar_data['atr']) and hasattr(self.config, 'USE_VOLATILITY_SIZING') and self.config.USE_VOLATILITY_SIZING:
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
            'pyramid_levels': []
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
                   f"ATR: {bar_data.get('atr', None):.2f}")
        
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
        
        # Update max/min price seen for trailing stop calculations
        if trade['side'] == 'buy':
            # For long positions, track the highest price seen
            if current_price > trade['max_price']:
                trade['max_price'] = current_price
                
                # Update trailing stop if applicable and ATR value is available
                if self.last_atr_value:
                    # Calculate new trailing stop using ATR
                    new_trailing_stop = trade['max_price'] - (self.last_atr_value * self.atr_trail_multiplier)
                    
                    # Only update if it would move the stop higher (for long positions)
                    if trade['trailing_stop'] is None or new_trailing_stop > trade['trailing_stop']:
                        trade['trailing_stop'] = new_trailing_stop
                        logger.info(f"Updated trailing stop to {new_trailing_stop:.2f} for {trade['side']} trade")
        else:  # short position
            # For short positions, track the lowest price seen
            if current_price < trade['min_price']:
                trade['min_price'] = current_price
                
                # Update trailing stop if applicable and ATR value is available
                if self.last_atr_value:
                    # Calculate new trailing stop using ATR
                    new_trailing_stop = trade['min_price'] + (self.last_atr_value * self.atr_trail_multiplier)
                    
                    # Only update if it would move the stop lower (for short positions)
                    if trade['trailing_stop'] is None or new_trailing_stop < trade['trailing_stop']:
                        trade['trailing_stop'] = new_trailing_stop
                        logger.info(f"Updated trailing stop to {new_trailing_stop:.2f} for {trade['side']} trade")
        
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
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Calculate average return percentage
            avg_return = sum(t.pnl_percent for t in trades if t.pnl_percent is not None) / total_trades
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
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