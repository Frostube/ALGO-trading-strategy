import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config_optimized import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, RISK_PER_TRADE,
    RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD, USE_ATR_STOPS,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, USE_TWO_LEG_STOP,
    TRAIL_ACTIVATION_PCT, TRAIL_ATR_MULTIPLIER, USE_SOFT_STOP,
    MAX_TRADES_PER_HOUR, MIN_BARS_BETWEEN_TRADES
)
from src.indicators.technical import apply_indicators, get_signal
from src.utils.logger import logger

class OptimizedStrategy:
    """BTC/USDT optimized trading strategy designed for 20%+ returns."""
    
    def __init__(self, db_session=None, account_balance=10000.0):
        self.db_session = db_session
        self.symbol = SYMBOL
        self.account_balance = account_balance
        self.active_trade = None
        self.consecutive_sl_count = 0
        self.consecutive_tp_count = 0  # Track consecutive winning trades
        self.last_signal = None
        self.higher_tf_trend = None  # Store higher timeframe trend
        self.last_signal_time = None  # For trade frequency control
        self.hourly_trade_count = 0  # For trade frequency control
        self.hourly_reset_time = datetime.now()  # For trade frequency control
        self.trailing_activated = False  # For two-leg stop
        self.win_streak = 0  # Track winning streak
        self.loss_streak = 0  # Track losing streak
        self.last_trade_profitable = None  # Track last trade result
        self.adaptive_sl_factor = 1.0  # Adaptive stop loss multiplier
        self.adaptive_tp_factor = 1.0  # Adaptive take profit multiplier
        
    def update(self, df, higher_tf_df=None):
        """
        Update the strategy with new data and generate signals.
        Uses price action, trend following and volatility-based position sizing.
        
        Args:
            df: DataFrame with OHLCV data
            higher_tf_df: Optional DataFrame with higher timeframe data
            
        Returns:
            dict: Current strategy state and signals
        """
        # Apply indicators to the data
        df_with_indicators = apply_indicators(df)
        
        # Higher timeframe trend confirmation
        if higher_tf_df is not None:
            higher_tf_indicators = apply_indicators(higher_tf_df)
            if not higher_tf_indicators.empty:
                self.higher_tf_trend = higher_tf_indicators.iloc[-1]['market_trend']
        
        # Reset hourly counter if needed
        current_time = datetime.now()
        if (current_time - self.hourly_reset_time).total_seconds() > 3600:
            self.hourly_trade_count = 0
            self.hourly_reset_time = current_time
        
        # Get the latest signal
        signal = get_signal(df_with_indicators, last_signal_time=self.last_signal_time, 
                           min_bars_between=MIN_BARS_BETWEEN_TRADES)
        
        # Enhanced Signal Logic - Only take trades aligned with higher timeframe trend
        if self.higher_tf_trend is not None:
            # Only long when higher timeframe trend is up
            if signal['signal'] == 'buy' and self.higher_tf_trend <= 0:
                signal['signal'] = 'neutral'
                signal['filtered_higher_tf'] = True
            # Only short when higher timeframe trend is down    
            elif signal['signal'] == 'sell' and self.higher_tf_trend >= 0:
                signal['signal'] = 'neutral'
                signal['filtered_higher_tf'] = True
        
        # Avoid trades after multiple consecutive losses
        if self.consecutive_sl_count >= 3 and signal['signal'] != 'neutral':
            logger.info(f"Skipping signal after {self.consecutive_sl_count} consecutive losses")
            signal['signal'] = 'neutral'
            signal['filtered_consecutive_losses'] = True
        
        # Adapt position sizing based on recent performance
        if self.win_streak >= 3:
            # Increase position size after winning streak
            self.adaptive_tp_factor = min(1.5, 1.0 + (self.win_streak * 0.1))
            logger.info(f"Win streak: {self.win_streak} - Increasing TP target factor to {self.adaptive_tp_factor}")
        elif self.loss_streak >= 2:
            # Decrease position size and tighten stops after losing streak
            self.adaptive_sl_factor = max(0.7, 1.0 - (self.loss_streak * 0.1))
            logger.info(f"Loss streak: {self.loss_streak} - Decreasing SL distance factor to {self.adaptive_sl_factor}")
        
        # Maximum trades per hour check
        if self.hourly_trade_count >= MAX_TRADES_PER_HOUR and signal['signal'] != 'neutral':
            signal['signal'] = 'neutral'
            signal['filtered_max_trades'] = True
        
        # Update last signal
        self.last_signal = signal
        
        # Update active trade if exists
        if self.active_trade:
            self._update_active_trade(df_with_indicators.iloc[-1])
        
        # Check for new trade signals if no active trade
        if not self.active_trade and signal['signal'] != 'neutral':
            self._open_trade(signal, df_with_indicators.iloc[-1])
            
            # Update trading frequency control
            if self.active_trade:  # If a trade was actually opened
                self.last_signal_time = signal['timestamp']
                self.hourly_trade_count += 1
        
        return {
            'signal': signal,
            'active_trade': self.active_trade,
            'account_balance': self.account_balance
        }
    
    def _update_active_trade(self, latest_bar):
        """
        Update the active trade status with enhanced trailing stops
        and dynamic take profit targets.
        
        Args:
            latest_bar: Latest price bar data
        """
        if not self.active_trade:
            return
            
        current_price = latest_bar['close']
        trade = self.active_trade
        
        # Calculate current unrealized profit/loss
        if trade['side'] == 'buy':
            unrealized_pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
        else:  # short
            unrealized_pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
        
        # Dynamic trailing stop activation
        if USE_TWO_LEG_STOP and not self.trailing_activated:
            # Activate trailing when price moves favorably by activation threshold
            if unrealized_pnl_pct >= TRAIL_ACTIVATION_PCT:
                self.trailing_activated = True
                
                # Set trailing stop based on ATR and current price
                atr_value = latest_bar.get('atr', current_price * 0.002)  # Default to 0.2% if ATR not available
                
                if trade['side'] == 'buy':
                    new_stop = current_price - (atr_value * TRAIL_ATR_MULTIPLIER)
                    # Only move stop loss up for longs
                    if new_stop > trade['stop_loss']:
                        trade['stop_loss'] = new_stop
                        logger.info(f"Trailing stop activated at {unrealized_pnl_pct*100:.2f}% profit. New SL: {new_stop:.2f}")
                else:  # sell
                    new_stop = current_price + (atr_value * TRAIL_ATR_MULTIPLIER)
                    # Only move stop loss down for shorts
                    if new_stop < trade['stop_loss']:
                        trade['stop_loss'] = new_stop
                        logger.info(f"Trailing stop activated at {unrealized_pnl_pct*100:.2f}% profit. New SL: {new_stop:.2f}")
        
        # Update trailing stop if already activated - make it tighter as profit increases
        elif USE_TWO_LEG_STOP and self.trailing_activated:
            atr_value = latest_bar.get('atr', current_price * 0.002)
            # Make trailing tighter as profit increases
            trail_factor = TRAIL_ATR_MULTIPLIER * max(0.5, 1.0 - (unrealized_pnl_pct * 4))
            
            if trade['side'] == 'buy':
                new_stop = current_price - (atr_value * trail_factor)
                # Only move stop loss up
                if new_stop > trade['stop_loss']:
                    trade['stop_loss'] = new_stop
            else:  # sell
                new_stop = current_price + (atr_value * trail_factor)
                # Only move stop loss down
                if new_stop < trade['stop_loss']:
                    trade['stop_loss'] = new_stop
        
        # Dynamic take profit adjustment based on market volatility
        if unrealized_pnl_pct > 0 and 'atr_pct' in latest_bar:
            volatility = latest_bar['atr_pct']
            if volatility > 0.5:  # Higher volatility
                if unrealized_pnl_pct > (TAKE_PROFIT_PCT * 0.7):  # If we're close to take profit
                    # Close trade early in high volatility
                    self._close_trade(current_price, 'volatility_tp')
                    return
        
        # Check if stop loss or take profit has been hit
        if trade['side'] == 'buy':
            # For long positions
            if current_price <= trade['stop_loss']:
                self._close_trade(current_price, 'stop_loss')
            elif current_price >= trade['take_profit']:
                self._close_trade(current_price, 'take_profit')
        else:
            # For short positions
            if current_price >= trade['stop_loss']:
                self._close_trade(current_price, 'stop_loss')
            elif current_price <= trade['take_profit']:
                self._close_trade(current_price, 'take_profit')
    
    def _open_trade(self, signal, bar_data):
        """
        Open a new trade with optimized entry timing and position sizing.
        
        Args:
            signal: Signal dictionary with trade direction
            bar_data: Current bar data
        """
        # Extract current price and other data
        current_price = bar_data['close']
        current_time = signal['timestamp']
        atr_value = bar_data.get('atr', current_price * 0.002)  # Default to 0.2% if ATR not available
        
        # Determine position type
        side = 'buy' if signal['signal'] == 'buy' else 'sell'
        
        # Determine stop loss and take profit levels
        if USE_ATR_STOPS and not pd.isna(atr_value):
            # Use ATR-based stops with adaptive factors
            sl_distance = atr_value * ATR_SL_MULTIPLIER * self.adaptive_sl_factor
            tp_distance = atr_value * ATR_TP_MULTIPLIER * self.adaptive_tp_factor
            
            if side == 'buy':
                stop_loss = current_price - sl_distance
                take_profit = current_price + tp_distance
            else:  # sell
                stop_loss = current_price + sl_distance
                take_profit = current_price - tp_distance
        else:
            # Use fixed percentage stops
            if side == 'buy':
                stop_loss = current_price * (1 - STOP_LOSS_PCT)
                take_profit = current_price * (1 + TAKE_PROFIT_PCT)
            else:  # sell
                stop_loss = current_price * (1 + STOP_LOSS_PCT)
                take_profit = current_price * (1 - TAKE_PROFIT_PCT)
        
        # Calculate risk/reward ratio
        if side == 'buy':
            risk = (current_price - stop_loss) / current_price
            reward = (take_profit - current_price) / current_price
        else:  # sell
            risk = (stop_loss - current_price) / current_price
            reward = (current_price - take_profit) / current_price
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Only take trades with good risk/reward ratio
        if risk_reward_ratio < 1.5:
            logger.info(f"Skipping {side} signal - insufficient risk/reward ratio: {risk_reward_ratio:.2f}")
            return
        
        # Calculate position size based on fixed risk percentage
        risk_amount = self.account_balance * RISK_PER_TRADE
        risk_per_unit = abs(current_price - stop_loss)
        position_size = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        # Adjust position size based on win/loss streak
        if self.win_streak >= 3:
            position_size *= min(1.5, 1.0 + (self.win_streak * 0.1))
        elif self.loss_streak >= 2:
            position_size *= max(0.5, 1.0 - (self.loss_streak * 0.1))
        
        # Create the trade
        self.active_trade = {
            'symbol': self.symbol,
            'side': side,
            'entry_time': current_time,
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'risk_reward': risk_reward_ratio,
            'strategy': signal.get('strategy', 'unknown'),
            'market_trend': signal.get('market_trend', 0),
            'higher_tf_trend': self.higher_tf_trend,
            'rsi': signal.get('rsi', 0),
            'atr': signal.get('atr', 0),
            'volume_spike': signal.get('volume_spike', False)
        }
        
        # Initialize trailing stop variables
        self.trailing_activated = False
        
        logger.info(f"Entering {side} position at {current_price:.2f} using {signal.get('strategy', 'unknown')} strategy")
        logger.info(f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, R/R: {risk_reward_ratio:.2f}")
    
    def _close_trade(self, current_price, reason):
        """
        Close an active trade and update performance metrics.
        
        Args:
            current_price: Current price at exit
            reason: Reason for closing the trade
        """
        if not self.active_trade:
            return
            
        trade = self.active_trade
        
        # Calculate P&L
        if trade['side'] == 'buy':
            pnl = (current_price - trade['entry_price']) * trade['position_size']
            pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
        else:  # sell
            pnl = (trade['entry_price'] - current_price) * trade['position_size']
            pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
        
        # Update account balance
        self.account_balance += pnl
        
        # Update win/loss streaks
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.consecutive_tp_count += 1
            self.consecutive_sl_count = 0
            self.last_trade_profitable = True
            
            # Reset adaptive factors on win
            if self.win_streak >= 3:
                self.adaptive_sl_factor = 1.0
        else:
            self.loss_streak += 1
            self.win_streak = 0
            self.consecutive_sl_count += 1
            self.consecutive_tp_count = 0
            self.last_trade_profitable = False
            
            # Alert on consecutive losses
            if self.consecutive_sl_count >= 3:
                logger.warning(f"⚠️ {self.consecutive_sl_count} consecutive losses detected")
        
        # Log trade
        exit_time = datetime.now()
        duration = exit_time - trade['entry_time'] if hasattr(trade['entry_time'], 'timestamp') else None
        
        logger.info(f"Closed {trade['side']} position at {current_price:.2f} ({reason})")
        logger.info(f"P&L: ${pnl:.2f} ({pnl_pct*100:.2f}%), Balance: ${self.account_balance:.2f}")
        
        # Add trade to database if session exists
        if self.db_session:
            try:
                from src.db.models import Trade
                db_trade = Trade(
                    symbol=trade['symbol'],
                    side=trade['side'],
                    entry_time=trade['entry_time'],
                    exit_time=exit_time,
                    entry_price=trade['entry_price'],
                    exit_price=current_price,
                    position_size=trade['position_size'],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=reason,
                    strategy=trade['strategy'],
                    market_trend=trade.get('market_trend', None),
                    rsi=trade.get('rsi', None)
                )
                self.db_session.add(db_trade)
                self.db_session.commit()
            except Exception as e:
                logger.error(f"Error recording trade to database: {str(e)}")
                if self.db_session:
                    self.db_session.rollback()
        
        # Reset active trade
        self.active_trade = None 