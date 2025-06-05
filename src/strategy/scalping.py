from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, RISK_PER_TRADE,
    RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD, USE_ATR_STOPS,
    ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER, USE_TWO_LEG_STOP,
    TRAIL_ACTIVATION_PCT, TRAIL_ATR_MULTIPLIER, USE_SOFT_STOP,
    MAX_TRADES_PER_HOUR, MIN_BARS_BETWEEN_TRADES, MIN_CONSECUTIVE_BARS_AGREE,
    LOG_FALSE_POSITIVES, MAX_TRADE_DURATION_MINUTES
)
from src.indicators.technical import apply_indicators, get_signal
from src.db.models import Trade, FalsePositive
from src.utils.logger import logger, log_trade, consecutive_sl_alert

class ScalpingStrategy:
    """BTC/USDT intra-day scalping strategy implementation."""
    
    def __init__(self, db_session, account_balance=1000.0):
        self.db_session = db_session
        self.symbol = SYMBOL
        self.account_balance = account_balance
        self.active_trade = None
        self.consecutive_sl_count = 0
        self.last_signal = None
        self.higher_tf_trend = None  # Store higher timeframe trend
        self.last_signal_time = None  # For trade frequency control
        self.hourly_trade_count = 0  # For trade frequency control
        self.hourly_reset_time = datetime.now()  # For trade frequency control
        self.trailing_activated = False  # For two-leg stop
        
    def update(self, df, higher_tf_df=None):
        """
        Update the strategy with new data and generate signals.
        
        Args:
            df: DataFrame with OHLCV data
            higher_tf_df: Optional DataFrame with higher timeframe data
            
        Returns:
            dict: Current strategy state and signals
        """
        # Compute indicators but keep any precomputed columns provided in df
        indicators = apply_indicators(df)
        df_with_indicators = df.copy()
        for col in indicators.columns:
            if col not in df_with_indicators.columns:
                df_with_indicators[col] = indicators[col]
        
        # If higher timeframe data is provided, use it for trend confirmation
        if higher_tf_df is not None:
            higher_tf_indicators = apply_indicators(higher_tf_df)
            if not higher_tf_indicators.empty:
                # Get the latest row
                latest_higher_tf = higher_tf_indicators.iloc[-1]
                # Store the higher timeframe trend
                self.higher_tf_trend = latest_higher_tf['market_trend']
        
        # Check if it's time to reset the hourly trade counter
        current_time = datetime.now()
        if (current_time - self.hourly_reset_time).total_seconds() > 3600:
            self.hourly_trade_count = 0
            self.hourly_reset_time = current_time
        
        # Get the latest signal, passing the last signal time for trade frequency control
        signal = get_signal(df_with_indicators, last_signal_time=self.last_signal_time, 
                           min_bars_between=MIN_BARS_BETWEEN_TRADES)
        
        # Override signal if higher timeframe trend doesn't align
        if self.higher_tf_trend is not None:
            if signal['signal'] == 'buy' and self.higher_tf_trend <= 0:
                signal['signal'] = 'neutral'  # No long positions in downtrend
            elif signal['signal'] == 'sell' and self.higher_tf_trend >= 0:
                signal['signal'] = 'neutral'  # No short positions in uptrend
        
        # Check if we've reached the maximum trades per hour
        if self.hourly_trade_count >= MAX_TRADES_PER_HOUR and signal['signal'] != 'neutral':
            logger.info(f"Maximum trades per hour reached ({MAX_TRADES_PER_HOUR}). Skipping signal.")
            signal['signal'] = 'neutral'
        
        # Check for consecutive bars agreement
        if signal['signal'] != 'neutral' and MIN_CONSECUTIVE_BARS_AGREE > 1:
            agree_count = 0
            for i in range(1, MIN_CONSECUTIVE_BARS_AGREE):
                if i < len(df_with_indicators):
                    prev_signal = get_signal(df_with_indicators, index=-1-i)
                    if prev_signal['signal'] == signal['signal']:
                        agree_count += 1
            
            if agree_count < MIN_CONSECUTIVE_BARS_AGREE - 1:
                logger.debug(f"Signal doesn't have {MIN_CONSECUTIVE_BARS_AGREE} consecutive bars agreement. Skipping.")
                signal['signal'] = 'neutral'
        
        self.last_signal = signal
        
        # Update active trade if exists
        if self.active_trade:
            self._update_active_trade(df_with_indicators.iloc[-1])
            
            # Check for false positives (trades that have been open too long)
            self._check_false_positive_trade()
        
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
        Update the active trade status with the latest price data.
        
        Args:
            latest_bar: Latest price bar data
        """
        current_price = latest_bar['close']
        trade = self.active_trade
        
        if not trade:
            return
        
        # Check for trailing stop activation
        if USE_TWO_LEG_STOP and not self.trailing_activated:
            # Calculate current profit percentage
            if trade['side'] == 'buy':
                profit_pct = (current_price - trade['entry_price']) / trade['entry_price']
            else:
                profit_pct = (trade['entry_price'] - current_price) / trade['entry_price']
            
            # If profit exceeds activation threshold, activate trailing stop
            if profit_pct >= TRAIL_ACTIVATION_PCT / 100:
                self.trailing_activated = True
                
                # Update stop loss to trailing level based on ATR
                atr_value = latest_bar.get('atr', None)
                if atr_value and not pd.isna(atr_value):
                    if trade['side'] == 'buy':
                        new_stop = current_price - (atr_value * TRAIL_ATR_MULTIPLIER)
                        # Only move stop loss up
                        if new_stop > trade['stop_loss']:
                            trade['stop_loss'] = new_stop
                            logger.info(f"Trailing stop activated at {profit_pct*100:.2f}% profit. "
                                       f"New SL: {new_stop:.2f} ({TRAIL_ATR_MULTIPLIER}x ATR)")
                    else:  # sell
                        new_stop = current_price + (atr_value * TRAIL_ATR_MULTIPLIER)
                        # Only move stop loss down
                        if new_stop < trade['stop_loss']:
                            trade['stop_loss'] = new_stop
                            logger.info(f"Trailing stop activated at {profit_pct*100:.2f}% profit. "
                                       f"New SL: {new_stop:.2f} ({TRAIL_ATR_MULTIPLIER}x ATR)")
        
        # Update trailing stop if already activated
        elif USE_TWO_LEG_STOP and self.trailing_activated:
            atr_value = latest_bar.get('atr', None)
            if atr_value and not pd.isna(atr_value):
                if trade['side'] == 'buy':
                    new_stop = current_price - (atr_value * TRAIL_ATR_MULTIPLIER)
                    # Only move stop loss up
                    if new_stop > trade['stop_loss']:
                        trade['stop_loss'] = new_stop
                        logger.debug(f"Updated trailing stop to {new_stop:.2f}")
                else:  # sell
                    new_stop = current_price + (atr_value * TRAIL_ATR_MULTIPLIER)
                    # Only move stop loss down
                    if new_stop < trade['stop_loss']:
                        trade['stop_loss'] = new_stop
                        logger.debug(f"Updated trailing stop to {new_stop:.2f}")
        
        # Check for soft stop alerts
        if USE_SOFT_STOP:
            # Calculate distance to stop as percentage
            if trade['side'] == 'buy':
                stop_distance_pct = (current_price - trade['stop_loss']) / current_price * 100
                if stop_distance_pct < 0.05:  # When price is within 0.05% of stop loss
                    logger.warning(f"Soft stop alert! Price is near stop loss (distance: {stop_distance_pct:.2f}%). "
                                  f"Current: {current_price:.2f}, SL: {trade['stop_loss']:.2f}")
            else:  # sell
                stop_distance_pct = (trade['stop_loss'] - current_price) / current_price * 100
                if stop_distance_pct < 0.05:  # When price is within 0.05% of stop loss
                    logger.warning(f"Soft stop alert! Price is near stop loss (distance: {stop_distance_pct:.2f}%). "
                                  f"Current: {current_price:.2f}, SL: {trade['stop_loss']:.2f}")
        
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
    
    def _check_false_positive_trade(self):
        """Check for trades that have been open too long without hitting TP or SL."""
        if not self.active_trade or not LOG_FALSE_POSITIVES:
            return
        
        # Calculate how long the trade has been open
        entry_time = self.active_trade['entry_time']
        current_time = datetime.now()
        trade_duration = (current_time - entry_time).total_seconds() / 60  # Duration in minutes
        
        # If trade has been open longer than threshold, log it as a false positive and close it
        if trade_duration > MAX_TRADE_DURATION_MINUTES:
            logger.warning(f"Trade has been open for {trade_duration:.1f} minutes without hitting TP or SL. "
                          f"Logging as false positive and closing.")
            
            # Log false positive
            self._log_false_positive(self.active_trade, trade_duration)
            
            # Close the trade
            current_price = self.active_trade['last_price'] if 'last_price' in self.active_trade else self.active_trade['entry_price']
            self._close_trade(current_price, 'timeout')
    
    def _log_false_positive(self, trade, duration_minutes):
        """Log a false positive trade for analysis."""
        try:
            # Create false positive record
            false_positive = FalsePositive(
                symbol=trade['symbol'],
                side=trade['side'],
                entry_time=trade['entry_time'],
                entry_price=trade['entry_price'],
                duration_minutes=duration_minutes,
                market_trend=trade.get('market_trend', None),
                higher_tf_trend=trade.get('higher_tf_trend', None),
                rsi_value=trade.get('rsi', None),
                atr_value=trade.get('atr', None)
            )
            
            # Save to database
            self.db_session.add(false_positive)
            self.db_session.commit()
            
            logger.info(f"Logged false positive trade: {trade['side']} at {trade['entry_price']}, "
                       f"duration: {duration_minutes:.1f} minutes")
            
        except Exception as e:
            logger.error(f"Error logging false positive: {str(e)}")
            self.db_session.rollback()
    
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

        side = signal['signal']  # 'buy' or 'sell'

        # Support passing a raw price instead of a bar dictionary for tests
        if isinstance(bar_data, (int, float)):
            current_price = bar_data
            bar_data = {}
        else:
            current_price = bar_data['close']
        
        # Calculate stop loss and take profit levels
        if USE_ATR_STOPS and 'atr' in bar_data and not pd.isna(bar_data['atr']):
            # Use ATR-based stops
            atr_value = bar_data['atr']
            
            if side == 'buy':
                stop_loss_price = current_price - (atr_value * ATR_SL_MULTIPLIER)
                take_profit_price = current_price + (atr_value * ATR_TP_MULTIPLIER)
            else:  # sell
                stop_loss_price = current_price + (atr_value * ATR_SL_MULTIPLIER)
                take_profit_price = current_price - (atr_value * ATR_TP_MULTIPLIER)
                
            logger.info(f"Using ATR-based stops: ATR={atr_value:.2f}, SL={ATR_SL_MULTIPLIER}xATR, TP={ATR_TP_MULTIPLIER}xATR")
        else:
            # Fallback to fixed percentage stops
            stop_loss_price = (
                current_price * (1 - STOP_LOSS_PCT) if side == 'buy' 
                else current_price * (1 + STOP_LOSS_PCT)
            )
            take_profit_price = (
                current_price * (1 + TAKE_PROFIT_PCT) if side == 'buy' 
                else current_price * (1 - TAKE_PROFIT_PCT)
            )
            
            logger.info(f"Using fixed percentage stops: SL={STOP_LOSS_PCT*100:.3f}%, TP={TAKE_PROFIT_PCT*100:.3f}%")
        
        # Calculate position size based on risk per trade
        risk_amount = self.account_balance * RISK_PER_TRADE
        
        # Calculate the potential loss per unit
        risk_per_unit = abs(current_price - stop_loss_price)
        
        # Calculate the number of units to trade
        position_size = risk_amount / risk_per_unit
        
        # Calculate notional value
        notional_value = position_size * current_price
        
        # Get market trend and RSI values
        market_trend = bar_data.get('market_trend', 0)
        rsi_value = bar_data.get('rsi', None)
        atr_value = bar_data.get('atr', None)
        
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
            'higher_tf_trend': self.higher_tf_trend,
            'market_trend': market_trend,
            'rsi': rsi_value,
            'atr': atr_value,
            'last_price': current_price,  # Track last price for false positive handling
            'adaptive_threshold_used': signal.get('adaptive_threshold_used', False),
            'micro_trend': signal.get('micro_trend', 0),
            'momentum_signal': True if side == 'buy' and signal.get('momentum_up', False) 
                               else True if side == 'sell' and signal.get('momentum_down', False) else False
        }
        
        # Reset trailing stop flag
        self.trailing_activated = False
        
        # Log the trade
        log_trade(self.active_trade)
        
        # Store in database
        db_trade = Trade(
            symbol=self.symbol,
            side=side,
            entry_time=self.active_trade['entry_time'],
            entry_price=current_price,
            amount=position_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
            rsi_value=rsi_value,
            atr_value=atr_value,
            market_trend=market_trend,
            higher_tf_trend=self.higher_tf_trend,
            micro_trend=signal.get('micro_trend', 0),
            momentum_confirmed=self.active_trade['momentum_signal']
        )
        self.db_session.add(db_trade)
        self.db_session.commit()
        
        # Update trade with ID from database
        self.active_trade['id'] = db_trade.id
        
        logger.info(f"Opened {side} trade at {current_price} with SL: {stop_loss_price}, TP: {take_profit_price}")

        # Safely format optional values for logging
        rsi_display = f"{rsi_value:.1f}" if rsi_value is not None else "n/a"
        atr_display = f"{atr_value:.2f}" if atr_value is not None else "n/a"

        logger.info(
            f"Market conditions: RSI={rsi_display}, ATR={atr_display}, "
            f"Trend={'Up' if market_trend > 0 else 'Down'}, "
            f"Higher TF={'Up' if (self.higher_tf_trend or 0) > 0 else 'Down' if (self.higher_tf_trend or 0) < 0 else 'Neutral'}, "
            f"Micro-trend={'Up' if signal.get('micro_trend', 0) > 0 else 'Down'}, "
            f"Momentum: {'Confirmed' if self.active_trade['momentum_signal'] else 'Not confirmed'}"
        )
        
        return self.active_trade
    
    def _close_trade(self, current_price, reason):
        """
        Close the active trade.
        
        Args:
            current_price: Current price
            reason: Reason for closing ('stop_loss', 'take_profit', 'manual', 'timeout', etc.)
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
            'pnl_percent': pnl_percent,
            'trailing_activated': self.trailing_activated
        }
        
        # Calculate trade duration
        trade_duration = (closed_trade['exit_time'] - closed_trade['entry_time']).total_seconds() / 60  # minutes
        
        # Log trade closing
        logger.info(f"Closed trade at {current_price} for {reason}, PnL: {pnl}, PnL%: {pnl_percent*100:.2f}%, "
                   f"Duration: {trade_duration:.1f} minutes, "
                   f"Trailing: {'Activated' if self.trailing_activated else 'Not activated'}")
        
        # Track consecutive stop losses
        if reason == 'stop_loss':
            self.consecutive_sl_count += 1
            consecutive_sl_alert(self.consecutive_sl_count)
        else:
            self.consecutive_sl_count = 0
        
        # Update in database
        try:
            trade = self.db_session.query(Trade).filter_by(id=self.active_trade['id']).first()
            if trade:
                trade.exit_time = closed_trade['exit_time']
                trade.exit_price = current_price
                trade.exit_reason = reason
                trade.pnl = pnl
                trade.pnl_percent = pnl_percent
                trade.duration_minutes = trade_duration
                trade.trailing_activated = self.trailing_activated
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error updating trade in database: {str(e)}")
            self.db_session.rollback()
        
        # Reset active trade and trailing flags
        self.active_trade = None
        self.trailing_activated = False
        
        return closed_trade
    
    def get_performance_summary(self, timeframe='daily'):
        """
        Get performance summary for the specified timeframe.
        
        Args:
            timeframe: 'daily', 'weekly', 'monthly'
            
        Returns:
            dict: Performance metrics
        """
        # Query completed trades
        try:
            if timeframe == 'daily':
                # Get trades from today
                today = datetime.now().date()
                trades = self.db_session.query(Trade).filter(
                    Trade.entry_time >= today,
                    Trade.exit_time.isnot(None)
                ).all()
            elif timeframe == 'weekly':
                # Implementation for weekly summary
                pass
            elif timeframe == 'monthly':
                # Implementation for monthly summary
                pass
            else:
                # Default to all trades
                trades = self.db_session.query(Trade).filter(
                    Trade.exit_time.isnot(None)
                ).all()
                
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t.pnl and t.pnl > 0)
            losing_trades = sum(1 for t in trades if t.pnl and t.pnl <= 0)
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
            avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            # Calculate average return percentage
            avg_return = sum(t.pnl_percent for t in trades if t.pnl_percent is not None) / total_trades if total_trades > 0 else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'pnl': total_pnl,
                'avg_pnl': avg_pnl,
                'avg_return': avg_return
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'pnl': 0,
                'avg_pnl': 0,
                'avg_return': 0
            } 