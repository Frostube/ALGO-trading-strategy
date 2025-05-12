from datetime import datetime
import pandas as pd
import numpy as np

from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, RISK_PER_TRADE,
    RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD
)
from src.indicators.technical import apply_indicators, get_signal
from src.db.models import Trade
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
        
    def update(self, df):
        """
        Update the strategy with new data and generate signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict: Current strategy state and signals
        """
        # Apply indicators to the data
        df_with_indicators = apply_indicators(df)
        
        # Get the latest signal
        signal = get_signal(df_with_indicators)
        self.last_signal = signal
        
        # Update active trade if exists
        if self.active_trade:
            self._update_active_trade(df_with_indicators.iloc[-1])
        
        # Check for new trade signals if no active trade
        if not self.active_trade and signal['signal'] != 'neutral':
            self._open_trade(signal, df_with_indicators.iloc[-1]['close'])
        
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
    
    def _open_trade(self, signal, current_price):
        """
        Open a new trade based on the signal.
        
        Args:
            signal: Signal dictionary
            current_price: Current price
        """
        if self.active_trade:
            logger.warning("Attempted to open trade while another is active")
            return
        
        side = signal['signal']  # 'buy' or 'sell'
        
        # Calculate position size based on risk per trade
        risk_amount = self.account_balance * RISK_PER_TRADE
        stop_loss_price = (
            current_price * (1 - STOP_LOSS_PCT) if side == 'buy' 
            else current_price * (1 + STOP_LOSS_PCT)
        )
        take_profit_price = (
            current_price * (1 + TAKE_PROFIT_PCT) if side == 'buy' 
            else current_price * (1 - TAKE_PROFIT_PCT)
        )
        
        # Calculate the potential loss per unit
        risk_per_unit = abs(current_price - stop_loss_price)
        
        # Calculate the number of units to trade
        position_size = risk_amount / risk_per_unit
        
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
            'notional_value': notional_value
        }
        
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
            take_profit=take_profit_price
        )
        self.db_session.add(db_trade)
        self.db_session.commit()
        
        # Update trade with ID from database
        self.active_trade['id'] = db_trade.id
        
        logger.info(f"Opened {side} trade at {current_price} with SL: {stop_loss_price}, TP: {take_profit_price}")
    
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
        
        # Log trade closing
        logger.info(f"Closed trade at {current_price} for {reason}, PnL: {pnl}, PnL%: {pnl_percent*100:.2f}%")
        
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
                self.db_session.commit()
        except Exception as e:
            logger.error(f"Error updating trade in database: {str(e)}")
            self.db_session.rollback()
        
        # Reset active trade
        self.active_trade = None
        
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