import pandas as pd
import numpy as np
from datetime import datetime

from src.strategy.base_strategy import BaseStrategy
from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, RISK_PER_TRADE,
    USE_ATR_STOPS, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    VOL_TARGET_PCT, MAX_POSITION_PCT, TRAIL_ACTIVATION_PCT,
    RSI_PERIOD, RSI_OVERSOLD, RSI_OVERBOUGHT, VOL_RATIO_MIN,
    MIN_BARS_BETWEEN_TRADES
)
from src.indicators.indicators import (
    calculate_ema, calculate_rsi, calculate_atr,
    calculate_volume_ratio, add_rsi, add_volume_indicators, add_atr
)
from src.utils.logger import logger
from src.utils.metrics import profit_factor

class RSIOscillatorStrategy(BaseStrategy):
    """
    RSI Oscillator strategy that uses RSI reversals and volume confirmation.
    """
    
    def __init__(self, symbol=SYMBOL, timeframe='4h', db_session=None, account_balance=1000.0, 
                history_days=365, auto_optimize=False, config=None,
                rsi_period=14, oversold=30, overbought=70, atr_sl_multiplier=1.0,
                risk_per_trade=0.0075, use_volatility_sizing=True, vol_target_pct=0.0075,
                enable_pyramiding=False, max_pyramid_entries=2, health_monitor=None,
                atr_trail_multiplier=1.25):
        # Initialize the base class first
        super().__init__(config)
        
        # Set strategy-specific attributes
        self.symbol = symbol
        self.timeframe = timeframe
        self.db_session = db_session
        self.account_balance = account_balance
        self.active_trade = None
        self.history_days = history_days
        self.health_monitor = health_monitor
        
        # RSI parameters
        self.rsi_period = rsi_period if rsi_period else RSI_PERIOD
        self.oversold = oversold if oversold else RSI_OVERSOLD
        self.overbought = overbought if overbought else RSI_OVERBOUGHT
        
        # Stop loss and risk management
        self.use_atr_stops = USE_ATR_STOPS
        self.atr_sl_multiplier = atr_sl_multiplier if atr_sl_multiplier else ATR_SL_MULTIPLIER
        self.atr_tp_multiplier = None  # No fixed TP, using trailing stops
        self.atr_trail_multiplier = atr_trail_multiplier
        self.stop_loss_pct = STOP_LOSS_PCT
        self.take_profit_pct = TAKE_PROFIT_PCT
        self.risk_per_trade = risk_per_trade
        
        # IMPROVEMENT 2: Dynamic position sizing based on signal probability
        self.base_risk_per_trade = risk_per_trade
        self.high_prob_risk_per_trade = risk_per_trade * 1.33  # Increase risk by 33% for high probability signals
        self.high_prob_threshold = 0.70  # Probability threshold for higher risk
        
        # Volume filter parameters
        self.use_volume_filter = True
        self.volume_threshold = VOL_RATIO_MIN
        
        # Volatility-based sizing
        self.use_volatility_sizing = use_volatility_sizing
        self.vol_target_pct = vol_target_pct
        self.max_position_pct = MAX_POSITION_PCT
        
        # Trade frequency
        self.min_bars_between_trades = MIN_BARS_BETWEEN_TRADES
        self.last_trade_index = -100  # Initialize to allow first trade
        self.last_trade_time = None  # Initialize last trade time
        
        # Pyramiding settings
        self.enable_pyramiding = enable_pyramiding
        self.max_pyramid_entries = max_pyramid_entries
        self.current_pyramid_entries = 0
        
        # Performance tracking
        self.trades = []
        self.win_count = 0
        self.loss_count = 0
        
        # Strategy type
        self.strategy_type = "rsi_oscillator"
        
        logger.info(f"Initialized RSI Oscillator Strategy with RSI{self.rsi_period} on {timeframe} timeframe")
    
    def generate_signals(self, df):
        """
        Generate trading signals based on RSI values and filters.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with signals added
        """
        # Apply indicators first
        df = self.apply_indicators(df)
        
        # Initialize signals column
        df['signal'] = 0
        
        # Generate buy signals when RSI is below oversold threshold
        buy_condition = (df['rsi'] < self.oversold) & (df['volume_ratio'] > 1.2)
        df.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals when RSI is above overbought threshold
        sell_condition = (df['rsi'] > self.overbought) & (df['volume_ratio'] > 1.2)
        df.loc[sell_condition, 'signal'] = -1
        
        # Add signal type for logging
        df['signal_type'] = 'none'
        df.loc[df['signal'] == 1, 'signal_type'] = 'buy'
        df.loc[df['signal'] == -1, 'signal_type'] = 'sell'
        
        return df
    
    def get_signal(self, df, index=-1):
        """
        Generate trading signal based on the latest indicators.
        
        Args:
            df: DataFrame with OHLCV data
            index: Index to get signal for (-1 for latest)
            
        Returns:
            String: 'buy', 'sell', or '' (empty string for no signal)
        """
        if df is None or len(df) < self.rsi_period + 1:
            return ''
        
        # If health monitor indicates poor performance, avoid trading
        if self.health_monitor and not self.health_monitor.is_trading_allowed(self.strategy_type, self.symbol):
            return ''
        
        # Use processed signals if available
        if 'signal' in df.columns:
            signal_val = df['signal'].iloc[index]
            if signal_val == 1:
                return 'buy'
            elif signal_val == -1:
                return 'sell'
            else:
                return ''
        
        # Process the data if signals not pre-calculated
        processed_df = self.generate_signals(df.copy())
        signal_val = processed_df['signal'].iloc[index] if len(processed_df) > 0 else 0
        
        if signal_val == 1:
            return 'buy'
        elif signal_val == -1:
            return 'sell'
        else:
            return ''
    
    def get_position_size(self, df, account_balance, index=-1):
        """
        Calculate appropriate position size based on account balance and risk parameters.
        
        Args:
            df: DataFrame with OHLCV data
            account_balance: Current account balance
            index: Index to calculate for (-1 for latest)
            
        Returns:
            float: Position size in base currency units
        """
        if df is None or len(df) == 0:
            return 0
        
        price = df['close'].iloc[index]
        
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return 0
        
        # Get ATR value for position sizing
        atr_value = df['atr'].iloc[index] if 'atr' in df.columns else (price * 0.02)  # Default to 2% if ATR not available
        
        # Get recent profit factor from health monitor if available
        pf_recent = None
        if hasattr(self, 'health_monitor') and self.health_monitor:
            pf_recent = self.health_monitor.get_profit_factor_last_n(40)
        
        # Use portfolio manager if available with edge-weighted sizing
        if hasattr(self, 'allocator') and self.allocator:
            # Get signal direction
            signal = 'buy' if df['signal'].iloc[index] == 1 else 'sell' if df['signal'].iloc[index] == -1 else None
            
            if signal:
                position_size, _, _ = self.allocator.calculate_position_size(
                    symbol=self.symbol,
                    current_price=price,
                    atr_value=atr_value,
                    strat_name="rsi_oscillator",
                    pf_recent=pf_recent,
                    side=signal
                )
                return position_size
        
        # If not using portfolio manager, continue with existing calculation
            
        # IMPROVEMENT 2: Calculate signal probability for dynamic position sizing
        if 'rsi' in df.columns:
            rsi_value = df['rsi'].iloc[index]
            
            # Initialize signal probability
            signal_prob = 0.5
            
            # Check if there's a signal column to determine direction
            if 'signal' in df.columns:
                signal = df['signal'].iloc[index]
                
                # For buy signals (RSI in oversold territory)
                if signal == 1:
                    # The deeper into oversold, the higher the probability
                    signal_prob = 0.5 + min(0.4, max(0, (self.oversold - rsi_value) / self.oversold * 0.5))
                # For sell signals (RSI in overbought territory)
                elif signal == -1:
                    # The deeper into overbought, the higher the probability
                    signal_prob = 0.5 + min(0.4, max(0, (rsi_value - self.overbought) / (100 - self.overbought) * 0.5))
            
            # Adjust risk based on signal probability
            current_risk = self.risk_per_trade
            if signal_prob >= self.high_prob_threshold:
                current_risk = self.high_prob_risk_per_trade
                logger.info(f"High probability signal detected (prob={signal_prob:.2f}), increasing risk from {self.base_risk_per_trade*100:.2f}% to {current_risk*100:.2f}%")
        else:
            current_risk = self.risk_per_trade
        
        # Calculate stop loss level
        if self.use_atr_stops and 'atr' in df.columns:
            atr = df['atr'].iloc[index]
            stop_distance_pct = atr * self.atr_sl_multiplier / price
        else:
            stop_distance_pct = self.stop_loss_pct
        
        # Ensure minimum stop distance
        stop_distance_pct = max(stop_distance_pct, 0.001)  # Minimum 0.1% stop
        
        # Risk amount (adjusted for signal probability)
        risk_amount = account_balance * current_risk
        
        # Position size based on risk
        position_value = risk_amount / stop_distance_pct
        
        # Cap position size at max percentage
        max_position_value = account_balance * self.max_position_pct
        position_value = min(position_value, max_position_value)
        
        # Calculate units
        units = position_value / price
        
        # Apply volatility scaling if enabled
        if self.use_volatility_sizing and 'atr' in df.columns:
            vol_scale_factor = self.vol_target_pct / (df['atr'].iloc[index] / price)
            units = units * vol_scale_factor
        
        return units
    
    def get_stop_loss(self, df, entry_price, signal, index=-1):
        """
        Calculate the stop loss level for a trade.
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price of the trade
            signal: 'buy' or 'sell'
            index: Index to calculate for (-1 for latest)
            
        Returns:
            float: Stop loss price
        """
        if df is None or len(df) == 0:
            return None
        
        # ATR-based stops
        if self.use_atr_stops and 'atr' in df.columns:
            atr = df['atr'].iloc[index]
            stop_distance = atr * self.atr_sl_multiplier
            
            if signal == 'buy':
                return entry_price - stop_distance
            else:  # sell
                return entry_price + stop_distance
        
        # Percentage-based stops
        else:
            if signal == 'buy':
                return entry_price * (1 - self.stop_loss_pct)
            else:  # sell
                return entry_price * (1 + self.stop_loss_pct)
    
    def get_take_profit(self, df, entry_price, signal, index=-1):
        """
        Calculate the take profit level for a trade.
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price of the trade
            signal: 'buy' or 'sell'
            index: Index to calculate for (-1 for latest)
            
        Returns:
            float: Take profit price or None for trailing stops
        """
        # Using trailing stops instead of fixed take profits
        if self.take_profit_pct is None:
            return None
        
        if signal == 'buy':
            return entry_price * (1 + self.take_profit_pct)
        else:  # sell
            return entry_price * (1 - self.take_profit_pct)
    
    def on_trade_executed(self, trade_type, entry_price, position_size, timestamp=None):
        """
        Handle trade execution event.
        
        Args:
            trade_type: 'buy' or 'sell'
            entry_price: Entry price of the trade
            position_size: Size of the position in base currency
            timestamp: Time of execution
        """
        logger.info(f"RSI Strategy executed {trade_type} trade at {entry_price}, size: {position_size}")
        
        # Reset pyramiding count if new trade
        if trade_type != self.active_trade:
            self.current_pyramid_entries = 1
        else:
            self.current_pyramid_entries += 1
        
        self.active_trade = trade_type
    
    def on_trade_exit(self, trade_type, entry_price, exit_price, position_size, pnl, timestamp=None):
        """
        Handle trade exit event.
        
        Args:
            trade_type: 'buy' or 'sell'
            entry_price: Entry price of the trade
            exit_price: Exit price of the trade
            position_size: Size of the position in base currency
            pnl: Profit/loss amount
            timestamp: Time of exit
        """
        logger.info(f"RSI Strategy exited {trade_type} trade: Entry {entry_price}, Exit {exit_price}, PnL {pnl:.2f}")
        
        # Track win/loss
        if pnl > 0:
            self.win_count += 1
        else:
            self.loss_count += 1
        
        # Add to trade history
        trade_record = {
            'type': trade_type,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'pnl': pnl,
            'timestamp': timestamp or datetime.now()
        }
        self.trades.append(trade_record)
        
        # Reset active trade
        self.active_trade = None
        self.current_pyramid_entries = 0
        
        # Update health monitor if available
        if self.health_monitor:
            self.health_monitor.record_trade(
                strategy_type=self.strategy_type,
                symbol=self.symbol,
                pnl=pnl,
                win=(pnl > 0)
            )
    
    def get_entry_reason(self, df, index=-1):
        """
        Get a human-readable reason for the entry signal
        
        Args:
            df: DataFrame with OHLCV data
            index: Index to get reason for (-1 for latest)
            
        Returns:
            String: Reason for the entry
        """
        if df is None or len(df) == 0:
            return "Insufficient data"
        
        signal = self.get_signal(df, index)
        if signal is None:
            return "No signal"
        
        # Current values
        current_rsi = df['rsi'].iloc[index]
        volume_ratio = df['volume_ratio'].iloc[index]
        
        # Previous values
        prev_rsi = df['rsi'].iloc[index-1] if index > 0 else None
        
        # Build reason text
        reason = f"RSI Oscillator: "
        
        if signal == 'buy':
            reason += f"RSI ({current_rsi:.1f}) moved up from oversold ({prev_rsi:.1f} to {current_rsi:.1f})"
        else:  # sell
            reason += f"RSI ({current_rsi:.1f}) moved down from overbought ({prev_rsi:.1f} to {current_rsi:.1f})"
        
        reason += f" + Volume({volume_ratio:.2f}x) > {self.volume_threshold}"
        
        return reason
    
    def debug_info(self, df, index=-1):
        """
        Get debug information about the current state for logging/debugging.
        
        Args:
            df: DataFrame with OHLCV data
            index: Index to get info for (-1 for latest)
            
        Returns:
            dict: Debug information
        """
        if df is None or len(df) == 0:
            return {"error": "No data"}
        
        return {
            "price": df['close'].iloc[index],
            "rsi": df['rsi'].iloc[index] if 'rsi' in df.columns else None,
            "volume_ratio": df['volume_ratio'].iloc[index] if 'volume_ratio' in df.columns else None,
            "atr": df['atr'].iloc[index] if 'atr' in df.columns else None,
            "signal": self.get_signal(df, index),
            "entry_reason": self.get_entry_reason(df, index),
            "strategy_type": self.strategy_type
        }
    
    def set_account(self, account):
        """
        Set the trading account for the strategy.
        
        Args:
            account: Trading account object
        """
        self.account = account 
    
    def apply_indicators(self, df):
        """
        Apply RSI and other necessary indicators to the dataframe
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        # Apply RSI
        df = add_rsi(df, period=self.rsi_period)
        
        # Apply volume indicators for signal filtering
        df = add_volume_indicators(df)
        
        # Add ATR for volatility-based position sizing
        df = add_atr(df, period=14)
        
        return df 

    def generate_signal(self, i, df):
        """
        Generate trading signal for a specific candle.
        
        Args:
            i: Index of the current candle
            df: DataFrame with indicators and previous signals
            
        Returns:
            Signal type: 'buy', 'sell', or '' (empty string for no signal)
        """
        # Skip if not enough bars between trades
        current_time = df.index[i]
        if self.last_trade_time and (current_time - self.last_trade_time).total_seconds() / 3600 < self.min_bars_between_trades * 4:
            return ''
            
        current_signal = df['signal'].iloc[i]
        
        if current_signal == 1:
            signal_reason = f"RSI({df['rsi'].iloc[i]:.1f}) < {self.oversold} (oversold)"
            logger.info(f"Generated BUY signal for {self.symbol}: {signal_reason}")
            self.last_trade_time = current_time
            return 'buy'
        elif current_signal == -1:
            signal_reason = f"RSI({df['rsi'].iloc[i]:.1f}) > {self.overbought} (overbought)"
            logger.info(f"Generated SELL signal for {self.symbol}: {signal_reason}")
            self.last_trade_time = current_time
            return 'sell'
        
        return '' 