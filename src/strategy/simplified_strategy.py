"""
Simplified High-Leverage Trading Strategy

A modified version of the high leverage strategy with more permissive
signal generation and configurable filter thresholds.
"""

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from ta.trend import EMAIndicator

class SimplifiedStrategy:
    """
    A simplified version of the high leverage strategy that generates
    more signals and has configurable filter thresholds.
    """
    
    def __init__(self, **kwargs):
        """Initialize the strategy with configurable parameters."""
        # Extract EMA parameters
        self.fast_ema = kwargs.get('fast_ema', 8)
        self.slow_ema = kwargs.get('slow_ema', 21)
        self.trend_ema = kwargs.get('trend_ema', 50)
        
        # Filter settings
        self.use_mtf_filter = kwargs.get('use_mtf_filter', True)
        self.mtf_signal_mode = kwargs.get('mtf_signal_mode', 'any')  # 'any', 'both', or 'weighted'
        self.mtf_timeframes = kwargs.get('mtf_timeframes', [])
        
        self.use_momentum_filter = kwargs.get('use_momentum_filter', True)
        self.momentum_period = kwargs.get('momentum_period', 14)
        self.momentum_threshold = kwargs.get('momentum_threshold', 40)
        self.momentum_lookback = kwargs.get('momentum_lookback', 3)
        
        self.use_volatility_sizing = kwargs.get('use_volatility_sizing', True)
        self.volatility_target = kwargs.get('volatility_target', 0.01)
        self.volatility_lookback = kwargs.get('volatility_lookback', 20)
        self.max_position_size = kwargs.get('max_position_size', 0.05)
        
        # Other parameters
        self.risk_per_trade = kwargs.get('risk_per_trade', 0.02)
        self.adaptive_vol_scaling = kwargs.get('adaptive_vol_scaling', False)
        
        # Debugging parameters
        self.force_signals = kwargs.get('force_signals', False)
        self.signal_threshold = kwargs.get('signal_threshold', 0.0)
        
        # Stats tracking
        self.confirmation_stats = {
            'mtf_confirmed': 0,
            'mtf_rejected': 0,
            'momentum_confirmed': 0,
            'momentum_rejected': 0,
            'volatility_adjustments': 0
        }
    
    def generate_signals(self, df, higher_tf_data=None):
        """
        Generate trading signals with configurable parameters.
        
        Args:
            df: DataFrame with OHLCV data
            higher_tf_data: Dictionary of {timeframe: DataFrame} with higher timeframe data
            
        Returns:
            DataFrame with signals added
        """
        # Make a copy to avoid modifying original
        signals = df.copy()
        
        # Calculate EMAs
        signals['ema_fast'] = EMAIndicator(close=signals['close'], window=self.fast_ema).ema_indicator()
        signals['ema_slow'] = EMAIndicator(close=signals['close'], window=self.slow_ema).ema_indicator()
        signals['ema_trend'] = EMAIndicator(close=signals['close'], window=self.trend_ema).ema_indicator()
        
        # Initialize signal column
        signals['signal'] = 0
        
        # Detect trend direction
        signals['trend_direction'] = np.where(signals['ema_trend'] > signals['ema_trend'].shift(1), 1, 
                                           np.where(signals['ema_trend'] < signals['ema_trend'].shift(1), -1, 0))
        
        # Calculate crossovers for more frequent signals
        for i in range(1, len(signals)):
            # Bullish crossover (fast crosses above slow)
            if signals['ema_fast'].iloc[i] > signals['ema_slow'].iloc[i] and signals['ema_fast'].iloc[i-1] <= signals['ema_slow'].iloc[i-1]:
                signals.loc[signals.index[i], 'signal'] = 1
            
            # Bearish crossover (fast crosses below slow)
            elif signals['ema_fast'].iloc[i] < signals['ema_slow'].iloc[i] and signals['ema_fast'].iloc[i-1] >= signals['ema_slow'].iloc[i-1]:
                signals.loc[signals.index[i], 'signal'] = -1
        
        # Force signals generation if needed for testing
        if self.force_signals and signals['signal'].abs().sum() < 10:
            # Generate more signals based on a simple threshold
            for i in range(self.trend_ema, len(signals)):
                # Skip if we already have a signal here
                if signals['signal'].iloc[i] != 0:
                    continue
                
                # Calculate percent difference between fast and slow EMAs
                ema_diff_pct = (signals['ema_fast'].iloc[i] - signals['ema_slow'].iloc[i]) / signals['ema_slow'].iloc[i] * 100
                
                # Generate buy signals when fast EMA is above slow EMA by threshold
                if ema_diff_pct > self.signal_threshold and signals['ema_fast'].iloc[i] > signals['ema_slow'].iloc[i]:
                    # Only add signal if far enough from previous one
                    last_signal = signals['signal'].iloc[max(0, i-10):i].abs().sum()
                    if last_signal == 0:  # No recent signals
                        signals.loc[signals.index[i], 'signal'] = 1
                
                # Generate sell signals when fast EMA is below slow EMA by threshold
                elif ema_diff_pct < -self.signal_threshold and signals['ema_fast'].iloc[i] < signals['ema_slow'].iloc[i]:
                    # Only add signal if far enough from previous one
                    last_signal = signals['signal'].iloc[max(0, i-10):i].abs().sum()
                    if last_signal == 0:  # No recent signals
                        signals.loc[signals.index[i], 'signal'] = -1
        
        # Add RSI for momentum filter
        rsi_indicator = RSIIndicator(close=signals['close'], window=self.momentum_period)
        signals['rsi'] = rsi_indicator.rsi()
        
        # Calculate ATR for volatility sizing
        atr_indicator = AverageTrueRange(high=signals['high'], low=signals['low'], 
                                        close=signals['close'], window=self.volatility_lookback)
        signals['atr'] = atr_indicator.average_true_range()
        
        # Calculate historical volatility
        signals['hist_vol'] = signals['close'].pct_change().rolling(self.volatility_lookback).std()
        
        # Calculate annualized volatility
        if isinstance(signals.index, pd.DatetimeIndex):
            # Detect timeframe from index
            if len(signals) > 1:
                seconds_diff = (signals.index[1] - signals.index[0]).total_seconds()
                if seconds_diff <= 3600:  # Hourly or less
                    annualization_factor = np.sqrt(365 * 24)
                elif seconds_diff <= 86400:  # Daily
                    annualization_factor = np.sqrt(252)
                else:  # Weekly or more
                    annualization_factor = np.sqrt(52)
            else:
                annualization_factor = np.sqrt(252)  # Default to daily
        else:
            annualization_factor = np.sqrt(252)  # Default to daily
            
        signals['annualized_vol'] = signals['hist_vol'] * annualization_factor
        
        # Detect volatility regimes
        signals['vol_regime'] = 'NORMAL'
        signals.loc[signals['annualized_vol'] < self.volatility_target * 0.7, 'vol_regime'] = 'LOW'
        signals.loc[signals['annualized_vol'] > self.volatility_target * 1.5, 'vol_regime'] = 'HIGH'
        
        # Calculate volatility adjustment factor
        signals['vol_adjustment'] = self.volatility_target / signals['annualized_vol'].fillna(self.volatility_target)
        
        # Cap adjustment factor for safety
        signals['vol_adjustment'] = signals['vol_adjustment'].clip(0.2, 3.0)
        
        return signals
    
    def check_multi_timeframe_alignment(self, df, idx, mtf_data):
        """
        Check if higher timeframes confirm the signal.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            mtf_data: Dictionary of {timeframe: DataFrame} with higher timeframe data
            
        Returns:
            Boolean indicating if filter passes
        """
        if not self.use_mtf_filter or not mtf_data:
            return True
        
        # Get current signal
        current_signal = df['signal'].iloc[idx]
        if current_signal == 0:
            return True  # No signal to confirm
        
        # Track which timeframes agreed
        agreements = []
        
        # Check agreement with each timeframe
        for tf, tf_df in mtf_data.items():
            # Generate signals for this timeframe if needed
            if 'signal' not in tf_df.columns:
                tf_df = self.generate_signals(tf_df)
            
            # Find the closest timeframe data point
            if isinstance(df.index, pd.DatetimeIndex) and isinstance(tf_df.index, pd.DatetimeIndex):
                # Find nearest timestamp
                current_time = df.index[idx]
                nearest_idx = tf_df.index.get_indexer([current_time], method='pad')[0]
                
                if nearest_idx >= 0:
                    # Check trend direction or signal agreement
                    ema_trend = tf_df['trend_direction'].iloc[nearest_idx] if 'trend_direction' in tf_df.columns else 0
                    mtf_signal = tf_df['signal'].iloc[nearest_idx]
                    
                    # Enhanced: Check both trend direction and actual signal with more permissive logic
                    # For long signals, either positive trend or recent buy signal is good
                    # For short signals, either negative trend or recent sell signal is good
                    if (current_signal > 0 and (ema_trend > 0 or mtf_signal > 0)) or \
                       (current_signal < 0 and (ema_trend < 0 or mtf_signal < 0)):
                        agreements.append(tf)
        
        # Calculate agreement score
        agreement_score = len(agreements) / len(mtf_data) if mtf_data else 0
        
        # Decision based on the mode
        if self.mtf_signal_mode == 'both':
            # All timeframes must agree
            confirmed = len(agreements) == len(mtf_data)
        elif self.mtf_signal_mode == 'any':
            # Any timeframe must agree (more permissive)
            confirmed = len(agreements) > 0
        else:  # 'weighted' mode
            # At least 50% of timeframes must agree
            confirmed = agreement_score >= 0.5
        
        # Update stats
        if confirmed:
            self.confirmation_stats['mtf_confirmed'] += 1
        else:
            self.confirmation_stats['mtf_rejected'] += 1
        
        return confirmed
    
    def check_momentum_filter(self, df, idx):
        """
        Check if momentum confirms the signal, with more permissive conditions.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            
        Returns:
            Boolean indicating if filter passes
        """
        if not self.use_momentum_filter or idx < self.momentum_lookback:
            return True
            
        # Get current signal
        current_signal = df['signal'].iloc[idx]
        if current_signal == 0:
            return True  # No signal to confirm
            
        # Get RSI value
        rsi = df['rsi'].iloc[idx]
        
        # Check RSI direction over lookback period
        rsi_direction = df['rsi'].iloc[idx] - df['rsi'].iloc[idx - self.momentum_lookback]
        
        # More permissive logic for momentum filter
        if current_signal > 0:  # Long signal
            # For long signals: RSI > threshold OR RSI trending up
            momentum_confirmed = (rsi > self.momentum_threshold) or (rsi_direction > 0)
        else:  # Short signal
            # For short signals: RSI < (100-threshold) OR RSI trending down
            momentum_confirmed = (rsi < (100 - self.momentum_threshold)) or (rsi_direction < 0)
        
        # Update stats
        if momentum_confirmed:
            self.confirmation_stats['momentum_confirmed'] += 1
        else:
            self.confirmation_stats['momentum_rejected'] += 1
            
        return momentum_confirmed
    
    def should_place_trade(self, df, idx, mtf_data=None):
        """
        Determine if a trade should be placed using configurable filters.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            mtf_data: Dictionary of higher timeframe data
            
        Returns:
            Boolean indicating if trade should be placed
        """
        # Check if there is a signal
        current_signal = df['signal'].iloc[idx]
        if current_signal == 0:
            return False
            
        # Check all filters
        mtf_aligned = self.check_multi_timeframe_alignment(df, idx, mtf_data)
        momentum_aligned = self.check_momentum_filter(df, idx)
        
        # Check volatility (less restrictive)
        valid_volatility = True
        if self.use_volatility_sizing and 'vol_regime' in df.columns:
            # Only filter out extreme volatility
            extreme_volatility = df['vol_regime'].iloc[idx] == 'HIGH' and df['annualized_vol'].iloc[idx] > self.volatility_target * 3
            valid_volatility = not extreme_volatility
        
        # Combined decision
        return mtf_aligned and momentum_aligned and valid_volatility
    
    def calculate_position_size(self, df, idx, account_balance, risk_pct):
        """
        Calculate position size with volatility targeting.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            account_balance: Current account balance
            risk_pct: Risk percentage per trade
            
        Returns:
            Position size in base currency
        """
        if not self.use_volatility_sizing:
            # Simple fixed percentage position sizing
            return account_balance * risk_pct / df['close'].iloc[idx]
        
        # Get current price
        current_price = df['close'].iloc[idx]
        
        # Get volatility adjustment
        vol_factor = df['vol_adjustment'].iloc[idx] if 'vol_adjustment' in df.columns else 1.0
        
        # Calculate adjusted risk percentage (less conservative)
        adjusted_risk = risk_pct * vol_factor
        
        # Cap maximum position size
        adjusted_risk = min(adjusted_risk, self.max_position_size)
        
        # Calculate position size
        position_size = (account_balance * adjusted_risk) / current_price
        
        # Update stats
        self.confirmation_stats['volatility_adjustments'] += 1
        
        return position_size 