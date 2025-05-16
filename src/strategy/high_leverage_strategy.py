"""
High-Leverage Trading Strategy

Implements three powerful enhancements that punch above their weight:
1. Multi-Timeframe Confirmation - align signals across timeframes
2. Momentum/RSI Filter - prevent trading against momentum
3. Volatility-Targeted Sizing - normalize risk across market conditions

CONFIGURATION NOTE:
This strategy now uses the "Fast & Loose" configuration by default, which includes:
- MTF Filter: "any" mode (at least one higher-timeframe must agree)
- Momentum Threshold: 35 (lowered from 50)
- Momentum Filter: More permissive (RSI above threshold OR trending up for longs)
- Volatility Filter: Disabled by default
- Extreme Volatility Check: Only filters out top 5% most volatile periods

These settings produce significantly more trade signals than the original configuration.
For a more conservative approach, parameters can be explicitly provided when initializing.
"""

import numpy as np
import pandas as pd
from src.strategy.enhanced_strategy import EnhancedConfirmationStrategy
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


class HighLeverageStrategy(EnhancedConfirmationStrategy):
    """
    A focused strategy implementing the three most impactful features:
    - Multi-Timeframe Confirmation
    - Momentum/RSI Filter
    - Volatility-Targeted Sizing
    
    This strategy extends the EnhancedConfirmationStrategy but prioritizes
    these three features with improved implementations.
    """
    
    def __init__(self, **kwargs):
        # Extract parameters that are specific to this class and shouldn't be passed to parent
        risk_per_trade = kwargs.pop('risk_per_trade', 0.02)
        use_mtf_filter = kwargs.pop('use_mtf_filter', True)
        mtf_timeframes = kwargs.pop('mtf_timeframes', ['4h', '1d'])
        mtf_signal_mode = kwargs.pop('mtf_signal_mode', 'any')
        mtf_alignment_weight = kwargs.pop('mtf_alignment_weight', 2.0)
        
        use_momentum_filter = kwargs.get('use_momentum_filter', True)  # Keep this for parent
        momentum_period = kwargs.pop('momentum_period', 14)
        momentum_threshold = kwargs.pop('momentum_threshold', 35)
        momentum_lookback = kwargs.pop('momentum_lookback', 3)
        momentum_weight = kwargs.pop('momentum_weight', 2.0)
        
        use_volatility_sizing = kwargs.pop('use_volatility_sizing', False)
        volatility_target = kwargs.pop('volatility_target', 0.01)
        volatility_lookback = kwargs.pop('volatility_lookback', 20)
        max_position_size = kwargs.pop('max_position_size', 0.05)
        adaptive_vol_scaling = kwargs.pop('adaptive_vol_scaling', True)
        
        # Initialize parent strategy with base parameters
        super().__init__(**kwargs)
        
        # Store our extracted parameters as instance variables
        self.risk_per_trade = risk_per_trade
        
        # Multi-Timeframe parameters
        self.use_mtf_filter = use_mtf_filter
        self.mtf_timeframes = mtf_timeframes
        self.mtf_signal_mode = mtf_signal_mode
        self.mtf_alignment_weight = mtf_alignment_weight
        
        # Momentum parameters (keep parent's use_momentum_filter)
        self.momentum_period = momentum_period
        self.momentum_threshold = momentum_threshold 
        self.momentum_lookback = momentum_lookback
        self.momentum_weight = momentum_weight
        
        # Volatility sizing parameters
        self.use_volatility_sizing = use_volatility_sizing
        self.volatility_target = volatility_target
        self.volatility_lookback = volatility_lookback
        self.max_position_size = max_position_size
        self.adaptive_vol_scaling = adaptive_vol_scaling
        
        # Tracking stats
        self.mtf_confirmations = 0
        self.mtf_rejections = 0
        self.momentum_confirmations = 0
        self.momentum_rejections = 0
        self.volatility_adjustments = 0
        
        # Initialize confirmation stats dictionary
        self.confirmation_stats = {
            'mtf_confirmed': 0,
            'mtf_rejected': 0,
            'momentum_confirmed': 0,
            'momentum_rejected': 0,
            'volatility_adjustments': 0
        }
        
        # For storing regime statistics
        self.volatility_regime_stats = {
            'LOW': {'percent_time': 0, 'trades': 0, 'win_rate': 0, 'profit_factor': 0},
            'NORMAL': {'percent_time': 0, 'trades': 0, 'win_rate': 0, 'profit_factor': 0},
            'HIGH': {'percent_time': 0, 'trades': 0, 'win_rate': 0, 'profit_factor': 0}
        }
    
    def generate_signals(self, df, higher_tf_data=None):
        """
        Generate trading signals with high-leverage features.
        
        Args:
            df: DataFrame with OHLCV data
            higher_tf_data: DataFrame with higher timeframe data
            
        Returns:
            DataFrame with signals added
        """
        # Generate base signals using parent class method
        signals = super().generate_signals(df, higher_tf_data)
        
        # Add RSI for momentum filter using ta library
        rsi_indicator = RSIIndicator(close=signals['close'], window=self.momentum_period)
        signals['rsi'] = rsi_indicator.rsi()
        
        # Calculate ATR for volatility sizing using ta library
        atr_indicator = AverageTrueRange(high=signals['high'], low=signals['low'], 
                                        close=signals['close'], window=self.volatility_lookback)
        signals['atr'] = atr_indicator.average_true_range()
        
        # Calculate historical volatility (alternative to ATR)
        signals['hist_vol'] = signals['close'].pct_change().rolling(self.volatility_lookback).std()
        
        # Calculate annualized volatility
        # Estimate trading days based on timeframe
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
        # Lower volatility = larger position, higher volatility = smaller position
        signals['vol_adjustment'] = self.volatility_target / signals['annualized_vol'].fillna(self.volatility_target)
        
        # Cap adjustment factor for safety
        signals['vol_adjustment'] = signals['vol_adjustment'].clip(0.2, 3.0)
        
        # Apply advanced, adaptive scaling when enabled
        if self.adaptive_vol_scaling:
            # Nonlinear scaling based on regime transitions
            signals['regime_transition'] = signals['vol_regime'] != signals['vol_regime'].shift(1)
            
            # Smooth transitions when regimes change (avoid abrupt position size changes)
            for i in range(1, len(signals)):
                if signals['regime_transition'].iloc[i]:
                    # If transitioning to higher volatility, reduce position faster
                    if signals['vol_regime'].iloc[i] == 'HIGH' and signals['vol_regime'].iloc[i-1] != 'HIGH':
                        signals.loc[signals.index[i], 'vol_adjustment'] *= 0.8
                    # If transitioning to lower volatility, increase position gradually
                    elif signals['vol_regime'].iloc[i] == 'LOW' and signals['vol_regime'].iloc[i-1] != 'LOW':
                        signals.loc[signals.index[i], 'vol_adjustment'] *= 0.9  # More conservative
        
        return signals
    
    def check_multi_timeframe_alignment(self, df, idx, mtf_data):
        """
        Enhanced multi-timeframe filter with weighted alignment.
        
        Args:
            df: DataFrame with current timeframe data
            idx: Current index
            mtf_data: Dictionary of higher timeframe data
            
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
                    
                    # Enhanced: Check both trend direction and actual signal
                    if (current_signal > 0 and (ema_trend > 0 or mtf_signal > 0)) or \
                       (current_signal < 0 and (ema_trend < 0 or mtf_signal < 0)):
                        agreements.append(tf)
        
        # Improved approach: Calculate a weighted agreement score
        agreement_score = len(agreements) / len(mtf_data)
        
        # Decision based on the mode
        if self.mtf_signal_mode == 'both':
            # All timeframes must agree (traditional approach)
            confirmed = len(agreements) == len(mtf_data)
        elif self.mtf_signal_mode == 'any':
            # Any timeframe must agree
            confirmed = len(agreements) > 0
        else:  # 'weighted' mode (improved approach)
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
        Enhanced momentum filter with lookback period.
        Modified to be more permissive in the "Fast & Loose" configuration.
        
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
        
        # More permissive logic with relaxed momentum requirements
        if current_signal > 0:  # Long signal
            # For longs, RSI should be above threshold OR trending up
            momentum_confirmed = (rsi > self.momentum_threshold) or (rsi_direction > 0)
        else:  # Short signal
            # For shorts, RSI should be below (100-threshold) OR trending down
            momentum_confirmed = (rsi < (100 - self.momentum_threshold)) or (rsi_direction < 0)
        
        # Update stats
        if momentum_confirmed:
            self.confirmation_stats['momentum_confirmed'] += 1
        else:
            self.confirmation_stats['momentum_rejected'] += 1
            
        return momentum_confirmed
    
    def calculate_position_size(self, df, idx, account_balance, risk_pct):
        """
        Calculate position size using volatility targeting.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            account_balance: Current account balance
            risk_pct: Risk percentage per trade (as decimal)
            
        Returns:
            Position size in base currency
        """
        if not self.use_volatility_sizing:
            # Use standard position sizing from parent class
            return super().calculate_position_size(df, idx, account_balance, risk_pct)
        
        # Get current price
        current_price = df['close'].iloc[idx]
        
        # Get volatility (ATR or historical vol)
        if 'atr' in df.columns and not np.isnan(df['atr'].iloc[idx]):
            current_volatility = df['atr'].iloc[idx] / current_price  # As percentage of price
        else:
            # Fallback to historical volatility
            current_volatility = df['hist_vol'].iloc[idx] if 'hist_vol' in df.columns else 0.01
        
        # Safety check
        if current_volatility <= 0:
            current_volatility = self.volatility_target
        
        # Get volatility adjustment factor
        vol_factor = df['vol_adjustment'].iloc[idx] if 'vol_adjustment' in df.columns else 1.0
        
        # Calculate adjusted risk percentage
        adjusted_risk = risk_pct * vol_factor
        
        # Cap maximum position size
        adjusted_risk = min(adjusted_risk, self.max_position_size)
        
        # Calculate position size
        position_size = (account_balance * adjusted_risk) / current_price
        
        # Update volatility adjustment stats
        self.confirmation_stats['volatility_adjustments'] += 1
        
        # Update regime stats
        if 'vol_regime' in df.columns:
            regime = df['vol_regime'].iloc[idx]
            if regime in self.volatility_regime_stats:
                self.volatility_regime_stats[regime]['trades'] += 1
        
        return position_size
    
    def should_place_trade(self, df, idx, mtf_data=None):
        """
        Determine if a trade should be placed using all high-leverage filters.
        
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
            
        # Check multi-timeframe alignment and momentum filters
        mtf_aligned = self.check_multi_timeframe_alignment(df, idx, mtf_data)
        momentum_aligned = self.check_momentum_filter(df, idx)
        
        # Volatility filter is now disabled by default
        # Only perform an extreme volatility check in rare cases
        valid_volatility = True
        if self.use_volatility_sizing and 'vol_regime' in df.columns:
            # Only filter out extremely high volatility (top 5% cases)
            extreme_volatility = df['vol_regime'].iloc[idx] == 'HIGH' and df['annualized_vol'].iloc[idx] > self.volatility_target * 3
            valid_volatility = not extreme_volatility
        
        # Combined decision - required filters must pass
        return mtf_aligned and momentum_aligned and valid_volatility
    
    def calculate_stop_loss(self, df, idx, entry_price, side):
        """
        Calculate volatility-adjusted stop loss.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            entry_price: Entry price
            side: Trade direction ('long' or 'short')
            
        Returns:
            Stop loss price
        """
        # Get volatility (ATR)
        atr = df['atr'].iloc[idx] if 'atr' in df.columns else 0
        
        # Volatility-adjusted stop distance
        if atr > 0:
            # Use ATR for stop distance
            atr_multiplier = self.atr_multiplier
            
            # Adjust ATR multiplier based on volatility regime
            if 'vol_regime' in df.columns:
                if df['vol_regime'].iloc[idx] == 'LOW':
                    atr_multiplier *= 1.2  # Wider stops in low vol
                elif df['vol_regime'].iloc[idx] == 'HIGH':
                    atr_multiplier *= 0.8  # Tighter stops in high vol
            
            stop_distance = atr * atr_multiplier
        else:
            # Fallback to percentage-based stop
            stop_distance = entry_price * self.stop_loss
        
        # Calculate stop loss price
        if side == 'long':
            stop_loss = entry_price - stop_distance
        else:  # short
            stop_loss = entry_price + stop_distance
        
        return stop_loss 