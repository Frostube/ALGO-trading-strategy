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

EXIT STRATEGY:
This strategy implements a comprehensive exit system with:
- Multi-Tier Take-Profit: Fixed TP plus trailing stop that tightens at R-multiple milestones
- Partial Scale-Outs: Take 50% profit at 1R and move stop to breakeven
- R-Multiple-Based Trailing Stops: Tighten trailing stop at 1R, 2R, 3R profit milestones
- Maximum Hold Period: Time-based exit to prevent holding positions too long

PATTERN FILTERS:
Added candlestick pattern detection and volume filters:
- Strong Bar Confirmation: Requires decisive candles with 70%+ body-to-range ratio
- Directional Alignment: Entry bar must close in the direction of the trade
- Engulfing, Hammer, and Doji pattern recognition for entry confirmation
- Volume spike detection to confirm significant market interest
- Optional pattern and volume requirements for higher-quality trade signals

POSITION SIZING:
Enhanced volatility-targeted position sizing:
- ATR-Based Risk: Size positions so dollar risk equals ATR × multiplier
- Regime-Adaptive: Adjust ATR multiplier based on volatility regime
- Position Caps: Prevent over-sizing in any market condition 
"""

import numpy as np
import pandas as pd
from src.strategy.enhanced_strategy import EnhancedConfirmationStrategy
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from datetime import timedelta


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
        atr_multiplier = kwargs.pop('atr_multiplier', 1.5)  # Default ATR multiplier for stop loss calculation
        
        # New exit strategy parameters
        take_profit_r = kwargs.pop('take_profit_r', 2.0)  # Take profit at 2R
        use_trailing_stop = kwargs.pop('use_trailing_stop', True)
        initial_trail_r = kwargs.pop('initial_trail_r', 1.0)  # Initial trailing stop at 1R distance
        r1_trail_pct = kwargs.pop('r1_trail_pct', 0.75)  # At 1R profit, trail tightens to 75% of initial
        r2_trail_pct = kwargs.pop('r2_trail_pct', 0.5)   # At 2R profit, trail tightens to 50% of initial
        r3_trail_pct = kwargs.pop('r3_trail_pct', 0.25)  # At 3R profit, trail tightens to 25% of initial
        use_partial_exit = kwargs.pop('use_partial_exit', True)
        partial_exit_r = kwargs.pop('partial_exit_r', 1.0)  # Take partial profit at 1R
        partial_exit_pct = kwargs.pop('partial_exit_pct', 0.5)  # Exit 50% of position
        max_hold_periods = kwargs.pop('max_hold_periods', 24)  # Maximum hold time in periods
        
        # New pattern and volume filter parameters
        use_pattern_filter = kwargs.pop('use_pattern_filter', True)
        pattern_strictness = kwargs.pop('pattern_strictness', 'medium')  # 'loose', 'medium', 'strict'
        require_engulfing = kwargs.pop('require_engulfing', False)
        require_doji = kwargs.pop('require_doji', False)
        require_hammer = kwargs.pop('require_hammer', False)
        use_volume_filter = kwargs.pop('use_volume_filter', True)
        volume_threshold = kwargs.pop('volume_threshold', 1.5)  # 150% of average volume
        volume_lookback = kwargs.pop('volume_lookback', 20)  # Look back 20 periods for avg volume
        
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
        self.atr_multiplier = atr_multiplier
        
        # Store exit strategy parameters
        self.take_profit_r = take_profit_r
        self.use_trailing_stop = use_trailing_stop
        self.initial_trail_r = initial_trail_r
        self.r1_trail_pct = r1_trail_pct
        self.r2_trail_pct = r2_trail_pct
        self.r3_trail_pct = r3_trail_pct
        self.use_partial_exit = use_partial_exit
        self.partial_exit_r = partial_exit_r
        self.partial_exit_pct = partial_exit_pct
        self.max_hold_periods = max_hold_periods
        
        # Store pattern and volume filter parameters
        self.use_pattern_filter = use_pattern_filter
        self.pattern_strictness = pattern_strictness
        self.require_engulfing = require_engulfing
        self.require_doji = require_doji
        self.require_hammer = require_hammer
        self.use_volume_filter = use_volume_filter
        self.volume_threshold = volume_threshold
        self.volume_lookback = volume_lookback
        
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
            'volatility_adjustments': 0,
            'volatility_confirmed': 0,
            'volatility_rejected': 0,
            'pattern_confirmed': 0,
            'pattern_rejected': 0,
            'volume_confirmed': 0,
            'volume_rejected': 0,
            'pivot_confirmed': 0,
            'pivot_rejected': 0,
            'total_signals': 0,
            'signals_passed': 0,
            'signals_rejected': 0,
            'weighted_scores': []
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
        
        # Add candlestick pattern detection
        if self.use_pattern_filter:
            self.detect_candlestick_patterns(signals)
            
        # Add volume filter
        if self.use_volume_filter:
            self.detect_volume_spikes(signals)
            
        return signals
    
    def detect_candlestick_patterns(self, df):
        """
        Detect key candlestick patterns.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with pattern columns added
        """
        # Initialize pattern columns
        df['doji'] = False
        df['engulfing'] = False
        df['hammer'] = False
        df['bullish_pattern'] = False
        df['bearish_pattern'] = False
        
        # Calculate body and shadow sizes
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_to_range_ratio'] = df['body_size'] / (df['high'] - df['low']).replace(0, np.nan)
        
        # Detect Doji patterns (small body relative to range)
        if self.pattern_strictness == 'strict':
            doji_threshold = 0.1  # Body is less than 10% of range
        elif self.pattern_strictness == 'medium':
            doji_threshold = 0.15  # Body is less than 15% of range
        else:  # loose
            doji_threshold = 0.2  # Body is less than 20% of range
            
        df['doji'] = df['body_to_range_ratio'] <= doji_threshold
        
        # Detect Engulfing patterns
        for i in range(1, len(df)):
            # Bullish Engulfing
            df.loc[df.index[i], 'engulfing'] = (
                # Current candle is bullish (close > open)
                (df['close'].iloc[i] > df['open'].iloc[i]) and
                # Previous candle is bearish (close < open)
                (df['close'].iloc[i-1] < df['open'].iloc[i-1]) and
                # Current body engulfs previous body
                (df['close'].iloc[i] > df['open'].iloc[i-1]) and
                (df['open'].iloc[i] < df['close'].iloc[i-1])
            ) or (
                # Bearish Engulfing
                # Current candle is bearish (close < open)
                (df['close'].iloc[i] < df['open'].iloc[i]) and
                # Previous candle is bullish (close > open)
                (df['close'].iloc[i-1] > df['open'].iloc[i-1]) and
                # Current body engulfs previous body
                (df['close'].iloc[i] < df['open'].iloc[i-1]) and
                (df['open'].iloc[i] > df['close'].iloc[i-1])
            )
            
        # Detect Hammer patterns
        for i in range(len(df)):
            # Get candle direction
            is_bullish = df['close'].iloc[i] > df['open'].iloc[i]
            
            # Calculate body and shadow ratios for hammer detection
            if df['body_size'].iloc[i] > 0:  # Avoid division by zero
                lower_shadow_ratio = df['lower_shadow'].iloc[i] / df['body_size'].iloc[i]
                upper_shadow_ratio = df['upper_shadow'].iloc[i] / df['body_size'].iloc[i]
                
                # Hammer criteria based on strictness
                if self.pattern_strictness == 'strict':
                    min_shadow_ratio = 2.0  # Lower shadow at least 2x body
                    max_upper_ratio = 0.2   # Upper shadow at most 0.2x body
                elif self.pattern_strictness == 'medium':
                    min_shadow_ratio = 1.5  # Lower shadow at least 1.5x body
                    max_upper_ratio = 0.3   # Upper shadow at most 0.3x body
                else:  # loose
                    min_shadow_ratio = 1.0  # Lower shadow at least as big as body
                    max_upper_ratio = 0.5   # Upper shadow at most half the body
                
                # Bullish Hammer (in downtrend)
                df.loc[df.index[i], 'hammer'] = (
                    lower_shadow_ratio >= min_shadow_ratio and
                    upper_shadow_ratio <= max_upper_ratio
                )
        
        # Combine patterns into bullish/bearish signals
        for i in range(1, len(df)):
            # Bullish patterns
            df.loc[df.index[i], 'bullish_pattern'] = (
                # Bullish Engulfing
                (df['engulfing'].iloc[i] and df['close'].iloc[i] > df['open'].iloc[i]) or
                # Hammer in downtrend (check for downtrend using past n candles)
                (df['hammer'].iloc[i] and df['close'].iloc[i-1] < df['close'].iloc[max(0, i-5):i].mean()) or
                # Doji after down move
                (df['doji'].iloc[i] and df['close'].iloc[i-1] < df['open'].iloc[i-1])
            )
            
            # Bearish patterns
            df.loc[df.index[i], 'bearish_pattern'] = (
                # Bearish Engulfing
                (df['engulfing'].iloc[i] and df['close'].iloc[i] < df['open'].iloc[i]) or
                # Inverted Hammer in uptrend
                (df['hammer'].iloc[i] and df['upper_shadow'].iloc[i] > df['lower_shadow'].iloc[i] and 
                 df['close'].iloc[i-1] > df['close'].iloc[max(0, i-5):i].mean()) or
                # Doji after up move
                (df['doji'].iloc[i] and df['close'].iloc[i-1] > df['open'].iloc[i-1])
            )
        
        return df
    
    def detect_volume_spikes(self, df):
        """
        Detect volume spikes for trade confirmation.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volume spike indicator added
        """
        if 'volume' not in df.columns:
            # Add dummy volume indicator if volume not available
            df['volume_spike'] = True
            return df
            
        # Calculate average volume over lookback period
        df['avg_volume'] = df['volume'].rolling(window=self.volume_lookback).mean()
        
        # Detect volume spikes
        df['volume_spike'] = df['volume'] >= (df['avg_volume'] * self.volume_threshold)
        
        # For first few candles without enough lookback, assume no spike
        df['volume_spike'].fillna(False, inplace=True)
        
        return df
    
    def check_pattern_filter(self, df, idx):
        """
        Check if candlestick pattern filter passes.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            
        Returns:
            Boolean indicating if filter passes
        """
        if not self.use_pattern_filter or idx < 1:
            return True
            
        # Get current signal
        current_signal = df['signal'].iloc[idx]
        if current_signal == 0:
            return True  # No signal to confirm
            
        # Enhanced Candlestick Confirmation Filter
        # Check for strong bar characteristics
        current_bar = df.iloc[idx-1]  # Use previous completed bar for confirmation
        
        # Calculate bar strength characteristics
        bar_range = current_bar['high'] - current_bar['low']
        if bar_range == 0:  # Avoid division by zero
            body_to_range_ratio = 0
        else:
            body_to_range_ratio = abs(current_bar['close'] - current_bar['open']) / bar_range
            
        # Check for decisive candle (strong body)
        is_strong_candle = body_to_range_ratio > 0.7
        
        # Directional alignment - bar should close in signal direction
        directional_aligned = False
        if current_signal > 0:  # Long signal
            directional_aligned = current_bar['close'] > current_bar['open']  # Bullish candle
        else:  # Short signal
            directional_aligned = current_bar['close'] < current_bar['open']  # Bearish candle
            
        # Check for existing pattern indicators if available
        pattern_matched = False
        if current_signal > 0:  # Long signal
            # Check for any bullish pattern
            if 'bullish_pattern' in df.columns:
                pattern_matched = df['bullish_pattern'].iloc[idx]
                
            # Apply additional requirements based on settings
            if self.require_engulfing and 'engulfing' in df.columns and not df['engulfing'].iloc[idx]:
                pattern_matched = False
            if self.require_doji and 'doji' in df.columns and not df['doji'].iloc[idx]:
                pattern_matched = False
            if self.require_hammer and 'hammer' in df.columns and not df['hammer'].iloc[idx]:
                pattern_matched = False
                
        else:  # Short signal
            # Check for any bearish pattern
            if 'bearish_pattern' in df.columns:
                pattern_matched = df['bearish_pattern'].iloc[idx]
                
            # Apply additional requirements
            if self.require_engulfing and 'engulfing' in df.columns and not df['engulfing'].iloc[idx]:
                pattern_matched = False
            if self.require_doji and 'doji' in df.columns and not df['doji'].iloc[idx]:
                pattern_matched = False
                
        # Combine filters: strong candle + directional alignment OR specific pattern match
        filter_passed = (is_strong_candle and directional_aligned) or pattern_matched
        
        # Update stats
        if filter_passed:
            self.confirmation_stats['pattern_confirmed'] += 1
        else:
            self.confirmation_stats['pattern_rejected'] += 1
            
        return filter_passed
    
    def check_volume_filter(self, df, idx):
        """
        Check if volume filter passes.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            
        Returns:
            Boolean indicating if filter passes
        """
        if not self.use_volume_filter or 'volume_spike' not in df.columns:
            return True
            
        # Get current signal
        current_signal = df['signal'].iloc[idx]
        if current_signal == 0:
            return True  # No signal to confirm
            
        # Check for volume spike
        volume_confirmed = df['volume_spike'].iloc[idx]
        
        # Update stats
        if volume_confirmed:
            self.confirmation_stats['volume_confirmed'] += 1
        else:
            self.confirmation_stats['volume_rejected'] += 1
            
        return volume_confirmed
    
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
    
    def calculate_position_size(self, df, idx, account_balance, risk_pct=None):
        """
        Calculate position size based on account risk and volatility.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            account_balance: Current account balance
            risk_pct: Risk percentage per trade (optional)
            
        Returns:
            Position size in base currency
        """
        # Use instance risk_per_trade if not provided
        if risk_pct is None:
            risk_pct = self.risk_per_trade
            
        # Calculate dollar risk amount
        risk_amount = account_balance * risk_pct
        
        # Current price
        current_price = df['close'].iloc[idx]
        
        # Get the ATR if available
        atr_value = df['atr'].iloc[idx] if 'atr' in df.columns else 0
        
        # Enhanced Volatility-Targeted Position Sizing
        if atr_value > 0 and self.use_volatility_sizing:
            # ATR-based stop distance
            atr_multiplier = self.atr_multiplier
            
            # Adjust multiplier based on volatility regime if available
            if 'vol_regime' in df.columns:
                vol_regime = df['vol_regime'].iloc[idx]
                if vol_regime == 'LOW':
                    # Wider stops for low volatility
                    atr_multiplier *= 1.2  
                elif vol_regime == 'HIGH':
                    # Tighter stops for high volatility
                    atr_multiplier *= 0.8
            
            # Calculate dollar-risk per unit based on ATR
            stop_distance = atr_value * atr_multiplier
            
            # Calculate position size to risk exactly risk_amount dollars
            # If ATR is $100 and we want to risk $200, we size 2 units
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            # Convert to position size in base currency
            position_size_base = position_size / current_price if current_price > 0 else 0
        else:
            # Fallback to simple percentage-based risk
            # Assume a default 2% stop loss if no ATR
            stop_distance = current_price * 0.02  
            position_size_base = risk_amount / stop_distance if stop_distance > 0 else 0
        
        # Apply reasonability checks
        # 1. Cap to max position size (percentage of account)
        max_position_value = account_balance * self.max_position_size
        position_size_base = min(position_size_base, max_position_value / current_price)
        
        # 2. Ensure non-negative size
        position_size_base = max(0, position_size_base)
        
        # Track volatility adjustments
        if atr_value > 0 and self.use_volatility_sizing:
            self.volatility_adjustments += 1
            self.confirmation_stats['volatility_adjustments'] += 1
            
        return position_size_base
    
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
        
        # Check pattern and volume filters
        pattern_confirmed = self.check_pattern_filter(df, idx)
        volume_confirmed = self.check_volume_filter(df, idx)
        
        # Volatility filter is now disabled by default
        # Only perform an extreme volatility check in rare cases
        valid_volatility = True
        if self.use_volatility_sizing and 'vol_regime' in df.columns:
            # Only filter out extremely high volatility (top 5% cases)
            extreme_volatility = df['vol_regime'].iloc[idx] == 'HIGH' and df['annualized_vol'].iloc[idx] > self.volatility_target * 3
            valid_volatility = not extreme_volatility
        
        # Combined decision - required filters must pass
        return (mtf_aligned and 
                momentum_aligned and 
                pattern_confirmed and 
                volume_confirmed and 
                valid_volatility)
    
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

    def calculate_take_profit(self, df, idx, entry_price, stop_loss, side):
        """
        Calculate take profit levels based on R-multiples.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            entry_price: Entry price
            stop_loss: Stop loss price
            side: Trade direction ('long' or 'short')
            
        Returns:
            Dictionary containing take profit levels
        """
        # Calculate R value (risk per share)
        if side == 'long':
            r_value = entry_price - stop_loss
        else:  # short
            r_value = stop_loss - entry_price
            
        # Safety check
        if r_value <= 0:
            r_value = entry_price * 0.01  # Default to 1% if calculation fails
            
        # Calculate take profit levels based on R-multiples
        take_profits = {}
        
        # Main take profit
        if side == 'long':
            take_profits['main'] = entry_price + (r_value * self.take_profit_r)
        else:  # short
            take_profits['main'] = entry_price - (r_value * self.take_profit_r)
            
        # Partial exit level
        if self.use_partial_exit:
            if side == 'long':
                take_profits['partial'] = entry_price + (r_value * self.partial_exit_r)
            else:  # short
                take_profits['partial'] = entry_price - (r_value * self.partial_exit_r)
        
        return take_profits
    
    def calculate_trailing_stop(self, df, idx, entry_price, stop_loss, current_price, side, r_value=None):
        """
        Calculate adaptive trailing stop based on profit achieved.
        
        Args:
            df: DataFrame with signal data
            idx: Current index
            entry_price: Entry price
            stop_loss: Initial stop loss price
            current_price: Current market price
            side: Trade direction ('long' or 'short')
            r_value: Risk per share (optional)
            
        Returns:
            Updated stop loss price
        """
        if not self.use_trailing_stop:
            return stop_loss
            
        # Calculate R value if not provided
        if r_value is None:
            if side == 'long':
                r_value = entry_price - stop_loss
            else:  # short
                r_value = stop_loss - entry_price
                
        # Safety check
        if r_value <= 0:
            r_value = entry_price * 0.01  # Default to 1% if calculation fails
            
        # Calculate current profit in R multiples
        if side == 'long':
            current_profit_r = (current_price - entry_price) / r_value
        else:  # short
            current_profit_r = (entry_price - current_price) / r_value
            
        # Determine trailing stop distance based on profit achieved
        trail_percentage = 1.0  # Default trail distance (100% of initial trail)
        
        # Enhanced R-Multiple-Based trailing stops
        if current_profit_r >= 3.0:
            # Very tight trail at 3R+ (25% of initial)
            trail_percentage = self.r3_trail_pct
        elif current_profit_r >= 2.0:
            # Tighter trail at 2R+ (50% of initial)
            trail_percentage = self.r2_trail_pct
        elif current_profit_r >= 1.0:
            # Standard trail at 1R+ (75% of initial)
            trail_percentage = self.r1_trail_pct
            
        # Calculate trailing stop distance
        trail_distance = self.initial_trail_r * r_value * trail_percentage
        
        # If we're up 1R+, don't let the stop go below breakeven
        if current_profit_r >= 1.0:
            if side == 'long':
                new_stop = max(current_price - trail_distance, entry_price)
                # Only move stop up, never down
                return max(new_stop, stop_loss)
            else:  # short
                new_stop = min(current_price + trail_distance, entry_price)
                # Only move stop down, never up
                return min(new_stop, stop_loss)
        else:
            # Standard trailing stop calculation
            if side == 'long':
                new_stop = current_price - trail_distance
                # Only move stop up, never down
                return max(new_stop, stop_loss)
            else:  # short
                new_stop = current_price + trail_distance
                # Only move stop down, never up
                return min(new_stop, stop_loss)
            
    def check_max_hold_time(self, entry_time, current_time, timeframe=None):
        """
        Check if the maximum hold time has been exceeded.
        
        Args:
            entry_time: Entry timestamp
            current_time: Current timestamp
            timeframe: Timeframe string (optional)
            
        Returns:
            Boolean indicating if max hold time is exceeded
        """
        # If times aren't datetime objects, assume periods and count them
        if not isinstance(entry_time, pd.Timestamp) or not isinstance(current_time, pd.Timestamp):
            periods_held = current_time - entry_time
            return periods_held >= self.max_hold_periods
            
        # If timeframe is provided, convert to timedelta
        if timeframe:
            if timeframe.endswith('m'):
                minutes = int(timeframe[:-1])
                max_delta = timedelta(minutes=minutes * self.max_hold_periods)
            elif timeframe.endswith('h'):
                hours = int(timeframe[:-1])
                max_delta = timedelta(hours=hours * self.max_hold_periods)
            elif timeframe.endswith('d'):
                days = int(timeframe[:-1])
                max_delta = timedelta(days=days * self.max_hold_periods)
            else:
                # Default to 24 hours if timeframe is unknown
                max_delta = timedelta(hours=24)
        else:
            # Default to 24 hours if timeframe is not provided
            max_delta = timedelta(hours=24)
            
        # Check if the time difference exceeds the maximum hold time
        return (current_time - entry_time) >= max_delta
    
    def manage_trade_exits(self, df, position, idx, timeframe=None):
        """
        Comprehensive exit management system.
        
        Args:
            df: DataFrame with signal data
            position: Current position information dictionary
            idx: Current index
            timeframe: Timeframe string (optional)
            
        Returns:
            Dictionary with exit decision and updated position
        """
        # Extract position details
        entry_price = position.get('entry_price', 0)
        stop_loss = position.get('stop_loss', 0)
        take_profits = position.get('take_profits', {})
        side = position.get('side', 'long')
        entry_time = position.get('entry_time', 0)
        position_size = position.get('size', 0)
        remaining_size = position.get('remaining_size', position_size)
        half_exited = position.get('half_exited', False)
        trail_tightened = position.get('trail_tightened', False)
        
        # Current market conditions
        current_price = df['close'].iloc[idx]
        current_time = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else idx
        
        # Initialize exit decision
        exit = {
            'exit_triggered': False,
            'exit_price': 0,
            'exit_reason': '',
            'exit_size': 0,
            'remaining_size': remaining_size,
            'half_exited': half_exited,
            'trail_tightened': trail_tightened
        }
        
        # Calculate R value
        if side == 'long':
            r_value = entry_price - stop_loss
        else:  # short
            r_value = stop_loss - entry_price
            
        # Calculate current profit in R multiples
        if side == 'long':
            current_profit_r = (current_price - entry_price) / r_value if r_value > 0 else 0
        else:  # short
            current_profit_r = (entry_price - current_price) / r_value if r_value > 0 else 0
            
        # New logic: Move stop to breakeven once we're at +1R and took partial profit
        if current_profit_r >= 1.0 and not half_exited and self.use_partial_exit:
            # Take 50% profit at +1R
            exit_size = remaining_size * self.partial_exit_pct
            exit['exit_triggered'] = True
            exit['exit_price'] = current_price
            exit['exit_reason'] = 'Partial Take Profit +1R'
            exit['exit_size'] = exit_size
            exit['remaining_size'] = remaining_size - exit_size
            exit['half_exited'] = True
            
            # Move stop to breakeven
            if side == 'long':
                stop_loss = max(entry_price, stop_loss)
            else:
                stop_loss = min(entry_price, stop_loss)
                
            # Update position info
            position['stop_loss'] = stop_loss
            return exit
            
        # New logic: Tighten trail at +2R if we haven't already
        if current_profit_r >= 2.0 and not trail_tightened:
            # Mark trail as tightened - actual tightening happens in calculate_trailing_stop
            exit['trail_tightened'] = True
            position['trail_tightened'] = True
            
        # Update trailing stop
        updated_stop = self.calculate_trailing_stop(
            df, idx, entry_price, stop_loss, current_price, side, r_value
        )
        
        # Check stop loss hit
        stop_triggered = False
        if side == 'long' and current_price <= updated_stop:
            stop_triggered = True
        elif side == 'short' and current_price >= updated_stop:
            stop_triggered = True
            
        if stop_triggered:
            exit['exit_triggered'] = True
            exit['exit_price'] = updated_stop
            exit['exit_reason'] = 'Stop Loss' if updated_stop == stop_loss else 'Trailing Stop'
            exit['exit_size'] = remaining_size
            exit['remaining_size'] = 0
            return exit
            
        # Check take profit hit (for remaining position after partial exit)
        if 'main' in take_profits:
            tp_triggered = False
            if side == 'long' and current_price >= take_profits['main']:
                tp_triggered = True
            elif side == 'short' and current_price <= take_profits['main']:
                tp_triggered = True
                
            if tp_triggered:
                exit['exit_triggered'] = True
                exit['exit_price'] = take_profits['main']
                exit['exit_reason'] = 'Take Profit'
                exit['exit_size'] = remaining_size
                exit['remaining_size'] = 0
                return exit
                
        # Check time-based exit
        if self.check_max_hold_time(entry_time, current_time, timeframe):
            exit['exit_triggered'] = True
            exit['exit_price'] = current_price
            exit['exit_reason'] = 'Time Stop'
            exit['exit_size'] = remaining_size
            exit['remaining_size'] = 0
            return exit
            
        # No exit triggered
        return exit 