import pandas as pd
import numpy as np
import logging
from src.strategy.pattern_filtered_strategy import PatternFilteredStrategy
from src.strategy.confirm import (
    add_rsi, add_macd, add_stochastic, add_vwap, add_bollinger_bands,
    is_rsi_aligned, is_macd_aligned, is_stoch_aligned,
    is_volume_spike, is_vwap_aligned, is_atr_favorable,
    is_bb_squeeze, is_near_pivot, is_momentum_confirmed
)

logger = logging.getLogger(__name__)

class EnhancedConfirmationStrategy(PatternFilteredStrategy):
    """
    Enhanced trading strategy that combines all previous filters with additional
    confirmation tools:
    
    1. Momentum Oscillators (RSI, MACD, Stochastic)
    2. Volume Filters (Volume spike, VWAP alignment)
    3. Volatility Regime Checks (ATR Percentile, Bollinger Band squeeze)
    4. Support/Resistance Pivot Zones
    
    This strategy represents the most comprehensive filtering approach, using
    multiple layers of confirmation to ensure only the highest-probability
    trades are taken.
    """
    
    def __init__(self, 
                 # Base parameters
                 fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
                 lookback_window=20, vol_window=14, use_trend_filter=True,
                 regime_params_file=None, daily_ema_period=200, 
                 enforce_trend_alignment=True, vol_threshold_percentile=80,
                 require_pattern_confirmation=True, doji_threshold=0.1,
                 # New confirmation parameters
                 use_momentum_filter=True, use_volume_filter=True,
                 use_volatility_filter=True, use_pivot_filter=True,
                 rsi_period=14, rsi_threshold=50,
                 volume_lookback=20, volume_factor=1.5,
                 atr_percentile=80, bb_squeeze_lookback=20,
                 pivot_lookback=20, pivot_proximity=1.0,
                 min_confirmations=3):
        """
        Initialize the enhanced confirmation strategy.
        
        Args:
            fast_ema, slow_ema, etc.: Parameters inherited from parent strategies
            use_momentum_filter: Whether to use momentum oscillator confirmations
            use_volume_filter: Whether to use volume-based confirmations
            use_volatility_filter: Whether to use volatility-based confirmations
            use_pivot_filter: Whether to check for pivots before trading
            rsi_period: Period for RSI calculation
            rsi_threshold: RSI threshold for trend alignment
            volume_lookback: Periods for volume average calculation
            volume_factor: Factor for volume spike detection
            atr_percentile: Maximum percentile for acceptable ATR
            bb_squeeze_lookback: Periods for BB squeeze detection
            pivot_lookback: Periods for pivot point detection
            pivot_proximity: Percentage distance for pivot proximity
            min_confirmations: Minimum number of confirmation filters required
        """
        super().__init__(
            fast_ema, slow_ema, trend_ema, atr_period, atr_multiplier,
            lookback_window, vol_window, use_trend_filter,
            regime_params_file, daily_ema_period, 
            enforce_trend_alignment, vol_threshold_percentile,
            require_pattern_confirmation, doji_threshold
        )
        
        # Store confirmation parameters
        self.use_momentum_filter = use_momentum_filter
        self.use_volume_filter = use_volume_filter
        self.use_volatility_filter = use_volatility_filter
        self.use_pivot_filter = use_pivot_filter
        
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.volume_lookback = volume_lookback
        self.volume_factor = volume_factor
        self.atr_percentile = atr_percentile
        self.bb_squeeze_lookback = bb_squeeze_lookback
        self.pivot_lookback = pivot_lookback
        self.pivot_proximity = pivot_proximity
        self.min_confirmations = min_confirmations
        
        # Track confirmation statistics
        self.confirmation_stats = {
            'total_signals': 0,
            'momentum_confirmed': 0,
            'momentum_rejected': 0,
            'volume_confirmed': 0,
            'volume_rejected': 0,
            'volatility_confirmed': 0,
            'volatility_rejected': 0,
            'pivot_confirmed': 0,
            'pivot_rejected': 0,
            'signals_passed': 0,
            'signals_rejected': 0
        }
        
        # Initialize daily data holder for MTF analysis
        self.daily_df = None
    
    def _calculate_indicators(self, df):
        """
        Calculate all required indicators for confirmations.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        # Calculate indicators based on which filters are enabled
        if self.use_momentum_filter:
            df = add_rsi(df, period=self.rsi_period)
            df = add_macd(df)
            df = add_stochastic(df)
            
        if self.use_volume_filter:
            df = add_vwap(df)
            
        if self.use_volatility_filter:
            df = add_bollinger_bands(df)
            
        return df
    
    def _check_momentum_confirmation(self, df, signal_type):
        """
        Check if momentum indicators confirm the signal.
        
        Args:
            df: DataFrame with indicators calculated
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if momentum confirms the signal
        """
        if not self.use_momentum_filter:
            return True
            
        # Check RSI alignment
        rsi_confirmed = is_rsi_aligned(df, signal_type, threshold=self.rsi_threshold)
        
        # Check MACD alignment (optional - less strict)
        macd_confirmed = is_macd_aligned(df, signal_type) if 'macd' in df.columns else True
        
        # For this strategy, we'll require RSI confirmation at minimum
        is_confirmed = rsi_confirmed
        
        self.confirmation_stats['total_signals'] += 1
        if is_confirmed:
            self.confirmation_stats['momentum_confirmed'] += 1
        else:
            self.confirmation_stats['momentum_rejected'] += 1
            
        logger.debug(f"Momentum confirmation: {is_confirmed} (RSI: {rsi_confirmed}, MACD: {macd_confirmed})")
        return is_confirmed
    
    def _check_volume_confirmation(self, df, signal_type):
        """
        Check if volume indicators confirm the signal.
        
        Args:
            df: DataFrame with indicators calculated
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if volume confirms the signal
        """
        if not self.use_volume_filter:
            return True
            
        # Check for volume spike
        volume_spike = is_volume_spike(df, lookback=self.volume_lookback, factor=self.volume_factor)
        
        # Check VWAP alignment
        vwap_aligned = is_vwap_aligned(df, signal_type) if 'vwap' in df.columns else True
        
        # For this strategy, we'll accept either volume confirmation
        is_confirmed = volume_spike or vwap_aligned
        
        if is_confirmed:
            self.confirmation_stats['volume_confirmed'] += 1
        else:
            self.confirmation_stats['volume_rejected'] += 1
            
        logger.debug(f"Volume confirmation: {is_confirmed} (Spike: {volume_spike}, VWAP: {vwap_aligned})")
        return is_confirmed
    
    def _check_volatility_confirmation(self, df, signal_type):
        """
        Check if volatility indicators confirm the signal.
        
        Args:
            df: DataFrame with indicators calculated
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if volatility conditions are favorable
        """
        if not self.use_volatility_filter:
            return True
            
        # Check if ATR is at acceptable levels
        atr_favorable = is_atr_favorable(df, period=self.atr_period, percentile=self.atr_percentile)
        
        # Check for Bollinger Band squeeze (optional)
        bb_squeeze = is_bb_squeeze(df, lookback=self.bb_squeeze_lookback) if 'bb_width' in df.columns else False
        
        # For this strategy, we primarily care about ATR levels
        is_confirmed = atr_favorable
        
        if is_confirmed:
            self.confirmation_stats['volatility_confirmed'] += 1
        else:
            self.confirmation_stats['volatility_rejected'] += 1
            
        logger.debug(f"Volatility confirmation: {is_confirmed} (ATR: {atr_favorable}, BB Squeeze: {bb_squeeze})")
        return is_confirmed
    
    def _check_pivot_confirmation(self, df, signal_type):
        """
        Check if the trade is not near a pivot point.
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if not near a pivot point (favorable for trading)
        """
        if not self.use_pivot_filter:
            return True
            
        # Check if price is near a pivot point (we want it NOT to be near pivot)
        near_pivot = is_near_pivot(df, lookback=self.pivot_lookback, proximity_pct=self.pivot_proximity)
        is_confirmed = not near_pivot
        
        if is_confirmed:
            self.confirmation_stats['pivot_confirmed'] += 1
        else:
            self.confirmation_stats['pivot_rejected'] += 1
            
        logger.debug(f"Pivot confirmation: {is_confirmed} (Near pivot: {near_pivot})")
        return is_confirmed
    
    def check_enhanced_confirmations(self, df, signal_type):
        """
        Run all confirmation checks and ensure minimum confirmations are met.
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if sufficient confirmations pass
        """
        # Calculate all required indicators
        df_with_indicators = self._calculate_indicators(df.copy())
        
        # Run all confirmation checks
        confirmations = []
        
        # 1. Momentum check
        momentum_confirmed = self._check_momentum_confirmation(df_with_indicators, signal_type)
        confirmations.append(momentum_confirmed)
        
        # 2. Volume check
        volume_confirmed = self._check_volume_confirmation(df_with_indicators, signal_type)
        confirmations.append(volume_confirmed)
        
        # 3. Volatility check
        volatility_confirmed = self._check_volatility_confirmation(df_with_indicators, signal_type)
        confirmations.append(volatility_confirmed)
        
        # 4. Pivot check
        pivot_confirmed = self._check_pivot_confirmation(df_with_indicators, signal_type)
        confirmations.append(pivot_confirmed)
        
        # Count active filters and confirmations
        active_filters = sum([
            self.use_momentum_filter,
            self.use_volume_filter,
            self.use_volatility_filter,
            self.use_pivot_filter
        ])
        passed_confirmations = sum(confirmations)
        
        # Calculate how many confirmations needed (either absolute or percentage)
        if isinstance(self.min_confirmations, float) and self.min_confirmations <= 1.0:
            # As a percentage of active filters
            required_confirmations = round(active_filters * self.min_confirmations)
        else:
            # As absolute number, but capped at active filters
            required_confirmations = min(int(self.min_confirmations), active_filters)
        
        # Ensure we have enough confirmations
        has_enough_confirmations = passed_confirmations >= required_confirmations
        
        # Track statistics
        if has_enough_confirmations:
            self.confirmation_stats['signals_passed'] += 1
        else:
            self.confirmation_stats['signals_rejected'] += 1
        
        logger.info(f"Enhanced confirmations: {passed_confirmations}/{active_filters} " 
                  f"({required_confirmations} required): {'PASS' if has_enough_confirmations else 'FAIL'}")
        
        return has_enough_confirmations
    
    def on_new_candle(self, candle_data, symbol=None):
        """
        Process new candle data with enhanced confirmations.
        
        Args:
            candle_data: Dictionary with latest candle OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            str: Signal type ('buy', 'sell', 'exit', 'none')
        """
        # Convert candle data to DataFrame if it's not already
        if not isinstance(candle_data, pd.DataFrame):
            if isinstance(candle_data, pd.Series):
                df = pd.DataFrame([candle_data.to_dict()])
            else:
                df = pd.DataFrame([candle_data])
        else:
            df = candle_data.copy()
            
        # Get signal from parent strategy (with all previous filters)
        signal = super().on_new_candle(candle_data, symbol)
        
        # If we have a buy or sell signal, apply enhanced confirmations
        if signal in ['buy', 'sell']:
            if not self.check_enhanced_confirmations(df, signal):
                logger.info(f"Signal {signal} rejected by enhanced confirmations")
                return 'none'
                
        return signal
    
    def backtest(self, df, symbol=None):
        """
        Backtest with all confirmations applied.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            DataFrame: DataFrame with signals
        """
        # Reset statistics
        self.confirmation_stats = {key: 0 for key in self.confirmation_stats}
        
        # Run backtest with parent strategy filters
        signals_df = super().backtest(df, symbol)
        
        # Make a copy to avoid modifying the original
        enhanced_df = signals_df.copy()
        
        # Calculate indicators for the entire dataset
        enhanced_df = self._calculate_indicators(enhanced_df)
        
        # Track filtered signals
        if 'enhanced_filtered' not in enhanced_df.columns:
            enhanced_df['enhanced_filtered'] = 0
        
        # Apply enhanced confirmations
        for i in range(max(5, self.lookback_window), len(enhanced_df)):
            # If we have a signal, check for confirmations
            if enhanced_df.iloc[i]['signal'] != 0:
                # Get subset of data up to current row
                subset_df = enhanced_df.iloc[:i+1]
                
                # Determine signal type
                signal_type = 'buy' if enhanced_df.iloc[i]['signal'] == 1 else 'sell'
                
                # Check enhanced confirmations
                if not self.check_enhanced_confirmations(subset_df, signal_type):
                    # Track the filtered signal
                    enhanced_df.loc[enhanced_df.index[i], 'enhanced_filtered'] = enhanced_df.iloc[i]['signal']
                    
                    # Remove the signal
                    enhanced_df.loc[enhanced_df.index[i], 'signal'] = 0
        
        # Log confirmation stats
        if self.confirmation_stats['total_signals'] > 0:
            momentum_pct = (self.confirmation_stats['momentum_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_momentum_filter else 'N/A'
            volume_pct = (self.confirmation_stats['volume_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_volume_filter else 'N/A'
            volatility_pct = (self.confirmation_stats['volatility_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_volatility_filter else 'N/A'
            pivot_pct = (self.confirmation_stats['pivot_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_pivot_filter else 'N/A'
            
            logger.info(f"Enhanced Confirmation Stats:")
            if self.use_momentum_filter:
                logger.info(f"Momentum: {self.confirmation_stats['momentum_confirmed']}/{self.confirmation_stats['total_signals']} ({momentum_pct:.1f}%)")
            if self.use_volume_filter:
                logger.info(f"Volume: {self.confirmation_stats['volume_confirmed']}/{self.confirmation_stats['total_signals']} ({volume_pct:.1f}%)")
            if self.use_volatility_filter:
                logger.info(f"Volatility: {self.confirmation_stats['volatility_confirmed']}/{self.confirmation_stats['total_signals']} ({volatility_pct:.1f}%)")
            if self.use_pivot_filter:
                logger.info(f"Pivot: {self.confirmation_stats['pivot_confirmed']}/{self.confirmation_stats['total_signals']} ({pivot_pct:.1f}%)")
                
            overall_pct = (self.confirmation_stats['signals_passed'] / (self.confirmation_stats['signals_passed'] + self.confirmation_stats['signals_rejected'])) * 100
            logger.info(f"Overall: {self.confirmation_stats['signals_passed']}/{self.confirmation_stats['signals_passed'] + self.confirmation_stats['signals_rejected']} ({overall_pct:.1f}%) passed all filters")
            
        return enhanced_df
        
    def get_info(self):
        """
        Get current strategy information.
        
        Returns:
            dict: Strategy parameters and state
        """
        info = super().get_info()
        info.update({
            'strategy_type': 'EnhancedConfirmationStrategy',
            'use_momentum_filter': self.use_momentum_filter,
            'use_volume_filter': self.use_volume_filter,
            'use_volatility_filter': self.use_volatility_filter,
            'use_pivot_filter': self.use_pivot_filter,
            'rsi_period': self.rsi_period,
            'rsi_threshold': self.rsi_threshold,
            'volume_lookback': self.volume_lookback,
            'volume_factor': self.volume_factor,
            'atr_percentile': self.atr_percentile,
            'min_confirmations': self.min_confirmations,
            'confirmation_stats': self.confirmation_stats
        })
        return info 