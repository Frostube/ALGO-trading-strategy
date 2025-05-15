import pandas as pd
import numpy as np
import logging
from src.strategy.trend_filtered_adaptive_ema import TrendFilteredAdaptiveEMA
from src.utils.candlestick_patterns import (
    is_doji, is_hammer, is_shooting_star, 
    is_bullish_engulfing, is_bearish_engulfing,
    is_bullish_pattern, is_bearish_pattern
)

logger = logging.getLogger(__name__)

class PatternFilteredStrategy(TrendFilteredAdaptiveEMA):
    """
    Enhanced trading strategy that combines regime adaptation, trend filtering,
    and candlestick pattern confirmation.
    
    Only generates signals when price action patterns confirm the trend and
    indicator-based signals.
    """
    
    def __init__(self, 
                 fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
                 lookback_window=20, vol_window=14, use_trend_filter=True,
                 regime_params_file=None, daily_ema_period=200, 
                 enforce_trend_alignment=True, vol_threshold_percentile=80,
                 require_pattern_confirmation=True, doji_threshold=0.1):
        """
        Initialize pattern-filtered strategy.
        
        Args:
            fast_ema, slow_ema, etc.: Parameters inherited from parent strategies
            require_pattern_confirmation: Whether to require candlestick pattern confirmation
            doji_threshold: Maximum ratio of body to range to qualify as a doji
        """
        super().__init__(
            fast_ema, slow_ema, trend_ema, atr_period, atr_multiplier,
            lookback_window, vol_window, use_trend_filter,
            regime_params_file, daily_ema_period, 
            enforce_trend_alignment, vol_threshold_percentile
        )
        
        self.require_pattern_confirmation = require_pattern_confirmation
        self.doji_threshold = doji_threshold
        self.recent_candles = []
        self.pattern_stats = {
            'total_signals': 0,
            'patterns_confirmed': 0,
            'patterns_rejected': 0,
            'bullish_patterns': 0,
            'bearish_patterns': 0
        }
        
    def _update_recent_candles(self, candle_data):
        """
        Update the recent candles list for pattern detection.
        
        Args:
            candle_data: Dictionary with candle OHLCV data
        """
        # Convert candle_data to proper format if it's not already
        if isinstance(candle_data, pd.DataFrame):
            candle = candle_data.iloc[-1].to_dict()
        elif isinstance(candle_data, pd.Series):
            candle = candle_data.to_dict()
        else:
            candle = candle_data
            
        # Append to recent candles, keeping only the last 5
        self.recent_candles.append(candle)
        if len(self.recent_candles) > 5:
            self.recent_candles = self.recent_candles[-5:]
            
    def check_pattern_confirmation(self, signal_type):
        """
        Check if the current price action confirms the signal.
        
        Args:
            signal_type: 'buy', 'sell', or other signal type
            
        Returns:
            bool: True if the pattern confirms the signal
        """
        if not self.require_pattern_confirmation:
            return True
            
        # We need at least 3 candles for most patterns
        if len(self.recent_candles) < 3:
            return False
            
        self.pattern_stats['total_signals'] += 1
        
        # For buy signals, check for bullish patterns
        if signal_type == 'buy':
            # Check for specific bullish patterns
            if is_bullish_pattern(self.recent_candles):
                self.pattern_stats['patterns_confirmed'] += 1
                self.pattern_stats['bullish_patterns'] += 1
                logger.info(f"Bullish pattern confirmed buy signal")
                return True
            else:
                self.pattern_stats['patterns_rejected'] += 1
                logger.info(f"No bullish pattern to confirm buy signal")
                return False
                
        # For sell signals, check for bearish patterns
        elif signal_type == 'sell':
            # Check for specific bearish patterns
            if is_bearish_pattern(self.recent_candles):
                self.pattern_stats['patterns_confirmed'] += 1
                self.pattern_stats['bearish_patterns'] += 1
                logger.info(f"Bearish pattern confirmed sell signal")
                return True
            else:
                self.pattern_stats['patterns_rejected'] += 1
                logger.info(f"No bearish pattern to confirm sell signal")
                return False
                
        # For other signals (like exit), no pattern confirmation needed
        return True
        
    def on_new_candle(self, candle_data, symbol=None):
        """
        Process new candle data and generate trading signals.
        
        Args:
            candle_data: Dictionary with latest candle OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            str: Signal type ('buy', 'sell', 'exit', 'none')
        """
        # Update recent candles list
        self._update_recent_candles(candle_data)
        
        # Get signal from parent strategy (with regime adaptation and trend filtering)
        signal = super().on_new_candle(candle_data, symbol)
        
        # If we have a buy or sell signal, check for pattern confirmation
        if signal in ['buy', 'sell']:
            if not self.check_pattern_confirmation(signal):
                return 'none'
                
        return signal
        
    def backtest(self, df, symbol=None):
        """
        Backtest with pattern filtering applied.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            DataFrame: DataFrame with signals
        """
        # Initialize the recent candles list
        self.recent_candles = []
        
        # Run backtest with parent strategy filters (regime adaptation and trend filtering)
        signals_df = super().backtest(df, symbol)
        
        # Make a copy to avoid modifying the original
        filtered_df = signals_df.copy()
        
        # Apply pattern filtering
        if self.require_pattern_confirmation:
            for i in range(3, len(filtered_df)):
                # If we have a signal, check for pattern confirmation
                if filtered_df.iloc[i]['signal'] != 0:
                    # Extract the recent candles
                    recent_candles = df.iloc[i-3:i+1].to_dict('records')
                    
                    # Determine signal type
                    signal_type = 'buy' if filtered_df.iloc[i]['signal'] == 1 else 'sell'
                    
                    # Check for pattern confirmation
                    self._update_recent_candles(filtered_df.iloc[i])
                    if not self.check_pattern_confirmation(signal_type):
                        # Track the filtered signal
                        if 'filtered_signal' not in filtered_df.columns:
                            filtered_df['filtered_signal'] = 0
                        filtered_df.loc[filtered_df.index[i], 'filtered_signal'] = filtered_df.iloc[i]['signal']
                        
                        # Remove the signal
                        filtered_df.loc[filtered_df.index[i], 'signal'] = 0
        
        # Log pattern stats
        if self.pattern_stats['total_signals'] > 0:
            confirmed_pct = (self.pattern_stats['patterns_confirmed'] / self.pattern_stats['total_signals']) * 100
            logger.info(f"Pattern Stats: {self.pattern_stats['patterns_confirmed']}/{self.pattern_stats['total_signals']} " 
                      f"({confirmed_pct:.1f}%) signals confirmed by patterns")
            logger.info(f"Bullish Patterns: {self.pattern_stats['bullish_patterns']}, " 
                      f"Bearish Patterns: {self.pattern_stats['bearish_patterns']}")
            
        return filtered_df
        
    def get_info(self):
        """
        Get current strategy information.
        
        Returns:
            dict: Strategy parameters and state
        """
        info = super().get_info()
        info.update({
            'strategy_type': 'PatternFilteredStrategy',
            'require_pattern_confirmation': self.require_pattern_confirmation,
            'doji_threshold': self.doji_threshold,
            'pattern_stats': self.pattern_stats
        })
        return info 