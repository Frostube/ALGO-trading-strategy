import pandas as pd
import numpy as np
import logging
from src.strategy.regime_adaptive_ema import RegimeAdaptiveEMAStrategy
from src.data.fetcher import fetch_ohlcv

logger = logging.getLogger(__name__)

class TrendFilteredAdaptiveEMA(RegimeAdaptiveEMAStrategy):
    """
    Enhanced regime-adaptive EMA strategy with additional trend and volatility filters.
    Adds daily EMA trend filter and ATR-based volatility thresholds to reduce false signals.
    """
    
    def __init__(self, 
                 fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
                 lookback_window=20, vol_window=14, use_trend_filter=True,
                 regime_params_file=None, 
                 daily_ema_period=200, 
                 enforce_trend_alignment=True,
                 vol_threshold_percentile=80):
        """
        Initialize trend-filtered regime-adaptive strategy.
        
        Args:
            fast_ema, slow_ema, trend_ema, atr_period, atr_multiplier: Base strategy parameters
            lookback_window, vol_window: Regime detection parameters
            use_trend_filter: Whether to use the basic trend filter
            regime_params_file: Path to CSV with regime-specific parameters
            daily_ema_period: Period for daily EMA trend filter
            enforce_trend_alignment: If True, only take trades in direction of daily trend
            vol_threshold_percentile: Percentile threshold for excessive volatility filter
        """
        super().__init__(fast_ema, slow_ema, trend_ema, atr_period, atr_multiplier,
                         lookback_window, vol_window, use_trend_filter, regime_params_file)
        
        self.daily_ema_period = daily_ema_period
        self.enforce_trend_alignment = enforce_trend_alignment
        self.vol_threshold_percentile = vol_threshold_percentile
        self.daily_ema = None
        self.vol_threshold = None
        self.daily_data = None
        
    def _fetch_daily_data(self, symbol):
        """
        Fetch daily candlestick data for longer-term trend analysis.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            DataFrame: Daily OHLCV data
        """
        try:
            # Fetch 300 days of daily data to ensure we have enough for 200 EMA
            daily_df = fetch_ohlcv(symbol, '1d', 300)
            if daily_df is not None and not daily_df.empty:
                return daily_df
        except Exception as e:
            logger.error(f"Error fetching daily data: {e}")
        
        return None
        
    def _calculate_daily_ema(self, symbol=None):
        """
        Calculate daily EMA for trend filtering.
        
        Args:
            symbol: Optional symbol for fetching data
            
        Returns:
            float: Current daily EMA value
        """
        if not self.daily_data and symbol:
            self.daily_data = self._fetch_daily_data(symbol)
            
        if self.daily_data is not None:
            # Calculate the daily EMA
            self.daily_data['ema'] = self.daily_data['close'].ewm(span=self.daily_ema_period, adjust=False).mean()
            
            # Get the most recent value
            return self.daily_data['ema'].iloc[-1]
            
        return None
        
    def _is_excessive_volatility(self, df):
        """
        Check if current volatility is excessively high.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            bool: True if volatility is above threshold
        """
        if df.empty or len(df) < self.vol_window:
            return False
            
        # Calculate normalized ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(self.vol_window).mean()
        norm_atr = atr / df['close']
        
        # Calculate volatility threshold if not already set
        if self.vol_threshold is None:
            self.vol_threshold = norm_atr.quantile(self.vol_threshold_percentile / 100)
            
        # Check if current volatility exceeds threshold
        current_vol = norm_atr.iloc[-1]
        return current_vol > self.vol_threshold
        
    def apply_trend_filter(self, df, symbol=None):
        """
        Apply daily trend filter to signals.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            symbol: Trading symbol for fetching daily data
            
        Returns:
            DataFrame: DataFrame with trend-filtered signals
        """
        if not self.enforce_trend_alignment:
            return df
            
        # Calculate daily EMA if not already cached
        if self.daily_ema is None and symbol:
            self.daily_ema = self._calculate_daily_ema(symbol)
            
        if self.daily_ema is None:
            logger.warning("Daily EMA not available for trend filtering")
            return df
            
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Current price
        current_price = filtered_df['close'].iloc[-1]
        
        # Get the current regime
        current_regime = self.current_regime or self.detect_regime(filtered_df)
        
        # Apply trend filter based on regime and daily EMA
        if current_regime.startswith('bull') and current_price < self.daily_ema:
            # In bull regime but below daily EMA, disable longs
            filtered_df.loc[filtered_df['signal'] == 1, 'signal'] = 0
            logger.info(f"Filtered long signal: price {current_price:.2f} below daily EMA {self.daily_ema:.2f}")
            
        elif current_regime.startswith('bear') and current_price > self.daily_ema:
            # In bear regime but above daily EMA, disable shorts
            filtered_df.loc[filtered_df['signal'] == -1, 'signal'] = 0
            logger.info(f"Filtered short signal: price {current_price:.2f} above daily EMA {self.daily_ema:.2f}")
            
        return filtered_df
        
    def apply_volatility_filter(self, df):
        """
        Filter signals during excessive volatility periods.
        
        Args:
            df: DataFrame with OHLCV data and signals
            
        Returns:
            DataFrame: DataFrame with volatility-filtered signals
        """
        # Check for excessive volatility
        if self._is_excessive_volatility(df):
            # Make a copy to avoid modifying the original
            filtered_df = df.copy()
            
            # Disable new signals during excessive volatility
            filtered_df['signal'] = 0
            logger.info("Filtered all signals due to excessive volatility")
            
            return filtered_df
            
        return df
        
    def backtest(self, df, symbol=None):
        """
        Backtest with trend and volatility filters applied.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            DataFrame: DataFrame with signals
        """
        # Run the base regime-adaptive backtest
        signals_df = super().backtest(df)
        
        # Apply the daily trend filter
        signals_df = self.apply_trend_filter(signals_df, symbol)
        
        # Apply the volatility filter
        signals_df = self.apply_volatility_filter(signals_df)
        
        return signals_df
        
    def on_new_candle(self, candle_data, symbol=None):
        """
        Process new candle data and generate trading signals.
        
        Args:
            candle_data: Dictionary with latest candle OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            str: Signal type ('buy', 'sell', 'exit', 'none')
        """
        # Update daily EMA if we have a symbol
        if symbol and (self.daily_ema is None or candle_data.get('is_new_day', False)):
            self.daily_ema = self._calculate_daily_ema(symbol)
            
        # Get base signal from parent class
        signal = super().on_new_candle(candle_data)
        
        # Apply trend filter
        current_price = candle_data.get('close', 0)
        current_regime = self.current_regime
        
        if self.enforce_trend_alignment and self.daily_ema is not None:
            if signal == 'buy' and current_regime.startswith('bull') and current_price < self.daily_ema:
                logger.info(f"Filtered buy signal: price {current_price:.2f} below daily EMA {self.daily_ema:.2f}")
                return 'none'
                
            if signal == 'sell' and current_regime.startswith('bear') and current_price > self.daily_ema:
                logger.info(f"Filtered sell signal: price {current_price:.2f} above daily EMA {self.daily_ema:.2f}")
                return 'none'
                
        # Check for excessive volatility
        if hasattr(self, 'recent_candles') and len(self.recent_candles) >= self.vol_window:
            recent_df = pd.DataFrame(self.recent_candles)
            if self._is_excessive_volatility(recent_df) and signal in ['buy', 'sell']:
                logger.info(f"Filtered {signal} signal due to excessive volatility")
                return 'none'
                
        return signal
        
    def get_info(self):
        """
        Get current strategy information.
        
        Returns:
            dict: Strategy parameters and state
        """
        info = super().get_info()
        info.update({
            'strategy_type': 'TrendFilteredAdaptiveEMA',
            'daily_ema_period': self.daily_ema_period,
            'daily_ema_value': self.daily_ema,
            'enforce_trend_alignment': self.enforce_trend_alignment,
            'vol_threshold': self.vol_threshold,
            'vol_threshold_percentile': self.vol_threshold_percentile
        })
        return info 