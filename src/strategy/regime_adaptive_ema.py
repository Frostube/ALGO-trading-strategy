import pandas as pd
import numpy as np
import logging
import os
from src.strategy.ema_crossover import EMACrossoverStrategy

logger = logging.getLogger(__name__)

class RegimeAdaptiveEMAStrategy(EMACrossoverStrategy):
    """
    Enhanced EMA Crossover strategy with regime-based parameter switching.
    Automatically selects the optimal parameters based on detected market regime.
    """
    
    def __init__(self, 
                 fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
                 lookback_window=20, vol_window=14, use_trend_filter=True,
                 regime_params_file=None):
        """
        Initialize with default parameters and regime-specific optimal parameters.
        
        Args:
            fast_ema: Default fast EMA period
            slow_ema: Default slow EMA period
            trend_ema: EMA period for trend filter
            atr_period: ATR lookback period
            atr_multiplier: Multiplier for ATR stop loss
            lookback_window: Window for trend detection (days)
            vol_window: Window for volatility calculation
            use_trend_filter: Whether to use trend filter
            regime_params_file: Path to CSV with best parameters for each regime
        """
        super().__init__(fast_ema, slow_ema, trend_ema, atr_period, atr_multiplier, use_trend_filter)
        
        self.lookback_window = lookback_window
        self.vol_window = vol_window
        self.current_regime = None
        self.best_params_by_regime = self._load_regime_params(regime_params_file)
        self.regime_history = []
        
    def _load_regime_params(self, params_file):
        """
        Load regime-specific parameters from CSV file.
        
        Returns:
            dict: Dictionary mapping regime names to parameter sets
        """
        default_params = {
            'bull_low_vol': {'fast_ema': 3, 'slow_ema': 15, 'atr_multiplier': 2.0},
            'bull_high_vol': {'fast_ema': 5, 'slow_ema': 20, 'atr_multiplier': 3.0},
            'bear_low_vol': {'fast_ema': 8, 'slow_ema': 25, 'atr_multiplier': 2.5},
            'bear_high_vol': {'fast_ema': 10, 'slow_ema': 30, 'atr_multiplier': 3.5},
        }
        
        if not params_file or not os.path.exists(params_file):
            logger.warning(f"Regime parameters file not found. Using default parameters.")
            return default_params
            
        try:
            # Load the CSV file with regime-specific parameters
            df = pd.read_csv(params_file)
            
            # Convert DataFrame to dictionary
            params_dict = {}
            for _, row in df.iterrows():
                regime = row['regime']
                params = {
                    'fast_ema': row['fast_ema'], 
                    'slow_ema': row['slow_ema'], 
                    'atr_multiplier': row['atr_multiplier']
                }
                params_dict[regime] = params
                
            logger.info(f"Loaded regime parameters for {len(params_dict)} regimes")
            return params_dict
            
        except Exception as e:
            logger.error(f"Error loading regime parameters: {e}")
            return default_params
            
    def detect_regime(self, df):
        """
        Detect current market regime based on trend and volatility.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            str: Detected regime ('bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol')
        """
        if len(df) < max(self.lookback_window, self.vol_window):
            return 'bull_low_vol'  # Default to bullish low volatility if not enough data
            
        # Copy the dataframe to avoid modifying the original
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate trend using simple moving average
        df['sma'] = df['close'].rolling(self.lookback_window).mean()
        df['trend'] = (df['close'] > df['sma']).astype(int)
        
        # Calculate volatility using ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        df['tr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(self.vol_window).mean()
        
        # Normalize ATR by price
        df['norm_atr'] = df['atr'] / df['close']
        
        # Determine if we're in a high volatility regime
        median_vol = df['norm_atr'].median()
        high_vol = df['norm_atr'].iloc[-1] > median_vol
        
        # Determine trend
        bullish = df['trend'].iloc[-1] == 1
        
        # Determine regime
        if bullish and high_vol:
            regime = 'bull_high_vol'
        elif bullish and not high_vol:
            regime = 'bull_low_vol'
        elif not bullish and high_vol:
            regime = 'bear_high_vol'
        else:
            regime = 'bear_low_vol'
            
        return regime
        
    def update_regime_parameters(self, current_regime):
        """
        Update strategy parameters based on detected regime.
        
        Args:
            current_regime: Detected market regime
        """
        if current_regime not in self.best_params_by_regime:
            logger.warning(f"Unknown regime {current_regime}. Using default parameters.")
            return
            
        # Get parameter set for current regime
        params = self.best_params_by_regime[current_regime]
        
        # Update strategy parameters
        self.fast_ema = params['fast_ema']
        self.slow_ema = params['slow_ema']
        self.atr_multiplier = params['atr_multiplier']
        
        logger.info(f"Switched to {current_regime} regime with parameters: {params}")
        
    def apply_indicators(self, df):
        """
        Apply indicators to dataframe and update regime parameters.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame: DataFrame with indicators added
        """
        # Detect market regime
        current_regime = self.detect_regime(df)
        
        # Update parameters if regime has changed
        if current_regime != self.current_regime:
            self.update_regime_parameters(current_regime)
            self.current_regime = current_regime
            
        # Track regime history
        self.regime_history.append({
            'timestamp': df.index[-1],
            'regime': current_regime,
            'fast_ema': self.fast_ema,
            'slow_ema': self.slow_ema,
            'atr_multiplier': self.atr_multiplier
        })
        
        # Apply standard indicators with updated parameters
        return super().apply_indicators(df)
        
    def get_info(self):
        """
        Get current strategy information.
        
        Returns:
            dict: Strategy parameters and state
        """
        info = super().get_info()
        info.update({
            'strategy_type': 'RegimeAdaptiveEMA',
            'current_regime': self.current_regime,
            'regime_parameters': self.best_params_by_regime.get(self.current_regime, {}),
            'lookback_window': self.lookback_window,
            'vol_window': self.vol_window
        })
        return info 