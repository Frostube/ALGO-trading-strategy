import pandas as pd
import numpy as np

from src.config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD, VOLUME_PERIOD, VOLUME_THRESHOLD,
    EMA_TREND, EMA_MICRO_TREND, ATR_PERIOD, USE_ATR_STOPS, 
    USE_ADAPTIVE_THRESHOLDS, ADAPTIVE_LOOKBACK, MIN_BARS_BETWEEN_TRADES
)
from src.utils.logger import logger

def apply_indicators(df):
    """
    Apply all technical indicators to the dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with indicators added
    """
    logger.debug("Applying technical indicators to data")
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Apply indicators
    df = calculate_ema(df, EMA_FAST, EMA_SLOW, EMA_TREND, EMA_MICRO_TREND)
    df = calculate_rsi(df, RSI_PERIOD)
    df = calculate_volume_ma(df, VOLUME_PERIOD)
    df = calculate_atr(df, ATR_PERIOD)
    df = calculate_momentum_indicators(df)
    
    if USE_ADAPTIVE_THRESHOLDS:
        df = calculate_adaptive_thresholds(df, lookback=ADAPTIVE_LOOKBACK)
    
    # Clean up NaN values (indicators typically have NaN at the beginning)
    df = df.dropna()
    
    return df

def calculate_ema(df, fast_period=9, slow_period=21, trend_period=200, micro_trend_period=50):
    """
    Calculate EMA indicators and crossover.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        trend_period: Longer-term trend EMA period
        micro_trend_period: Micro trend EMA period for slope detection
        
    Returns:
        DataFrame with EMA indicators added
    """
    # Calculate EMAs
    df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate long-term trend EMA
    df[f'ema_{trend_period}'] = df['close'].ewm(span=trend_period, adjust=False).mean()
    
    # Calculate micro-trend EMA for slope detection
    df[f'ema_{micro_trend_period}'] = df['close'].ewm(span=micro_trend_period, adjust=False).mean()
    
    # Calculate micro-trend EMA slope (positive: uptrend, negative: downtrend)
    df['ema_micro_slope'] = df[f'ema_{micro_trend_period}'].diff(5) / 5
    df['ema_micro_direction'] = np.where(df['ema_micro_slope'] > 0, 1, -1)
    
    # Determine trend direction (1 for bullish, -1 for bearish)
    df['ema_trend'] = 0
    df.loc[df[f'ema_{fast_period}'] > df[f'ema_{slow_period}'], 'ema_trend'] = 1
    df.loc[df[f'ema_{fast_period}'] < df[f'ema_{slow_period}'], 'ema_trend'] = -1
    
    # Determine long-term trend
    df['market_trend'] = 0
    df.loc[df['close'] > df[f'ema_{trend_period}'], 'market_trend'] = 1
    df.loc[df['close'] < df[f'ema_{trend_period}'], 'market_trend'] = -1
    
    # Detect crossover points
    df['ema_crossover'] = df['ema_trend'].diff().fillna(0)
    
    return df

def calculate_rsi(df, period=2):
    """
    Calculate RSI indicator.
    
    Args:
        df: DataFrame with OHLCV data
        period: RSI period
        
    Returns:
        DataFrame with RSI indicator added
    """
    # Calculate price changes
    df['price_change'] = df['close'].diff()
    
    # Calculate gains and losses
    df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
    df['loss'] = df['price_change'].apply(lambda x: abs(x) if x < 0 else 0)
    
    # Calculate average gains and losses
    df['avg_gain'] = df['gain'].rolling(window=period).mean()
    df['avg_loss'] = df['loss'].rolling(window=period).mean()
    
    # Calculate relative strength
    df['rs'] = df['avg_gain'] / df['avg_loss']
    
    # Calculate RSI
    df['rsi'] = 100 - (100 / (1 + df['rs']))
    
    # Clean up temporary columns
    df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
    
    return df

def calculate_atr(df, period=14):
    """
    Calculate Average True Range (ATR) for volatility-based stops.
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        
    Returns:
        DataFrame with ATR added
    """
    # Calculate True Range
    df['tr1'] = abs(df['high'] - df['low'])
    df['tr2'] = abs(df['high'] - df['close'].shift())
    df['tr3'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(window=period).mean()
    
    # Calculate percentage ATR (ATR relative to price)
    df['atr_pct'] = df['atr'] / df['close'] * 100
    
    # Clean up temporary columns
    df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
    
    return df

def calculate_volume_ma(df, period=20):
    """
    Calculate volume moving average and volume spike indicator.
    
    Args:
        df: DataFrame with OHLCV data
        period: Volume MA period
        
    Returns:
        DataFrame with volume indicators added
    """
    # Calculate volume moving average
    df['volume_ma'] = df['volume'].rolling(window=period).mean()
    
    # Calculate volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Volume spike indicator (True when volume > threshold * MA)
    df['volume_spike'] = df['volume_ratio'] > VOLUME_THRESHOLD
    
    return df

def calculate_momentum_indicators(df):
    """
    Calculate momentum confirmation indicators.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with momentum indicators added
    """
    # Calculate previous 5-period high and low
    df['prev_5_high'] = df['high'].rolling(window=5).max().shift(1)
    df['prev_5_low'] = df['low'].rolling(window=5).min().shift(1)
    
    # Momentum confirmation flags
    df['momentum_up'] = df['close'] > df['prev_5_high']
    df['momentum_down'] = df['close'] < df['prev_5_low']
    
    # Calculate bar since last entry/exit counter
    df['bars_since_signal'] = np.nan  # Will be set dynamically during signal generation
    
    return df

def calculate_adaptive_thresholds(df, lookback=100):
    """
    Calculate adaptive RSI and volume thresholds based on percentiles.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to look back for percentile calculation
        
    Returns:
        DataFrame with adaptive thresholds added
    """
    # Calculate RSI adaptive thresholds (10th percentile for oversold, 90th for overbought)
    df['rsi_lower_threshold'] = df['rsi'].rolling(window=lookback).quantile(0.10)
    df['rsi_upper_threshold'] = df['rsi'].rolling(window=lookback).quantile(0.90)
    
    # Calculate adaptive volume threshold (80th percentile)
    df['adaptive_volume_threshold'] = df['volume_ratio'].rolling(window=lookback).quantile(0.80)
    
    # Use adaptive volume spike threshold
    df['adaptive_volume_spike'] = df['volume_ratio'] > df['adaptive_volume_threshold']
    
    return df

def get_signal(df, index=-1, last_signal_time=None, min_bars_between=MIN_BARS_BETWEEN_TRADES):
    """
    Generate trading signal based on the latest indicators.
    
    Args:
        df: DataFrame with indicators applied
        index: Index to get signal for (-1 for latest)
        last_signal_time: Timestamp of the last signal (for trade frequency control)
        min_bars_between: Minimum number of bars between trades
        
    Returns:
        dict with signal information
    """
    # Get the row for the specified index
    if isinstance(index, int):
        row = df.iloc[index]
    else:
        row = df.loc[index]
    
    # Initialize signal
    signal = {
        'timestamp': row.name,
        'close': row['close'],
        'ema_trend': row['ema_trend'],
        'market_trend': row.get('market_trend', 0),
        'micro_trend': row.get('ema_micro_direction', 0),
        'rsi': row['rsi'],
        'volume_spike': row['volume_spike'] if not USE_ADAPTIVE_THRESHOLDS else row['adaptive_volume_spike'],
        'atr': row.get('atr', None),
        'atr_pct': row.get('atr_pct', None),
        'momentum_up': row.get('momentum_up', False),
        'momentum_down': row.get('momentum_down', False),
        'signal': 'neutral'
    }
    
    # Check if enough time has passed since last signal
    bars_since_allowed = True
    if last_signal_time is not None:
        try:
            # Find how many bars since last signal
            idx = df.index.get_loc(row.name)
            last_idx = df.index.get_loc(last_signal_time)
            bars_since = idx - last_idx
            bars_since_allowed = bars_since >= min_bars_between
        except:
            # If we can't find the index, assume it's OK
            bars_since_allowed = True
    
    # Get RSI thresholds (fixed or adaptive)
    if USE_ADAPTIVE_THRESHOLDS and 'rsi_lower_threshold' in row and 'rsi_upper_threshold' in row:
        rsi_oversold = row['rsi_lower_threshold']
        rsi_overbought = row['rsi_upper_threshold']
    else:
        rsi_oversold = RSI_PERIOD * 5  # Dynamic RSI threshold based on period
        rsi_overbought = 100 - (RSI_PERIOD * 5)  # Dynamic RSI threshold
    
    # Long signal with all conditions
    if (bars_since_allowed and
        row['market_trend'] > 0 and 
        row['ema_trend'] > 0 and 
        row['ema_micro_direction'] > 0 and  # Micro-trend slope is positive
        row['rsi'] < rsi_oversold and
        signal['volume_spike'] and
        row['momentum_up']):  # Momentum confirmation
        signal['signal'] = 'buy'
    
    # Short signal with all conditions
    elif (bars_since_allowed and
          row['market_trend'] < 0 and 
          row['ema_trend'] < 0 and 
          row['ema_micro_direction'] < 0 and  # Micro-trend slope is negative
          row['rsi'] > rsi_overbought and 
          signal['volume_spike'] and
          row['momentum_down']):  # Momentum confirmation
        signal['signal'] = 'sell'
    
    return signal 