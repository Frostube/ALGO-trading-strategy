import pandas as pd
import numpy as np

from src.config import EMA_FAST, EMA_SLOW, RSI_PERIOD, VOLUME_PERIOD, VOLUME_THRESHOLD
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
    df = calculate_ema(df, EMA_FAST, EMA_SLOW)
    df = calculate_rsi(df, RSI_PERIOD)
    df = calculate_volume_ma(df, VOLUME_PERIOD)
    
    # Clean up NaN values (indicators typically have NaN at the beginning)
    df = df.dropna()
    
    return df

def calculate_ema(df, fast_period=9, slow_period=21):
    """
    Calculate EMA indicators and crossover.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        DataFrame with EMA indicators added
    """
    # Calculate EMAs
    df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate crossover signal (1 when fast crosses above slow, -1 when fast crosses below slow, 0 otherwise)
    df['ema_crossover'] = 0
    df.loc[df[f'ema_{fast_period}'] > df[f'ema_{slow_period}'], 'ema_trend'] = 1
    df.loc[df[f'ema_{fast_period}'] < df[f'ema_{slow_period}'], 'ema_trend'] = -1
    
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

def get_signal(df, index=-1):
    """
    Generate trading signal based on the latest indicators.
    
    Args:
        df: DataFrame with indicators applied
        index: Index to get signal for (-1 for latest)
        
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
        'rsi': row['rsi'],
        'volume_spike': row['volume_spike'],
        'signal': 'neutral'
    }
    
    # Long signal: EMA trend is up (fast > slow), RSI is oversold (< 10), and volume spike
    if row['ema_trend'] > 0 and row['rsi'] < 10 and row['volume_spike']:
        signal['signal'] = 'buy'
    
    # Short signal: EMA trend is down (fast < slow), RSI is overbought (> 90), and volume spike
    elif row['ema_trend'] < 0 and row['rsi'] > 90 and row['volume_spike']:
        signal['signal'] = 'sell'
    
    return signal 