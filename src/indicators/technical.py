import pandas as pd
import numpy as np

from src.config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD, VOLUME_PERIOD, VOLUME_THRESHOLD,
    EMA_TREND, ATR_PERIOD, USE_ATR_STOPS
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
    df = calculate_ema(df, EMA_FAST, EMA_SLOW, EMA_TREND)
    df = calculate_rsi(df, RSI_PERIOD)
    df = calculate_volume_ma(df, VOLUME_PERIOD)
    df = calculate_atr(df, ATR_PERIOD)
    
    # Clean up NaN values (indicators typically have NaN at the beginning)
    df = df.dropna()
    
    return df

def calculate_ema(df, fast_period=9, slow_period=21, trend_period=200):
    """
    Calculate EMA indicators and crossover.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        trend_period: Longer-term trend EMA period
        
    Returns:
        DataFrame with EMA indicators added
    """
    # Calculate EMAs
    df[f'ema_{fast_period}'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df[f'ema_{slow_period}'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    
    # Calculate long-term trend EMA
    df[f'ema_{trend_period}'] = df['close'].ewm(span=trend_period, adjust=False).mean()
    
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
        'market_trend': row.get('market_trend', 0),
        'rsi': row['rsi'],
        'volume_spike': row['volume_spike'],
        'atr': row.get('atr', None),
        'signal': 'neutral'
    }
    
    # Long signal: Market trend is up, EMA trend is up, RSI is oversold, and volume spike
    if (row['market_trend'] > 0 and 
        row['ema_trend'] > 0 and 
        row['rsi'] < RSI_PERIOD * 5 and  # Dynamic RSI threshold based on period
        row['volume_spike']):
        signal['signal'] = 'buy'
    
    # Short signal: Market trend is down, EMA trend is down, RSI is overbought, and volume spike
    elif (row['market_trend'] < 0 and 
          row['ema_trend'] < 0 and 
          row['rsi'] > 100 - (RSI_PERIOD * 5) and  # Dynamic RSI threshold
          row['volume_spike']):
        signal['signal'] = 'sell'
    
    return signal 