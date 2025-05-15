import pandas as pd
import numpy as np

def calculate_ema(series, period=14):
    """
    Calculate Exponential Moving Average for a price series.
    
    Args:
        series: Price series to use (usually close prices)
        period: EMA period
        
    Returns:
        Series of EMA values
    """
    return series.ewm(span=period, adjust=False).mean()

def calculate_sma(series, period=14):
    """
    Calculate Simple Moving Average for a price series.
    
    Args:
        series: Price series to use (usually close prices)
        period: SMA period
        
    Returns:
        Series of SMA values
    """
    return series.rolling(window=period).mean()

def calculate_rsi(series, period=14):
    """
    Calculate Relative Strength Index for a price series.
    
    Args:
        series: Price series to use (usually close prices)
        period: RSI period
        
    Returns:
        Series of RSI values
    """
    # Calculate price change
    delta = series.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Handle first values more accurately with EMA
    if len(avg_gain) > period:
        avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(df, period=14):
    """
    Calculate Average True Range for OHLC data.
    
    Args:
        df: DataFrame with OHLC data
        period: ATR period
        
    Returns:
        Series of ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate true range
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = tr.rolling(window=period).mean()
    
    return atr

def calculate_volume_ratio(df, period=20):
    """
    Calculate volume ratio compared to average volume.
    
    Args:
        df: DataFrame with volume data
        period: Period for volume comparison
        
    Returns:
        Series of volume ratio values
    """
    volume = df['volume']
    avg_volume = volume.rolling(window=period).mean()
    
    # Avoid division by zero
    avg_volume = avg_volume.replace(0, np.nan)
    
    # Calculate ratio
    volume_ratio = volume / avg_volume
    
    return volume_ratio

def calculate_bollinger_bands(series, period=20, num_std=2):
    """
    Calculate Bollinger Bands for a price series.
    
    Args:
        series: Price series to use (usually close prices)
        period: Bollinger Bands period
        num_std: Number of standard deviations for bands
        
    Returns:
        DataFrame with middle band, upper band, and lower band
    """
    # Calculate middle band (SMA)
    middle_band = calculate_sma(series, period)
    
    # Calculate standard deviation
    std = series.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Combine into a DataFrame
    bands = pd.DataFrame({
        'middle': middle_band,
        'upper': upper_band,
        'lower': lower_band
    })
    
    return bands

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate MACD (Moving Average Convergence Divergence) for a price series.
    
    Args:
        series: Price series to use (usually close prices)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        DataFrame with MACD line, signal line, and histogram
    """
    # Calculate fast and slow EMAs
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = calculate_ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Combine into a DataFrame
    macd = pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })
    
    return macd

def calculate_stochastic_oscillator(df, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator for OHLC data.
    
    Args:
        df: DataFrame with OHLC data
        k_period: %K period
        d_period: %D period
        
    Returns:
        DataFrame with %K and %D values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    # Avoid division by zero
    range_diff = highest_high - lowest_low
    range_diff = range_diff.replace(0, np.nan)
    
    k = 100 * ((close - lowest_low) / range_diff)
    
    # Calculate %D (SMA of %K)
    d = k.rolling(window=d_period).mean()
    
    # Combine into a DataFrame
    stoch = pd.DataFrame({
        'k': k,
        'd': d
    })
    
    return stoch

def add_rsi(df, period=14):
    """
    Add RSI indicator to a dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        period: RSI period
        
    Returns:
        DataFrame with RSI column added
    """
    df['rsi'] = calculate_rsi(df['close'], period)
    return df

def add_volume_indicators(df, period=20):
    """
    Add volume-based indicators to a dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        period: Volume averaging period
        
    Returns:
        DataFrame with volume indicators added
    """
    df['volume_ratio'] = calculate_volume_ratio(df, period)
    return df

def add_atr(df, period=14):
    """
    Add ATR indicator to a dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR period
        
    Returns:
        DataFrame with ATR column added
    """
    df['atr'] = calculate_atr(df, period)
    return df

def add_ema(df, column='close', periods=[8, 21, 50, 200]):
    """
    Add EMA indicators to a dataframe.
    
    Args:
        df: DataFrame with OHLCV data
        column: Column to calculate EMA for (usually 'close')
        periods: List of EMA periods to calculate
        
    Returns:
        DataFrame with EMA columns added
    """
    for period in periods:
        df[f'ema_{period}'] = calculate_ema(df[column], period)
    return df 