import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

def add_rsi(df, period=14):
    """
    Add RSI to DataFrame (using Wilder's smoothing).
    
    Args:
        df: DataFrame with OHLCV data
        period: RSI calculation period
        
    Returns:
        DataFrame with 'rsi' column added
    """
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        up = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        down = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
        df['rsi'] = 100 * up / (up + down)
    return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """
    Add MACD to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        DataFrame with 'macd', 'macd_signal', and 'macd_hist' columns added
    """
    if 'macd' not in df.columns:
        df['macd'] = df['close'].ewm(span=fast_period).mean() - df['close'].ewm(span=slow_period).mean()
        df['macd_signal'] = df['macd'].ewm(span=signal_period).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
    return df

def add_stochastic(df, k_period=14, d_period=3, slowing=3):
    """
    Add Stochastic Oscillator to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        k_period: %K period
        d_period: %D period
        slowing: %K slowing period
        
    Returns:
        DataFrame with 'stoch_k' and 'stoch_d' columns added
    """
    if 'stoch_k' not in df.columns:
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        
        # Apply slowing period to %K
        df['stoch_k'] = df['stoch_k'].rolling(window=slowing).mean()
        
        # Calculate %D
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
    return df

def add_vwap(df):
    """
    Add Volume Weighted Average Price (VWAP) to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with 'vwap' column added
    """
    if 'vwap' not in df.columns:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    return df

def add_bollinger_bands(df, period=20, std_dev=2):
    """
    Add Bollinger Bands to DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        period: Moving average period
        std_dev: Number of standard deviations
        
    Returns:
        DataFrame with 'bb_upper', 'bb_middle', and 'bb_lower' columns added
    """
    if 'bb_middle' not in df.columns:
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        rolling_std = df['close'].rolling(window=period).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * std_dev)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    return df

def is_rsi_aligned(df, signal_type, threshold=50):
    """
    Check if RSI confirms the trading signal.
    
    Args:
        df: DataFrame with RSI already calculated
        signal_type: 'buy' or 'sell'
        threshold: RSI threshold (default 50)
        
    Returns:
        bool: True if RSI confirms the signal
    """
    if 'rsi' not in df.columns:
        df = add_rsi(df)
        
    current_rsi = df['rsi'].iloc[-1]
    
    if signal_type == 'buy':
        return current_rsi > threshold
    elif signal_type == 'sell':
        return current_rsi < threshold
    return False

def is_macd_aligned(df, signal_type):
    """
    Check if MACD histogram confirms the trading signal.
    
    Args:
        df: DataFrame with MACD already calculated
        signal_type: 'buy' or 'sell'
        
    Returns:
        bool: True if MACD confirms the signal
    """
    if 'macd_hist' not in df.columns:
        df = add_macd(df)
        
    current_hist = df['macd_hist'].iloc[-1]
    prev_hist = df['macd_hist'].iloc[-2]
    
    # Check for histogram crossing above zero (bullish)
    if signal_type == 'buy':
        return current_hist > 0 and prev_hist <= 0
    # Check for histogram crossing below zero (bearish)
    elif signal_type == 'sell':
        return current_hist < 0 and prev_hist >= 0
    return False

def is_stoch_aligned(df, signal_type, overbought=80, oversold=20):
    """
    Check if Stochastic Oscillator confirms the trading signal.
    
    Args:
        df: DataFrame with Stochastic already calculated
        signal_type: 'buy' or 'sell'
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        bool: True if Stochastic confirms the signal
    """
    if 'stoch_k' not in df.columns or 'stoch_d' not in df.columns:
        df = add_stochastic(df)
        
    k = df['stoch_k'].iloc[-1]
    d = df['stoch_d'].iloc[-1]
    prev_k = df['stoch_k'].iloc[-2]
    
    # Buy when K crosses above D in oversold territory
    if signal_type == 'buy':
        return k > d and prev_k <= d and k < oversold
    # Sell when K crosses below D in overbought territory
    elif signal_type == 'sell':
        return k < d and prev_k >= d and k > overbought
    return False

def is_volume_spike(df, lookback=20, factor=1.5):
    """
    Check if the current volume is significantly higher than recent average.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods for volume average
        factor: Multiplier for volume average
        
    Returns:
        bool: True if current volume exceeds average * factor
    """
    volume_ma = df['volume'].rolling(lookback).mean().iloc[-1]
    current_volume = df['volume'].iloc[-1]
    return current_volume > volume_ma * factor

def is_vwap_aligned(df, signal_type):
    """
    Check if price position relative to VWAP confirms the signal.
    
    Args:
        df: DataFrame with VWAP already calculated
        signal_type: 'buy' or 'sell'
        
    Returns:
        bool: True if VWAP position confirms the signal
    """
    if 'vwap' not in df.columns:
        df = add_vwap(df)
        
    current_price = df['close'].iloc[-1]
    current_vwap = df['vwap'].iloc[-1]
    
    if signal_type == 'buy':
        return current_price > current_vwap
    elif signal_type == 'sell':
        return current_price < current_vwap
    return False

def is_atr_favorable(df, period=14, percentile=80):
    """
    Check if current ATR is below the given percentile of historical ATR.
    
    Args:
        df: DataFrame with OHLCV data
        period: ATR calculation period
        percentile: Maximum percentile for acceptable ATR
        
    Returns:
        bool: True if current ATR is below the threshold
    """
    # Calculate True Range
    tr = pd.DataFrame()
    tr['h-l'] = df['high'] - df['low']
    tr['h-pc'] = abs(df['high'] - df['close'].shift(1))
    tr['l-pc'] = abs(df['low'] - df['close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = tr['tr'].rolling(period).mean()
    
    # Calculate the historical percentile
    atr_values = df['atr'].dropna().values
    threshold = np.percentile(atr_values, percentile)
    
    return df['atr'].iloc[-1] < threshold

def is_bb_squeeze(df, lookback=20, squeeze_threshold=0.05):
    """
    Check if there's a Bollinger Band squeeze (narrow bands).
    
    Args:
        df: DataFrame with Bollinger Bands already calculated
        lookback: Number of periods to check for squeeze
        squeeze_threshold: Maximum band width to consider as a squeeze
        
    Returns:
        bool: True if bands are in a squeeze
    """
    if 'bb_width' not in df.columns:
        df = add_bollinger_bands(df)
    
    min_width = df['bb_width'].rolling(lookback).min().iloc[-1]
    current_width = df['bb_width'].iloc[-1]
    
    # Check if we're in a squeeze or just ending one
    return current_width < squeeze_threshold or (current_width / min_width < 1.5 and min_width < squeeze_threshold)

def is_mtf_aligned(self, short_tf_df, long_tf_df, signal_type, fast_col='ema_fast', slow_col='ema_slow'):
    """
    Check if multiple timeframes are aligned for the trade direction.
    
    Args:
        short_tf_df: DataFrame for shorter timeframe
        long_tf_df: DataFrame for longer timeframe
        signal_type: 'buy' or 'sell'
        fast_col: Column name for fast EMA
        slow_col: Column name for slow EMA
        
    Returns:
        bool: True if multiple timeframes are aligned
    """
    # Trend direction in higher timeframe
    ht_bullish = long_tf_df[fast_col].iloc[-1] > long_tf_df[slow_col].iloc[-1]
    
    # Check for alignment
    if signal_type == 'buy':
        return ht_bullish
    elif signal_type == 'sell':
        return not ht_bullish
    return False

def find_pivot_points(df, lookback=20, order=5):
    """
    Find support and resistance pivot points.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to look back
        order: The order parameter for argrelextrema
        
    Returns:
        tuple: (support_levels, resistance_levels)
    """
    # Get the subset of recent data
    recent_data = df.iloc[-lookback:]
    
    # Find local maxima and minima
    local_max_idx = argrelextrema(recent_data['high'].values, np.greater, order=order)[0]
    local_min_idx = argrelextrema(recent_data['low'].values, np.less, order=order)[0]
    
    resistance_levels = recent_data.iloc[local_max_idx]['high'].values
    support_levels = recent_data.iloc[local_min_idx]['low'].values
    
    return support_levels, resistance_levels

def is_near_pivot(df, lookback=20, proximity_pct=1.0):
    """
    Check if the current price is near a support or resistance level.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Number of periods to look back for pivots
        proximity_pct: Percentage distance to consider "near"
        
    Returns:
        bool: True if price is near a pivot point
    """
    current_price = df['close'].iloc[-1]
    support_levels, resistance_levels = find_pivot_points(df, lookback)
    
    # Check if price is within proximity_pct% of any pivot level
    for level in np.concatenate([support_levels, resistance_levels]):
        if abs(current_price - level) / level * 100 < proximity_pct:
            return True
    
    return False

def is_momentum_confirmed(df, signal_type, rsi_threshold=50, check_macd=True, check_stoch=False):
    """
    Comprehensive check for momentum confirmation using multiple oscillators.
    
    Args:
        df: DataFrame with OHLCV data
        signal_type: 'buy' or 'sell'
        rsi_threshold: RSI threshold for trend alignment
        check_macd: Whether to check MACD histogram
        check_stoch: Whether to check Stochastic crossover
        
    Returns:
        bool: True if momentum confirms the signal
    """
    # Ensure indicators are calculated
    df = add_rsi(df)
    confirmations = []
    
    # RSI confirmation
    rsi_confirm = is_rsi_aligned(df, signal_type, rsi_threshold)
    confirmations.append(rsi_confirm)
    
    # MACD confirmation (optional)
    if check_macd:
        df = add_macd(df)
        macd_confirm = is_macd_aligned(df, signal_type)
        confirmations.append(macd_confirm)
    
    # Stochastic confirmation (optional)
    if check_stoch:
        df = add_stochastic(df)
        stoch_confirm = is_stoch_aligned(df, signal_type)
        confirmations.append(stoch_confirm)
    
    # Return True if majority of checked indicators confirm
    return sum(confirmations) >= len(confirmations) / 2 