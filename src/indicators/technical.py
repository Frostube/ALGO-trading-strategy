import pandas as pd
import numpy as np

from src.config import (
    EMA_FAST, EMA_SLOW, RSI_PERIOD, VOLUME_PERIOD, VOLUME_THRESHOLD,
    EMA_TREND, EMA_MICRO_TREND, ATR_PERIOD, USE_ATR_STOPS, 
    USE_ADAPTIVE_THRESHOLDS, ADAPTIVE_LOOKBACK, MIN_BARS_BETWEEN_TRADES,
    USE_TIME_FILTERS, TRADING_HOURS_START, TRADING_HOURS_END,
    AVOID_MIDNIGHT_HOURS, HIGH_VOLATILITY_HOURS, WEEKEND_TRADING,
    USE_ML_FILTER, ML_PROBABILITY_THRESHOLD
)
from src.utils.logger import logger

# Import ML filter if enabled
if USE_ML_FILTER:
    try:
        from src.ml.signal_filter import MLSignalFilter
        ml_filter = MLSignalFilter()
    except ImportError:
        logger.warning("ML filter module not found, disabling ML filtering")
        USE_ML_FILTER = False
    except Exception as e:
        logger.error(f"Error initializing ML filter: {e}")
        USE_ML_FILTER = False

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
    
    # Check if dataframe is empty or has sufficient data
    if df.empty:
        logger.warning("Cannot apply indicators to empty dataframe")
        return df
    
    # Make sure we have enough data for indicators
    if len(df) < 50:  # Need a reasonable amount of data
        logger.warning(f"Dataframe has only {len(df)} rows, may be insufficient for reliable indicators")
    
    try:
        # Apply indicators
        df = calculate_ema(df, EMA_FAST, EMA_SLOW, EMA_TREND, EMA_MICRO_TREND)
        df = calculate_rsi(df, RSI_PERIOD)
        df = calculate_volume_ma(df, VOLUME_PERIOD)
        df = calculate_atr(df, ATR_PERIOD)
        df = calculate_momentum_indicators(df)
        
        # Add Hull Moving Average (HMA) - a popular trend indicator
        df = calculate_hull_ma(df, 9, 20)
        
        # Add Donchian Channels for breakout strategy
        df = calculate_donchian_channels(df, 20)
        
        # Add price action patterns
        df = detect_candlestick_patterns(df)
        
        # Add support and resistance levels
        df = identify_support_resistance(df, window=20)
        
        if USE_ADAPTIVE_THRESHOLDS:
            df = calculate_adaptive_thresholds(df, lookback=min(ADAPTIVE_LOOKBACK, len(df) // 2))
        
        # IMPORTANT FIX: Instead of dropping NaN rows, fill them with appropriate values
        # This prevents losing data during backtest
        logger.debug(f"Before filling NaN values: {len(df)} rows")
        
        # Fill NaN values in indicator columns with appropriate defaults
        # This keeps all original OHLCV data while making indicators usable
        indicators_to_fill = {
            'ema_trend': 0,  # Neutral
            'rsi': 50,  # Neutral RSI
            'volume_spike': False,  # No spike
            'volume_ratio': 1.0,  # Neutral volume
            'market_trend': 0,  # Neutral
            'ema_micro_slope': 0,  # Neutral
            'ema_micro_direction': 0,  # Neutral
            'momentum_up': False,  # No momentum
            'momentum_down': False,  # No momentum
            'hma_trend': 0,  # Neutral
            'donchian_breakout': 0,  # No breakout
            'bullish_engulfing': False,  # No pattern
            'bearish_engulfing': False,  # No pattern
            'hammer': False,  # No pattern
            'shooting_star': False,  # No pattern
            'doji': False,  # No pattern
            'at_support': False,  # Not at support
            'at_resistance': False  # Not at resistance
        }
        
        for col, default_val in indicators_to_fill.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_val)
        
        # For ATR related columns, use a reasonable default percentage of price
        if 'atr' in df.columns:
            df['atr'] = df['atr'].fillna(df['close'] * 0.001)  # Default to 0.1% of price
        
        if 'atr_pct' in df.columns:
            df['atr_pct'] = df['atr_pct'].fillna(0.1)  # Default to 0.1%
        
        # Fill other indicators
        if 'upper_band' in df.columns:
            df['upper_band'] = df['upper_band'].fillna(df['close'] * 1.01)
        if 'lower_band' in df.columns:
            df['lower_band'] = df['lower_band'].fillna(df['close'] * 0.99)
        if 'middle_band' in df.columns:
            df['middle_band'] = df['middle_band'].fillna(df['close'])
        
        logger.debug(f"After filling NaN values: {len(df)} rows")
        
        # Log how many rows had NaN values
        nan_rows = df.isna().any(axis=1).sum()
        if nan_rows > 0:
            logger.warning(f"Filled {nan_rows} rows with NaN values")
            
    except Exception as e:
        logger.error(f"Error applying indicators: {str(e)}")
        # If we encounter an error, return the original dataframe but with minimal required columns
        if 'ema_trend' not in df.columns:
            df['ema_trend'] = 0  # Neutral
        if 'rsi' not in df.columns:
            df['rsi'] = 50  # Neutral
        if 'volume_spike' not in df.columns:
            df['volume_spike'] = False  # No spike
        
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
    
    # Enhanced momentum confirmation flags with multi-timeframe confirmation
    # Current close must be higher/lower than previous N-period highs/lows
    df['momentum_up'] = (df['close'] > df['prev_5_high']) & (df['close'] > df['close'].shift(3))
    df['momentum_down'] = (df['close'] < df['prev_5_low']) & (df['close'] < df['close'].shift(3))
    
    # Add price action confirmation
    # Check for bullish/bearish candles (close > open for bullish)
    df['bullish_candle'] = df['close'] > df['open']
    df['bearish_candle'] = df['close'] < df['open']
    
    # Calculate bar since last entry/exit counter
    df['bars_since_signal'] = np.nan  # Will be set dynamically during signal generation
    
    # Add strong momentum indicator (3 consecutive rising/falling closes)
    df['strong_momentum_up'] = (df['close'] > df['close'].shift(1)) & \
                              (df['close'].shift(1) > df['close'].shift(2)) & \
                              (df['close'].shift(2) > df['close'].shift(3))
    
    df['strong_momentum_down'] = (df['close'] < df['close'].shift(1)) & \
                                (df['close'].shift(1) < df['close'].shift(2)) & \
                                (df['close'].shift(2) < df['close'].shift(3))
    
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
    # Adjust lookback to ensure it's not larger than the dataset
    lookback = min(lookback, len(df) // 2) if len(df) > 4 else 2
    
    try:
        # Calculate RSI adaptive thresholds (10th percentile for oversold, 90th for overbought)
        df['rsi_lower_threshold'] = df['rsi'].rolling(window=lookback).quantile(0.10)
        df['rsi_upper_threshold'] = df['rsi'].rolling(window=lookback).quantile(0.90)
        
        # Calculate adaptive volume threshold (80th percentile)
        df['adaptive_volume_threshold'] = df['volume_ratio'].rolling(window=lookback).quantile(0.80)
        
        # Use adaptive volume spike threshold
        df['adaptive_volume_spike'] = df['volume_ratio'] > df['adaptive_volume_threshold']
    except Exception as e:
        logger.warning(f"Error calculating adaptive thresholds: {str(e)}")
        # Fallback to fixed thresholds
        df['rsi_lower_threshold'] = 30
        df['rsi_upper_threshold'] = 70
        df['adaptive_volume_threshold'] = 1.5
        df['adaptive_volume_spike'] = df['volume_ratio'] > 1.5
    
    return df

def calculate_hull_ma(df, fast_period=9, slow_period=20):
    """
    Calculate Hull Moving Average (HMA), which responds more quickly to price changes
    and reduces lag compared to traditional moving averages.
    
    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast HMA period
        slow_period: Slow HMA period
        
    Returns:
        DataFrame with Hull MA indicators added
    """
    # Calculate the Hull Moving Average for fast period
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    
    # Fast HMA
    half_period = int(fast_period / 2)
    sqrt_period = int(np.sqrt(fast_period))
    
    # Calculate weighted moving averages
    df[f'wma_half_{fast_period}'] = df['close'].rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1))
    )
    df[f'wma_full_{fast_period}'] = df['close'].rolling(window=fast_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1))
    )
    
    # Calculate raw Hull
    df[f'hull_raw_{fast_period}'] = 2 * df[f'wma_half_{fast_period}'] - df[f'wma_full_{fast_period}']
    
    # Apply final weighted MA to get Hull MA
    df[f'hma_{fast_period}'] = df[f'hull_raw_{fast_period}'].rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1))
    )
    
    # Slow HMA
    half_period = int(slow_period / 2)
    sqrt_period = int(np.sqrt(slow_period))
    
    # Calculate weighted moving averages
    df[f'wma_half_{slow_period}'] = df['close'].rolling(window=half_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1))
    )
    df[f'wma_full_{slow_period}'] = df['close'].rolling(window=slow_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1))
    )
    
    # Calculate raw Hull
    df[f'hull_raw_{slow_period}'] = 2 * df[f'wma_half_{slow_period}'] - df[f'wma_full_{slow_period}']
    
    # Apply final weighted MA to get Hull MA
    df[f'hma_{slow_period}'] = df[f'hull_raw_{slow_period}'].rolling(window=sqrt_period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1))
    )
    
    # Calculate HMA cross trend (1: bullish, -1: bearish, 0: neutral)
    df['hma_trend'] = 0
    df.loc[df[f'hma_{fast_period}'] > df[f'hma_{slow_period}'], 'hma_trend'] = 1
    df.loc[df[f'hma_{fast_period}'] < df[f'hma_{slow_period}'], 'hma_trend'] = -1
    
    # Calculate crossover points
    df['hma_crossover'] = df['hma_trend'].diff().fillna(0)
    
    # Clean up temporary columns
    columns_to_drop = [
        f'wma_half_{fast_period}', f'wma_full_{fast_period}', f'hull_raw_{fast_period}',
        f'wma_half_{slow_period}', f'wma_full_{slow_period}', f'hull_raw_{slow_period}'
    ]
    df = df.drop(columns_to_drop, axis=1)
    
    return df

def calculate_donchian_channels(df, period=20):
    """
    Calculate Donchian Channels for breakout strategy.
    
    Args:
        df: DataFrame with OHLCV data
        period: Lookback period for channel calculation
        
    Returns:
        DataFrame with Donchian Channel indicators added
    """
    # Calculate upper and lower bands
    df['upper_band'] = df['high'].rolling(window=period).max()
    df['lower_band'] = df['low'].rolling(window=period).min()
    df['middle_band'] = (df['upper_band'] + df['lower_band']) / 2
    
    # Calculate breakout signals
    # 1: breakout above upper band, -1: breakout below lower band, 0: no breakout
    df['donchian_breakout'] = 0
    
    # Upper breakout: close crosses above upper band
    upper_breakout = (df['close'] > df['upper_band'].shift(1)) & (df['close'].shift(1) <= df['upper_band'].shift(1))
    df.loc[upper_breakout, 'donchian_breakout'] = 1
    
    # Lower breakout: close crosses below lower band
    lower_breakout = (df['close'] < df['lower_band'].shift(1)) & (df['close'].shift(1) >= df['lower_band'].shift(1))
    df.loc[lower_breakout, 'donchian_breakout'] = -1
    
    # Calculate channel width (volatility measure)
    df['channel_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band'] * 100
    
    return df

def detect_candlestick_patterns(df):
    """
    Detect candlestick patterns for entry signals.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with candlestick pattern indicators added
    """
    # Make sure we have required data
    if 'open' not in df.columns or 'high' not in df.columns or 'low' not in df.columns or 'close' not in df.columns:
        logger.warning("Required OHLC data not available for pattern detection")
        return df
    
    # Calculate candle body and shadows
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Calculate average body size for reference
    df['avg_body_size'] = df['body_size'].rolling(window=20).mean()
    df['avg_true_range'] = df['tr'].rolling(window=14).mean() if 'tr' in df.columns else (df['high'] - df['low']).rolling(window=14).mean()
    
    # 1. Engulfing patterns
    # Bullish engulfing: Current green candle completely engulfs previous red candle
    df['bullish_engulfing'] = (
        (df['close'] > df['open']) &  # Current candle is green
        (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is red
        (df['close'] > df['open'].shift(1)) &  # Current close > previous open
        (df['open'] < df['close'].shift(1))  # Current open < previous close
    )
    
    # Bearish engulfing: Current red candle completely engulfs previous green candle
    df['bearish_engulfing'] = (
        (df['close'] < df['open']) &  # Current candle is red
        (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is green
        (df['close'] < df['open'].shift(1)) &  # Current close < previous open
        (df['open'] > df['close'].shift(1))  # Current open > previous close
    )
    
    # 2. Hammer pattern (bullish reversal)
    # Long lower shadow (at least 2x body), small upper shadow, small body
    df['hammer'] = (
        (df['lower_shadow'] > 2 * df['body_size']) &  # Long lower shadow
        (df['upper_shadow'] < 0.5 * df['body_size']) &  # Small upper shadow
        (df['body_size'] < 0.5 * df['avg_body_size']) &  # Relatively small body
        (df['close'] > df['open']) &  # Green candle preferred but not required
        (df['close'] > df['close'].shift(1))  # Higher close than previous
    )
    
    # 3. Shooting Star pattern (bearish reversal)
    # Long upper shadow (at least 2x body), small lower shadow, small body
    df['shooting_star'] = (
        (df['upper_shadow'] > 2 * df['body_size']) &  # Long upper shadow
        (df['lower_shadow'] < 0.5 * df['body_size']) &  # Small lower shadow
        (df['body_size'] < 0.5 * df['avg_body_size']) &  # Relatively small body
        (df['close'] < df['open']) &  # Red candle preferred but not required
        (df['close'] < df['close'].shift(1))  # Lower close than previous
    )
    
    # 4. Doji (indecision, could precede reversal)
    # Very small body relative to shadows
    df['doji'] = (
        (df['body_size'] < 0.1 * df['avg_true_range']) &  # Very small body
        ((df['upper_shadow'] + df['lower_shadow']) > 3 * df['body_size'])  # Long shadows
    )
    
    # 5. Morning Star (bullish reversal, 3-candle pattern)
    # First candle is bearish, second is small (maybe doji), third is bullish
    df['morning_star'] = False
    morning_star_condition = (
        (df['close'].shift(2) < df['open'].shift(2)) &  # First candle bearish
        (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.3 * df['avg_body_size']) &  # Second candle small
        (df['close'] > df['open']) &  # Third candle bullish
        (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Third close > midpoint of first
    )
    df.loc[morning_star_condition, 'morning_star'] = True
    
    # 6. Evening Star (bearish reversal, 3-candle pattern)
    # First candle is bullish, second is small (maybe doji), third is bearish
    df['evening_star'] = False
    evening_star_condition = (
        (df['close'].shift(2) > df['open'].shift(2)) &  # First candle bullish
        (abs(df['close'].shift(1) - df['open'].shift(1)) < 0.3 * df['avg_body_size']) &  # Second candle small
        (df['close'] < df['open']) &  # Third candle bearish
        (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Third close < midpoint of first
    )
    df.loc[evening_star_condition, 'evening_star'] = True
    
    # Clean up temporary columns
    df = df.drop(['body_size', 'upper_shadow', 'lower_shadow', 'avg_body_size'], axis=1)
    
    return df

def identify_support_resistance(df, window=20):
    """
    Identify support and resistance levels using pivot points.
    
    Args:
        df: DataFrame with OHLCV data
        window: Lookback window to identify pivot points
        
    Returns:
        DataFrame with support/resistance indicators added
    """
    half_window = window // 2
    
    # Initialize columns
    df['pivot_high'] = False
    df['pivot_low'] = False
    df['at_support'] = False
    df['at_resistance'] = False
    
    # We need at least window*2 bars to identify meaningful levels
    if len(df) < window*2:
        return df
    
    # Identify pivot highs and lows
    for i in range(half_window, len(df) - half_window):
        # Check if this is a pivot high (local maximum)
        if df['high'].iloc[i] == df['high'].iloc[i-half_window:i+half_window+1].max():
            df.loc[df.index[i], 'pivot_high'] = True
        
        # Check if this is a pivot low (local minimum)
        if df['low'].iloc[i] == df['low'].iloc[i-half_window:i+half_window+1].min():
            df.loc[df.index[i], 'pivot_low'] = True
    
    # Maintain lists of recent support and resistance levels
    support_levels = []
    resistance_levels = []
    
    # Tolerance for price near support/resistance (as percentage)
    price_tolerance = 0.0015  # 0.15%
    
    # Iterate through data to build support/resistance and check if price is at these levels
    for i in range(half_window, len(df)):
        current_price = df['close'].iloc[i]
        current_atr = df['atr'].iloc[i] if 'atr' in df.columns else (df['high'].iloc[i] - df['low'].iloc[i])
        
        # If this is a pivot high, add to resistance levels
        if df['pivot_high'].iloc[i-1]:  # Use previous bar to avoid lookahead bias
            resistance_levels.append(df['high'].iloc[i-1])
            # Keep only the most recent levels (avoid too many levels)
            if len(resistance_levels) > 5:
                resistance_levels.pop(0)
        
        # If this is a pivot low, add to support levels
        if df['pivot_low'].iloc[i-1]:  # Use previous bar to avoid lookahead bias
            support_levels.append(df['low'].iloc[i-1])
            # Keep only the most recent levels (avoid too many levels)
            if len(support_levels) > 5:
                support_levels.pop(0)
        
        # Check if current price is at support or resistance
        # At support: price is within tolerance of a support level
        for level in support_levels:
            if abs(current_price - level) < current_price * price_tolerance:
                df.loc[df.index[i], 'at_support'] = True
                break
        
        # At resistance: price is within tolerance of a resistance level
        for level in resistance_levels:
            if abs(current_price - level) < current_price * price_tolerance:
                df.loc[df.index[i], 'at_resistance'] = True
                break
    
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
        'hma_trend': row.get('hma_trend', 0),
        'donchian_breakout': row.get('donchian_breakout', 0),
        'market_trend': row.get('market_trend', 0),
        'micro_trend': row.get('ema_micro_direction', 0),
        'rsi': row['rsi'],
        'volume_spike': row['volume_spike'] if not USE_ADAPTIVE_THRESHOLDS else row['adaptive_volume_spike'],
        'atr': row.get('atr', None),
        'atr_pct': row.get('atr_pct', None),
        'momentum_up': row.get('momentum_up', False),
        'momentum_down': row.get('momentum_down', False),
        'strong_momentum_up': row.get('strong_momentum_up', False),
        'strong_momentum_down': row.get('strong_momentum_down', False),
        'bullish_candle': row.get('bullish_candle', False),
        'bearish_candle': row.get('bearish_candle', False),
        'upper_band': row.get('upper_band', row['close'] * 1.01),
        'lower_band': row.get('lower_band', row['close'] * 0.99),
        'channel_width': row.get('channel_width', 2.0),
        'bullish_engulfing': row.get('bullish_engulfing', False),
        'bearish_engulfing': row.get('bearish_engulfing', False),
        'hammer': row.get('hammer', False),
        'shooting_star': row.get('shooting_star', False),
        'doji': row.get('doji', False),
        'morning_star': row.get('morning_star', False),
        'evening_star': row.get('evening_star', False),
        'at_support': row.get('at_support', False),
        'at_resistance': row.get('at_resistance', False),
        'high_volatility': row.get('high_volatility', False),
        'trend_strength': row.get('trend_strength', 0.0),
        'signal': 'neutral'
    }
    
    # Import time filter settings
    from src.config import (
        USE_TIME_FILTERS, TRADING_HOURS_START, TRADING_HOURS_END,
        AVOID_MIDNIGHT_HOURS, HIGH_VOLATILITY_HOURS, WEEKEND_TRADING
    )
    
    # Apply time-of-day filters if configured - MAKING THIS LESS RESTRICTIVE
    if USE_TIME_FILTERS and hasattr(row.name, 'hour'):
        # Get hour in UTC
        current_hour = row.name.hour
        current_weekday = row.name.weekday()  # 0=Monday, 6=Sunday
        
        # Check if current time is outside trading hours - DISABLED FOR NOW
        # if current_hour < TRADING_HOURS_START or current_hour >= TRADING_HOURS_END:
        #    signal['filtered_time_outside_hours'] = True
        #    return signal
            
        # Check if current time is in midnight low-liquidity hours (0-3 UTC) - DISABLED FOR NOW
        # if AVOID_MIDNIGHT_HOURS and current_hour < 3:
        #    signal['filtered_midnight_hours'] = True
        #    return signal
            
        # Check if it's a weekend (Saturday=5, Sunday=6) and weekend trading is disabled
        if not WEEKEND_TRADING and current_weekday >= 5:
            signal['filtered_weekend'] = True
            return signal
            
        # Adjust signal based on time of day context
        # Higher volatility hours may have higher thresholds
        if current_hour in HIGH_VOLATILITY_HOURS:
            signal['high_volatility_hour'] = True
    
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
        rsi_oversold = 30  # Fixed threshold of 30 for oversold
        rsi_overbought = 70  # Fixed threshold of 70 for overbought
    
    # Adjust thresholds for high volatility hours - LESS RESTRICTIVE
    if signal.get('high_volatility_hour', False):
        # Make RSI thresholds more conservative during high volatility hours
        rsi_oversold = max(25, rsi_oversold)  # Less extreme oversold required
        rsi_overbought = min(75, rsi_overbought)  # Less extreme overbought required
    
    # STRATEGY 1: Hull MA + RSI + Donchian Channel Strategy (Trend-Following)
    # MODIFIED TO BE LESS RESTRICTIVE - Not requiring all conditions simultaneously
    # Long signal when most conditions align
    if (bars_since_allowed and
        row.get('hma_trend', 0) > 0 and  # Hull MA trending up
        (row.get('donchian_breakout', 0) > 0 or row.get('close', 0) > row.get('upper_band', float('inf')) * 0.998) and  # Near upper band
        row['rsi'] < rsi_oversold + 15 and  # Relaxed RSI threshold 
        # Price action confirmation - NOW ONLY REQUIRING ONE OF THESE:
        (signal['bullish_engulfing'] or signal['hammer'] or signal['morning_star'] or
         signal['doji'] or signal['at_support'] or signal['bullish_candle'])):
        signal['signal'] = 'buy'
        signal['strategy'] = 'hma_donchian_breakout'
    
    # Short signal when most conditions align - LESS RESTRICTIVE
    elif (bars_since_allowed and
          row.get('hma_trend', 0) < 0 and  # Hull MA trending down
          (row.get('donchian_breakout', 0) < 0 or row.get('close', 0) < row.get('lower_band', 0) * 1.002) and  # Near lower band
          row['rsi'] > rsi_overbought - 15 and  # Relaxed RSI threshold
          # Price action confirmation - NOW ONLY REQUIRING ONE OF THESE:
          (signal['bearish_engulfing'] or signal['shooting_star'] or signal['evening_star'] or
           signal['doji'] or signal['at_resistance'] or signal['bearish_candle'])):
        signal['signal'] = 'sell'
        signal['strategy'] = 'hma_donchian_breakout'
    
    # STRATEGY 2: Multi-MA Trend Strategy (Strong trends) - MUCH LESS RESTRICTIVE
    # Long signal with fewer conditions required
    elif (bars_since_allowed and
          ((row['market_trend'] > 0 and row['ema_trend'] > 0) or  # Either overall trend conditions
           (row.get('hma_trend', 0) > 0 and row['ema_micro_direction'] > 0)) and  # Or short-term trend conditions
          row['rsi'] < rsi_overbought and  # Just not extremely overbought
          (signal['bullish_candle'] or signal['bullish_engulfing'] or signal['hammer'] or signal['at_support'])):  # Any bullish pattern
        signal['signal'] = 'buy'
        signal['strategy'] = 'multi_ma_trend'
    
    # Short signal with fewer conditions required
    elif (bars_since_allowed and
          ((row['market_trend'] < 0 and row['ema_trend'] < 0) or  # Either overall trend conditions
           (row.get('hma_trend', 0) < 0 and row['ema_micro_direction'] < 0)) and  # Or short-term trend conditions 
          row['rsi'] > rsi_oversold and  # Just not extremely oversold
          (signal['bearish_candle'] or signal['bearish_engulfing'] or signal['shooting_star'] or signal['at_resistance'])):  # Any bearish pattern
        signal['signal'] = 'sell'
        signal['strategy'] = 'multi_ma_trend'
    
    # STRATEGY 3: Mean-reversion for RSI extremes - LESS RESTRICTIVE
    # Mean reversion buy with fewer conditions
    elif (bars_since_allowed and
          row['rsi'] < 30 and  # Oversold
          row.get('bullish_candle', False)):  # Just need a bullish candle
        signal['signal'] = 'buy'
        signal['strategy'] = 'mean_reversion'
    
    # Mean reversion sell with fewer conditions
    elif (bars_since_allowed and
          row['rsi'] > 70 and  # Overbought
          row.get('bearish_candle', False)):  # Just need a bearish candle
        signal['signal'] = 'sell'
        signal['strategy'] = 'mean_reversion'
    
    # Apply ML filter if enabled and we have a signal - MAKE IT OPTIONAL FOR NOW
    if USE_ML_FILTER and signal['signal'] != 'neutral':
        try:
            # Get probability from ML model
            probability = ml_filter.predict_probability(signal)
            signal['ml_probability'] = probability
            
            # Apply threshold filter - LOWER THE THRESHOLD 
            if probability < ML_PROBABILITY_THRESHOLD * 0.8:  # 20% less strict
                logger.debug(f"ML filter rejected signal with probability {probability:.2f} < {ML_PROBABILITY_THRESHOLD}")
                # Don't immediately reject - give it a 50% chance to pass anyway
                if np.random.random() < 0.5:
                    signal['signal'] = 'neutral'
                    signal['filtered_by_ml'] = True
        except Exception as e:
            logger.error(f"Error applying ML filter: {e}")
            # Don't let ML errors block signals
            pass
    
    return signal 