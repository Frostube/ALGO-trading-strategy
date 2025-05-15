import numpy as np
import pandas as pd

def is_doji(candle, threshold=0.1):
    """
    Returns True if the candle is a doji (open ≈ close relative to the full range).
    
    Args:
        candle: Dictionary or DataFrame row with OHLCV data
        threshold: Maximum ratio of body to range to qualify as a doji
        
    Returns:
        bool: True if candle is a doji
    """
    body = abs(candle['close'] - candle['open'])
    range_ = candle['high'] - candle['low']
    if range_ == 0:
        return False
    return (body / range_) <= threshold

def is_hammer(candle, body_to_tail=0.3, body_threshold=0.3):
    """
    Returns True if the candle is a hammer (small body, long lower shadow).
    
    Args:
        candle: Dictionary or DataFrame row with OHLCV data
        body_to_tail: Maximum ratio of body to lower shadow
        body_threshold: Maximum ratio of body to total range
        
    Returns:
        bool: True if candle is a hammer
    """
    is_bullish = candle['close'] > candle['open']
    body = abs(candle['close'] - candle['open'])
    
    # Calculate lower shadow
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    
    # Calculate upper shadow
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    
    # Total range
    total_range = candle['high'] - candle['low']
    
    if total_range == 0:
        return False
        
    # A hammer has:
    # 1. Small upper shadow
    # 2. Long lower shadow
    # 3. Small body relative to total range
    return (lower_shadow >= body / body_to_tail and  # Long lower shadow
            upper_shadow <= lower_shadow * 0.3 and   # Short upper shadow
            body <= total_range * body_threshold)    # Small body

def is_shooting_star(candle, body_to_tail=0.3, body_threshold=0.3):
    """
    Returns True if the candle is a shooting star (small body, long upper shadow).
    
    Args:
        candle: Dictionary or DataFrame row with OHLCV data
        body_to_tail: Maximum ratio of body to upper shadow
        body_threshold: Maximum ratio of body to total range
        
    Returns:
        bool: True if candle is a shooting star
    """
    is_bearish = candle['close'] < candle['open']
    body = abs(candle['close'] - candle['open'])
    
    # Calculate lower shadow
    lower_shadow = min(candle['open'], candle['close']) - candle['low']
    
    # Calculate upper shadow
    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
    
    # Total range
    total_range = candle['high'] - candle['low']
    
    if total_range == 0:
        return False
        
    # A shooting star has:
    # 1. Small lower shadow
    # 2. Long upper shadow
    # 3. Small body relative to total range
    return (upper_shadow >= body / body_to_tail and  # Long upper shadow
            lower_shadow <= upper_shadow * 0.3 and   # Short lower shadow
            body <= total_range * body_threshold)    # Small body

def is_bullish_engulfing(prev_candle, curr_candle):
    """
    Returns True if the current candle is a bullish engulfing pattern.
    
    Args:
        prev_candle: Dictionary or DataFrame row with previous candle OHLCV data
        curr_candle: Dictionary or DataFrame row with current candle OHLCV data
        
    Returns:
        bool: True if pattern is a bullish engulfing
    """
    # Previous candle is bearish (close < open)
    prev_bearish = prev_candle['close'] < prev_candle['open']
    
    # Current candle is bullish (close > open)
    curr_bullish = curr_candle['close'] > curr_candle['open']
    
    # Current candle's body engulfs previous candle's body
    body_engulfing = (curr_candle['open'] <= prev_candle['close'] and 
                      curr_candle['close'] >= prev_candle['open'])
    
    return prev_bearish and curr_bullish and body_engulfing

def is_bearish_engulfing(prev_candle, curr_candle):
    """
    Returns True if the current candle is a bearish engulfing pattern.
    
    Args:
        prev_candle: Dictionary or DataFrame row with previous candle OHLCV data
        curr_candle: Dictionary or DataFrame row with current candle OHLCV data
        
    Returns:
        bool: True if pattern is a bearish engulfing
    """
    # Previous candle is bullish (close > open)
    prev_bullish = prev_candle['close'] > prev_candle['open']
    
    # Current candle is bearish (close < open)
    curr_bearish = curr_candle['close'] < curr_candle['open']
    
    # Current candle's body engulfs previous candle's body
    body_engulfing = (curr_candle['open'] >= prev_candle['close'] and 
                      curr_candle['close'] <= prev_candle['open'])
    
    return prev_bullish and curr_bearish and body_engulfing

def is_bullish_harami(prev_candle, curr_candle):
    """
    Returns True if the current candle is a bullish harami pattern.
    
    Args:
        prev_candle: Dictionary or DataFrame row with previous candle OHLCV data
        curr_candle: Dictionary or DataFrame row with current candle OHLCV data
        
    Returns:
        bool: True if pattern is a bullish harami
    """
    # Previous candle is bearish (close < open)
    prev_bearish = prev_candle['close'] < prev_candle['open']
    
    # Current candle is bullish (close > open)
    curr_bullish = curr_candle['close'] > curr_candle['open']
    
    # Current candle's body is inside previous candle's body
    body_inside = (curr_candle['open'] >= prev_candle['close'] and 
                   curr_candle['close'] <= prev_candle['open'])
    
    return prev_bearish and curr_bullish and body_inside

def is_bearish_harami(prev_candle, curr_candle):
    """
    Returns True if the current candle is a bearish harami pattern.
    
    Args:
        prev_candle: Dictionary or DataFrame row with previous candle OHLCV data
        curr_candle: Dictionary or DataFrame row with current candle OHLCV data
        
    Returns:
        bool: True if pattern is a bearish harami
    """
    # Previous candle is bullish (close > open)
    prev_bullish = prev_candle['close'] > prev_candle['open']
    
    # Current candle is bearish (close < open)
    curr_bearish = curr_candle['close'] < curr_candle['open']
    
    # Current candle's body is inside previous candle's body
    body_inside = (curr_candle['open'] <= prev_candle['close'] and 
                   curr_candle['close'] >= prev_candle['open'])
    
    return prev_bullish and curr_bearish and body_inside

def is_morning_star(candles, doji_threshold=0.1):
    """
    Returns True if the last three candles form a morning star pattern.
    
    Args:
        candles: List of dictionaries or DataFrame rows with OHLCV data for three consecutive candles
        doji_threshold: Threshold for middle candle to be considered a doji
        
    Returns:
        bool: True if pattern is a morning star
    """
    if len(candles) < 3:
        return False
        
    # First candle is bearish with a large body
    first_bearish = candles[-3]['close'] < candles[-3]['open']
    first_body = abs(candles[-3]['close'] - candles[-3]['open'])
    first_range = candles[-3]['high'] - candles[-3]['low']
    first_large_body = first_body > first_range * 0.6
    
    # Second candle is a small body or doji
    second_body = abs(candles[-2]['close'] - candles[-2]['open'])
    second_range = candles[-2]['high'] - candles[-2]['low']
    second_small_body = second_range == 0 or (second_body / second_range) < doji_threshold
    
    # Gap down between first and second candles
    gap_down = max(candles[-2]['open'], candles[-2]['close']) < candles[-3]['close']
    
    # Third candle is bullish with a large body
    third_bullish = candles[-1]['close'] > candles[-1]['open']
    third_body = abs(candles[-1]['close'] - candles[-1]['open'])
    third_range = candles[-1]['high'] - candles[-1]['low']
    third_large_body = third_body > third_range * 0.6
    
    # Third candle closes well into the first candle's body
    third_closes_into_first = candles[-1]['close'] > (candles[-3]['open'] + candles[-3]['close']) / 2
    
    return (first_bearish and second_small_body and third_bullish and 
            gap_down and third_large_body and third_closes_into_first)

def is_evening_star(candles, doji_threshold=0.1):
    """
    Returns True if the last three candles form an evening star pattern.
    
    Args:
        candles: List of dictionaries or DataFrame rows with OHLCV data for three consecutive candles
        doji_threshold: Threshold for middle candle to be considered a doji
        
    Returns:
        bool: True if pattern is an evening star
    """
    if len(candles) < 3:
        return False
        
    # First candle is bullish with a large body
    first_bullish = candles[-3]['close'] > candles[-3]['open']
    first_body = abs(candles[-3]['close'] - candles[-3]['open'])
    first_range = candles[-3]['high'] - candles[-3]['low']
    first_large_body = first_body > first_range * 0.6
    
    # Second candle is a small body or doji
    second_body = abs(candles[-2]['close'] - candles[-2]['open'])
    second_range = candles[-2]['high'] - candles[-2]['low']
    second_small_body = second_range == 0 or (second_body / second_range) < doji_threshold
    
    # Gap up between first and second candles
    gap_up = min(candles[-2]['open'], candles[-2]['close']) > candles[-3]['close']
    
    # Third candle is bearish with a large body
    third_bearish = candles[-1]['close'] < candles[-1]['open']
    third_body = abs(candles[-1]['close'] - candles[-1]['open'])
    third_range = candles[-1]['high'] - candles[-1]['low']
    third_large_body = third_body > third_range * 0.6
    
    # Third candle closes well into the first candle's body
    third_closes_into_first = candles[-1]['close'] < (candles[-3]['open'] + candles[-3]['close']) / 2
    
    return (first_bullish and second_small_body and third_bearish and 
            gap_up and third_large_body and third_closes_into_first)

def is_bullish_pattern(candles):
    """
    Check if the last candles form any bullish pattern.
    
    Args:
        candles: List of dictionaries or DataFrame rows with OHLCV data
        
    Returns:
        bool: True if any bullish pattern is detected
    """
    if len(candles) < 3:
        return False
        
    # Check for single-candle bullish patterns
    hammer = is_hammer(candles[-1])
    
    # Check for two-candle bullish patterns
    bull_engulfing = is_bullish_engulfing(candles[-2], candles[-1])
    bull_harami = is_bullish_harami(candles[-2], candles[-1])
    
    # Check for three-candle bullish patterns
    morning_star = is_morning_star(candles)
    
    return hammer or bull_engulfing or bull_harami or morning_star

def is_bearish_pattern(candles):
    """
    Check if the last candles form any bearish pattern.
    
    Args:
        candles: List of dictionaries or DataFrame rows with OHLCV data
        
    Returns:
        bool: True if any bearish pattern is detected
    """
    if len(candles) < 3:
        return False
        
    # Check for single-candle bearish patterns
    shooting_star = is_shooting_star(candles[-1])
    
    # Check for two-candle bearish patterns
    bear_engulfing = is_bearish_engulfing(candles[-2], candles[-1])
    bear_harami = is_bearish_harami(candles[-2], candles[-1])
    
    # Check for three-candle bearish patterns
    evening_star = is_evening_star(candles)
    
    return shooting_star or bear_engulfing or bear_harami or evening_star 