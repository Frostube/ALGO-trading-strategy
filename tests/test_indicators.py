import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indicators.technical import calculate_ema, calculate_rsi, calculate_volume_ma, get_signal

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing indicators."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=100, freq='1min')
    df = pd.DataFrame({
        'open': [100.0 + i * 0.1 for i in range(100)],
        'high': [101.0 + i * 0.1 for i in range(100)],
        'low': [99.0 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000.0 + i * 10 for i in range(100)]
    }, index=dates)
    return df

def test_ema_calculation(sample_data):
    """Test EMA calculation."""
    # Apply EMA calculation
    df = calculate_ema(sample_data.copy(), fast_period=9, slow_period=21)
    
    # Check if EMA columns exist
    assert 'ema_9' in df.columns
    assert 'ema_21' in df.columns
    assert 'ema_trend' in df.columns
    assert 'ema_crossover' in df.columns
    
    # Check values (using pandas built-in EMA for verification)
    expected_ema9 = sample_data['close'].ewm(span=9, adjust=False).mean()
    expected_ema21 = sample_data['close'].ewm(span=21, adjust=False).mean()
    
    # Allow small floating point differences
    pd.testing.assert_series_equal(
        df['ema_9'].round(4), 
        expected_ema9.round(4), 
        check_names=False
    )
    pd.testing.assert_series_equal(
        df['ema_21'].round(4), 
        expected_ema21.round(4), 
        check_names=False
    )
    
    # Verify trend calculation
    assert all(df.loc[df['ema_9'] > df['ema_21'], 'ema_trend'] == 1)
    assert all(df.loc[df['ema_9'] < df['ema_21'], 'ema_trend'] == -1)

def test_rsi_calculation(sample_data):
    """Test RSI calculation."""
    # Apply RSI calculation
    df = calculate_rsi(sample_data.copy(), period=2)
    
    # Check if RSI column exists
    assert 'rsi' in df.columns
    
    # RSI should be between 0 and 100
    assert df['rsi'].min() >= 0
    assert df['rsi'].max() <= 100
    
    # For our sample data with incrementing prices, RSI should be high
    assert df['rsi'].iloc[-1] > 70

def test_volume_ma_calculation(sample_data):
    """Test volume moving average calculation."""
    # Apply volume MA calculation
    df = calculate_volume_ma(sample_data.copy(), period=20)
    
    # Check if volume MA columns exist
    assert 'volume_ma' in df.columns
    assert 'volume_ratio' in df.columns
    assert 'volume_spike' in df.columns
    
    # Check values
    expected_volume_ma = sample_data['volume'].rolling(window=20).mean()
    pd.testing.assert_series_equal(
        df['volume_ma'].dropna().round(4), 
        expected_volume_ma.dropna().round(4), 
        check_names=False
    )
    
    # Check volume ratio calculation
    assert all(df['volume_ratio'] == df['volume'] / df['volume_ma'])
    
    # Set a test case for volume spike
    df.loc[50, 'volume'] = df.loc[50, 'volume_ma'] * 2  # Create a volume spike
    df = calculate_volume_ma(df, period=20)  # Recalculate
    assert df.loc[50, 'volume_spike'] == True

def test_get_signal(sample_data):
    """Test signal generation."""
    # Prepare data with indicators
    df = sample_data.copy()
    df = calculate_ema(df, fast_period=9, slow_period=21)
    df = calculate_rsi(df, period=2)
    df = calculate_volume_ma(df, period=20)
    
    # Test neutral signal (default case)
    signal = get_signal(df)
    assert signal['signal'] == 'neutral'
    
    # Test buy signal
    # Conditions: ema_trend > 0, rsi < 10, volume_spike = True
    df.loc[50, 'ema_trend'] = 1
    df.loc[50, 'rsi'] = 5
    df.loc[50, 'volume_spike'] = True
    signal = get_signal(df, index=50)
    assert signal['signal'] == 'buy'
    
    # Test sell signal
    # Conditions: ema_trend < 0, rsi > 90, volume_spike = True
    df.loc[60, 'ema_trend'] = -1
    df.loc[60, 'rsi'] = 95
    df.loc[60, 'volume_spike'] = True
    signal = get_signal(df, index=60)
    assert signal['signal'] == 'sell' 