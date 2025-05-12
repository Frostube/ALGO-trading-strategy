import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.scalping import ScalpingStrategy
from src.db.models import Trade

@pytest.fixture
def sample_data():
    """Create sample OHLCV data with indicators for testing strategy."""
    dates = pd.date_range(start=datetime.now() - timedelta(days=1), periods=100, freq='1min')
    df = pd.DataFrame({
        'open': [100.0 + i * 0.1 for i in range(100)],
        'high': [101.0 + i * 0.1 for i in range(100)],
        'low': [99.0 + i * 0.1 for i in range(100)],
        'close': [100.5 + i * 0.1 for i in range(100)],
        'volume': [1000.0 + i * 10 for i in range(100)],
        # Pre-calculated indicators
        'ema_9': [100.5 + i * 0.12 for i in range(100)],
        'ema_21': [100.5 + i * 0.1 for i in range(100)],
        'ema_trend': [1 for _ in range(100)],
        'ema_crossover': [0 for _ in range(100)],
        'rsi': [50.0 for _ in range(100)],
        'volume_ma': [950.0 + i * 10 for i in range(100)],
        'volume_ratio': [1.05 for _ in range(100)],
        'volume_spike': [False for _ in range(100)]
    }, index=dates)
    return df

@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    mock_session = MagicMock()
    mock_session.add = MagicMock()
    mock_session.commit = MagicMock()
    mock_session.query.return_value.filter_by.return_value.first.return_value = None
    return mock_session

def test_strategy_initialization(mock_db_session):
    """Test strategy initialization."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    assert strategy.db_session == mock_db_session
    assert strategy.symbol == "BTC/USDT"
    assert strategy.account_balance == 1000.0
    assert strategy.active_trade is None
    assert strategy.consecutive_sl_count == 0

def test_update_no_active_trade_no_signal(sample_data, mock_db_session):
    """Test strategy update with no active trade and no signal."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    # Create a neutral signal
    sample_data.loc[:, 'rsi'] = 50  # Not oversold or overbought
    
    result = strategy.update(sample_data)
    
    assert result['signal']['signal'] == 'neutral'
    assert result['active_trade'] is None
    assert result['account_balance'] == 1000.0

def test_update_with_buy_signal(sample_data, mock_db_session):
    """Test strategy update with buy signal."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    # Create a buy signal: RSI < 10, EMA trend > 0, volume spike
    sample_data.loc[50, 'rsi'] = 5
    sample_data.loc[50, 'ema_trend'] = 1
    sample_data.loc[50, 'volume_spike'] = True
    
    # Patch the _open_trade method to avoid actual trade opening
    with patch.object(strategy, '_open_trade') as mock_open_trade:
        result = strategy.update(sample_data)
        
        # Check if _open_trade was called with the right parameters
        mock_open_trade.assert_called_once()
        # Check the signal
        assert result['signal']['signal'] == 'buy'

def test_update_with_sell_signal(sample_data, mock_db_session):
    """Test strategy update with sell signal."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    # Create a sell signal: RSI > 90, EMA trend < 0, volume spike
    sample_data.loc[50, 'rsi'] = 95
    sample_data.loc[50, 'ema_trend'] = -1
    sample_data.loc[50, 'volume_spike'] = True
    
    # Patch the _open_trade method to avoid actual trade opening
    with patch.object(strategy, '_open_trade') as mock_open_trade:
        result = strategy.update(sample_data)
        
        # Check if _open_trade was called with the right parameters
        mock_open_trade.assert_called_once()
        # Check the signal
        assert result['signal']['signal'] == 'sell'

def test_open_trade(sample_data, mock_db_session):
    """Test opening a trade."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    # Create a signal
    signal = {'signal': 'buy'}
    current_price = 100.0
    
    # Create a Trade mock
    mock_trade = MagicMock(spec=Trade)
    mock_trade.id = 1
    mock_db_session.add.return_value = None
    
    # When the session commits, update the mock trade ID
    def side_effect():
        # Set the ID property of the Trade object that was added to the session
        added_trade = mock_db_session.add.call_args[0][0]
        added_trade.id = 1
    
    mock_db_session.commit.side_effect = side_effect
    
    # Call the method
    strategy._open_trade(signal, current_price)
    
    # Check the active trade
    assert strategy.active_trade is not None
    assert strategy.active_trade['side'] == 'buy'
    assert strategy.active_trade['entry_price'] == current_price
    assert strategy.active_trade['id'] == 1
    assert 'stop_loss' in strategy.active_trade
    assert 'take_profit' in strategy.active_trade
    
    # Verify the database was updated
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()

def test_close_trade(mock_db_session):
    """Test closing a trade."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    # Set up an active trade
    strategy.active_trade = {
        'id': 1,
        'symbol': 'BTC/USDT',
        'side': 'buy',
        'entry_time': datetime.now(),
        'entry_price': 100.0,
        'amount': 1.0,
        'stop_loss': 99.0,
        'take_profit': 101.0,
        'notional_value': 100.0
    }
    
    # Set up the mock query to return a trade object
    mock_trade = MagicMock(spec=Trade)
    mock_trade.id = 1
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_trade
    
    # Close the trade with profit
    current_price = 101.0
    reason = 'take_profit'
    
    closed_trade = strategy._close_trade(current_price, reason)
    
    # Check the results
    assert strategy.active_trade is None  # Trade should be cleared
    assert closed_trade['exit_price'] == current_price
    assert closed_trade['exit_reason'] == reason
    assert closed_trade['pnl'] > 0  # Should be profitable
    assert strategy.account_balance > 1000.0  # Balance should increase
    
    # Verify database was updated
    mock_db_session.query.assert_called_once()
    mock_db_session.commit.assert_called_once()

def test_get_performance_summary(mock_db_session):
    """Test getting performance summary."""
    strategy = ScalpingStrategy(mock_db_session, account_balance=1000.0)
    
    # Create mock trades
    mock_trades = [
        MagicMock(pnl=10.0, pnl_percent=0.01),
        MagicMock(pnl=-5.0, pnl_percent=-0.005),
        MagicMock(pnl=7.0, pnl_percent=0.007)
    ]
    
    # Set up the query to return these trades
    mock_db_session.query.return_value.filter.return_value.all.return_value = mock_trades
    
    # Get the performance summary
    summary = strategy.get_performance_summary()
    
    # Check the results
    assert summary['total_trades'] == 3
    assert summary['winning_trades'] == 2
    assert summary['losing_trades'] == 1
    assert summary['win_rate'] == 2/3
    assert summary['pnl'] == 12.0  # Sum of all PnLs
    assert summary['avg_pnl'] == 4.0  # 12/3
    assert summary['avg_return'] == 0.004  # (0.01 - 0.005 + 0.007) / 3 