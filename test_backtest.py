import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest
from unittest.mock import patch

# Ensure src package is on the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtest.ema_backtest import run_ema_backtest

@pytest.fixture
def sample_ohlcv():
    """Provide simple OHLCV data for backtest."""
    start = datetime.now() - timedelta(hours=50)
    dates = pd.date_range(start=start, periods=50, freq="1h")
    data = {
        "open": 100 + np.arange(50, dtype=float),
        "high": 101 + np.arange(50, dtype=float),
        "low": 99 + np.arange(50, dtype=float),
        "close": 100 + np.arange(50, dtype=float),
        "volume": np.random.uniform(1, 10, size=50)
    }
    df = pd.DataFrame(data, index=dates)
    return df


def test_run_ema_backtest_returns_metrics(sample_ohlcv):
    """Ensure run_ema_backtest outputs a metrics dictionary."""
    with patch("src.backtest.ema_backtest.fetch_historical_data", return_value=sample_ohlcv), \
         patch("src.backtest.ema_backtest._simulate_trades", return_value=([10000, 10050], [{'pnl': 50}])):
        results = run_ema_backtest(
            symbol="BTC/USDT",
            timeframe="1h",
            days=2,
            initial_balance=10000.0,
            plot=False,
            optimize=False,
        )

    assert isinstance(results, dict)
    expected_keys = [
        "total_return",
        "win_rate",
        "sharpe_ratio",
        "max_drawdown",
        "total_trades",
        "profit_factor",
        "avg_win",
        "avg_loss",
        "equity_curve",
        "strategy_params",
    ]
    for key in expected_keys:
        assert key in results

    numeric_keys = [
        "total_return",
        "win_rate",
        "sharpe_ratio",
        "max_drawdown",
        "profit_factor",
        "avg_win",
        "avg_loss",
    ]
    for key in numeric_keys:
        assert isinstance(results[key], (int, float))
    assert isinstance(results["total_trades"], int)
    assert isinstance(results["equity_curve"], list)
    assert isinstance(results["strategy_params"], dict)
    assert results["strategy_params"]["fast_ema"] is not None
    assert results["strategy_params"]["slow_ema"] is not None
