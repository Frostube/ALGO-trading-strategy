import sys
import os
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategy.bollinger_breakout import BollingerBreakoutStrategy


def _create_df(last_close: float) -> pd.DataFrame:
    dates = pd.date_range(datetime.now() - timedelta(minutes=20), periods=21, freq='1T')
    data = {
        'open': [100.0]*20 + [last_close],
        'high': [100.0]*20 + [last_close],
        'low': [100.0]*20 + [last_close],
        'close': [100.0]*20 + [last_close],
        'volume': [1.0]*21
    }
    return pd.DataFrame(data, index=dates)


def test_buy_signal():
    df = _create_df(101.0)
    strat = BollingerBreakoutStrategy()
    signal = strat.get_signal(df)
    assert signal == 'buy'


def test_sell_signal():
    df = _create_df(99.0)
    strat = BollingerBreakoutStrategy()
    signal = strat.get_signal(df)
    assert signal == 'sell'


def test_no_signal_when_no_breakout():
    df = _create_df(100.0)  # price within bands
    strat = BollingerBreakoutStrategy()
    signal = strat.get_signal(df)
    assert signal == ''
