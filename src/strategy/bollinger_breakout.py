import pandas as pd

from .base_strategy import BaseStrategy
from .confirm import add_bollinger_bands, is_bb_squeeze


class BollingerBreakoutStrategy(BaseStrategy):
    """Simple breakout strategy using Bollinger Band squeeze detection."""

    def __init__(self, bb_period=20, bb_std=2, squeeze_lookback=20,
                 squeeze_threshold=0.05, config=None):
        super().__init__(config=config)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_threshold = squeeze_threshold

    def apply_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply Bollinger Bands to the DataFrame."""
        df = add_bollinger_bands(df, period=self.bb_period, std_dev=self.bb_std)
        return df

    def get_signal(self, df: pd.DataFrame) -> str:
        """Return 'buy', 'sell' or '' based on breakout conditions."""
        if df is None or len(df) < self.bb_period + 1:
            return ''

        df = self.apply_indicators(df.copy())
        if not is_bb_squeeze(df, lookback=self.squeeze_lookback,
                             squeeze_threshold=self.squeeze_threshold):
            return ''

        last = df.iloc[-1]
        if last['close'] > last['bb_upper']:
            return 'buy'
        if last['close'] < last['bb_lower']:
            return 'sell'
        return ''
