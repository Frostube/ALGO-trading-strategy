import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange settings
EXCHANGE = 'binance'
SYMBOL = 'BTC/USDT'
FUTURES = True
TIMEFRAME = '1m'
HIGHER_TIMEFRAME = '5m'  # For multi-timeframe analysis

# API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# Database settings
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Indicator parameters
EMA_FAST = 12  # Changed from 9 to 12
EMA_SLOW = 26  # Changed from 21 to 26
EMA_TREND = 200  # Long-term trend filter
EMA_MICRO_TREND = 50  # Micro-trend for slope detection
RSI_PERIOD = 5  # Changed from 2 to 5
RSI_LONG_THRESHOLD = 30  # Changed from 10 to 30 (less aggressive entries)
RSI_SHORT_THRESHOLD = 70  # Changed from 90 to 70 (less aggressive entries)
VOLUME_PERIOD = 20
VOLUME_THRESHOLD = 1.5
ATR_PERIOD = 14  # For volatility-based stop losses

# Adaptive threshold settings
USE_ADAPTIVE_THRESHOLDS = True  # Use percentile-based thresholds for RSI and volume
ADAPTIVE_LOOKBACK = 100  # Number of periods for adaptive threshold calculation

# Trade frequency controls
MIN_BARS_BETWEEN_TRADES = 5  # Minimum number of bars between trades
MAX_TRADES_PER_HOUR = 3  # Maximum number of trades per hour
MIN_CONSECUTIVE_BARS_AGREE = 2  # Minimum consecutive bars where conditions must agree

# Risk management settings
RISK_PER_TRADE = 0.01  # 1% of account per trade
STOP_LOSS_PCT = 0.0015  # 0.15% (fixed percentage, used if USE_ATR_STOPS is False)
TAKE_PROFIT_PCT = 0.0030  # 0.30% (fixed percentage, used if USE_ATR_STOPS is False)
USE_ATR_STOPS = True  # Use ATR-based stops instead of fixed percentage
ATR_SL_MULTIPLIER = 1.5  # Stop loss at 1.5 * ATR
ATR_TP_MULTIPLIER = 3.0  # Take profit at 3.0 * ATR

# Two-leg stop settings
USE_TWO_LEG_STOP = True  # Use two-leg stop strategy (initial SL then trailing)
TRAIL_ACTIVATION_PCT = 0.15  # Percentage move to activate trailing (0.15%)
TRAIL_ATR_MULTIPLIER = 0.5  # Trail at 0.5× ATR once activated
USE_SOFT_STOP = True  # Use "soft stop" alerts before hard stop

# Backtesting settings
SLIPPAGE = 0.0004  # 0.04%
COMMISSION = 0.0002  # 0.02%
BACKTEST_TRAIN_SPLIT = 0.8  # 80% training, 20% testing

# API settings
API_HOST = '127.0.0.1'
API_PORT = 5000

# Historical data settings
HISTORICAL_DATA_DAYS = 30  # Number of days of historical data to fetch

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FILE = 'logs/trading.log'

# Performance logging settings
LOG_FALSE_POSITIVES = True  # Log trades that never hit TP or SL
MAX_TRADE_DURATION_MINUTES = 120  # Maximum trade duration in minutes before considering it a false positive 