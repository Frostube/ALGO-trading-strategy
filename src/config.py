import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange settings
EXCHANGE = 'binance'
SYMBOL = 'BTC/USDT'
FUTURES = True
TIMEFRAME = '4h'  # Changed from 1m to 4h for longer-term trend following
HIGHER_TIMEFRAME = '1d'  # Changed from 5m to 1d for multi-timeframe analysis

# API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# Database settings
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Indicator parameters (Optimized for 4h timeframe)
EMA_FAST = 10  # Changed from 9 to 10 (more suitable for 4h timeframe)
EMA_SLOW = 40  # Changed from 16 to 40 (more suitable for 4h timeframe)
EMA_TREND = 200  # Added 200 EMA for trend filter
RSI_PERIOD = 14  # Changed back to standard 14 (was 7)
RSI_LONG_THRESHOLD = 30  # Changed from 25 to standard 30
RSI_SHORT_THRESHOLD = 70  # Changed from 69 to standard 70
VOLUME_PERIOD = 20  # Standard volume period
VOLUME_THRESHOLD = 1.5  # Standard volume threshold
ATR_PERIOD = 14  # Standard ATR period

# Order execution settings
ORDER_TIMEOUT_MS = 700  # Maker order timeout in milliseconds

# Risk management settings (Optimized for 4h timeframe)
RISK_PER_TRADE = 0.0075  # Changed from 0.01 to 0.75% of account per trade
STOP_LOSS_PCT = 0.02  # Increased to 2% for longer timeframe (fallback if ATR not available)
TAKE_PROFIT_PCT = None  # Removed fixed take profit - using trailing stops instead
USE_ATR_STOPS = True  # Always use ATR-based stops
ATR_SL_MULTIPLIER = 1.0  # Changed from 1.3 to 1.0 - using 1× ATR for stop loss
ATR_TP_MULTIPLIER = None  # No fixed take profit multiplier - using trailing stops instead

# Trailing stop settings
USE_TRAILING_STOP = True  # Enable trailing stops
TRAIL_ATR_MULTIPLIER = 1.0  # Use 1× ATR for trailing stops
TRAIL_ACTIVATION_PCT = 0.005  # Activate trailing stop after 0.5% move in our favor

# Volatility-based position sizing
USE_VOLATILITY_SIZING = True  # New setting for volatility-targeted position sizing
VOL_TARGET_PCT = 0.0075  # Target 0.75% volatility per trade
VOL_LOOKBACK = 20  # Use 20 periods for volatility calculation
MAX_POSITION_PCT = 0.20  # Cap position size at 20% of account

# Trade frequency controls (adjusted for 4h timeframe)
MIN_BARS_BETWEEN_TRADES = 2  # Allow trades every 2 bars (8 hours)
MAX_TRADES_PER_DAY = 3  # Limit to 3 trades per day
MIN_CONSECUTIVE_BARS_AGREE = 2  # Require 2 consecutive bars to agree on direction

# Dynamic trailing stop settings
TRAIL_VOLATILITY_ADJUSTMENT = True  # Adjust trailing stops based on volatility
TRAIL_PROFIT_LOCK_PCT = 0.5  # Lock in percentage of profit (50%)
TRAIL_ACCELERATION = True  # Tighten trail as profit increases

# Market regime settings
VOLATILITY_WINDOW = 50  # Longer window for volatility calculation (was 20)
TREND_STRENGTH_THRESHOLD = 0.005  # Minimum trend strength to consider market trending
HIGH_VOLATILITY_THRESHOLD = 1.5  # Relative multiple of avg volatility to consider "high"
AVOID_HIGH_VOLATILITY = False  # Don't avoid high volatility (longer timeframe can handle it)
TREND_CONFIRMATION_WINDOW = 5  # Number of bars needed to confirm trend (was 3)

# Time-of-day trading filters - Disabled for 4h timeframe
USE_TIME_FILTERS = False
TRADING_HOURS_START = 0
TRADING_HOURS_END = 24
AVOID_MIDNIGHT_HOURS = False
HIGH_VOLATILITY_HOURS = []
WEEKEND_TRADING = True

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

# Machine Learning filter settings
USE_ML_FILTER = False  # Disable ML filter temporarily until we have enough trades
ML_PROBABILITY_THRESHOLD = 0.5  # Lowered from 0.6 to be less restrictive
ML_RETRAIN_FREQUENCY = 100  # Retrain ML model after this many new trades
ML_MIN_TRADES_FOR_TRAINING = 30  # Lowered from 50 to enable ML sooner 