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

# Indicator parameters (Optimized)
EMA_FAST = 9  # Optimized from 5 to 9
EMA_SLOW = 16  # Optimized from 13 to 16
EMA_TREND = 50  # Unchanged - for medium-term trend detection
EMA_MICRO_TREND = 21  # Unchanged - for micro trend detection
RSI_PERIOD = 7  # Optimized from 6 to 7
RSI_LONG_THRESHOLD = 25  # Optimized from 30 to 25 (more aggressive entries)
RSI_SHORT_THRESHOLD = 69  # Optimized from 70 to 69 (slightly more aggressive)
VOLUME_PERIOD = 19  # Optimized from 15 to 19 (longer lookback)
VOLUME_THRESHOLD = 1.37  # Optimized from 1.5 to 1.37 (slightly less restrictive)
ATR_PERIOD = 10  # Unchanged

# HMA parameters (Optimized)
HMA_FAST_PERIOD = 14  # Optimized value
HMA_SLOW_PERIOD = 30  # Optimized value

# Adaptive threshold settings
USE_ADAPTIVE_THRESHOLDS = True  # Use percentile-based thresholds for RSI and volume
ADAPTIVE_LOOKBACK = 100  # Number of periods for adaptive threshold calculation

# Trade frequency controls
MIN_BARS_BETWEEN_TRADES = 2  # Decreased from 5 to allow more frequent trading
MAX_TRADES_PER_HOUR = 6  # Increased from 3 to allow more trades
MIN_CONSECUTIVE_BARS_AGREE = 1  # Decreased from 2 to enter trades faster

# Risk management settings (Optimized)
RISK_PER_TRADE = 0.01  # 1% of account per trade
STOP_LOSS_PCT = 0.0035  # Increased from 0.00293 to 0.35%
TAKE_PROFIT_PCT = 0.007  # Increased from 0.00608 to 0.7%
USE_ATR_STOPS = True  # Use ATR-based stops instead of fixed percentage
ATR_SL_MULTIPLIER = 1.3  # Increased from 1.12 to allow more room
ATR_TP_MULTIPLIER = 3.0  # Decreased from 3.85 for faster profit taking

# Two-leg stop settings
USE_TWO_LEG_STOP = True  # Use two-leg stop strategy (initial SL then trailing)
TRAIL_ACTIVATION_PCT = 0.001  # Decreased from 0.0015 to activate trailing sooner
TRAIL_ATR_MULTIPLIER = 0.7  # Increased from 0.5 to give more room
USE_SOFT_STOP = True  # Use "soft stop" alerts before hard stop

# Dynamic trailing stop settings
TRAIL_VOLATILITY_ADJUSTMENT = True  # Adjust trailing stops based on volatility
TRAIL_PROFIT_LOCK_PCT = 0.5  # Lock in percentage of profit (50%)
TRAIL_ACCELERATION = True  # Tighten trail as profit increases

# Market regime settings
VOLATILITY_WINDOW = 20  # Window for volatility calculation
TREND_STRENGTH_THRESHOLD = 0.002  # Minimum trend strength to consider market trending
HIGH_VOLATILITY_THRESHOLD = 1.2  # Relative multiple of avg volatility to consider "high"
AVOID_HIGH_VOLATILITY = True  # Avoid trading in high volatility periods
TREND_CONFIRMATION_WINDOW = 3  # Number of bars needed to confirm trend

# Time-of-day trading filters - Make less restrictive
USE_TIME_FILTERS = False  # Disabled time filters to generate more signals
TRADING_HOURS_START = 0  # Allow trading at any hour
TRADING_HOURS_END = 24  # Allow trading at any hour
AVOID_MIDNIGHT_HOURS = False  # Allow trading during midnight hours
HIGH_VOLATILITY_HOURS = [8, 9, 14, 15, 16, 20, 21]  # Hours with typically higher volatility (UTC)
WEEKEND_TRADING = True  # Allow weekend trading

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