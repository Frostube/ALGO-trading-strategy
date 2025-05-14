import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange settings
EXCHANGE = 'binance'
SYMBOL = 'BTC/USDT'
FUTURES = True
TIMEFRAME = '5m'  # Changed from 1m to 5m for more reliable signals
HIGHER_TIMEFRAME = '15m'  # Changed from 5m to 15m for better trend confirmation

# API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# Database settings
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Indicator parameters (Highly Optimized for BTC/USDT 5m timeframe)
EMA_FAST = 8  # Optimized from 9 to 8 for faster response
EMA_SLOW = 21  # Optimized from 16 to 21 for stronger trend confirmation
EMA_TREND = 50  # For medium-term trend detection
EMA_MICRO_TREND = 13  # Optimized from 21 to 13 for faster signal generation
RSI_PERIOD = 6  # Optimized from 7 to 6 for faster reaction
RSI_LONG_THRESHOLD = 35  # Changed from 25 to 35 to be less aggressive on entries
RSI_SHORT_THRESHOLD = 65  # Changed from 69 to 65 to be less aggressive
VOLUME_PERIOD = 14  # Optimized from 19 to 14
VOLUME_THRESHOLD = 1.5  # Increased from 1.37 to 1.5 for stronger volume confirmation
ATR_PERIOD = 14  # Optimized from 10 to 14

# HMA parameters (Optimized)
HMA_FAST_PERIOD = 9  # Optimized value
HMA_SLOW_PERIOD = 21  # Optimized value

# Adaptive threshold settings
USE_ADAPTIVE_THRESHOLDS = True  # Use percentile-based thresholds for RSI and volume
ADAPTIVE_LOOKBACK = 100  # Number of periods for adaptive threshold calculation

# Trade frequency controls - More selective
MIN_BARS_BETWEEN_TRADES = 3  # Increased from 2 to 3 to be more selective
MAX_TRADES_PER_HOUR = 4  # Decreased from 6 to 4 for more selectivity
MIN_CONSECUTIVE_BARS_AGREE = 2  # Increased from 1 to 2 for stronger confirmation

# Risk management settings (More conservative)
RISK_PER_TRADE = 0.02  # Increased from 0.01 to 0.02 (2% of account per trade)
STOP_LOSS_PCT = 0.0025  # Tighter stop loss - 0.25% (from 0.35%)
TAKE_PROFIT_PCT = 0.01  # Increased take profit target - 1.0% (from 0.7%)
USE_ATR_STOPS = True  # Use ATR-based stops instead of fixed percentage
ATR_SL_MULTIPLIER = 1.5  # Increased from 1.3 to 1.5
ATR_TP_MULTIPLIER = 4.0  # Increased from 3.0 to 4.0 for larger profits

# Two-leg stop settings
USE_TWO_LEG_STOP = True  # Use two-leg stop strategy (initial SL then trailing)
TRAIL_ACTIVATION_PCT = 0.002  # Increased from 0.001 to 0.002 (0.2%)
TRAIL_ATR_MULTIPLIER = 0.6  # Changed from 0.7 to 0.6 (tighter trailing)
USE_SOFT_STOP = True  # Use "soft stop" alerts before hard stop

# Dynamic trailing stop settings
TRAIL_VOLATILITY_ADJUSTMENT = True  # Adjust trailing stops based on volatility
TRAIL_PROFIT_LOCK_PCT = 0.6  # Lock in percentage of profit (increased from 0.5 to 0.6)
TRAIL_ACCELERATION = True  # Tighten trail as profit increases

# Market regime settings
VOLATILITY_WINDOW = 20  # Window for volatility calculation
TREND_STRENGTH_THRESHOLD = 0.0015  # Decreased from 0.002 to 0.0015
HIGH_VOLATILITY_THRESHOLD = 1.5  # Increased from 1.2 to 1.5 to be more restrictive
AVOID_HIGH_VOLATILITY = True  # Avoid trading in high volatility periods
TREND_CONFIRMATION_WINDOW = 2  # Decreased from 3 to 2

# Time-of-day trading filters - Market hours focused
USE_TIME_FILTERS = True  # Enable time filters
TRADING_HOURS_START = 8  # Start trading at 8 UTC (major market hours)
TRADING_HOURS_END = 20  # End trading at 20 UTC
AVOID_MIDNIGHT_HOURS = True  # Avoid trading during midnight hours
HIGH_VOLATILITY_HOURS = [8, 9, 14, 15, 16, 20]  # Hours with typically higher volatility (UTC)
WEEKEND_TRADING = False  # Disable weekend trading for more reliability

# Backtesting settings
SLIPPAGE = 0.0005  # 0.05% (increased from 0.04%)
COMMISSION = 0.0002  # 0.02%
BACKTEST_TRAIN_SPLIT = 0.7  # 70% training, 30% testing

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
MAX_TRADE_DURATION_MINUTES = 120  # Maximum trade duration in minutes

# Machine Learning filter settings
USE_ML_FILTER = False  # Disable ML filter
ML_PROBABILITY_THRESHOLD = 0.5
ML_RETRAIN_FREQUENCY = 100
ML_MIN_TRADES_FOR_TRAINING = 30 