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
BINANCE_TESTNET_API_KEY = os.getenv('BINANCE_TESTNET_API_KEY', '')
BINANCE_TESTNET_SECRET_KEY = os.getenv('BINANCE_TESTNET_SECRET_KEY', '')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# Database settings
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Indicator parameters (Optimized for 4h timeframe)
EMA_FAST = 5  # Optimal fast EMA from grid search
EMA_SLOW = 12  # Optimal slow EMA from grid search
EMA_TREND = 200  # Added 200 EMA for trend filter

# RSI parameters
RSI_PERIOD = 14  # Standard RSI period
RSI_OVERSOLD = 30  # Optimal RSI oversold threshold from grid search
RSI_OVERBOUGHT = 70  # Optimal RSI overbought threshold from grid search
RSI_LONG_THRESHOLD = 40  # RSI threshold for long entries in scalping strategy
RSI_SHORT_THRESHOLD = 60  # RSI threshold for short entries in scalping strategy

# Volume parameters
VOLUME_PERIOD = 20  # Standard volume period
VOL_RATIO_MIN = 1.2  # Optimal volume threshold from grid search
VOLUME_THRESHOLD = 1.5  # Standard volume threshold
ATR_PERIOD = 14  # Standard ATR period

# Order execution settings
ORDER_TIMEOUT_MS = 700  # Maker order timeout in milliseconds

# Risk management settings (Optimized for 4h timeframe)
RISK_PER_TRADE = 0.0075  # Risk 0.75% of account per trade
STOP_LOSS_PCT = 0.03  # Default stop loss (3%)
TAKE_PROFIT_PCT = 0.06  # Default take profit (6%)
USE_ATR_STOPS = True  # Always use ATR-based stops
ATR_SL_MULTIPLIER = 1.0
ATR_TP_MULTIPLIER = 3.0

# Trailing stop settings
USE_TRAILING_STOP = True  # Enable trailing stops
TRAIL_ATR_MULTIPLIER = 1.0
TRAIL_ACTIVATION_PCT = 0.02  # Start trailing at 2% profit

# Two-leg stop settings (for scalping strategy)
USE_TWO_LEG_STOP = True  # Use two-leg stop loss strategy in scalping

# Volatility-based position sizing
USE_VOLATILITY_SIZING = True  # New setting for volatility-targeted position sizing
VOL_TARGET_PCT = 0.0075  # Target 0.75% daily volatility
VOL_LOOKBACK = 20  # Use 20 periods for volatility calculation
MAX_POSITION_PCT = 0.30  # Maximum position size as % of equity

# Trade frequency controls (adjusted for 4h timeframe)
MAX_TRADES_PER_DAY = 6  # Maximum trades per day
MAX_TRADES_PER_HOUR = 2  # Maximum trades per hour
MIN_BARS_BETWEEN_TRADES = 1  # Minimum bars between trades (from grid search)
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
SLIPPAGE = 0.0005  # Estimated slippage
COMMISSION = 0.0006  # 0.06% taker fee on most exchanges
BACKTEST_TRAIN_SPLIT = 0.8  # % of historical data to use for training

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
USE_ML_FILTER = False
ML_MODEL_TYPE = "RandomForest"  # Model type: RandomForest or XGBoost
ML_PROBABILITY_THRESHOLD = 0.5  # Lowered from 0.6 to be less restrictive
ML_RETRAIN_FREQUENCY = 100  # Retrain ML model after this many new trades
ML_MIN_TRADES_FOR_TRAINING = 50  # Need at least 50 trades to train model

# Performance Targets
WIN_RATE_TARGET = 0.45  # Target win rate of 45%
PF_TARGET = 1.4  # Target profit factor of 1.4
DRAWDOWN_TARGET = 0.15  # Target max drawdown of 15%
TRADES_PER_MONTH_TARGET = 10  # Target 10 trades per month per symbol

# Risk management parameters
USE_SOFT_STOP = True  # Enable soft stop alerts for manual intervention

# Optional adaptive threshold settings used by technical indicators
USE_ADAPTIVE_THRESHOLDS = False
ADAPTIVE_LOOKBACK = 100
