import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Exchange settings
EXCHANGE = 'binance'
SYMBOL = 'BTC/USDT'
FUTURES = True
TIMEFRAME = '1m'

# API credentials
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
SLACK_CHANNEL = os.getenv('SLACK_CHANNEL')

# Database settings
DATABASE_URL = 'sqlite:///data/trading_data.db'

# Indicator parameters
EMA_FAST = 9
EMA_SLOW = 21
RSI_PERIOD = 2
RSI_LONG_THRESHOLD = 10
RSI_SHORT_THRESHOLD = 90
VOLUME_PERIOD = 20
VOLUME_THRESHOLD = 1.5

# Risk management settings
RISK_PER_TRADE = 0.01  # 1% of account per trade
STOP_LOSS_PCT = 0.0015  # 0.15%
TAKE_PROFIT_PCT = 0.0030  # 0.30%
TRAILING_STOP = True

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