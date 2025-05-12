import os
import logging
import json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from src.config import SLACK_BOT_TOKEN, SLACK_CHANNEL, LOG_LEVEL, LOG_FILE

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Configure logging
logger = logging.getLogger("trading_bot")
logger.setLevel(getattr(logging, LOG_LEVEL))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(getattr(logging, LOG_LEVEL))
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# File handler
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10485760, backupCount=5)
file_handler.setLevel(getattr(logging, LOG_LEVEL))
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# Add handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Initialize Slack client if token is provided
slack_client = None
if SLACK_BOT_TOKEN:
    slack_client = WebClient(token=SLACK_BOT_TOKEN)

def log_trade(trade_data):
    """Log trade information to file and send notification to Slack."""
    # Log to file
    trade_str = json.dumps(trade_data, default=str)
    logger.info(f"TRADE: {trade_str}")
    
    # Send to Slack if configured
    if slack_client and SLACK_CHANNEL:
        try:
            # Format message for Slack
            message = f"*TRADE EXECUTED*\n"
            message += f"Symbol: {trade_data['symbol']}\n"
            message += f"Side: {trade_data['side']}\n"
            message += f"Entry Price: {trade_data['entry_price']}\n"
            message += f"Stop Loss: {trade_data['stop_loss']}\n"
            message += f"Take Profit: {trade_data['take_profit']}\n"
            message += f"Time: {trade_data['timestamp']}\n"
            
            slack_client.chat_postMessage(
                channel=SLACK_CHANNEL,
                text=message
            )
        except SlackApiError as e:
            logger.error(f"Error sending message to Slack: {e.response['error']}")

def log_alert(message, error=False):
    """Log alerts and errors, sending critical notifications to Slack."""
    if error:
        logger.error(message)
    else:
        logger.warning(message)
    
    # Send to Slack if configured and is an error or alert
    if slack_client and SLACK_CHANNEL:
        try:
            slack_client.chat_postMessage(
                channel=SLACK_CHANNEL,
                text=f"{'🚨 ERROR' if error else '⚠️ ALERT'}: {message}"
            )
        except SlackApiError as e:
            logger.error(f"Error sending message to Slack: {e.response['error']}")

def log_daily_summary(performance_data):
    """Log daily performance summary."""
    summary_str = json.dumps(performance_data, default=str)
    logger.info(f"DAILY SUMMARY: {summary_str}")
    
    if slack_client and SLACK_CHANNEL:
        try:
            # Format message for Slack
            message = f"*DAILY PERFORMANCE SUMMARY*\n"
            message += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            message += f"Total Trades: {performance_data['total_trades']}\n"
            message += f"Win Rate: {performance_data['win_rate']*100:.2f}%\n"
            message += f"Profit/Loss: {performance_data['pnl']:.4f}\n"
            message += f"Avg. Return: {performance_data['avg_return']*100:.2f}%\n"
            
            slack_client.chat_postMessage(
                channel=SLACK_CHANNEL,
                text=message
            )
        except SlackApiError as e:
            logger.error(f"Error sending message to Slack: {e.response['error']}")

def consecutive_sl_alert(count):
    """Send alert when multiple consecutive stop losses are hit."""
    if count >= 2:
        log_alert(f"WARNING: {count} consecutive stop losses hit!", error=False) 