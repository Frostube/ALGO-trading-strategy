#!/usr/bin/env python3
"""
Notification Utility Module

Provides functionality for sending notifications through various channels
(Slack, email, etc.) about important trading events and alerts.
"""
import os
import json
import requests
from datetime import datetime
from pathlib import Path

from src.utils.logger import logger
from src.config import SLACK_BOT_TOKEN, SLACK_CHANNEL

def send_notification(message, channel=None, title=None, severity="info"):
    """
    Send a notification through the configured channels.
    
    Args:
        message: Message to send
        channel: Override default channel (optional)
        title: Optional title for the notification
        severity: Severity level ('info', 'warning', 'error')
        
    Returns:
        bool: True if notification was sent successfully
    """
    # Log the notification
    logger.info(f"NOTIFICATION ({severity}): {message}")
    
    # Save notification to file
    _save_notification(message, title, severity)
    
    # Send to Slack if configured
    slack_result = False
    if SLACK_BOT_TOKEN and (SLACK_CHANNEL or channel):
        slack_result = send_slack_notification(message, channel, title, severity)
    
    return slack_result

def send_slack_notification(message, channel=None, title=None, severity="info"):
    """
    Send a notification to Slack.
    
    Args:
        message: Message to send
        channel: Override default channel (optional)
        title: Optional title for the notification
        severity: Severity level ('info', 'warning', 'error')
        
    Returns:
        bool: True if notification was sent successfully
    """
    if not SLACK_BOT_TOKEN:
        logger.warning("Cannot send Slack notification: SLACK_BOT_TOKEN not configured")
        return False
    
    # Use provided channel or default
    target_channel = channel or SLACK_CHANNEL
    
    if not target_channel:
        logger.warning("Cannot send Slack notification: No channel specified")
        return False
    
    try:
        # Determine emoji based on severity
        if severity == "warning":
            emoji = ":warning:"
        elif severity == "error":
            emoji = ":rotating_light:"
        else:
            emoji = ":chart_with_upwards_trend:"
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Format the message
        formatted_title = f"{emoji} {title}" if title else f"{emoji} Trading Alert"
        formatted_message = {
            "channel": target_channel,
            "text": formatted_title,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": formatted_title
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:* {timestamp}"
                        }
                    ]
                }
            ]
        }
        
        # Send the message
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={
                "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
                "Content-Type": "application/json"
            },
            data=json.dumps(formatted_message)
        )
        
        # Check the response
        response_data = response.json()
        if response.status_code == 200 and response_data.get("ok"):
            logger.info(f"Slack notification sent to {target_channel}")
            return True
        else:
            logger.warning(f"Failed to send Slack notification: {response_data.get('error', 'Unknown error')}")
            return False
        
    except Exception as e:
        logger.error(f"Error sending Slack notification: {str(e)}")
        return False

def _save_notification(message, title=None, severity="info"):
    """
    Save notification to a local file for record-keeping.
    
    Args:
        message: Notification message
        title: Optional title
        severity: Severity level
    """
    try:
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Notification file path
        notifications_file = logs_dir / "notifications.log"
        
        # Format the notification
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_title = title or "Trading Alert"
        formatted_message = f"[{timestamp}] [{severity.upper()}] {formatted_title}: {message}\n"
        
        # Append to file
        with open(notifications_file, "a") as f:
            f.write(formatted_message)
            
    except Exception as e:
        logger.error(f"Error saving notification to file: {str(e)}")
        
def get_recent_notifications(count=10):
    """
    Get recent notifications from the log file.
    
    Args:
        count: Number of notifications to retrieve
        
    Returns:
        list: Recent notifications
    """
    notifications = []
    try:
        notifications_file = Path("logs/notifications.log")
        
        if not notifications_file.exists():
            return []
        
        with open(notifications_file, "r") as f:
            lines = f.readlines()
            
        # Return the most recent notifications
        return lines[-count:]
    except Exception as e:
        logger.error(f"Error reading notifications: {str(e)}")
        return [] 