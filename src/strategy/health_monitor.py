#!/usr/bin/env python3
"""
Strategy Health Monitor Module

This module provides real-time monitoring of strategy performance to detect when a
strategy's edge is degrading. It can automatically pause trading when performance
metrics fall below specified thresholds.
"""
import json
import os
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path
import numpy as np

from src.utils.logger import logger
from src.utils.notification import send_notification

class HealthMonitor:
    """
    Monitors the health of a trading strategy by tracking its recent performance.
    Can automatically pause trading if the performance degrades below thresholds.
    """
    
    def __init__(self, strategy_name, symbol, window_size=40, 
                min_profit_factor=1.0, min_win_rate=0.35,
                notification_enabled=True):
        """
        Initialize the health monitor.
        
        Args:
            strategy_name: Name of the strategy being monitored
            symbol: Symbol being traded
            window_size: Number of trades to include in rolling window (default: 40)
            min_profit_factor: Minimum acceptable profit factor (default: 1.0)
            min_win_rate: Minimum acceptable win rate (default: 0.35)
            notification_enabled: Whether to send notifications (default: True)
        """
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.window_size = window_size
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        self.notification_enabled = notification_enabled
        
        # Initialize the rolling window
        self.trades = deque(maxlen=window_size)
        
        # Trading status
        self.is_trading_paused = False
        self.pause_until = None
        
        # Statistics
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        
        # Tracking file path
        self.status_file = Path(f"data/health_monitor_{strategy_name}_{symbol.replace('/', '_')}.json")
        
        # Load existing status if available
        self._load_status()
        
        logger.info(f"Health monitor initialized for {strategy_name} on {symbol} "
                   f"with window size {window_size}, min PF {min_profit_factor}, "
                   f"min win rate {min_win_rate*100:.1f}%")
    
    def _load_status(self):
        """Load health monitor status from file if available."""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                
                # Reconstruct the trades deque from saved list
                trades_list = status.get('trades', [])
                self.trades = deque(trades_list, maxlen=self.window_size)
                
                # Load statistics
                self.win_count = status.get('win_count', 0)
                self.loss_count = status.get('loss_count', 0)
                self.total_profit = status.get('total_profit', 0)
                self.total_loss = status.get('total_loss', 0)
                
                # Check if trading is paused
                self.is_trading_paused = status.get('is_trading_paused', False)
                pause_until_str = status.get('pause_until', None)
                if pause_until_str:
                    self.pause_until = datetime.fromisoformat(pause_until_str)
                    
                    # Check if pause period has expired
                    if datetime.now() > self.pause_until:
                        self.is_trading_paused = False
                        self.pause_until = None
                        logger.info(f"Trading resumed for {self.strategy_name} on {self.symbol} (pause period expired)")
                
                logger.info(f"Loaded health monitor status: {len(self.trades)} trades in window, "
                           f"trading {'paused' if self.is_trading_paused else 'active'}")
                
            except Exception as e:
                logger.error(f"Error loading health monitor status: {str(e)}")
                # Initialize with empty state
                self.trades = deque(maxlen=self.window_size)
    
    def _save_status(self):
        """Save health monitor status to file."""
        # Create directory if it doesn't exist
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            status = {
                'strategy_name': self.strategy_name,
                'symbol': self.symbol,
                'window_size': self.window_size,
                'trades': list(self.trades),
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'total_profit': self.total_profit,
                'total_loss': self.total_loss,
                'is_trading_paused': self.is_trading_paused,
                'pause_until': self.pause_until.isoformat() if self.pause_until else None,
                'updated_at': datetime.now().isoformat()
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving health monitor status: {str(e)}")
    
    def on_trade_closed(self, trade):
        """
        Process a closed trade and update the health monitor.
        
        Args:
            trade: Dict containing trade information
        
        Returns:
            bool: True if trading should continue, False if it should be paused
        """
        # Extract trade result
        is_win = trade['pnl'] > 0
        profit_loss = trade['pnl']
        
        # Add to the trades window (1 for win, 0 for loss)
        self.trades.append(1 if is_win else 0)
        
        # Update statistics
        if is_win:
            self.win_count += 1
            self.total_profit += profit_loss
        else:
            self.loss_count += 1
            self.total_loss += abs(profit_loss)
        
        # Save updated status
        self._save_status()
        
        # Check if we have enough trades to evaluate health
        if len(self.trades) >= self.window_size:
            return self._check_health()
        
        # Not enough trades yet to determine health
        return True
    
    def _check_health(self):
        """
        Check if the strategy's performance is healthy.
        
        Returns:
            bool: True if healthy, False if unhealthy
        """
        # Calculate win rate
        win_rate = sum(self.trades) / len(self.trades)
        
        # Calculate profit factor
        profit_factor = self.total_profit / max(self.total_loss, 0.0001)
        
        # Check against thresholds
        is_healthy = (profit_factor >= self.min_profit_factor and 
                     win_rate >= self.min_win_rate)
        
        # Log current health status
        logger.info(f"Strategy health check: "
                   f"PF={profit_factor:.2f} (min {self.min_profit_factor}), "
                   f"Win={win_rate*100:.1f}% (min {self.min_win_rate*100:.1f}%), "
                   f"Status: {'HEALTHY' if is_healthy else 'DEGRADED'}")
        
        # If unhealthy, pause trading
        if not is_healthy and not self.is_trading_paused:
            self._pause_trading()
            return False
        
        return True
    
    def _pause_trading(self, hours=24):
        """
        Pause trading for the specified number of hours.
        
        Args:
            hours: Number of hours to pause trading (default: 24)
        """
        self.is_trading_paused = True
        self.pause_until = datetime.now() + timedelta(hours=hours)
        
        # Save status
        self._save_status()
        
        # Log pause
        logger.warning(f"⚠️ Pausing {self.strategy_name} on {self.symbol} until {self.pause_until.strftime('%Y-%m-%d %H:%M')} "
                      f"due to degraded performance (PF={self.get_profit_factor():.2f}, "
                      f"Win={self.get_win_rate()*100:.1f}%)")
        
        # Send notification if enabled
        if self.notification_enabled:
            try:
                message = (f"⚠️ Strategy paused: {self.strategy_name} on {self.symbol}\n"
                          f"PF: {self.get_profit_factor():.2f}, Win rate: {self.get_win_rate()*100:.1f}%\n"
                          f"Paused until {self.pause_until.strftime('%Y-%m-%d %H:%M')}")
                
                send_notification(message)
            except Exception as e:
                logger.error(f"Error sending notification: {str(e)}")
    
    def can_trade(self):
        """
        Check if trading is allowed.
        
        Returns:
            bool: True if trading is allowed, False if paused
        """
        # If not paused, can trade
        if not self.is_trading_paused:
            return True
        
        # Check if pause period has expired
        if datetime.now() > self.pause_until:
            self.is_trading_paused = False
            self.pause_until = None
            logger.info(f"Trading resumed for {self.strategy_name} on {self.symbol} (pause period expired)")
            self._save_status()
            return True
        
        return False
    
    def get_win_rate(self):
        """Get the current win rate."""
        if not self.trades:
            return 0
        return sum(self.trades) / len(self.trades)
    
    def get_profit_factor(self):
        """Get the current profit factor."""
        if self.total_loss == 0:
            return float('inf') if self.total_profit > 0 else 0
        return self.total_profit / self.total_loss
    
    def reset(self):
        """Reset the health monitor."""
        self.trades.clear()
        self.win_count = 0
        self.loss_count = 0
        self.total_profit = 0
        self.total_loss = 0
        self.is_trading_paused = False
        self.pause_until = None
        self._save_status()
        logger.info(f"Health monitor reset for {self.strategy_name} on {self.symbol}")


def send_notification(message):
    """
    Send a notification message.
    This is a placeholder for an actual notification system.
    
    Args:
        message: Message to send
    """
    logger.info(f"NOTIFICATION: {message}")
    # Actual implementation would use Slack, email, etc. 