import numpy as np
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger('trading_bot')

class StrategyHealthMonitor:
    """
    Monitors the health of trading strategies by tracking performance metrics
    and determining if trading should be paused when performance deteriorates.
    """
    
    def __init__(self, window_size=40, min_profit_factor=1.0, min_win_rate=0.35):
        """
        Initialize the health monitor.
        
        Args:
            window_size: Number of trades to use for rolling window metrics
            min_profit_factor: Minimum acceptable profit factor
            min_win_rate: Minimum acceptable win rate
        """
        self.window_size = window_size
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        
        # Initialize trade history
        self.trades = defaultdict(lambda: defaultdict(list))
        self.status = defaultdict(lambda: defaultdict(lambda: "ACTIVE"))
        
        logger.info(f"Health monitor initialized with window size {window_size}, min PF {min_profit_factor}, min win rate {min_win_rate*100}%")
    
    def record_trade(self, strategy_type, symbol, pnl, win):
        """
        Record a trade result in the history.
        
        Args:
            strategy_type: Type of strategy (e.g., 'ema_crossover', 'rsi_oscillator')
            symbol: Trading symbol (e.g., 'BTC/USDT')
            pnl: Profit/loss amount
            win: Boolean indicating if the trade was a win
        """
        # Create a trade record
        trade = {
            'timestamp': datetime.now(),
            'pnl': pnl,
            'win': win
        }
        
        # Add to trade history for this strategy/symbol
        self.trades[strategy_type][symbol].append(trade)
        
        # Keep only the most recent window_size trades
        if len(self.trades[strategy_type][symbol]) > self.window_size:
            self.trades[strategy_type][symbol] = self.trades[strategy_type][symbol][-self.window_size:]
        
        # Update health status
        self._update_status(strategy_type, symbol)
    
    def _update_status(self, strategy_type, symbol):
        """
        Update the health status for a strategy/symbol pair.
        
        Args:
            strategy_type: Type of strategy
            symbol: Trading symbol
        """
        trades = self.trades[strategy_type][symbol]
        
        # Need minimum number of trades to evaluate
        if len(trades) < max(5, self.window_size // 4):
            self.status[strategy_type][symbol] = "ACTIVE"
            return
        
        # Calculate performance metrics
        wins = sum(1 for t in trades if t['win'])
        losses = len(trades) - wins
        
        win_rate = wins / len(trades) if len(trades) > 0 else 0
        
        total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Check if performance is below thresholds
        if profit_factor < self.min_profit_factor or win_rate < self.min_win_rate:
            self.status[strategy_type][symbol] = "DEGRADED"
            logger.info(f"Strategy health check: PF={profit_factor:.2f} (min {self.min_profit_factor}), Win={win_rate*100:.1f}% (min {self.min_win_rate*100:.1f}%), Status: DEGRADED")
        else:
            self.status[strategy_type][symbol] = "ACTIVE"
            logger.info(f"Strategy health check: PF={profit_factor:.2f} (min {self.min_profit_factor}), Win={win_rate*100:.1f}% (min {self.min_win_rate*100:.1f}%), Status: ACTIVE")
    
    def is_trading_allowed(self, strategy_type, symbol):
        """
        Check if trading is allowed for a strategy/symbol pair.
        
        Args:
            strategy_type: Type of strategy
            symbol: Trading symbol
            
        Returns:
            Boolean: True if trading is allowed, False otherwise
        """
        # If no trades recorded, allow trading
        if not self.trades[strategy_type][symbol]:
            return True
        
        # Check if status is active
        return self.status[strategy_type][symbol] == "ACTIVE"
    
    def get_statistics(self, strategy_type, symbol):
        """
        Get performance statistics for a strategy/symbol pair.
        
        Args:
            strategy_type: Type of strategy
            symbol: Trading symbol
            
        Returns:
            dict: Statistics including win rate, profit factor, etc.
        """
        trades = self.trades[strategy_type][symbol]
        
        if not trades:
            return {
                'trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'status': "ACTIVE"
            }
        
        wins = sum(1 for t in trades if t['win'])
        losses = len(trades) - wins
        
        win_rate = wins / len(trades) if len(trades) > 0 else 0
        
        total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': total_profit - total_loss,
            'status': self.status[strategy_type][symbol]
        }
    
    def reset(self, strategy_type=None, symbol=None):
        """
        Reset the monitor for a strategy/symbol pair, or all if not specified.
        
        Args:
            strategy_type: Type of strategy to reset, or None for all
            symbol: Trading symbol to reset, or None for all
        """
        if strategy_type is None:
            # Reset all
            self.trades = defaultdict(lambda: defaultdict(list))
            self.status = defaultdict(lambda: defaultdict(lambda: "ACTIVE"))
        elif symbol is None:
            # Reset all symbols for this strategy
            self.trades[strategy_type] = defaultdict(list)
            self.status[strategy_type] = defaultdict(lambda: "ACTIVE")
        else:
            # Reset specific strategy/symbol pair
            self.trades[strategy_type][symbol] = []
            self.status[strategy_type][symbol] = "ACTIVE" 