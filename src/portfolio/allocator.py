#!/usr/bin/env python3
"""
Portfolio allocator for managing capital allocation across multiple strategies and symbols.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PortfolioAllocator:
    """
    Allocates capital across multiple trading strategies and symbols based on performance metrics.
    Supports equal-weight, performance-based (Sharpe ratio), and custom weighting schemes.
    Implements volatility targeting for risk management.
    """
    
    def __init__(self, strategies=None, target_volatility=0.10, max_allocation=0.30):
        """
        Initialize the portfolio allocator.
        
        Args:
            strategies (list): List of strategy instances
            target_volatility (float): Target portfolio volatility (e.g., 0.10 = 10%)
            max_allocation (float): Maximum allocation to any single strategy (e.g., 0.30 = 30%)
        """
        self.strategies = strategies or []
        self.target_volatility = target_volatility
        self.max_allocation = max_allocation
        
        # Initialize weights equally
        self.strategy_weights = {} 
        self.symbol_weights = {}
        
        if self.strategies:
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {s.name: equal_weight for s in strategies}
        
        # Performance tracking
        self.performance_history = {}
        self.volatility_history = {}
        self.last_rebalance = datetime.now()
    
    def add_strategy(self, strategy):
        """Add a strategy to the portfolio."""
        if strategy not in self.strategies:
            self.strategies.append(strategy)
            
            # Reset weights to equal
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {s.name: equal_weight for s in self.strategies}
            
            # Initialize performance history
            if strategy.name not in self.performance_history:
                self.performance_history[strategy.name] = []
    
    def set_symbols(self, symbols):
        """Set the trading symbols and initialize with equal weights."""
        if symbols:
            equal_weight = 1.0 / len(symbols)
            self.symbol_weights = {symbol: equal_weight for symbol in symbols}
    
    def set_symbol_weights(self, weights):
        """Set custom symbol weights."""
        if not weights:
            return
            
        total = sum(weights.values())
        if total <= 0:
            return
            
        # Normalize weights to sum to 1.0
        self.symbol_weights = {k: v/total for k, v in weights.items()}
    
    def set_strategy_weights(self, weights):
        """Set custom strategy weights."""
        if not weights:
            return
            
        total = sum(weights.values())
        if total <= 0:
            return
            
        # Normalize weights to sum to 1.0
        self.strategy_weights = {k: v/total for k, v in weights.items()}
    
    def record_performance(self, strategy_name, symbol, returns):
        """
        Record performance for a strategy-symbol combination.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading symbol
            returns (list): List of return values
        """
        key = f"{strategy_name}_{symbol}"
        
        if key not in self.performance_history:
            self.performance_history[key] = []
            
        self.performance_history[key].extend(returns)
        
        # Keep only the most recent returns (cap at 252 trading days)
        if len(self.performance_history[key]) > 252:
            self.performance_history[key] = self.performance_history[key][-252:]
    
    def record_volatility(self, strategy_name, symbol, volatility):
        """Record volatility for a strategy-symbol combination."""
        key = f"{strategy_name}_{symbol}"
        
        if key not in self.volatility_history:
            self.volatility_history[key] = []
            
        self.volatility_history[key].append(volatility)
        
        # Keep only the most recent values (cap at 60 values)
        if len(self.volatility_history[key]) > 60:
            self.volatility_history[key] = self.volatility_history[key][-60:]
    
    def update_weights_sharpe(self, lookback_days=30):
        """
        Update weights based on recent Sharpe ratios.
        
        Args:
            lookback_days (int): Number of days to look back for performance
            
        Returns:
            dict: Updated strategy weights
        """
        sharpes = {}
        
        for key, returns in self.performance_history.items():
            # Skip if not enough data
            if len(returns) < lookback_days:
                sharpes[key] = 0
                continue
                
            # Get most recent returns
            recent_returns = returns[-lookback_days:]
            returns_array = np.array(recent_returns)
            
            # Calculate Sharpe ratio (annualized)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array) or 1e-10  # Avoid division by zero
            sharpe = mean_return / std_return * np.sqrt(252)
            
            # Only use positive Sharpe ratios
            sharpes[key] = max(0, sharpe)
        
        # Compute weights proportional to Sharpe
        total_sharpe = sum(sharpes.values()) or 1  # Avoid division by zero
        weights = {key: value/total_sharpe for key, value in sharpes.items()}
        
        # Extract strategy-specific weights
        strategy_weights = {}
        for key, weight in weights.items():
            strategy_name = key.split('_')[0]
            
            if strategy_name not in strategy_weights:
                strategy_weights[strategy_name] = 0
                
            strategy_weights[strategy_name] += weight
        
        # Cap maximum allocation per strategy
        for strategy, weight in strategy_weights.items():
            strategy_weights[strategy] = min(weight, self.max_allocation)
            
        # Normalize again to ensure sum to 1.0
        total = sum(strategy_weights.values()) or 1
        strategy_weights = {k: v/total for k, v in strategy_weights.items()}
        
        self.strategy_weights = strategy_weights
        return self.strategy_weights
    
    def update_weights_win_rate(self, lookback_days=30):
        """Update weights based on recent win rates."""
        win_rates = {}
        
        for key, returns in self.performance_history.items():
            # Skip if not enough data
            if len(returns) < lookback_days:
                win_rates[key] = 0
                continue
                
            # Get most recent returns
            recent_returns = returns[-lookback_days:]
            
            # Calculate win rate
            wins = sum(1 for r in recent_returns if r > 0)
            win_rate = wins / len(recent_returns) if recent_returns else 0
            
            # Apply a minimum threshold (ignore strategies with poor win rates)
            win_rates[key] = max(0, win_rate - 0.4)  # Minimum 40% win rate to get allocation
        
        # Compute weights proportional to win rates
        total_win_rate = sum(win_rates.values()) or 1  # Avoid division by zero
        weights = {key: value/total_win_rate for key, value in win_rates.items()}
        
        # Extract strategy-specific weights
        strategy_weights = {}
        for key, weight in weights.items():
            strategy_name = key.split('_')[0]
            
            if strategy_name not in strategy_weights:
                strategy_weights[strategy_name] = 0
                
            strategy_weights[strategy_name] += weight
        
        # Cap maximum allocation per strategy
        for strategy, weight in strategy_weights.items():
            strategy_weights[strategy] = min(weight, self.max_allocation)
            
        # Normalize again to ensure sum to 1.0
        total = sum(strategy_weights.values()) or 1
        strategy_weights = {k: v/total for k, v in strategy_weights.items()}
        
        self.strategy_weights = strategy_weights
        return self.strategy_weights
    
    def get_position_size(self, strategy_name, symbol, equity, volatility):
        """
        Calculate position size targeting specific volatility.
        
        Args:
            strategy_name (str): Name of the strategy
            symbol (str): Trading symbol
            equity (float): Current account equity
            volatility (float): Current asset volatility (e.g., ATR/price)
            
        Returns:
            float: Position size in units of the base currency
        """
        # Get weights
        strategy_weight = self.strategy_weights.get(strategy_name, 0)
        symbol_weight = self.symbol_weights.get(symbol, 0)
        
        # Combined weight for this strategy-symbol combo
        combined_weight = strategy_weight * symbol_weight
        
        # Calculate volatility-targeted position size
        vol_target = self.target_volatility * combined_weight
        
        # K-factor for volatility targeting (Kelly-inspired)
        k_factor = vol_target / (volatility + 1e-10)  # Avoid division by zero
        
        # Base position size as percentage of equity
        position_size = equity * k_factor
        
        # Cap at max allocation
        max_size = equity * self.max_allocation
        return min(position_size, max_size)
    
    def should_rebalance(self, rebalance_period_days=7):
        """Check if we should rebalance the portfolio based on time."""
        elapsed = datetime.now() - self.last_rebalance
        return elapsed > timedelta(days=rebalance_period_days)
    
    def rebalance(self, method="sharpe", lookback_days=30):
        """
        Rebalance the portfolio using the specified method.
        
        Args:
            method (str): Weighting method - "equal", "sharpe", "win_rate" or "custom"
            lookback_days (int): Days to look back for performance metrics
            
        Returns:
            dict: Updated weights
        """
        if method == "equal" and self.strategies:
            # Equal weight
            equal_weight = 1.0 / len(self.strategies)
            self.strategy_weights = {s.name: equal_weight for s in self.strategies}
        
        elif method == "sharpe":
            # Sharpe ratio weighted
            self.update_weights_sharpe(lookback_days)
        
        elif method == "win_rate":
            # Win rate weighted
            self.update_weights_win_rate(lookback_days)
        
        elif method == "custom":
            # Custom weights already set
            pass
        
        # Update rebalance timestamp
        self.last_rebalance = datetime.now()
        
        return self.strategy_weights
    
    def get_portfolio_stats(self):
        """
        Get statistics for the entire portfolio.
        
        Returns:
            dict: Portfolio statistics
        """
        if not self.performance_history:
            return {
                "sharpe_ratio": 0,
                "win_rate": 0,
                "volatility": 0,
                "max_drawdown": 0
            }
        
        # Combine returns across all strategies
        all_returns = []
        for returns in self.performance_history.values():
            all_returns.extend(returns)
        
        if not all_returns:
            return {
                "sharpe_ratio": 0,
                "win_rate": 0,
                "volatility": 0,
                "max_drawdown": 0
            }
        
        # Calculate stats
        returns_array = np.array(all_returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array) or 1e-10
        sharpe = mean_return / std_return * np.sqrt(252)
        
        win_rate = sum(1 for r in all_returns if r > 0) / len(all_returns)
        
        # Calculate drawdown
        cumulative = np.cumprod(1 + returns_array)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            "sharpe_ratio": sharpe,
            "win_rate": win_rate,
            "volatility": std_return * np.sqrt(252),  # Annualized
            "max_drawdown": max_drawdown
        } 