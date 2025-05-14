#!/usr/bin/env python3
"""
Market regime detector for identifying market conditions and adapting strategy behavior.
"""

import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime, timedelta

class MarketRegime(Enum):
    """Enum for market regime types."""
    RANGING = "ranging"
    NORMAL = "normal"
    TRENDING = "trending"
    UNKNOWN = "unknown"

class MarketRegimeDetector:
    """
    Detects market regimes (ranging, trending, etc.) based on price volatility
    and other technical indicators. Helps strategies adapt to changing market
    conditions by switching between mean-reversion and trend-following approaches.
    """
    
    def __init__(self, lookback_days=30, vol_range_threshold=0.03, vol_trend_threshold=0.08):
        """
        Initialize the market regime detector.
        
        Args:
            lookback_days (int): Number of days to look back for regime detection
            vol_range_threshold (float): Volatility threshold below which market is considered ranging
            vol_trend_threshold (float): Volatility threshold above which market is considered trending
        """
        self.lookback = lookback_days
        self.vol_range_threshold = vol_range_threshold
        self.vol_trend_threshold = vol_trend_threshold
        self.price_history = []
        self.returns_history = []
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_history = []
        self.last_update = None
        self.regime_strategies = {
            MarketRegime.RANGING: [],
            MarketRegime.NORMAL: [],
            MarketRegime.TRENDING: []
        }
    
    def set_regime_strategies(self, ranging_strategies=None, normal_strategies=None, trending_strategies=None):
        """
        Set up strategies appropriate for each market regime.
        
        Args:
            ranging_strategies (list): List of strategy names for ranging markets
            normal_strategies (list): List of strategy names for normal markets
            trending_strategies (list): List of strategy names for trending markets
        """
        if ranging_strategies:
            self.regime_strategies[MarketRegime.RANGING] = ranging_strategies
        if normal_strategies:
            self.regime_strategies[MarketRegime.NORMAL] = normal_strategies
        if trending_strategies:
            self.regime_strategies[MarketRegime.TRENDING] = trending_strategies
    
    def add_price(self, price, timestamp=None):
        """
        Add a price point to the history.
        
        Args:
            price (float): Current price
            timestamp (datetime): Timestamp for this price point
        """
        if not timestamp:
            timestamp = datetime.now()
            
        self.price_history.append((timestamp, price))
        
        # Calculate return if we have at least two prices
        if len(self.price_history) > 1:
            prev_price = self.price_history[-2][1]
            current_return = price / prev_price - 1
            self.returns_history.append((timestamp, current_return))
        
        # Keep history within lookback period
        cutoff_time = timestamp - timedelta(days=self.lookback)
        self.price_history = [(t, p) for t, p in self.price_history if t >= cutoff_time]
        self.returns_history = [(t, r) for t, r in self.returns_history if t >= cutoff_time]
    
    def detect_regime(self):
        """
        Detect the current market regime based on volatility.
        
        Returns:
            MarketRegime: The detected market regime
        """
        if len(self.returns_history) < 10:  # Need minimum data points
            return MarketRegime.UNKNOWN
        
        # Extract returns values
        returns = [r for _, r in self.returns_history]
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(returns)
        annualized_vol = volatility * np.sqrt(252)  # Annualize
        
        # Detect regime based on volatility thresholds
        if annualized_vol < self.vol_range_threshold:
            regime = MarketRegime.RANGING
        elif annualized_vol > self.vol_trend_threshold:
            regime = MarketRegime.TRENDING
        else:
            regime = MarketRegime.NORMAL
        
        # Update current regime and history
        self.current_regime = regime
        self.regime_history.append((datetime.now(), regime))
        self.last_update = datetime.now()
        
        return regime
    
    def get_current_regime(self):
        """
        Get the current market regime. If no recent update, detect again.
        
        Returns:
            MarketRegime: The current market regime
        """
        # If we haven't updated in a while, re-detect
        if not self.last_update or (datetime.now() - self.last_update) > timedelta(hours=6):
            return self.detect_regime()
        
        return self.current_regime
    
    def get_regime_stats(self):
        """
        Get statistics on time spent in each regime.
        
        Returns:
            dict: Percentage of time spent in each regime
        """
        if not self.regime_history:
            return {regime.value: 0 for regime in MarketRegime}
        
        # Count occurrences of each regime
        regime_counts = {}
        for _, regime in self.regime_history:
            if regime.value not in regime_counts:
                regime_counts[regime.value] = 0
            regime_counts[regime.value] += 1
        
        # Calculate percentages
        total = len(self.regime_history)
        regime_percentages = {regime: count / total for regime, count in regime_counts.items()}
        
        # Ensure all regimes are included
        for regime in MarketRegime:
            if regime.value not in regime_percentages:
                regime_percentages[regime.value] = 0
        
        return regime_percentages
    
    def is_regime_change(self, new_price):
        """
        Check if adding this price would cause a regime change.
        
        Args:
            new_price (float): New price to check
            
        Returns:
            bool: True if regime would change, False otherwise
        """
        # Store current regime
        current = self.current_regime
        
        # Add new price temporarily
        self.add_price(new_price)
        new_regime = self.detect_regime()
        
        # Check if regime changed
        changed = current != new_regime
        
        # Remove the temporary addition (roll back)
        if len(self.price_history) > 0:
            self.price_history.pop()
        if len(self.returns_history) > 0:
            self.returns_history.pop()
        
        return changed
    
    def get_trending_metrics(self):
        """
        Calculate trending market metrics.
        
        Returns:
            dict: Dict containing trend metrics
        """
        if len(self.price_history) < 20:
            return {"trend_strength": 0, "direction": 0}
        
        # Extract prices
        prices = [p for _, p in self.price_history]
        
        # Calculate average directional movement
        price_changes = np.diff(prices)
        adm = np.sum(np.abs(price_changes)) / len(price_changes)
        
        # Calculate trend direction (-1 for down, 1 for up)
        direction = 1 if np.sum(price_changes) > 0 else -1
        
        # Calculate trend strength
        trend_strength = adm / (np.std(prices) + 1e-10)
        
        return {
            "trend_strength": trend_strength,
            "direction": direction
        }
    
    def get_ranging_metrics(self):
        """
        Calculate ranging market metrics.
        
        Returns:
            dict: Dict containing ranging metrics
        """
        if len(self.price_history) < 20:
            return {"range_width": 0, "position_in_range": 0}
        
        # Extract prices
        prices = [p for _, p in self.price_history]
        
        # Calculate range bounds
        high = max(prices)
        low = min(prices)
        
        # Calculate range width as percentage
        range_width = (high - low) / ((high + low) / 2)
        
        # Calculate position within range (0 = bottom, 1 = top)
        current_price = prices[-1]
        position_in_range = (current_price - low) / (high - low + 1e-10)
        
        return {
            "range_width": range_width,
            "position_in_range": position_in_range
        }
    
    def get_recommended_strategies(self, available_strategies):
        """
        Get recommended strategies for the current market regime.
        
        Args:
            available_strategies (dict): Dict mapping strategy names to strategy objects
            
        Returns:
            list: List of recommended strategy names for the current regime
        """
        regime = self.get_current_regime()
        
        if regime == MarketRegime.RANGING:
            # For ranging markets, prefer mean reversion strategies
            preferred = self.regime_strategies.get(MarketRegime.RANGING, ["mean_reversion", "rsi_momentum"])
        elif regime == MarketRegime.TRENDING:
            # For trending markets, prefer trend following strategies
            preferred = self.regime_strategies.get(MarketRegime.TRENDING, ["ema_crossover", "donchian_breakout"])
        else:
            # For normal markets or unknown, use a mix
            preferred = self.regime_strategies.get(MarketRegime.NORMAL, ["ema_crossover", "volume_breakout"])
        
        # Filter to only include available strategies
        return [strat for strat in preferred if strat in available_strategies] 