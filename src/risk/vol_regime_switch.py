#!/usr/bin/env python3
"""
Realized Volatility Regime Switch

This module calculates realized volatility over a specified window and
determines the current market regime to adjust trading strategy parameters.

Market regimes:
- QUIET (<3% volatility): Use mean-reversion strategies, reduced position size
- NORMAL (3-6% volatility): Use trending strategies, standard position size
- EXPLOSIVE (>6% volatility): Use breakout strategies, full position size
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from src.utils.logger import logger

# Volatility thresholds for regime classification
QUIET_THRESHOLD = 3.0  # Below 3% = quiet market
EXPLOSIVE_THRESHOLD = 6.0  # Above 6% = explosive market

# Risk adjustments by regime (multipliers applied to base risk)
RISK_ADJUSTMENTS = {
    'QUIET': 0.50,     # 50% of base risk in quiet markets
    'NORMAL': 0.75,    # 75% of base risk in normal markets
    'EXPLOSIVE': 1.00  # 100% of base risk in explosive markets
}

# Strategy selection by regime
REGIME_STRATEGIES = {
    'QUIET': 'mean_reversion',
    'NORMAL': 'ema_crossover',
    'EXPLOSIVE': 'breakout'
}

class VolatilityRegimeMonitor:
    """
    Monitor and classify market regimes based on realized volatility.
    """
    
    def __init__(self, lookback_days=30, update_interval_hours=4):
        """
        Initialize the volatility regime monitor.
        
        Args:
            lookback_days: Number of days to use for volatility calculation
            update_interval_hours: How often to recalculate volatility
        """
        self.lookback_days = lookback_days
        self.update_interval_hours = update_interval_hours
        self.last_update = {}
        self.current_regimes = {}
        self.volatility_history = {}
        self.regime_history = {}
        
    def calculate_realized_volatility(self, df, annualize=True):
        """
        Calculate realized volatility from price data.
        
        Args:
            df: DataFrame with 'close' column
            annualize: Whether to annualize the volatility
            
        Returns:
            Realized volatility in percent
        """
        # Calculate log returns
        returns = np.log(df['close'] / df['close'].shift(1)).dropna()
        
        # Calculate standard deviation of returns
        vol = returns.std()
        
        # Annualize if requested (sqrt of periods per year)
        if annualize:
            # Determine period from dataframe
            if 'timestamp' in df.columns:
                # Try to determine period from timestamp differences
                period_hours = (df['timestamp'].iloc[1] - df['timestamp'].iloc[0]).total_seconds() / 3600
            else:
                # Assume period based on index frequency
                if df.index.freq == 'D':
                    period_hours = 24
                elif df.index.freq == 'H':
                    period_hours = 1
                else:
                    # Default to 2h for this project
                    period_hours = 2
            
            # Calculate periods per year
            periods_per_year = 365 * 24 / period_hours
            vol = vol * np.sqrt(periods_per_year)
        
        # Convert to percentage
        return vol * 100
    
    def determine_regime(self, volatility):
        """
        Determine market regime based on volatility.
        
        Args:
            volatility: Realized volatility in percent
            
        Returns:
            String: 'QUIET', 'NORMAL', or 'EXPLOSIVE'
        """
        if volatility < QUIET_THRESHOLD:
            return 'QUIET'
        elif volatility > EXPLOSIVE_THRESHOLD:
            return 'EXPLOSIVE'
        else:
            return 'NORMAL'
    
    def update_regime(self, symbol, df):
        """
        Update the volatility regime for a symbol.
        
        Args:
            symbol: Trading symbol
            df: DataFrame with historical price data
            
        Returns:
            Dict with volatility info
        """
        now = datetime.now()
        
        # Check if update is needed
        if symbol in self.last_update:
            hours_since_update = (now - self.last_update[symbol]).total_seconds() / 3600
            if hours_since_update < self.update_interval_hours:
                return {
                    'symbol': symbol,
                    'volatility': self.volatility_history[symbol][-1],
                    'regime': self.current_regimes[symbol],
                    'last_update': self.last_update[symbol]
                }
        
        # Get the lookback window
        lookback_bars = self.lookback_days * 24 // 2  # Assuming 2h bars
        recent_data = df.tail(lookback_bars)
        
        # Calculate volatility
        volatility = self.calculate_realized_volatility(recent_data)
        
        # Determine regime
        regime = self.determine_regime(volatility)
        
        # Update histories
        if symbol not in self.volatility_history:
            self.volatility_history[symbol] = []
            self.regime_history[symbol] = []
        
        self.volatility_history[symbol].append(volatility)
        self.regime_history[symbol].append(regime)
        
        # Trim histories to keep last 90 days
        max_history = 90 * 24 // self.update_interval_hours
        if len(self.volatility_history[symbol]) > max_history:
            self.volatility_history[symbol] = self.volatility_history[symbol][-max_history:]
            self.regime_history[symbol] = self.regime_history[symbol][-max_history:]
        
        # Update current regime
        self.current_regimes[symbol] = regime
        self.last_update[symbol] = now
        
        logger.info(f"Updated volatility regime for {symbol}: {volatility:.2f}% = {regime}")
        
        return {
            'symbol': symbol,
            'volatility': volatility,
            'regime': regime,
            'last_update': now
        }
    
    def get_risk_adjustment(self, symbol):
        """
        Get the risk adjustment factor based on current regime.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Risk adjustment multiplier
        """
        if symbol in self.current_regimes:
            regime = self.current_regimes[symbol]
            return RISK_ADJUSTMENTS[regime]
        return 0.75  # Default to NORMAL if no regime set
    
    def get_strategy_for_regime(self, symbol):
        """
        Get the recommended strategy for the current regime.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Strategy name
        """
        if symbol in self.current_regimes:
            regime = self.current_regimes[symbol]
            return REGIME_STRATEGIES[regime]
        return 'ema_crossover'  # Default if no regime set
    
    def should_enable_pyramiding(self, symbol):
        """
        Determine if pyramiding should be enabled based on regime.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Boolean: True if pyramiding should be enabled
        """
        if symbol in self.current_regimes:
            # Enable pyramiding in EXPLOSIVE regime only
            return self.current_regimes[symbol] == 'EXPLOSIVE'
        return False  # Default to disabled if no regime set
    
    def plot_volatility_history(self, symbol, save_path=None):
        """
        Plot the volatility history for a symbol.
        
        Args:
            symbol: Trading symbol
            save_path: Path to save the plot image
            
        Returns:
            Path to saved plot or None
        """
        if symbol not in self.volatility_history or len(self.volatility_history[symbol]) < 2:
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot volatility line
        ax.plot(self.volatility_history[symbol], label=f'{symbol} 30-day Volatility (%)')
        
        # Add horizontal lines for regime thresholds
        ax.axhline(y=QUIET_THRESHOLD, color='g', linestyle='--', alpha=0.7, label=f'Quiet Threshold ({QUIET_THRESHOLD}%)')
        ax.axhline(y=EXPLOSIVE_THRESHOLD, color='r', linestyle='--', alpha=0.7, label=f'Explosive Threshold ({EXPLOSIVE_THRESHOLD}%)')
        
        # Shade regime regions
        regime_changes = []
        last_regime = None
        for i, regime in enumerate(self.regime_history[symbol]):
            if regime != last_regime:
                regime_changes.append((i, regime))
                last_regime = regime
        
        # Add colored background for each regime period
        for i in range(len(regime_changes)):
            start_idx = regime_changes[i][0]
            if i < len(regime_changes) - 1:
                end_idx = regime_changes[i+1][0]
            else:
                end_idx = len(self.volatility_history[symbol])
            
            regime = regime_changes[i][1]
            if regime == 'QUIET':
                color = 'green'
                alpha = 0.1
            elif regime == 'EXPLOSIVE':
                color = 'red'
                alpha = 0.1
            else:  # NORMAL
                color = 'blue'
                alpha = 0.1
                
            ax.axvspan(start_idx, end_idx, color=color, alpha=alpha)
        
        # Add labels and legend
        ax.set_title(f'30-Day Realized Volatility for {symbol}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save or show
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            plt.tight_layout()
            plt.show()
            return None

# Integration with portfolio manager - add these methods to your existing class
def realized_vol_pct(self, symbol, lookback_days=30):
    """
    Calculate realized volatility for a symbol.
    
    Args:
        symbol: Trading symbol
        lookback_days: Days to look back
        
    Returns:
        Volatility in percent
    """
    if symbol not in self.data or 'close' not in self.data[symbol]:
        return None
    
    # Get close prices for the lookback period
    # Assuming 2h bars, so 12 bars per day
    bars_needed = lookback_days * 12  
    close_prices = self.data[symbol]["close"][-bars_needed:]
    
    # Calculate log returns
    returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
    # Calculate annualized volatility
    vol = returns.std() * np.sqrt(12 * 365) * 100  # 12 bars per day × 365 days
    
    return vol

def current_risk_pct(self, symbol):
    """
    Get current risk percentage based on volatility regime.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Risk percentage
    """
    vol = self.realized_vol_pct(symbol)
    
    if vol is None:
        return self.base_risk
    
    if vol < 3.0:  # quiet market
        return 0.50 * self.base_risk
    elif vol > 6.0:  # wild market
        return 1.00 * self.base_risk
    else:
        return 0.75 * self.base_risk 