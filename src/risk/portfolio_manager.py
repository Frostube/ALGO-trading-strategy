#!/usr/bin/env python3
"""
Portfolio Risk Manager Module

Manages risk allocation across multiple assets, implements volatility-based
position sizing, and enforces risk limits to avoid overexposure.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

from src.utils.logger import logger

# Edge-weighted position sizing configuration
# Strategies with profit factor >= pf_hi will use risk_hi, others use risk_lo
EDGE_WEIGHTS = {
    "ema_crossover": {"pf_hi": 1.2, "risk_hi": 0.010, "risk_lo": 0.005},
    "rsi_oscillator": {"pf_hi": 1.2, "risk_hi": 0.012, "risk_lo": 0.007},
}

class PortfolioRiskManager:
    """
    Manages risk allocation across a multi-asset portfolio.
    
    Features:
    - Volatility-targeted position sizing
    - Global risk cap across all positions
    - Adaptive sizing based on market regime
    - Edge-weighted position sizing based on recent performance
    """
    
    def __init__(self, account_equity, max_portfolio_risk=0.015, risk_per_trade=0.0075,
                 max_position_pct=0.20, vol_lookback=30):
        """
        Initialize the portfolio risk manager.
        
        Args:
            account_equity: Total account equity
            max_portfolio_risk: Maximum total portfolio risk (default: 0.015 = 1.5%)
            risk_per_trade: Risk per trade as fraction of equity (default: 0.0075 = 0.75%)
            max_position_pct: Maximum position size as fraction of equity (default: 0.20 = 20%)
            vol_lookback: Days to look back for volatility calculation (default: 30)
        """
        self.account_equity = account_equity
        self.max_portfolio_risk = max_portfolio_risk
        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.vol_lookback = vol_lookback
        
        # Active positions dictionary: {symbol: position_data}
        self.active_positions = {}
        
        # Volatility regimes for each symbol: {symbol: regime_data}
        self.volatility_regimes = {}
        
        logger.info(f"Portfolio risk manager initialized with equity ${account_equity}, "
                   f"max risk {max_portfolio_risk*100:.2f}%, "
                   f"trade risk {risk_per_trade*100:.2f}%, "
                   f"max position {max_position_pct*100:.2f}%")
    
    def update_account_equity(self, new_equity):
        """Update the account equity value."""
        self.account_equity = new_equity
    
    def dynamic_risk_pct(self, strat_name, pf_recent):
        """
        Calculate dynamic risk percentage based on strategy's recent performance.
        
        Args:
            strat_name: Strategy name ('ema_crossover', 'rsi_oscillator', etc.)
            pf_recent: Recent profit factor from health monitor
            
        Returns:
            float: Risk percentage to use for position sizing
        """
        cfg = EDGE_WEIGHTS.get(strat_name, None)
        if cfg is None:
            return self.risk_per_trade  # Default if strategy not in config
        
        # Use higher risk for strategies with good recent performance
        return cfg["risk_hi"] if pf_recent >= cfg["pf_hi"] else cfg["risk_lo"]
    
    def calculate_position_size(self, symbol, current_price, atr_value, strat_name=None, pf_recent=None, side='long'):
        """
        Calculate the position size for a new trade based on volatility-targeted risk
        and dynamic edge-based position sizing.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            atr_value: ATR value in price units
            strat_name: Strategy name for edge-weighted sizing
            pf_recent: Recent profit factor for the strategy
            side: Trade direction ('long' or 'short')
            
        Returns:
            tuple: (quantity, notional_value, can_open) - position quantity, 
                   notional value, and whether the position can be opened
        """
        # Calculate risk percentage - use dynamic risk if strategy info provided
        if strat_name and pf_recent is not None:
            risk_pct = self.dynamic_risk_pct(strat_name, pf_recent)
            logger.info(f"Using dynamic risk for {strat_name}: {risk_pct*100:.3f}% (PF={pf_recent:.2f})")
        else:
            risk_pct = self.risk_per_trade
        
        # Calculate risk in dollar terms
        dollar_risk = self.account_equity * risk_pct
        
        # Calculate price volatility
        atr_pct = atr_value / current_price
        
        # Check volatility regime and adjust risk if necessary
        regime = self.get_volatility_regime(symbol)
        regime_factor = self._get_regime_factor(regime)
        
        # Adjust risk based on volatility regime
        adjusted_dollar_risk = dollar_risk * regime_factor
        
        # Calculate position size (dividing by ATR for volatility targeting)
        position_size = adjusted_dollar_risk / atr_value
        
        # Calculate notional value
        notional_value = position_size * current_price
        
        # Check if position would exceed maximum size
        max_notional = self.account_equity * self.max_position_pct
        if notional_value > max_notional:
            logger.warning(f"Position size for {symbol} capped at {self.max_position_pct*100:.1f}% of equity "
                          f"(${max_notional:.2f})")
            position_size = max_notional / current_price
            notional_value = max_notional
        
        # Check if opening this position would exceed portfolio risk limit
        new_risk = atr_value * position_size  # Risk in dollar terms (ATR × quantity)
        can_open = self.can_open_position(symbol, new_risk)
        
        # Round position size to appropriate precision
        # For most crypto, 3 decimal places is good
        position_size = round(position_size, 3)
        
        # Log position sizing details
        logger.info(f"Position size for {symbol} ({side}): {position_size} coins, "
                   f"${notional_value:.2f} notional, ${new_risk:.2f} risk, "
                   f"vol regime: {regime}, risk%: {risk_pct*100:.2f}, can open: {can_open}")
        
        return position_size, notional_value, can_open
    
    def get_volatility_regime(self, symbol):
        """
        Get the current volatility regime for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: Volatility regime ('low', 'normal', or 'high')
        """
        # If we have a cached regime less than 4 hours old, use it
        if symbol in self.volatility_regimes:
            regime_data = self.volatility_regimes[symbol]
            last_update = regime_data.get('updated_at')
            
            if last_update:
                last_update_dt = datetime.fromisoformat(last_update)
                if datetime.now() - last_update_dt < timedelta(hours=4):
                    return regime_data.get('regime', 'normal')
        
        # Otherwise, calculate a new regime
        try:
            # This would typically use historical price data
            # For now, default to 'normal' and update this in a real implementation
            regime = 'normal'
            
            # Store the regime
            self.volatility_regimes[symbol] = {
                'regime': regime,
                'updated_at': datetime.now().isoformat()
            }
            
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating volatility regime for {symbol}: {str(e)}")
            return 'normal'  # Default to normal regime
    
    def calculate_volatility_regime(self, symbol, price_data):
        """
        Calculate the volatility regime based on historical data.
        
        Args:
            symbol: Trading pair symbol
            price_data: DataFrame with price data
            
        Returns:
            str: Volatility regime ('low', 'normal', or 'high')
        """
        try:
            # Calculate 30-day realized volatility
            if len(price_data) < 30:
                return 'normal'  # Not enough data
                
            # Calculate daily returns
            returns = price_data['close'].pct_change().dropna()
            
            # Calculate annualized volatility
            vol = returns.std() * np.sqrt(365)  # Annualized
            
            # Determine regime based on volatility thresholds
            if vol < 0.03:  # Less than 3% daily vol
                regime = 'low'
            elif vol > 0.06:  # More than 6% daily vol
                regime = 'high'
            else:
                regime = 'normal'
                
            # Store the regime
            self.volatility_regimes[symbol] = {
                'regime': regime,
                'volatility': vol,
                'updated_at': datetime.now().isoformat()
            }
            
            logger.info(f"Volatility regime for {symbol}: {regime} (vol: {vol*100:.2f}%)")
            
            return regime
            
        except Exception as e:
            logger.error(f"Error calculating volatility regime: {str(e)}")
            return 'normal'  # Default to normal regime
    
    def _get_regime_factor(self, regime):
        """
        Get the risk adjustment factor based on volatility regime.
        
        Args:
            regime: Volatility regime ('low', 'normal', or 'high')
            
        Returns:
            float: Risk adjustment factor
        """
        if regime == 'low':
            return 0.5  # Reduce risk in low volatility (ranging) markets
        elif regime == 'high':
            return 1.0  # Normal risk in high volatility (trending) markets
        else:
            return 0.75  # Slightly reduced risk in normal markets
    
    def register_position(self, symbol, side, quantity, entry_price, atr_value, stop_loss=None):
        """
        Register a new position with the risk manager.
        
        Args:
            symbol: Trading pair symbol
            side: Trade direction ('long' or 'short')
            quantity: Position size in coins
            entry_price: Entry price
            atr_value: ATR value at entry
            stop_loss: Stop loss price (optional)
            
        Returns:
            bool: True if registration successful
        """
        # Calculate risk in dollar terms
        if stop_loss:
            # Risk based on stop loss
            risk_per_coin = abs(entry_price - stop_loss)
            risk = risk_per_coin * quantity
        else:
            # Risk based on ATR
            risk = atr_value * quantity
        
        # Calculate notional value
        notional = entry_price * quantity
        
        # Store position data
        self.active_positions[symbol] = {
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'notional': notional,
            'risk': risk,
            'atr': atr_value,
            'stop_loss': stop_loss,
            'entry_time': datetime.now().isoformat()
        }
        
        # Log position registration
        logger.info(f"Registered {side} position for {symbol}: {quantity} coins, "
                   f"${notional:.2f} notional, ${risk:.2f} risk")
        
        return True
    
    def update_position(self, symbol, current_price, stop_loss=None, quantity=None):
        """
        Update an existing position with new data.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            stop_loss: New stop loss price (optional)
            quantity: Updated position size (optional)
            
        Returns:
            bool: True if update successful
        """
        if symbol not in self.active_positions:
            logger.warning(f"Cannot update position for {symbol}: Position not found")
            return False
        
        position = self.active_positions[symbol]
        
        # Update quantity if provided
        if quantity is not None:
            position['quantity'] = quantity
        
        # Update stop loss if provided
        if stop_loss is not None:
            position['stop_loss'] = stop_loss
        
        # Recalculate notional value
        position['notional'] = current_price * position['quantity']
        
        # Recalculate risk
        if stop_loss:
            # Risk based on stop loss
            risk_per_coin = abs(current_price - stop_loss)
            position['risk'] = risk_per_coin * position['quantity']
        else:
            # Risk based on ATR
            position['risk'] = position['atr'] * position['quantity']
        
        # Update timestamp
        position['updated_at'] = datetime.now().isoformat()
        
        return True
    
    def close_position(self, symbol):
        """
        Remove a position from the active positions registry.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if successful
        """
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info(f"Closed position for {symbol} in risk manager")
            return True
        
        logger.warning(f"Cannot close position for {symbol}: Position not found")
        return False
    
    def can_open_position(self, symbol, new_position_risk):
        """
        Check if a new position can be opened without exceeding risk limits.
        
        Args:
            symbol: Trading pair symbol for the new position
            new_position_risk: Risk of the new position in dollar terms
            
        Returns:
            bool: True if position can be opened
        """
        # Calculate current total portfolio risk
        current_risk = self.get_total_portfolio_risk()
        
        # Calculate risk if we add the new position
        total_risk = current_risk + new_position_risk
        
        # Check if opening this position would exceed maximum portfolio risk
        max_risk = self.account_equity * self.max_portfolio_risk
        can_open = total_risk <= max_risk
        
        if not can_open:
            logger.warning(f"Cannot open position for {symbol}: "
                          f"Would exceed portfolio risk limit of {self.max_portfolio_risk*100:.2f}%. "
                          f"Current risk: ${current_risk:.2f}, New position risk: ${new_position_risk:.2f}, "
                          f"Total would be: ${total_risk:.2f} > ${max_risk:.2f} max")
            logger.info("Queued signal – risk cap hit")
        
        return can_open
    
    def get_total_portfolio_risk(self):
        """
        Calculate the total risk across all active positions.
        
        Returns:
            float: Total portfolio risk in dollar terms
        """
        return sum(pos['risk'] for pos in self.active_positions.values())
    
    def get_portfolio_exposure(self):
        """
        Calculate the total portfolio exposure.
        
        Returns:
            float: Total portfolio exposure as percentage of equity
        """
        total_notional = sum(pos['notional'] for pos in self.active_positions.values())
        return total_notional / self.account_equity if self.account_equity > 0 else 0
    
    def position_sizing_for_pyramiding(self, symbol, current_price, atr_value, pyramid_level=0, max_levels=2):
        """
        Calculate position size for pyramiding (adding to a winning position).
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            atr_value: ATR value
            pyramid_level: Current pyramid level (0 = initial, 1 = first add, etc.)
            max_levels: Maximum pyramid levels allowed
            
        Returns:
            tuple: (quantity, notional_value, can_open) - position quantity, 
                   notional value, and whether the position can be opened
        """
        # Check if we're already at max pyramid levels
        if pyramid_level >= max_levels:
            return 0, 0, False
        
        # Calculate base position size
        base_size, base_notional, can_open = self.calculate_position_size(
            symbol, current_price, atr_value
        )
        
        if not can_open:
            return 0, 0, False
        
        # Scale position size based on pyramid level
        # First pyramid: 50% of initial, second: 25% of initial
        scale_factor = 0.5 ** pyramid_level
        
        pyramid_size = base_size * scale_factor
        notional_value = pyramid_size * current_price
        
        # Check if this additional risk would exceed portfolio limits
        new_risk = atr_value * pyramid_size
        can_add = self.can_open_position(symbol, new_risk)
        
        # Round position size
        pyramid_size = round(pyramid_size, 3)
        
        logger.info(f"Pyramid position size for {symbol} (level {pyramid_level+1}): "
                   f"{pyramid_size} coins, ${notional_value:.2f} notional, can add: {can_add}")
        
        return pyramid_size, notional_value, can_add
    
    def should_enable_pyramiding(self, symbol):
        """
        Determine if pyramiding should be enabled based on market regime.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if pyramiding should be enabled
        """
        # Check the volatility regime
        regime = self.get_volatility_regime(symbol)
        
        # In low volatility regimes, disable pyramiding
        # In normal or high volatility regimes, enable pyramiding
        return regime != 'low' 