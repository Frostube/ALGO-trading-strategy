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
from src.data.fetcher import fetch_ohlcv
from src.risk.vol_regime_switch import VolatilityRegimeMonitor

# Edge-weighted position sizing configuration
# Strategies with profit factor >= pf_hi will use risk_hi, others use risk_lo
EDGE_WEIGHTS = {
    "ema_crossover": {"pf_hi": 1.2, "risk_hi": 0.010, "risk_lo": 0.005},
    "rsi_oscillator": {"pf_hi": 1.2, "risk_hi": 0.012, "risk_lo": 0.007},
}

# Volatility regime thresholds
VOL_CALM = 0.03  # 3% realized volatility threshold for calm markets
VOL_STORM = 0.06  # 6% realized volatility threshold for volatile markets

class PortfolioRiskManager:
    """
    Manages risk allocation across a multi-asset portfolio.
    
    Features:
    - Volatility-targeted position sizing
    - Global risk cap across all positions
    - Adaptive sizing based on market regime
    - Edge-weighted position sizing based on recent performance
    """
    
    def __init__(self, account_equity=10000.0, risk_per_trade=0.01, 
                 max_pos_pct=0.2, max_portfolio_risk=0.05, max_correlated_risk=0.08,
                 use_volatility_sizing=True, cache_dir="data"):
        """
        Initialize the portfolio risk manager.
        
        Args:
            account_equity: Total account equity
            risk_per_trade: Default risk per trade (% of equity)
            max_pos_pct: Maximum position size (% of equity)
            max_portfolio_risk: Maximum total portfolio risk (% of equity)
            max_correlated_risk: Maximum risk for correlated assets (% of equity)
            use_volatility_sizing: Whether to size positions based on volatility
            cache_dir: Directory to cache market data
        """
        self.account_equity = account_equity
        self.risk_per_trade = risk_per_trade
        self.base_risk = risk_per_trade  # Store base risk for scaling
        self.max_pos_pct = max_pos_pct
        self.max_portfolio_risk = max_portfolio_risk
        self.max_correlated_risk = max_correlated_risk
        self.use_volatility_sizing = use_volatility_sizing
        self.active_positions = {}
        self.cache_dir = cache_dir
        self.vol_cache = {}  # Cache for realized volatility calculations
        self.regime_cache = {}  # Cache for market regime classifications
        self.vol_monitor = VolatilityRegimeMonitor(lookback_days=30)
        
        self._ensure_cache_dir()
        
        logger.info(f"Portfolio risk manager initialized with equity ${account_equity}, "
                   f"max risk {max_portfolio_risk*100:.2f}%, "
                   f"trade risk {risk_per_trade*100:.2f}%, "
                   f"max position {max_pos_pct*100:.2f}%")
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            cache_path.mkdir(parents=True)
    
    def get_available_risk(self):
        """
        Calculate available risk capacity based on current positions.
        
        Returns:
            float: Available risk as percentage of equity
        """
        current_risk = sum(pos['risk_allocation'] for pos in self.active_positions.values())
        return max(0, self.max_portfolio_risk - current_risk)
    
    def dynamic_risk_pct(self, strat_name, pf_recent):
        """
        Calculate dynamic risk percentage based on strategy edge (profit factor).
        
        Args:
            strat_name: Strategy name ('ema_crossover' or 'rsi_oscillator')
            pf_recent: Recent profit factor (e.g., last 40 trades)
            
        Returns:
            float: Risk percentage (0.01 = 1% of equity)
        """
        cfg = EDGE_WEIGHTS.get(strat_name, None)
        if cfg is None:
            return self.risk_per_trade  # default
        return cfg["risk_hi"] if pf_recent >= cfg["pf_hi"] else cfg["risk_lo"]
    
    def calculate_realized_volatility(self, symbol, lookback=30, timeframe="1d"):
        """
        Calculate realized volatility over a specified lookback period.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            lookback: Lookback period in days
            timeframe: Data timeframe
            
        Returns:
            float: Realized volatility as a decimal (e.g., 0.05 = 5%)
        """
        # Check cache first
        cache_key = f"{symbol}_{lookback}_{timeframe}"
        if cache_key in self.vol_cache:
            cache_time, vol = self.vol_cache[cache_key]
            # Use cached value if less than 4 hours old
            if (datetime.now() - cache_time).total_seconds() < 14400:  # 4 hours
                return vol
        
        try:
            # Fetch data
            df = fetch_ohlcv(symbol, timeframe, days=lookback+5)  # Add buffer days
            
            if df is None or len(df) < lookback:
                logger.warning(f"Insufficient data for volatility calculation: {symbol}")
                return 0.05  # Default to 5% if data insufficient
            
            # Calculate daily returns
            df['return'] = df['close'].pct_change()
            
            # Calculate annualized volatility
            daily_vol = df['return'].std()
            
            # Convert to period volatility based on timeframe
            if timeframe == "1d":
                period_vol = daily_vol
            elif timeframe == "4h":
                period_vol = daily_vol * np.sqrt(6)  # 6 4-hour periods in a day
            elif timeframe == "1h":
                period_vol = daily_vol * np.sqrt(24)  # 24 hours in a day
            else:
                # Default to daily
                period_vol = daily_vol
            
            # Cache the result
            self.vol_cache[cache_key] = (datetime.now(), period_vol)
            
            return period_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.05  # Default to 5% if calculation fails
    
    def current_regime(self, symbol, lookback=30):
        """
        Determine the current market regime based on realized volatility.
        
        Args:
            symbol: Trading pair symbol
            lookback: Lookback period in days
            
        Returns:
            str: Market regime ('calm', 'normal', or 'storm')
        """
        # Check cache first
        cache_key = f"{symbol}_{lookback}"
        if cache_key in self.regime_cache:
            cache_time, regime = self.regime_cache[cache_key]
            # Use cached value if less than 4 hours old
            if (datetime.now() - cache_time).total_seconds() < 14400:  # 4 hours
                return regime
        
        # Calculate realized volatility
        vol = self.calculate_realized_volatility(symbol, lookback)
        
        # Determine regime
        if vol < VOL_CALM:
            regime = "calm"
        elif vol > VOL_STORM:
            regime = "storm"
        else:
            regime = "normal"
        
        # Cache the result
        self.regime_cache[cache_key] = (datetime.now(), regime)
        
        logger.info(f"Market regime for {symbol}: {regime.upper()} (vol={vol:.2%})")
        
        return regime
    
    def adjust_size_for_regime(self, base_qty, symbol):
        """
        Adjust position size based on current market regime.
        
        Args:
            base_qty: Base position size
            symbol: Trading pair symbol
            
        Returns:
            float: Adjusted position size
        """
        regime = self.current_regime(symbol)
        
        if regime == "calm":
            # Reduce position size in calm markets
            return base_qty * 0.5
        elif regime == "storm":
            # Keep full size in volatile markets
            return base_qty
        else:
            # Normal market conditions
            return base_qty * 0.75
    
    def should_enable_pyramiding(self, symbol):
        """
        Determine if pyramiding should be enabled based on current market regime.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            bool: True if pyramiding should be enabled
        """
        # Use the volatility monitor to determine if pyramiding should be enabled
        return self.vol_monitor.should_enable_pyramiding(symbol)
    
    def current_risk_pct(self, symbol):
        """
        Get the current risk percentage based on volatility regime.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Risk percentage adjusted for current market regime
        """
        # Get base risk adjustment from volatility monitor
        risk_adjustment = self.vol_monitor.get_risk_adjustment(symbol)
        
        # Apply adjustment to base risk
        adjusted_risk = self.base_risk * risk_adjustment
        
        logger.debug(f"Risk adjustment for {symbol}: {risk_adjustment} × {self.base_risk:.4f} = {adjusted_risk:.4f}")
        
        return adjusted_risk
    
    def calculate_position_size(self, symbol, current_price, atr_value, strat_name="generic", pf_recent=1.0, side=None):
        """
        Calculate optimal position size based on risk parameters and market conditions.
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            atr_value: ATR value for volatility sizing
            strat_name: Strategy name for dynamic risk allocation
            pf_recent: Recent profit factor for edge-weighted sizing
            side: Trade direction ('buy' or 'sell')
            
        Returns:
            float: Position size in base currency units
        """
        # Step 1: Calculate base risk percentage based on strategy edge and current regime
        risk_pct = self.current_risk_pct(symbol)
        
        # If we have profit factor data, further adjust based on edge
        if pf_recent > 0:
            cfg = EDGE_WEIGHTS.get(strat_name, None)
            if cfg is not None and pf_recent >= cfg["pf_hi"]:
                # Boost risk for strategies with proven edge
                risk_pct = max(risk_pct, cfg["risk_hi"])
        
        # Step 2: Calculate dollar risk amount
        dollar_risk = self.account_equity * risk_pct
        
        # Step 3: Calculate position size based on ATR value
        qty = round(dollar_risk / atr_value, 3)
        
        # Step 4: Adjust position size based on market regime
        qty = self.adjust_size_for_regime(qty, symbol)
        
        # Step 5: Apply maximum position size constraint
        max_qty = self.account_equity * self.max_pos_pct / current_price
        qty = min(qty, max_qty)
        
        # Log the details
        regime = self.current_regime(symbol)
        logger.info(f"Position sizing: {symbol} ({regime} regime) - ${dollar_risk:.2f} risk, {risk_pct:.2%} of equity")
        logger.info(f"Final position: {qty:.6f} units ({qty * current_price:.2f} USD)")
        
        return qty
    
    def register_position(self, symbol, position_size, current_price, risk_amount):
        """
        Register a new position in the portfolio.
        
        Args:
            symbol: Trading pair symbol
            position_size: Position size in base currency
            current_price: Current market price
            risk_amount: Dollar risk amount
        """
        if symbol in self.active_positions:
            logger.warning(f"Position already exists for {symbol}, updating")
        
        self.active_positions[symbol] = {
            'size': position_size,
            'entry_price': current_price,
            'notional_value': position_size * current_price,
            'risk_allocation': risk_amount / self.account_equity
        }
    
    def close_position(self, symbol):
        """
        Remove a position from the portfolio.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info(f"Closed position for {symbol}")
    
    def update_account_equity(self, new_equity):
        """
        Update the account equity value.
        
        Args:
            new_equity: New account equity value
        """
        self.account_equity = new_equity
        logger.info(f"Updated account equity to ${new_equity:.2f}")
    
    def get_position(self, symbol):
        """
        Get position details for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            dict: Position details or None if position doesn't exist
        """
        return self.active_positions.get(symbol, None)
    
    def get_total_exposure(self):
        """
        Calculate total portfolio exposure.
        
        Returns:
            float: Total notional exposure as percentage of equity
        """
        total_notional = sum(pos['notional_value'] for pos in self.active_positions.values())
        return total_notional / self.account_equity
    
    def get_total_risk(self):
        """
        Calculate total portfolio risk.
        
        Returns:
            float: Total risk allocation as percentage of equity
        """
        return sum(pos['risk_allocation'] for pos in self.active_positions.values())
    
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
            position['size'] = quantity
        
        # Update stop loss if provided
        if stop_loss is not None:
            position['stop_loss'] = stop_loss
        
        # Recalculate notional value
        position['notional_value'] = current_price * position['size']
        
        # Recalculate risk
        if stop_loss:
            # Risk based on stop loss
            risk_per_coin = abs(current_price - stop_loss)
            position['risk_allocation'] = risk_per_coin * position['size'] / self.account_equity
        else:
            # Risk based on ATR
            position['risk_allocation'] = position['risk_allocation']
        
        # Update timestamp
        position['updated_at'] = datetime.now().isoformat()
        
        return True
    
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
        current_risk = self.get_total_risk()
        
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
    
    def allocate_capital(self, strategy, symbols, available_capital):
        """
        Allocate capital across multiple symbols based on volatility and opportunity.
        
        Args:
            strategy: Trading strategy instance
            symbols: List of symbols to trade
            available_capital: Capital available for allocation
            
        Returns:
            dict: Capital allocation by symbol
        """
        allocations = {}
        
        for symbol in symbols:
            # Update regime for this symbol if we have data
            if hasattr(strategy, 'data') and symbol in strategy.data:
                self.vol_monitor.update_regime(symbol, strategy.data[symbol])
                
            # Set strategy parameters based on current regime
            strategy.enable_pyramiding = self.should_enable_pyramiding(symbol)
            
            # Allocate based on opportunity score and volatility
            allocations[symbol] = available_capital / len(symbols)  # Equal allocation for now
        
        return allocations 