"""
Risk management utilities for the trading bot including trailing stop and circuit breaker functionality.
"""
from datetime import datetime, timedelta

class TrailingStop:
    """
    Trailing stop implementation that activates once a certain profit threshold is reached.
    Moves the stop loss to follow the price at a specified distance.
    """
    
    def __init__(self, activation_pct=0.0015, trail_distance_pct=0.0010):
        """
        Initialize the trailing stop.
        
        Args:
            activation_pct: Percentage of profit at which to activate trailing stop (default 0.15%)
            trail_distance_pct: Distance to maintain behind price (default 0.10%)
        """
        self.activation_pct = activation_pct
        self.trail_distance_pct = trail_distance_pct
        self.activated = False
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.current_stop = 0
    
    def update(self, trade, current_price):
        """
        Update the trailing stop based on current price.
        
        Args:
            trade: Current trade dictionary with entry_price, side, etc.
            current_price: Current market price
            
        Returns:
            dict: Updated trade with potentially new stop_loss
        """
        if not trade:
            return None
        
        side = trade['side']
        entry_price = trade['entry_price']
        stop_loss = trade['stop_loss']
        
        # Long position logic
        if side == 'buy':
            # Track highest price seen
            if current_price > self.highest_price:
                self.highest_price = current_price
            
            # Calculate current profit percentage
            profit_pct = (current_price - entry_price) / entry_price
            
            # Check if trailing stop should be activated
            if not self.activated and profit_pct >= self.activation_pct:
                self.activated = True
                # Initialize trailing stop at activation
                new_stop = current_price * (1 - self.trail_distance_pct)
                # Only update if new stop is higher than original
                if new_stop > stop_loss:
                    trade['stop_loss'] = new_stop
                    self.current_stop = new_stop
            
            # If already activated, update trailing stop if price moved higher
            elif self.activated:
                new_stop = self.highest_price * (1 - self.trail_distance_pct)
                # Only update if new stop is higher than current
                if new_stop > self.current_stop:
                    trade['stop_loss'] = new_stop
                    self.current_stop = new_stop
        
        # Short position logic
        elif side == 'sell':
            # Track lowest price seen
            if current_price < self.lowest_price:
                self.lowest_price = current_price
            
            # Calculate current profit percentage
            profit_pct = (entry_price - current_price) / entry_price
            
            # Check if trailing stop should be activated
            if not self.activated and profit_pct >= self.activation_pct:
                self.activated = True
                # Initialize trailing stop at activation
                new_stop = current_price * (1 + self.trail_distance_pct)
                # Only update if new stop is lower than original
                if new_stop < stop_loss:
                    trade['stop_loss'] = new_stop
                    self.current_stop = new_stop
            
            # If already activated, update trailing stop if price moved lower
            elif self.activated:
                new_stop = self.lowest_price * (1 + self.trail_distance_pct)
                # Only update if new stop is lower than current
                if new_stop < self.current_stop:
                    trade['stop_loss'] = new_stop
                    self.current_stop = new_stop
        
        return trade

class CircuitBreaker:
    """
    Circuit breaker to pause trading after consecutive stop losses.
    """
    
    def __init__(self, max_consecutive_losses=3, cooldown_minutes=30):
        """
        Initialize the circuit breaker.
        
        Args:
            max_consecutive_losses: Number of consecutive losses before pausing (default 3)
            cooldown_minutes: Minutes to pause trading after triggered (default 30)
        """
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_minutes = cooldown_minutes
        self.consecutive_losses = 0
        self.pause_until = None
    
    def update(self, trade_result):
        """
        Update the circuit breaker based on trade result.
        
        Args:
            trade_result: Dictionary with trade result information
            
        Returns:
            bool: True if trading should be allowed, False if paused
        """
        # If we're in a cooldown period
        if self.pause_until and datetime.now() < self.pause_until:
            return False
        
        # If cooldown has ended
        if self.pause_until and datetime.now() >= self.pause_until:
            self.pause_until = None
            self.consecutive_losses = 0
            return True
        
        # Process new trade result
        if trade_result and trade_result.get('exit_reason') == 'stop_loss':
            self.consecutive_losses += 1
            
            # Check if circuit breaker should be triggered
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.pause_until = datetime.now() + timedelta(minutes=self.cooldown_minutes)
                return False
        elif trade_result and trade_result.get('exit_reason') != 'stop_loss':
            # Reset counter on any non-stop-loss exit
            self.consecutive_losses = 0
        
        return True

class VolatilityAdjuster:
    """
    Adjusts position sizes and stop distances based on market volatility.
    """
    
    def __init__(self, lookback_periods=100, volatility_factor=1.5):
        """
        Initialize the volatility adjuster.
        
        Args:
            lookback_periods: Number of periods to calculate volatility (default 100)
            volatility_factor: Factor to adjust risk by in high volatility (default 1.5)
        """
        self.lookback_periods = lookback_periods
        self.volatility_factor = volatility_factor
        self.baseline_volatility = None
        self.current_volatility = None
    
    def calculate_atr(self, df, periods=14):
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLCV data
            periods: ATR period
            
        Returns:
            float: Current ATR value
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr1 = abs(high[1:] - low[1:])
        tr2 = abs(high[1:] - close[:-1])
        tr3 = abs(low[1:] - close[:-1])
        
        tr = [max(tr1[i], tr2[i], tr3[i]) for i in range(len(tr1))]
        
        # Simple moving average of TR
        atr = sum(tr[-periods:]) / periods
        return atr
    
    def update(self, df):
        """
        Update volatility measurements.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            dict: Volatility information
        """
        # Calculate current ATR
        current_atr = self.calculate_atr(df)
        
        # On first run, set baseline
        if self.baseline_volatility is None:
            self.baseline_volatility = current_atr
        
        # Update current volatility
        self.current_volatility = current_atr
        
        # Calculate relative volatility (compared to baseline)
        relative_volatility = self.current_volatility / self.baseline_volatility if self.baseline_volatility else 1.0
        
        # Determine if we're in a high volatility regime
        high_volatility = relative_volatility > self.volatility_factor
        
        return {
            'baseline_volatility': self.baseline_volatility,
            'current_volatility': self.current_volatility,
            'relative_volatility': relative_volatility,
            'high_volatility': high_volatility
        }
    
    def adjust_risk(self, risk_params, volatility_info):
        """
        Adjust risk parameters based on volatility.
        
        Args:
            risk_params: Dictionary with risk parameters
            volatility_info: Dictionary with volatility information
            
        Returns:
            dict: Adjusted risk parameters
        """
        adjusted_params = risk_params.copy()
        
        # If in high volatility, adjust risk parameters
        if volatility_info['high_volatility']:
            # Reduce position size
            adjusted_params['risk_per_trade'] = risk_params['risk_per_trade'] * 0.5
            
            # Widen stop-loss
            adjusted_params['stop_loss_pct'] = risk_params['stop_loss_pct'] * 1.5
            
            # Widen take-profit
            adjusted_params['take_profit_pct'] = risk_params['take_profit_pct'] * 1.5
        
        return adjusted_params 