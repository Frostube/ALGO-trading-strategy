#!/usr/bin/env python3
"""
Exchange Fee & Funding Cost Model

This module implements realistic exchange fee calculations, funding rate simulation,
and slippage modeling for accurate P&L estimation in backtesting and live trading.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
from src.utils.logger import logger

# Standard fee structure
TAKER_FEE = 0.0004  # 0.04% for market orders/taker fills
MAKER_FEE = 0.0002  # 0.02% for limit orders with immediate execution
MAKER_REBATE = -0.0001  # -0.01% rebate for true maker orders that provide liquidity

# Default funding rate (per 8 hours) when historical data not available
DEFAULT_FUNDING_RATE = 0.0001  # 0.01% per 8 hours, conservative estimate

# Slippage estimates
LIMIT_SLIPPAGE = 0.0002  # 0.02% for limit orders
MARKET_SLIPPAGE = 0.0005  # 0.05% for market orders/fallbacks

class FeeModel:
    """
    Exchange fee and funding cost model.
    """
    
    def __init__(self, exchange_name='binance', use_historical_funding=True):
        """
        Initialize the fee model.
        
        Args:
            exchange_name: Name of the exchange to model
            use_historical_funding: Whether to fetch historical funding rates
        """
        self.exchange_name = exchange_name
        self.use_historical_funding = use_historical_funding
        self.funding_history = {}
        self.exchange_volume_30d = 0  # For VIP tier calculation
        
        # Initialize exchange connection for funding rate fetching if needed
        if use_historical_funding:
            try:
                self.exchange = getattr(ccxt, exchange_name)({
                    'enableRateLimit': True
                })
            except Exception as e:
                logger.error(f"Error initializing exchange connection: {str(e)}")
                self.exchange = None
                self.use_historical_funding = False
    
    def get_fee_rate(self, is_taker=True, order_type='market'):
        """
        Get the applicable fee rate based on order type and volume tier.
        
        Args:
            is_taker: Whether the order is taking liquidity
            order_type: 'market', 'limit', or 'maker_limit' (true maker)
            
        Returns:
            Fee rate as a decimal (positive for fee, negative for rebate)
        """
        # True maker orders that provide liquidity may get a rebate
        if order_type == 'maker_limit' and not is_taker:
            return MAKER_REBATE
            
        # Standard market or immediate-execution limit orders
        if is_taker or order_type == 'market':
            return TAKER_FEE
        else:
            return MAKER_FEE
    
    def fetch_funding_history(self, symbol, start_time, end_time):
        """
        Fetch historical funding rates for a symbol.
        
        Args:
            symbol: Trading symbol
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with timestamp and funding rate
        """
        if not self.use_historical_funding or self.exchange is None:
            # Return dummy data with default funding rate
            dummy_times = pd.date_range(start=start_time, end=end_time, freq='8H')
            return pd.DataFrame({
                'timestamp': dummy_times,
                'funding_rate': [DEFAULT_FUNDING_RATE] * len(dummy_times)
            })
        
        try:
            # Create cache key
            cache_key = f"{symbol}_{start_time.strftime('%Y%m%d')}_{end_time.strftime('%Y%m%d')}"
            
            # Return cached data if available
            if cache_key in self.funding_history:
                return self.funding_history[cache_key]
            
            # Convert symbol format if needed (e.g., BTC/USDT -> BTCUSDT)
            exchange_symbol = symbol.replace('/', '')
            
            # Fetch funding rate history from exchange
            funding_history = self.exchange.fetch_funding_rate_history(
                exchange_symbol, 
                since=int(start_time.timestamp() * 1000),
                limit=1000  # Adjust as needed
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(funding_history)
            
            # Standardize column names
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:
                df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
            
            if 'fundingRate' in df.columns:
                df['funding_rate'] = df['fundingRate']
            
            # Select relevant columns
            df = df[['timestamp', 'funding_rate']]
            
            # Cache the result
            self.funding_history[cache_key] = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching funding history for {symbol}: {str(e)}")
            
            # Return dummy data with default funding rate
            dummy_times = pd.date_range(start=start_time, end=end_time, freq='8H')
            return pd.DataFrame({
                'timestamp': dummy_times,
                'funding_rate': [DEFAULT_FUNDING_RATE] * len(dummy_times)
            })
    
    def get_current_funding_rate(self, symbol):
        """
        Get the current funding rate for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current funding rate as a decimal
        """
        if not self.use_historical_funding or self.exchange is None:
            return DEFAULT_FUNDING_RATE
        
        try:
            # Convert symbol format if needed
            exchange_symbol = symbol.replace('/', '')
            
            # Fetch current funding rate
            ticker = self.exchange.fetch_ticker(exchange_symbol)
            
            if 'fundingRate' in ticker:
                return ticker['fundingRate']
            else:
                return DEFAULT_FUNDING_RATE
                
        except Exception as e:
            logger.error(f"Error fetching current funding rate for {symbol}: {str(e)}")
            return DEFAULT_FUNDING_RATE
    
    def calculate_funding_cost(self, position_size, funding_rate, position_value):
        """
        Calculate funding payment/charge for a position.
        
        Args:
            position_size: Position size in base currency (negative for shorts)
            funding_rate: Current funding rate
            position_value: USD value of position
            
        Returns:
            Funding cost in USD (positive means you pay, negative means you receive)
        """
        # Long positions pay positive funding, receive negative funding
        # Short positions pay negative funding, receive positive funding
        payment = position_value * funding_rate * np.sign(position_size)
        
        return payment
    
    def apply_slippage(self, price, order_type, side):
        """
        Apply realistic slippage to order execution price.
        
        Args:
            price: Base execution price
            order_type: 'limit' or 'market'
            side: 'buy' or 'sell'
            
        Returns:
            Adjusted execution price
        """
        # Determine slippage based on order type
        slippage = LIMIT_SLIPPAGE if order_type == 'limit' else MARKET_SLIPPAGE
        
        # Apply slippage based on side (buy orders get worse prices, sell orders get worse prices)
        multiplier = (1 + slippage) if side == 'buy' else (1 - slippage)
        
        # Return adjusted price
        return price * multiplier
    
    def execute_order(self, symbol, side, quantity, price=None, order_type='limit'):
        """
        Simulate order execution with fees and slippage.
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            price: Limit price (None for market orders)
            order_type: 'market', 'limit', or 'maker_limit'
            
        Returns:
            Dict with execution details including fees
        """
        # Use market price if not provided
        if price is None:
            # In a real implementation, we would fetch the current market price
            raise ValueError("Price must be provided (market orders not implemented)")
        
        # Determine order type and apply slippage
        is_market = (order_type == 'market')
        is_taker = is_market or (order_type == 'limit')  # Assume limit orders are taker by default
        
        # Apply slippage to execution price
        actual_price = self.apply_slippage(price, order_type, side)
        
        # Calculate order value
        order_value = quantity * actual_price
        
        # Apply exchange fees
        fee_rate = self.get_fee_rate(is_taker=is_taker, order_type=order_type)
        fee_amount = order_value * fee_rate
        
        # Return execution details
        return {
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'original_price': price,
            'executed_price': actual_price,
            'slippage_pct': abs(actual_price - price) / price * 100,
            'executed_quantity': quantity,
            'is_taker': is_taker,
            'fee_rate': fee_rate,
            'fee_amount': fee_amount,
            'order_value': order_value,
            'net_value': order_value + fee_amount,  # Include fee in net value
            'execution_time': datetime.now()
        }
    
    def update_positions_with_funding(self, positions, current_time):
        """
        Apply funding payments to open positions.
        Called every 8 hours in backtest, on funding timestamp in live.
        
        Args:
            positions: Dict of positions by symbol
            current_time: Current simulation time
            
        Returns:
            Updated positions dict with funding applied
        """
        funding_applied = False
        
        # Check if it's funding time (every 8 hours at 00:00, 08:00, 16:00 UTC)
        if current_time.hour % 8 == 0 and current_time.minute == 0:
            funding_applied = True
            
            for symbol, position in positions.items():
                if position['size'] != 0:  # Only apply funding to open positions
                    # Get current funding rate
                    current_funding_rate = self.get_current_funding_rate(symbol)
                    
                    # Calculate funding cost
                    position_value = abs(position['size'] * position['mark_price'])
                    funding_cost = self.calculate_funding_cost(
                        position['size'], 
                        current_funding_rate,
                        position_value
                    )
                    
                    # Apply to position P&L
                    position['realized_pnl'] -= funding_cost
                    
                    # Track funding payments
                    if 'funding_payments' not in position:
                        position['funding_payments'] = funding_cost
                    else:
                        position['funding_payments'] += funding_cost
                    
                    logger.info(f"Applied funding payment for {symbol}: {funding_cost:.2f} USD " +
                               f"(rate: {current_funding_rate*100:.4f}%)")
        
        return positions, funding_applied

# Utility function to calculate net P&L after fees and funding
def net_pnl(gross_pnl, side, qty, price, hours_held, maker_first=True):
    """
    Calculate net P&L after fees and estimated funding.
    
    Args:
        gross_pnl: Gross P&L
        side: 'buy' or 'sell'
        qty: Position size
        price: Average position price
        hours_held: Hours position was held
        maker_first: Whether entry was maker order
        
    Returns:
        Net P&L after fees and funding
    """
    # Entry fee
    entry_fee_rate = MAKER_FEE if maker_first else TAKER_FEE
    
    # Exit fee - assume taker for exits
    exit_fee_rate = TAKER_FEE
    
    # Calculate fee amounts
    total_fee_rate = entry_fee_rate + exit_fee_rate
    fee = price * abs(qty) * total_fee_rate
    
    # Estimate funding costs (number of 8-hour periods × funding rate)
    funding_periods = hours_held / 8
    funding = abs(qty) * price * DEFAULT_FUNDING_RATE * funding_periods
    
    # Calculate net P&L
    return gross_pnl - fee - funding 