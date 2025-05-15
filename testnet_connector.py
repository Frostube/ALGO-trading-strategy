#!/usr/bin/env python3
"""
Binance Testnet Connector

This module provides a connector class for interacting with the
Binance Testnet API using ccxt. It handles API key management,
sandbox mode configuration, and provides utility methods for trading.
"""
import os
import ccxt
from src.utils.logger import logger

class BinanceTestnet:
    """
    Connector for Binance Testnet API.
    
    This class provides methods for interacting with the Binance Testnet
    for testing trading strategies without using real funds.
    """
    
    def __init__(self, api_key=None, api_secret=None):
        """
        Initialize the Binance Testnet connector.
        
        Args:
            api_key (str, optional): Binance Testnet API key.
            api_secret (str, optional): Binance Testnet API secret.
        """
        # Default test API credentials
        self.api_key = api_key or os.getenv('BINANCE_TESTNET_API_KEY') or "nyRjeAYWuNencCkqRNcPnhkJ4LlJPlP99IZxFdrg8TdVDIYpmbmJF0GYL5NvXOXd"
        self.api_secret = api_secret or os.getenv('BINANCE_TESTNET_SECRET_KEY') or "go8C76Hn7CoOPadXE7Z1UpDdAW3d2sG0A5f7QADrmBoQUMiBc3tkG1"
        
        # Initialize the exchange
        self.exchange = ccxt.binance({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
                'defaultType': 'spot',  # Default to spot trading
                'recvWindow': 60000,    # Use larger recvWindow to avoid timestamp issues
            }
        })
        
        # Enable sandbox mode for testnet
        self.exchange.setSandboxMode(True)
        
        # Try to load markets right away to check connectivity
        try:
            self.exchange.load_markets()
            logger.info("Successfully connected to Binance Testnet")
        except Exception as e:
            logger.error(f"Failed to connect to Binance Testnet: {str(e)}")
            raise
    
    def get_balance(self):
        """Get account balance."""
        try:
            return self.exchange.fetch_balance()
        except Exception as e:
            logger.error(f"Error fetching balance: {str(e)}")
            raise
    
    def get_ticker(self, symbol='BTC/USDT'):
        """Get current ticker data for a symbol."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {str(e)}")
            raise
    
    def place_market_order(self, symbol, side, amount):
        """
        Place a market order.
        
        Args:
            symbol (str): Trading pair, e.g. 'BTC/USDT'
            side (str): 'buy' or 'sell'
            amount (float): Quantity to trade
            
        Returns:
            dict: Order details
        """
        try:
            return self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
        except Exception as e:
            logger.error(f"Error placing market {side} order for {amount} {symbol}: {str(e)}")
            raise
    
    def place_limit_order(self, symbol, side, amount, price):
        """
        Place a limit order.
        
        Args:
            symbol (str): Trading pair, e.g. 'BTC/USDT'
            side (str): 'buy' or 'sell'
            amount (float): Quantity to trade
            price (float): Limit price
            
        Returns:
            dict: Order details
        """
        try:
            return self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price
            )
        except Exception as e:
            logger.error(f"Error placing limit {side} order for {amount} {symbol} @ {price}: {str(e)}")
            raise
    
    def get_open_orders(self, symbol=None):
        """Get open orders, optionally filtered by symbol."""
        try:
            return self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.error(f"Error fetching open orders: {str(e)}")
            raise
    
    def cancel_order(self, order_id, symbol):
        """Cancel an open order."""
        try:
            return self.exchange.cancel_order(order_id, symbol)
        except Exception as e:
            logger.error(f"Error canceling order {order_id} for {symbol}: {str(e)}")
            raise
    
    def get_ohlcv(self, symbol='BTC/USDT', timeframe='1h', limit=100):
        """Get OHLCV candlestick data."""
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            logger.error(f"Error fetching OHLCV data for {symbol}: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    try:
        # Create connector
        connector = BinanceTestnet()
        
        # Get and print ticker
        ticker = connector.get_ticker()
        print(f"BTC/USDT Price: ${ticker['last']:.2f}")
        
        # Try to get balance
        balance = connector.get_balance()
        print("\nAccount Balance:")
        for currency, data in balance['total'].items():
            if data > 0:
                print(f"{currency}: {data}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc() 