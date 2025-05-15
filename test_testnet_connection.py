#!/usr/bin/env python3
"""
Test Binance Testnet Connection

This script tests the connection to Binance Testnet and demonstrates
how to fetch public data from the exchange.
"""
import sys
import ccxt
from src.utils.logger import setup_logger

def main():
    # Set up logging
    logger = setup_logger(debug=True)
    
    try:
        # Initialize the exchange with testnet configuration
        logger.info("Initializing Binance Testnet connection...")
        
        # Create a basic exchange instance without credentials for public endpoints
        exchange = ccxt.binance({
            "enableRateLimit": True,
        })
        
        # Enable sandbox/testnet mode
        exchange.setSandboxMode(True)
        
        # Print exchange URLs for verification
        logger.info(f"Using Binance Testnet URLs: {exchange.urls['test']}")
        
        # Test connection by fetching public data
        logger.info("Fetching exchange status...")
        status = exchange.fetch_status()
        logger.info(f"Exchange status: {status['status']}")
        
        # Fetch ticker
        logger.info("Fetching BTC/USDT ticker data...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        
        print("\n=== BTC/USDT Market Data (Testnet) ===")
        print(f"Last Price: ${ticker['last']:.2f}")
        print(f"24h High: ${ticker['high']:.2f}")
        print(f"24h Low: ${ticker['low']:.2f}")
        print(f"24h Volume: {ticker['baseVolume']:.2f} BTC")
        print(f"24h Change: {ticker['percentage']:.2f}%")
        
        # Fetch recent trades
        logger.info("Fetching recent trades...")
        trades = exchange.fetch_trades('BTC/USDT', limit=5)
        
        print("\n=== Recent BTC/USDT Trades ===")
        for trade in trades:
            ts = exchange.iso8601(trade['timestamp'])
            side = trade['side']
            price = trade['price']
            amount = trade['amount']
            print(f"{ts} | {side.upper()} | {amount} BTC @ ${price}")
        
        # Fetch order book
        logger.info("Fetching order book...")
        orderbook = exchange.fetch_order_book('BTC/USDT', limit=5)
        
        print("\n=== BTC/USDT Order Book ===")
        print("Bids:")
        for bid in orderbook['bids'][:5]:
            print(f"${bid[0]:.2f} | {bid[1]}")
            
        print("\nAsks:")
        for ask in orderbook['asks'][:5]:
            print(f"${ask[0]:.2f} | {ask[1]}")
        
        print("\nSuccessfully connected to Binance Testnet and verified public API access!")
        
        # Inform user about private endpoint requirements
        print("\nNote: To access private endpoints (balance, orders, etc.), you need to create")
        print("a Binance Testnet account and get API keys from https://testnet.binance.vision/")
        
        return 0
    
    except ccxt.NetworkError as e:
        logger.error(f"Network error connecting to Binance Testnet: {str(e)}")
        return 1
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error with Binance Testnet: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Error connecting to Binance Testnet: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 