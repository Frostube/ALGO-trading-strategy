#!/usr/bin/env python3
"""
Test Binance Testnet Trading Operations

This script tests basic trading operations on Binance Testnet
using CCXT with the API credentials from the environment.
"""
import os
import sys
from time import sleep
import ccxt
from dotenv import load_dotenv
from testnet_api_key import TESTNET_API_KEY, TESTNET_SECRET_KEY
from src.utils.logger import setup_logger

def main():
    # Set up logging
    logger = setup_logger(debug=True)
    
    try:
        # Initialize Binance with testnet configuration
        logger.info("Initializing Binance Testnet connection...")
        
        exchange = ccxt.binance({
            "apiKey": TESTNET_API_KEY,
            "secret": TESTNET_SECRET_KEY,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",  # Using spot trading for simplicity
                "adjustForTimeDifference": True,
                "recvWindow": 60000
            }
        })
        
        # Enable sandbox/testnet mode
        exchange.setSandboxMode(True)
        
        # Fetch account balance
        logger.info("Fetching account balance...")
        balance = exchange.fetch_balance()
        
        # Display non-zero balances
        print("\n=== Testnet Account Balance ===")
        for currency, data in balance['total'].items():
            if data > 0:
                print(f"{currency}: {data}")
        
        # Get current market price for BTC/USDT
        logger.info("Fetching current BTC/USDT price...")
        ticker = exchange.fetch_ticker('BTC/USDT')
        current_price = ticker['last']
        print(f"\nCurrent BTC/USDT price: ${current_price:.2f}")
        
        # Create a market buy order
        symbol = 'BTC/USDT'
        
        # Calculate amount based on USDT balance (use 10% of USDT balance)
        usdt_balance = balance['total'].get('USDT', 0)
        
        if usdt_balance < 10:
            print("Insufficient USDT balance for trading. Please fund your testnet account.")
            return 1
        
        # Use 10% of USDT balance
        usdt_to_use = usdt_balance * 0.1
        
        # Calculate BTC amount based on current price (with 5% buffer)
        btc_amount = (usdt_to_use / current_price) * 0.95
        
        # Binance minimum order size for BTC is 0.000001 BTC
        btc_amount = max(btc_amount, 0.000001)
        
        # Format to 6 decimal places (Binance BTC precision)
        btc_amount = float(f"{btc_amount:.6f}")
        
        print(f"\nPlacing market buy order for {btc_amount} BTC (~${usdt_to_use:.2f})")
        
        # Place a market buy order
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side='buy',
            amount=btc_amount
        )
        
        logger.info(f"Market buy order placed: {order['id']}")
        print(f"Order executed: {order['id']}")
        print(f"Status: {order['status']}")
        
        # Wait a moment for the order to be processed
        sleep(2)
        
        # Check updated balance
        balance_after = exchange.fetch_balance()
        btc_balance = balance_after['total'].get('BTC', 0)
        usdt_balance_after = balance_after['total'].get('USDT', 0)
        
        print(f"\nUpdated balances after buy:")
        print(f"BTC: {btc_balance}")
        print(f"USDT: {usdt_balance_after}")
        
        # Wait a moment, then place a limit sell order at 5% above current price
        sleep(1)
        
        sell_price = current_price * 1.05
        
        # Always sell slightly less than we have to avoid precision issues
        sell_amount = btc_balance * 0.95
        sell_amount = float(f"{sell_amount:.6f}")
        
        print(f"\nPlacing limit sell order for {sell_amount} BTC at ${sell_price:.2f}")
        
        # Place a limit sell order
        sell_order = exchange.create_order(
            symbol=symbol,
            type='limit',
            side='sell',
            amount=sell_amount,
            price=sell_price
        )
        
        logger.info(f"Limit sell order placed: {sell_order['id']}")
        print(f"Sell order placed: {sell_order['id']}")
        print(f"Status: {sell_order['status']}")
        
        # Check open orders
        sleep(1)
        open_orders = exchange.fetch_open_orders(symbol)
        
        print(f"\nOpen orders: {len(open_orders)}")
        for order in open_orders:
            print(f"ID: {order['id']}, Type: {order['type']}, Side: {order['side']}, "
                  f"Amount: {order['amount']}, Price: {order['price']}")
        
        print("\nTest completed successfully! You can log into the Binance Testnet website to see your orders.")
        print("Visit: https://testnet.binance.vision/")
        
        return 0
        
    except ccxt.NetworkError as e:
        logger.error(f"Network error: {str(e)}")
        return 1
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error: {str(e)}")
        return 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 