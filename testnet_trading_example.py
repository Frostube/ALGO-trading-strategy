#!/usr/bin/env python3
"""
Binance Testnet Trading Example

This script demonstrates how to use the Binance Testnet
for testing algorithmic trading strategies without risking real funds.

The example implements a simple EMA crossover strategy.
"""
import os
import sys
import pandas as pd
import numpy as np
import time
from datetime import datetime
import ccxt
from testnet_api_key import TESTNET_API_KEY, TESTNET_SECRET_KEY
from src.utils.logger import setup_logger

# Initialize logging
logger = setup_logger(debug=True)

class TestnetTradingBot:
    """Simple trading bot using Binance Testnet."""
    
    def __init__(self, symbol="BTC/USDT", timeframe="15m"):
        """
        Initialize the trading bot.
        
        Args:
            symbol: Trading pair to trade
            timeframe: Candle timeframe to use
        """
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Initialize exchange with testnet configuration
        self.exchange = ccxt.binance({
            'apiKey': TESTNET_API_KEY,
            'secret': TESTNET_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
                'adjustForTimeDifference': True,
                'createMarketBuyOrderRequiresPrice': False,
                'recvWindow': 60000
            }
        })
        
        # Enable sandbox mode for testnet
        self.exchange.setSandboxMode(True)
        
        # Strategy parameters
        self.short_window = 9    # Fast EMA period
        self.long_window = 21    # Slow EMA period
        
        # Risk management parameters
        self.position_size_pct = 0.1  # 10% of available balance
        self.take_profit_pct = 0.03   # 3% profit target
        self.stop_loss_pct = 0.02     # 2% stop loss
        
        # Track position
        self.in_position = False
        self.entry_price = 0
        
        # Initialize data
        self.data = pd.DataFrame()
        
        # Connect and verify access
        try:
            logger.info("Connecting to Binance Testnet...")
            self.exchange.load_markets()
            logger.info(f"Connected to Binance Testnet with {len(self.exchange.markets)} markets available")
            
            # Check if we can access public data
            ticker = self.exchange.fetch_ticker(self.symbol)
            logger.info(f"Current {self.symbol} price: ${ticker['last']:.2f}")
            
            # Attempt to fetch balance to verify API key works
            try:
                balance = self.exchange.fetch_balance()
                available_currencies = [currency for currency, amount in balance['total'].items() if amount > 0]
                logger.info(f"Account has balances in: {', '.join(available_currencies)}")
                
                # Check if we have enough balance to trade
                quote_currency = self.symbol.split('/')[1]  # USDT in BTC/USDT
                available_balance = balance['free'].get(quote_currency, 0)
                logger.info(f"Available {quote_currency} balance: {available_balance:.2f}")
                
                if available_balance < 10:
                    logger.warning(f"Low {quote_currency} balance. Consider funding testnet account.")
                
            except Exception as e:
                logger.error(f"Unable to fetch account balance: {str(e)}")
                logger.warning("Will run in public data mode only (no trading)")
        
        except Exception as e:
            logger.error(f"Error connecting to Binance Testnet: {str(e)}")
            raise
    
    def fetch_data(self, limit=100):
        """Fetch OHLCV data for analysis."""
        try:
            logger.info(f"Fetching {limit} {self.timeframe} candles for {self.symbol}...")
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate EMAs
            df['ema_short'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
            df['ema_long'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
            
            # Calculate signals
            df['signal'] = 0
            df.loc[df['ema_short'] > df['ema_long'], 'signal'] = 1  # Buy signal
            df.loc[df['ema_short'] < df['ema_long'], 'signal'] = -1  # Sell signal
            
            # Store data
            self.data = df
            
            logger.info(f"Fetched data from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise
    
    def calculate_position_size(self):
        """Calculate position size based on available balance."""
        try:
            balance = self.exchange.fetch_balance()
            quote_currency = self.symbol.split('/')[1]  # USDT in BTC/USDT
            available_balance = balance['free'].get(quote_currency, 0)
            
            # Use percentage of available balance
            amount_to_use = available_balance * self.position_size_pct
            
            # Get current price
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker['last']
            
            # Calculate amount in base currency (BTC in BTC/USDT)
            base_amount = amount_to_use / current_price
            
            # Format to appropriate precision
            # Get market info for precision
            market = self.exchange.market(self.symbol)
            amount_precision = market['precision']['amount']
            
            # Round to proper precision
            formatted_amount = float(f"{{:.{amount_precision}f}}".format(base_amount))
            
            logger.info(f"Calculated position size: {formatted_amount} (${amount_to_use:.2f})")
            return formatted_amount
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.001  # Default small amount as fallback
    
    def check_exit_conditions(self, current_price):
        """Check if take profit or stop loss conditions are met."""
        if not self.in_position or self.entry_price == 0:
            return False
        
        # Calculate % change from entry
        pct_change = (current_price - self.entry_price) / self.entry_price
        
        # Check take profit
        if pct_change >= self.take_profit_pct:
            logger.info(f"Take profit triggered: {pct_change:.2%} gain")
            return True
        
        # Check stop loss
        if pct_change <= -self.stop_loss_pct:
            logger.info(f"Stop loss triggered: {pct_change:.2%} loss")
            return True
        
        return False
    
    def execute_strategy(self):
        """Execute the EMA crossover strategy."""
        # Fetch fresh data
        self.fetch_data()
        
        if self.data.empty:
            logger.error("No data available to execute strategy")
            return
        
        # Get current signal
        current_signal = self.data['signal'].iloc[-1]
        previous_signal = self.data['signal'].iloc[-2] if len(self.data) > 1 else 0
        
        # Get current price
        current_price = self.data['close'].iloc[-1]
        
        # Check for signal change
        signal_changed = current_signal != previous_signal
        
        try:
            # If we're in a position, check for exit conditions
            if self.in_position:
                if current_signal == -1 or self.check_exit_conditions(current_price):
                    # Time to sell
                    logger.info(f"Sell signal at {current_price:.2f}")
                    
                    # Get current balance
                    balance = self.exchange.fetch_balance()
                    base_currency = self.symbol.split('/')[0]  # BTC in BTC/USDT
                    available_base = balance['free'].get(base_currency, 0)
                    
                    if available_base > 0:
                        # Place a market sell order
                        sell_amount = available_base * 0.99  # Sell 99% to avoid precision issues
                        
                        # Format to appropriate precision
                        market = self.exchange.market(self.symbol)
                        amount_precision = market['precision']['amount']
                        sell_amount = float(f"{{:.{amount_precision}f}}".format(sell_amount))
                        
                        logger.info(f"Placing market sell order for {sell_amount} {base_currency}")
                        
                        try:
                            order = self.exchange.create_order(
                                symbol=self.symbol,
                                type='market',
                                side='sell',
                                amount=sell_amount
                            )
                            
                            logger.info(f"Sell order executed: {order['id']}")
                            
                            # Calculate profit/loss
                            pnl_pct = (current_price - self.entry_price) / self.entry_price
                            logger.info(f"Position closed with {pnl_pct:.2%} {'profit' if pnl_pct > 0 else 'loss'}")
                            
                            # Reset position tracking
                            self.in_position = False
                            self.entry_price = 0
                            
                        except Exception as e:
                            logger.error(f"Error executing sell order: {str(e)}")
                    else:
                        logger.warning(f"No {base_currency} available to sell")
            
            # Check for buy signal
            elif current_signal == 1 and signal_changed:
                # Time to buy
                logger.info(f"Buy signal at {current_price:.2f}")
                
                # Calculate position size
                buy_amount = self.calculate_position_size()
                
                if buy_amount > 0:
                    try:
                        # Place a market buy order
                        logger.info(f"Placing market buy order for {buy_amount} {self.symbol.split('/')[0]}")
                        
                        order = self.exchange.create_order(
                            symbol=self.symbol,
                            type='market',
                            side='buy',
                            amount=buy_amount
                        )
                        
                        logger.info(f"Buy order executed: {order['id']}")
                        
                        # Update position tracking
                        self.in_position = True
                        self.entry_price = current_price
                        
                    except Exception as e:
                        logger.error(f"Error executing buy order: {str(e)}")
                else:
                    logger.warning("Insufficient funds to place buy order")
            
            # No signal change
            else:
                status = "LONG" if self.in_position else "FLAT"
                logger.info(f"No signal change. Current position: {status}")
                
                if self.in_position:
                    pnl_pct = (current_price - self.entry_price) / self.entry_price
                    logger.info(f"Unrealized P&L: {pnl_pct:.2%}")
        
        except Exception as e:
            logger.error(f"Error in strategy execution: {str(e)}")

def main():
    """Main function to run the trading bot."""
    try:
        bot = TestnetTradingBot(symbol="BTC/USDT", timeframe="15m")
        
        print("\nBinance Testnet Trading Bot")
        print("==========================")
        print("This bot executes a simple EMA crossover strategy on the Binance Testnet.")
        print("Press Ctrl+C to exit.")
        print()
        
        # Initial data fetch
        data = bot.fetch_data(limit=50)
        
        # Main loop
        iteration = 1
        while True:
            print(f"\nIteration {iteration} - {datetime.now()}")
            print("-" * 40)
            
            # Execute strategy
            bot.execute_strategy()
            
            # Wait before next iteration (15 seconds)
            print(f"Waiting 15 seconds before next update...")
            time.sleep(15)
            iteration += 1
            
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
        return 0
    except Exception as e:
        logger.error(f"Error running bot: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 