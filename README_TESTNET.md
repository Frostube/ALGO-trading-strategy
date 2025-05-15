# Using Binance Testnet with Trading Bots

This document explains how to integrate Binance Testnet with your algorithmic trading bot.

## Overview

Binance Testnet allows you to test your trading bot without risking real funds. It mimics the real Binance exchange but uses test tokens instead. This is ideal for:

- Testing trading strategies
- Debugging order execution logic
- Validating risk management rules
- Verifying API integration

## Files We've Created

1. **testnet_api_key.py** - Stores Binance Testnet API credentials
2. **testnet_connector.py** - A connector class for Binance Testnet API
3. **test_data_fetcher.py** - Tests fetching historical data from Binance Testnet
4. **testnet_trading_example.py** - Example trading bot using Binance Testnet

## Setup Instructions

### 1. Register for Binance Testnet

Visit [Binance Testnet](https://testnet.binance.vision/) and register for an account.

### 2. Get API Keys

Generate API keys from the Binance Testnet dashboard. You'll receive:
- API Key
- Secret Key

### 3. Set Up Your Environment

You can store your API keys:

a) In a `.env` file (preferred, don't commit to Git):
```
BINANCE_TESTNET_API_KEY=your_api_key
BINANCE_TESTNET_SECRET_KEY=your_secret_key
```

b) In the `testnet_api_key.py` file (less secure but simpler):
```python
TESTNET_API_KEY = "your_api_key"
TESTNET_SECRET_KEY = "your_secret_key"
```

## Usage

### Fetching Historical Data

The `DataFetcher` class has been updated to support testnet mode:

```python
from src.data.fetcher import DataFetcher

# Initialize with testnet mode
fetcher = DataFetcher(use_testnet=True)

# Fetch historical data from testnet
btc_data = fetcher.fetch_historical_data(
    symbol="BTC/USDT",
    timeframe="4h",
    days=7
)

print(f"Retrieved {len(btc_data)} candles from Binance Testnet")
```

### Setting Up CCXT with Testnet

When initializing CCXT, use the `setSandboxMode` method to enable testnet:

```python
import ccxt
from testnet_api_key import TESTNET_API_KEY, TESTNET_SECRET_KEY

# Initialize exchange with testnet configuration
exchange = ccxt.binance({
    'apiKey': TESTNET_API_KEY,
    'secret': TESTNET_SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',  # Or 'future' for futures trading
        'adjustForTimeDifference': True,  # Handle time synchronization
        'recvWindow': 60000  # Use larger recvWindow to avoid timestamp issues
    }
})

# Enable sandbox mode for testnet
exchange.setSandboxMode(True)
```

### Running the Example Bot

We've created an example trading bot using a simple EMA crossover strategy:

```bash
python testnet_trading_example.py
```

This bot:
- Connects to Binance Testnet
- Fetches OHLCV data for BTC/USDT
- Calculates EMA indicators
- Generates buy/sell signals
- Executes trades with risk management

## Troubleshooting

### Common Issues and Solutions

1. **Signature Validation Error**:
   ```
   Error: binance {"code":-1022,"msg":"Signature for this request is not valid."}
   ```
   
   Solution:
   - Double check your API key and secret
   - Create fresh keys on the Binance Testnet website
   - Make sure you're enabling sandbox mode with `exchange.setSandboxMode(True)`

2. **Timestamp Errors**:
   ```
   Error: binance {"code":-1021,"msg":"Timestamp for this request was 1000ms ahead of the server's time."}
   ```
   
   Solution:
   - Use the `adjustForTimeDifference` option in CCXT
   - Add `'recvWindow': 60000` to your options

3. **Insufficient Balance**:

   Solution:
   - Visit the Binance Testnet website
   - Use the faucet to request more test funds
   - Try trading pairs with lower value coins

## Notes on Public vs Private Endpoints

In our testing, we've found:

- **Public Endpoints**: Always work with Binance Testnet (market data, tickers, orderbooks)
- **Private Endpoints**: May require generating API keys directly from the Binance Testnet site

If you only need market data, you can still use Testnet without authentication by skipping the private endpoints.

## Next Steps

1. Read the [Binance Testnet Documentation](https://testnet.binance.vision/hc/en-us)
2. Explore the CCXT documentation for [sandbox mode](https://github.com/ccxt/ccxt/wiki/Manual#sandbox-mode)
3. Try implementing your own strategies using our example template
4. Compare performance between testnet and backtest results

Remember that testnet data may sometimes differ from real-market data, so use it primarily for testing functionality rather than optimizing strategy parameters. 