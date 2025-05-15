# Binance Testnet Guide

This guide explains how to use the Binance Testnet in your algorithmic trading bot.

## What is Binance Testnet?

Binance Testnet is a sandbox environment that mimics the real Binance exchange without using real funds. It's perfect for testing trading strategies, order execution, and API integration.

## Setting Up

### 1. Register for a Testnet Account

Visit [Binance Testnet](https://testnet.binance.vision/) and sign up for an account.

### 2. Get API Keys

After logging in, generate API keys from the dashboard. You'll receive:
- API Key
- Secret Key

### 3. Store Your Keys Securely

The preferred approach is to use environment variables:

```python
# In .env file (add to .gitignore)
BINANCE_TESTNET_API_KEY=your_api_key_here
BINANCE_TESTNET_SECRET_KEY=your_secret_key_here
```

## Using the Testnet in Your Code

We've created a `BinanceTestnet` connector class to simplify testnet usage:

```python
from testnet_connector import BinanceTestnet

# Initialize with your API keys
connector = BinanceTestnet(
    api_key="your_api_key",
    api_secret="your_secret_key"
)

# Get current Bitcoin price
ticker = connector.get_ticker("BTC/USDT")
print(f"Current BTC price: ${ticker['last']}")

# Place a test market order
order = connector.place_market_order(
    symbol="BTC/USDT",
    side="buy",
    amount=0.001
)
```

## Using With Data Fetcher

Our `DataFetcher` class has been updated to support testnet mode:

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

## Testing Connectivity

Run the test script to verify your connection to the Binance Testnet:

```bash
python test_data_fetcher.py
```

## Common Errors and Solutions

1. **Signature is not valid**: This usually indicates an issue with your API keys or timestamp. Make sure you're using the correct keys for the testnet.

2. **IP not allowed**: You may need to whitelist your IP in the Binance Testnet dashboard.

3. **Insufficient balance**: Testnet accounts come with some test funds, but you may need to request more from the faucet on the testnet website.

## Next Steps

1. Follow the examples in the test scripts to see how to place orders and manage trades
2. Update your trading bot to toggle between live and testnet mode using a configuration parameter
3. Run full strategy backtests on testnet data to validate your models

Remember that testnet data may sometimes differ from the real market, so use it for testing functionality rather than optimizing strategy parameters. 