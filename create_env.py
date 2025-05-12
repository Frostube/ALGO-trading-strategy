#!/usr/bin/env python3
# Script to create a .env file with Binance testnet API keys

with open('.env', 'w') as f:
    f.write("""# API Keys
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
OPENAI_API_KEY=

# Slack integration
SLACK_BOT_TOKEN=
SLACK_CHANNEL=

# Logging
LOG_LEVEL=INFO
""") 