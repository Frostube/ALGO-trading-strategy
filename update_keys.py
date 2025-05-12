#!/usr/bin/env python3
# Script to update .env with Binance testnet API keys

import os

# Define the API keys
testnet_api_key = "Gpm4scWfI6msBE7M2LrQq3FaMBNX3cVkpP1A6EbfTDR7k3esbTr0DQaTcmEnEABG"
testnet_secret_key = "8BKj0vgOOMy8VUqtqB0AAB0Q1X2SWmGOcHxqb98LfQxRrrLar5e1hpyoUkTZDPIw"

# Create or update .env file
env_content = f"""
# API Keys
BINANCE_API_KEY={testnet_api_key}
BINANCE_SECRET_KEY={testnet_secret_key}

# OpenAI key
OPENAI_API_KEY=

# Slack integration (optional)
SLACK_BOT_TOKEN=
SLACK_CHANNEL=
"""

# Write to .env file
with open('.env', 'w') as f:
    f.write(env_content)

print("Updated .env file with Binance Testnet API keys") 