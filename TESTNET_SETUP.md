# Setting Up Binance Testnet for Your Trading Bot

This guide will help you set up Binance testnet API keys for your trading bot without needing KYC verification.

## Step 1: Get Binance Testnet API Keys

1. Visit [Binance Testnet](https://testnet.binance.vision/)
2. Create an account or log in with GitHub
3. Navigate to the "Generate HMAC_SHA256 Key" section
4. Click "Generate" to create a new API key and secret
5. Save both the API key and Secret key

## Step 2: Update Your Environment Variables

1. Open the `.env` file in the project root
2. Replace the placeholder values with your actual keys:
   ```
   BINANCE_API_KEY=your_actual_testnet_api_key
   BINANCE_SECRET_KEY=your_actual_testnet_secret_key
   ```
3. Save the file

## Step 3: Test Your Application

1. The dashboard has been configured to use the testnet by default
2. Make sure the "Use Binance Testnet" checkbox is selected in the sidebar
3. Run the dashboard using:
   ```
   python -m streamlit run src/dashboard/backtest_dashboard.py
   ```

## Important Notes

- Testnet accounts have unlimited paper funds
- The testnet environment behaves like real Binance but with simulated market data
- API keys expire periodically; generate new ones if you encounter authentication errors
- No real trading occurs in testnet mode
