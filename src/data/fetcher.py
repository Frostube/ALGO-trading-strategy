import os
import asyncio
import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from collections import defaultdict

from src.config import (
    EXCHANGE, SYMBOL, TIMEFRAME, HIGHER_TIMEFRAME, BINANCE_API_KEY, 
    BINANCE_SECRET_KEY, FUTURES, HISTORICAL_DATA_DAYS
)
from src.db.models import OHLCV, init_db
from src.utils.logger import logger

class DataFetcher:
    """Fetch and manage historical and real-time market data."""
    
    def __init__(self, use_testnet=False):
        # Initialize exchange
        self.exchange_id = EXCHANGE
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self.use_testnet = use_testnet
        
        # Set up REST exchange for historical data
        exchange_config = {
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if FUTURES else 'spot'
            }
        }
        
        # Configure testnet if requested
        if use_testnet:
            exchange_config['options']['test'] = True  # Enable test mode
            
        self.rest_exchange = getattr(ccxt, self.exchange_id)(exchange_config)
        
        # Configure testnet URLs if requested
        if use_testnet:
            self.rest_exchange.urls['api'] = {
                'public': 'https://testnet.binance.vision/api',
                'private': 'https://testnet.binance.vision/api',
                'v3': 'https://testnet.binance.vision/api/v3',
                'v1': 'https://testnet.binance.vision/api/v1'
            }
            # Disable endpoints not available in testnet
            self.rest_exchange.options['recvWindow'] = 5000
            logger.info("Configured for Binance Testnet API")
        
        # Initialize database session
        self.db_session = init_db()
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize data cache for multiple timeframes
        self.data_cache = defaultdict(pd.DataFrame)
        
    async def initialize_ws_exchange(self):
        """Initialize WebSocket exchange for real-time data."""
        self.ws_exchange = getattr(ccxtpro, self.exchange_id)({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if FUTURES else 'spot'
            }
        })
        
        # Configure testnet if requested
        if self.use_testnet:
            self.ws_exchange.urls['api'] = {
                'public': 'https://testnet.binance.vision/api',
                'private': 'https://testnet.binance.vision/api'
            }
        
    def fetch_historical_data(self, days=None, timeframe=None):
        """
        Fetch historical data for the configured symbol and timeframe.
        
        Args:
            days: Number of days of historical data to fetch
            timeframe: Specific timeframe to fetch (default: use instance timeframe)
        
        Returns:
            DataFrame with historical OHLCV data
        """
        days = days or HISTORICAL_DATA_DAYS
        tf = timeframe or self.timeframe
        
        logger.info(f"Fetching {days} days of historical data for {self.symbol} ({tf})")
        
        # Use different approach for testnet
        if self.use_testnet:
            try:
                # For testnet, try to use a simple endpoint that's more likely to work
                logger.info("Attempting to fetch data from Binance Testnet...")
                batch = self.rest_exchange.fetch_ohlcv(
                    symbol=self.symbol,
                    timeframe=tf,
                    limit=500  # Request a reasonable amount
                )
                
                if batch and len(batch) > 0:
                    ohlcv_data = batch
                    logger.info(f"Successfully fetched {len(batch)} candles from testnet")
                else:
                    logger.warning(f"No data available on testnet for {self.symbol} with {tf}")
                    ohlcv_data = self._generate_mock_data(days, tf)
            except Exception as e:
                logger.warning(f"Testnet fetch error: {str(e)}")
                # Generate mock data as a fallback
                logger.info("Generating mock data for testing purposes")
                ohlcv_data = self._generate_mock_data(days, tf)
        else:
            # Standard approach for production
            try:
                # Calculate start time
                since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
                
                # Fetch OHLCV data
                ohlcv_data = []
                
                # Binance API has a limit on number of candles per request, so fetch in batches
                current_since = since
                while True:
                    logger.debug(f"Fetching batch from {datetime.fromtimestamp(current_since/1000)}")
                    batch = self.rest_exchange.fetch_ohlcv(
                        symbol=self.symbol,
                        timeframe=tf,
                        since=current_since,
                        limit=1000  # Binance limit
                    )
                    
                    if not batch:
                        break
                    
                    ohlcv_data.extend(batch)
                    
                    # If we got less than 1000 candles, we've reached the end
                    if len(batch) < 1000:
                        break
                    
                    # Update since for next batch (last candle timestamp + 1 minute)
                    current_since = batch[-1][0] + 60000
                    
                    # Avoid rate limits
                    self.rest_exchange.sleep(1)
            
            except Exception as e:
                logger.error(f"Error fetching historical data for {tf}: {str(e)}")
                return pd.DataFrame()  # Return empty DataFrame
        
        # Check if we got any data
        if not ohlcv_data:
            logger.error(f"No data retrieved for {self.symbol} with timeframe {tf}")
            return pd.DataFrame()  # Return empty DataFrame
        
        # Store the data in database with timeframe
        try:
            self._store_ohlcv_data(ohlcv_data, tf)
        except Exception as e:
            logger.warning(f"Could not store data in database: {str(e)}")
        
        # Convert to DataFrame and cache for this timeframe
        self.data_cache[tf] = self._convert_to_dataframe(ohlcv_data)
        
        logger.info(f"Fetched {len(ohlcv_data)} historical candles for timeframe {tf}")
        return self.data_cache[tf]
    
    def _store_ohlcv_data(self, ohlcv_data, timeframe):
        """Store OHLCV data in the database."""
        try:
            # Batch insert for better performance
            batch_size = 1000
            for i in range(0, len(ohlcv_data), batch_size):
                batch = ohlcv_data[i:i+batch_size]
                ohlcv_objects = [
                    OHLCV.from_ccxt(self.symbol, candle, timeframe=timeframe) 
                    for candle in batch
                ]
                self.db_session.bulk_save_objects(ohlcv_objects)
            
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing OHLCV data: {str(e)}")
            raise
    
    def _convert_to_dataframe(self, ohlcv_data):
        """Convert OHLCV data to a pandas DataFrame."""
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    
    async def watch_ohlcv(self, timeframe=None):
        """
        Watch OHLCV data in real-time using WebSocket.
        
        Args:
            timeframe: Specific timeframe to watch (default: use instance timeframe)
        
        Yields:
            New candle data as a DataFrame
        """
        tf = timeframe or self.timeframe
        
        if not hasattr(self, 'ws_exchange'):
            await self.initialize_ws_exchange()
        
        logger.info(f"Starting real-time OHLCV watching for {self.symbol} ({tf})")
        
        try:
            while True:
                ohlcv = await self.ws_exchange.watch_ohlcv(self.symbol, tf)
                
                # Process the new candle
                yield self._process_new_candle(ohlcv, tf)
                
                # If we're watching the lower timeframe, periodically update higher timeframe
                if tf == self.timeframe and HIGHER_TIMEFRAME:
                    # Update higher timeframe every few iterations based on the ratio
                    # between timeframes (e.g., update 5m data every 5 1m candles)
                    now = datetime.now()
                    # Simple approach: update on the minute where higher timeframe would complete
                    if now.minute % int(HIGHER_TIMEFRAME[:-1]) == 0 and now.second < 5:
                        await self._update_higher_timeframe()
                
        except Exception as e:
            logger.error(f"Error watching OHLCV for {tf}: {str(e)}")
            # Close the connection
            if hasattr(self, 'ws_exchange'):
                await self.ws_exchange.close()
            raise
    
    async def _update_higher_timeframe(self):
        """Update the higher timeframe data via REST API."""
        try:
            # Fetch just the last few candles of the higher timeframe
            candles = self.rest_exchange.fetch_ohlcv(
                symbol=self.symbol,
                timeframe=HIGHER_TIMEFRAME,
                limit=20  # Last 20 candles should be enough
            )
            
            if candles:
                # Convert to DataFrame
                df = self._convert_to_dataframe(candles)
                
                # Update cache
                if self.data_cache[HIGHER_TIMEFRAME].empty:
                    self.data_cache[HIGHER_TIMEFRAME] = df
                else:
                    # Merge with existing data, keeping only unique timestamps
                    combined = pd.concat([self.data_cache[HIGHER_TIMEFRAME], df])
                    self.data_cache[HIGHER_TIMEFRAME] = combined[~combined.index.duplicated(keep='last')]
                
                logger.debug(f"Updated higher timeframe data ({HIGHER_TIMEFRAME})")
                
                # Store in database
                self._store_ohlcv_data(candles, HIGHER_TIMEFRAME)
        
        except Exception as e:
            logger.error(f"Error updating higher timeframe: {str(e)}")
    
    def _process_new_candle(self, ohlcv, timeframe):
        """Process a new candle from the WebSocket feed."""
        # Convert the candle to a DataFrame row
        candle_df = pd.DataFrame([ohlcv], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'], unit='ms')
        candle_df.set_index('timestamp', inplace=True)
        
        # Store in database
        candle_obj = OHLCV.from_ccxt(self.symbol, ohlcv, timeframe=timeframe)
        try:
            self.db_session.add(candle_obj)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing new candle: {str(e)}")
        
        # Update the cache for this timeframe
        if self.data_cache[timeframe].empty:
            self.data_cache[timeframe] = candle_df
        else:
            # If this timestamp already exists, replace it, otherwise append
            if candle_df.index[0] in self.data_cache[timeframe].index:
                self.data_cache[timeframe].loc[candle_df.index[0]] = candle_df.iloc[0]
            else:
                self.data_cache[timeframe] = pd.concat([self.data_cache[timeframe], candle_df])
        
        # Return the processed candle
        return candle_df
    
    def get_latest_data(self, n_periods=100, timeframe=None):
        """
        Get the latest n periods of data from cache.
        
        Args:
            n_periods: Number of periods to return
            timeframe: Specific timeframe to get data for (default: use instance timeframe)
            
        Returns:
            DataFrame with the latest n periods of data
        """
        tf = timeframe or self.timeframe
        
        if self.data_cache[tf].empty:
            self.fetch_historical_data(timeframe=tf)
        
        return self.data_cache[tf].tail(n_periods)
    
    def close(self):
        """Close connections and release resources."""
        self.db_session.close()
        
    async def close_async(self):
        """Close async connections."""
        if hasattr(self, 'ws_exchange'):
            await self.ws_exchange.close()
    
    def _generate_mock_data(self, days, timeframe):
        """Generate mock OHLCV data for testing when testnet fails."""
        # Current time
        now = datetime.now()
        
        # Calculate number of candles based on timeframe
        minutes_per_candle = int(timeframe[:-1]) if timeframe.endswith('m') else \
                            int(timeframe[:-1]) * 60 if timeframe.endswith('h') else \
                            int(timeframe[:-1]) * 60 * 24  # days
        
        total_minutes = days * 24 * 60
        num_candles = total_minutes // minutes_per_candle
        
        # Limit to a reasonable number
        num_candles = min(num_candles, 5000)
        
        # Generate mock data
        mock_data = []
        base_price = 50000  # Starting BTC price
        volume = 10  # Base volume
        
        for i in range(num_candles):
            # Calculate timestamp for this candle
            timestamp = int((now - timedelta(minutes=minutes_per_candle * (num_candles - i))).timestamp() * 1000)
            
            # Add some randomness to price movements
            price_change = (np.random.random() - 0.5) * 100
            base_price += price_change
            
            # Generate OHLCV values with some randomness
            open_price = base_price
            close_price = base_price + (np.random.random() - 0.5) * 50
            high_price = max(open_price, close_price) + np.random.random() * 30
            low_price = min(open_price, close_price) - np.random.random() * 30
            
            # Random volume with occasional spikes
            candle_volume = volume * (1 + np.random.random())
            if np.random.random() < 0.1:  # 10% chance of volume spike
                candle_volume *= 3
            
            # Add the candle to our mock data
            mock_data.append([
                timestamp,
                float(open_price),
                float(high_price),
                float(low_price),
                float(close_price),
                float(candle_volume)
            ])
        
        logger.info(f"Generated {len(mock_data)} mock candles for testing")
        return mock_data 