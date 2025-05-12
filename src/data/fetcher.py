import os
import asyncio
import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from src.config import (
    EXCHANGE, SYMBOL, TIMEFRAME, BINANCE_API_KEY, 
    BINANCE_SECRET_KEY, FUTURES, HISTORICAL_DATA_DAYS
)
from src.db.models import OHLCV, init_db
from src.utils.logger import logger

class DataFetcher:
    """Fetch and manage historical and real-time market data."""
    
    def __init__(self):
        # Initialize exchange
        self.exchange_id = EXCHANGE
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        
        # Set up REST exchange for historical data
        self.rest_exchange = getattr(ccxt, self.exchange_id)({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if FUTURES else 'spot'
            }
        })
        
        # Initialize database session
        self.db_session = init_db()
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize data cache
        self.data_cache = pd.DataFrame()
        
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
        
    def fetch_historical_data(self, days=None):
        """Fetch historical data for the configured symbol and timeframe."""
        days = days or HISTORICAL_DATA_DAYS
        logger.info(f"Fetching {days} days of historical data for {self.symbol}")
        
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
                    timeframe=self.timeframe,
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
            
            # Store the data in database
            self._store_ohlcv_data(ohlcv_data)
            
            # Convert to DataFrame and cache
            self.data_cache = self._convert_to_dataframe(ohlcv_data)
            
            logger.info(f"Fetched {len(ohlcv_data)} historical candles")
            return self.data_cache
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
    
    def _store_ohlcv_data(self, ohlcv_data):
        """Store OHLCV data in the database."""
        try:
            # Batch insert for better performance
            batch_size = 1000
            for i in range(0, len(ohlcv_data), batch_size):
                batch = ohlcv_data[i:i+batch_size]
                ohlcv_objects = [OHLCV.from_ccxt(self.symbol, candle) for candle in batch]
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
    
    async def watch_ohlcv(self):
        """Watch OHLCV data in real-time using WebSocket."""
        if not hasattr(self, 'ws_exchange'):
            await self.initialize_ws_exchange()
        
        logger.info(f"Starting real-time OHLCV watching for {self.symbol}")
        
        try:
            while True:
                ohlcv = await self.ws_exchange.watch_ohlcv(self.symbol, self.timeframe)
                
                # Process the new candle
                yield self._process_new_candle(ohlcv)
                
        except Exception as e:
            logger.error(f"Error watching OHLCV: {str(e)}")
            # Close the connection
            if hasattr(self, 'ws_exchange'):
                await self.ws_exchange.close()
            raise
    
    def _process_new_candle(self, ohlcv):
        """Process a new candle from the WebSocket feed."""
        # Convert the candle to a DataFrame row
        candle_df = pd.DataFrame([ohlcv], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        candle_df['timestamp'] = pd.to_datetime(candle_df['timestamp'], unit='ms')
        candle_df.set_index('timestamp', inplace=True)
        
        # Store in database
        candle_obj = OHLCV.from_ccxt(self.symbol, ohlcv)
        try:
            self.db_session.add(candle_obj)
            self.db_session.commit()
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing new candle: {str(e)}")
        
        # Update the cache
        self.data_cache = pd.concat([self.data_cache, candle_df])
        
        # Return the processed candle
        return candle_df
    
    def get_latest_data(self, n_periods=100):
        """Get the latest n periods of data from cache."""
        if self.data_cache.empty:
            self.fetch_historical_data()
        
        return self.data_cache.tail(n_periods)
    
    def close(self):
        """Close connections and release resources."""
        self.db_session.close()
        
    async def close_async(self):
        """Close async connections."""
        if hasattr(self, 'ws_exchange'):
            await self.ws_exchange.close() 