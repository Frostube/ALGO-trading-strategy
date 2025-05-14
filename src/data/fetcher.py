import os
import asyncio
import ccxt.pro as ccxtpro
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pathlib
import time
from sqlalchemy.orm import Session
from sqlalchemy import and_, func
from collections import defaultdict

from src.config import (
    EXCHANGE, SYMBOL, TIMEFRAME, HIGHER_TIMEFRAME, BINANCE_API_KEY, 
    BINANCE_SECRET_KEY, FUTURES, HISTORICAL_DATA_DAYS
)
from src.db.models import OHLCV, init_db
from src.utils.logger import logger

def fetch_ohlcv_full(ccxt_exch, symbol, tf, since_ms, limit=1000):
    """
    Fetch complete OHLCV data by making multiple API calls for pagination.
    
    Args:
        ccxt_exch: Initialized CCXT exchange instance
        symbol: Trading pair symbol
        tf: Timeframe (e.g. '1h', '4h', '1d')
        since_ms: Start timestamp in milliseconds
        limit: Maximum number of candles per request (default 1000)
        
    Returns:
        List of OHLCV candles
    """
    out = []
    retry_count = 0
    max_retries = 5
    
    while True:
        try:
            batch = ccxt_exch.fetch_ohlcv(symbol, tf, since=since_ms, limit=limit)
            if not batch:
                break
                
            since_ms = batch[-1][0] + 1  # Next ms after last candle
            out.extend(batch)
            
            if len(batch) < limit:  # Final page
                break
                
            # Reset retry counter after successful call
            retry_count = 0
            
            # Respect rate limits
            time.sleep(ccxt_exch.rateLimit / 1000)
            
        except (ConnectionResetError, ConnectionError, TimeoutError) as e:
            retry_count += 1
            logger.warning(f"Connection error while fetching data: {str(e)}. Retry {retry_count}/{max_retries}")
            
            if retry_count >= max_retries:
                logger.error(f"Max retries reached, returning partial data with {len(out)} candles")
                break
                
            # Exponential backoff
            backoff_time = 2 ** retry_count
            logger.info(f"Waiting {backoff_time} seconds before retry...")
            time.sleep(backoff_time)
            
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV data: {str(e)}")
            if len(out) > 0:
                logger.info(f"Returning partial data with {len(out)} candles")
            break
    
    return out

def fetch_ohlcv(symbol="BTC/USDT", tf="1h", days=90, cache_dir="data"):
    """
    Fetch OHLCV data from CCXT exchange and cache to disk.
    Will use cached data if available and fresh (< 1 hour old).
    
    Args:
        symbol: Trading pair symbol
        tf: Timeframe (e.g. '1h', '4h', '1d')
        days: Number of days of history to fetch
        cache_dir: Directory to cache data
        
    Returns:
        DataFrame with OHLCV data
    """
    # Create cache filename
    cache_f = pathlib.Path(cache_dir) / f"{symbol.replace('/','_')}_{tf}_{days}d.json"
    
    # Use cached data if recent enough
    if cache_f.exists() and (time.time() - cache_f.stat().st_mtime) < 3600:
        logger.info(f"Loading cached data for {symbol} ({tf}) from {cache_f}")
        df = pd.read_json(cache_f)
        # Convert timestamp to datetime
        if 'ts' in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], unit="ms")
            df.set_index("ts", inplace=True)
        elif 'timestamp' in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
        return df

    # Fetch from exchange
    try:
        logger.info(f"Fetching {days} days of {tf} data for {symbol} from exchange")
        exch = ccxt.binance()
        since = exch.milliseconds() - days*24*60*60*1000
        
        # Use fetch_ohlcv_full to get all candles without the 1000 bar limit
        ohlcv = fetch_ohlcv_full(exch, symbol, tf, since)
        
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Convert types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Save cache
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_json(cache_f, orient="records")
        
        logger.info(f"Fetched and cached {len(df)} candles for {symbol} ({tf})")
        return df
    except Exception as e:
        logger.error(f"Error fetching OHLCV via CCXT: {e}")
        # If the cache exists but is old, use it as a fallback
        if cache_f.exists():
            logger.warning(f"Using stale cache for {symbol} ({tf})")
            df = pd.read_json(cache_f)
            if 'ts' in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], unit="ms")
                df.set_index("ts", inplace=True)
            elif 'timestamp' in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
            return df
        # Last resort: return empty DataFrame
        logger.error(f"Failed to fetch data for {symbol} ({tf})")
        return pd.DataFrame()

class DataFetcher:
    """Fetch and manage historical and real-time market data."""
    
    def __init__(self, use_testnet=False, use_mock=False):
        # Initialize exchange
        self.exchange_id = EXCHANGE
        self.symbol = SYMBOL
        self.timeframe = TIMEFRAME
        self.use_testnet = use_testnet
        self.use_mock = use_mock
        
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
        
    def fetch_historical_data(self, symbol=SYMBOL, timeframe=TIMEFRAME, days=90):
        """
        Fetch historical OHLCV data. Prioritizes real data over mock data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candlestick timeframe
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with historical OHLCV data
        """
        if isinstance(symbol, str) and '/' in symbol:
            formatted_symbol = symbol  # Already in CCXT format (BTC/USDT)
        else:
            # Convert from Binance format (BTCUSDT) to CCXT format (BTC/USDT)
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                formatted_symbol = f"{base}/USDT"
            else:
                formatted_symbol = symbol  # Use as-is if unknown format
                
        logger.info(f"Fetching {days} days of historical data for {formatted_symbol} ({timeframe})")
        
        if not self.use_mock:
            try:
                # Try to fetch real data via CCXT
                df = fetch_ohlcv(symbol=formatted_symbol, tf=timeframe, days=days)
                if not df.empty:
                    logger.info(f"Successfully fetched {len(df)} historical candles for {formatted_symbol}")
                    return df
            except Exception as e:
                logger.warning(f"Error fetching real data: {e}")
                logger.warning("Falling back to mock data")
                
        # If we get here, either use_mock is True or we failed to get real data
        logger.info("Generating mock data for testing purposes")
        return self.generate_mock_data(days=days, timeframe=timeframe)
    
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
    
    def generate_mock_data(self, days=30, timeframe='1h'):
        """
        Generate mock price data for testing purposes.
        Only use for unit testing or when real data is unavailable.
        
        Args:
            days: Number of days to generate
            timeframe: Timeframe to generate
            
        Returns:
            DataFrame with mock OHLCV data
        """
        logger.warning("Using mock data - results will not reflect real market behavior!")
        
        # Set more reasonable volatility for testing
        base_volatility = 0.008  # 0.8% volatility per candle (more realistic)
        volume_volatility = 0.3
        
        # Calculate number of candles based on timeframe
        candles_per_day = {
            '1m': 1440,
            '3m': 480,
            '5m': 288,
            '15m': 96,
            '30m': 48,
            '1h': 24,
            '2h': 12,
            '4h': 6,
            '6h': 4,
            '8h': 3,
            '12h': 2,
            '1d': 1
        }
        
        candles_per_day_val = candles_per_day.get(timeframe, 24)
        num_candles = days * candles_per_day_val
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, periods=num_candles)
        
        # Generate random price data with more realistic trends
        close_prices = [20000]  # Start with $20,000 per BTC
        
        # Create more realistic trends with periods of consistent direction
        trend_direction = 1  # 1 for up, -1 for down
        trend_length = 0
        max_trend_length = 48  # Maximum length of a trend (2 days for hourly)
        
        for i in range(1, num_candles):
            # Determine if we should switch trend direction
            if trend_length >= max_trend_length or np.random.random() < 0.05:  # 5% chance to switch trend
                trend_direction = -trend_direction
                trend_length = 0
                # Moderate trend reversal (not too extreme)
                price_change = close_prices[-1] * np.random.normal(trend_direction * base_volatility * 1.5, base_volatility * 0.5)
            else:
                # Regular trend continuation
                price_change = close_prices[-1] * np.random.normal(trend_direction * base_volatility * 0.5, base_volatility)
                trend_length += 1
            
            # Add occasional larger moves (but not unrealistic ones)
            if np.random.random() < 0.02:  # 2% chance of a larger move
                price_change *= np.random.uniform(1.5, 2.0)
            
            new_price = close_prices[-1] + price_change
            
            # Ensure price stays positive and reasonable
            new_price = max(new_price, close_prices[-1] * 0.95)  # Don't drop more than 5% in a single candle
            new_price = min(new_price, close_prices[-1] * 1.05)  # Don't rise more than 5% in a single candle
            
            close_prices.append(new_price)
        
        # Generate OHLC data from close prices with realistic spread
        open_prices = []
        high_prices = []
        low_prices = []
        volumes = []
        
        for i in range(num_candles):
            if i == 0:
                open_price = close_prices[i] * (1 - np.random.random() * 0.005)  # 0.5% max gap
            else:
                open_price = close_prices[i-1] * (1 + np.random.normal(0, base_volatility * 0.3))
            
            # Generate high and low with more realistic range
            price_range = close_prices[i] * base_volatility * 1.5  # Typical range is 1-2% of price
            high_price = max(open_price, close_prices[i]) + abs(np.random.normal(0, price_range * 0.3))
            low_price = min(open_price, close_prices[i]) - abs(np.random.normal(0, price_range * 0.3))
            
            # Ensure high is always highest and low is always lowest
            high_price = max(high_price, open_price, close_prices[i])
            low_price = min(low_price, open_price, close_prices[i])
            
            # Generate volume with correlation to price volatility
            price_move = abs(close_prices[i] - open_price) / open_price
            volume_base = 50 + 500 * (price_move / base_volatility)  # Higher volume on larger moves
            volume = volume_base * np.random.lognormal(0, volume_volatility)
            
            open_prices.append(open_price)
            high_prices.append(high_price)
            low_prices.append(low_price)
            volumes.append(volume)
        
        # Create DataFrame from generated data
        mock_df = pd.DataFrame({
            'timestamp': timestamps,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        mock_df['timestamp'] = pd.to_datetime(mock_df['timestamp'])
        mock_df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated {len(mock_df)} mock candles with realistic parameters")
        return mock_df 