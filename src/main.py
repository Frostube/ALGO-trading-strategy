#!/usr/bin/env python3
import os
import asyncio
import signal
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import requests

from src.data.fetcher import DataFetcher
from src.strategy.scalping import ScalpingStrategy
from src.execution.trader import Trader
from src.db.models import init_db, Trade
from src.utils.logger import logger, log_daily_summary
from src.config import TIMEFRAME, HIGHER_TIMEFRAME

# Global variables for clean shutdown
running = True
data_fetcher = None
trader = None
strategy = None

async def shutdown():
    """Gracefully shutdown the bot."""
    global running, data_fetcher, trader
    
    logger.info("Shutting down...")
    running = False
    
    # Close all positions
    if trader:
        logger.info("Closing all positions...")
        await trader.close_all_positions()
        await trader.close()
    
    # Close data connections
    if data_fetcher:
        logger.info("Closing data connections...")
        await data_fetcher.close_async()
    
    logger.info("Shutdown complete")

def signal_handler(sig, frame):
    """Handle Ctrl+C."""
    logger.info("Interrupt received, shutting down...")
    asyncio.create_task(shutdown())

async def send_ai_feedback(strategy, recent_bars, db_session):
    """Send trading data to AI feedback API."""
    try:
        # Get performance summary
        perf = strategy.get_performance_summary('daily')
        
        # Get recent trades from database
        recent_trades = db_session.query(Trade).filter(
            Trade.exit_time.isnot(None)
        ).order_by(
            Trade.exit_time.desc()
        ).limit(20).all()
        
        # Format trades for API
        trades_data = []
        for trade in recent_trades:
            trades_data.append({
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': float(trade.entry_price),
                'exit_price': float(trade.exit_price),
                'pnl': float(trade.pnl),
                'exit_reason': trade.exit_reason
            })
        
        # Format bars for API
        bars_data = []
        for idx, row in recent_bars.iterrows():
            bars_data.append({
                'timestamp': idx.isoformat(),
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })
        
        # Prepare payload
        payload = {
            'recent_trades': trades_data,
            'last_50_bars': bars_data[-50:],
            'perf': {
                'win_rate': perf['win_rate'],
                'avg_return': perf['avg_return']
            }
        }
        
        # Send to API
        response = requests.post('http://127.0.0.1:5000/feedback', json=payload)
        response.raise_for_status()
        
        # Log the adjustments
        adjustments = response.json().get('adjustments', [])
        if adjustments:
            logger.info(f"Received {len(adjustments)} adjustments from AI")
            for adjustment in adjustments:
                logger.info(f"AI Adjustment: {adjustment['type']} - {adjustment.get('parameter', '')}: "
                          f"{adjustment.get('old_value', '')} -> {adjustment.get('new_value', '')}")
                logger.info(f"Reason: {adjustment.get('reason', '')}")
        
        return adjustments
    
    except Exception as e:
        logger.error(f"Error sending AI feedback: {str(e)}")
        return []

async def main(args):
    """Main trading loop."""
    global running, data_fetcher, trader, strategy
    
    # Set up signal handlers for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Initialize database session
    db_session = init_db()
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Initialize strategy
    strategy = ScalpingStrategy(db_session, account_balance=args.balance)
    
    # Initialize trader
    trader = Trader(db_session, paper_trading=not args.live)
    
    # Fetch initial historical data
    logger.info(f"Fetching initial historical data ({TIMEFRAME})...")
    historical_data = data_fetcher.fetch_historical_data(timeframe=TIMEFRAME)
    
    # Fetch higher timeframe data
    logger.info(f"Fetching higher timeframe data ({HIGHER_TIMEFRAME})...")
    higher_tf_data = data_fetcher.fetch_historical_data(timeframe=HIGHER_TIMEFRAME)
    
    # Initialize AI feedback timer
    last_ai_feedback = datetime.now()
    
    # Main trading loop
    logger.info("Starting main trading loop...")
    async for new_candle in data_fetcher.watch_ohlcv(timeframe=TIMEFRAME):
        if not running:
            break
        
        # Get the latest data
        latest_data = data_fetcher.get_latest_data(100, timeframe=TIMEFRAME)
        
        # Get the latest higher timeframe data
        latest_higher_tf_data = data_fetcher.get_latest_data(50, timeframe=HIGHER_TIMEFRAME)
        
        logger.debug(f"Processing candle: {latest_data.iloc[-1].name}, " 
                     f"Higher TF: {latest_higher_tf_data.iloc[-1].name if not latest_higher_tf_data.empty else 'None'}")
        
        # Update strategy with new data
        strategy_update = strategy.update(latest_data, higher_tf_df=latest_higher_tf_data)
        
        # Log current market conditions
        latest_bar = latest_data.iloc[-1]
        higher_tf_bar = latest_higher_tf_data.iloc[-1] if not latest_higher_tf_data.empty else None
        
        logger.debug(f"Market conditions: Price={latest_bar['close']:.2f}, "
                    f"Signal={strategy_update['signal']['signal']}, "
                    f"EMA Trend={'Up' if strategy_update['signal']['ema_trend'] > 0 else 'Down'}, "
                    f"Market Trend={'Up' if strategy_update['signal']['market_trend'] > 0 else 'Down'}, "
                    f"Higher TF Trend={'Up' if higher_tf_bar is not None and strategy.higher_tf_trend > 0 else 'Down' if higher_tf_bar is not None else 'Unknown'}")
        
        # Update active trades with current price
        current_price = latest_data.iloc[-1]['close']
        closed_trades = await trader.update_trades(current_price)
        
        # Handle any closed trades
        for trade in closed_trades:
            logger.info(f"Trade closed: {trade['side']} at {trade['exit_price']}, "
                       f"PnL: {trade['pnl']}, Reason: {trade['exit_reason']}")
        
        # Check for new signals if no active trade
        if not trader.get_active_trade() and strategy_update['signal']['signal'] != 'neutral':
            logger.info(f"Signal detected: {strategy_update['signal']['signal']}")
            
            # Execute the trade
            trade = await trader.execute_trade(
                strategy_update['signal'],
                strategy_update['account_balance']
            )
            
            if trade:
                logger.info(f"Trade opened: {trade['side']} at {trade['entry_price']}, "
                           f"Stop Loss: {trade['stop_loss']}, Take Profit: {trade['take_profit']}")
        
        # Check if it's time to get AI feedback (hourly)
        if datetime.now() - last_ai_feedback > timedelta(hours=1):
            logger.info("Requesting AI feedback...")
            adjustments = await send_ai_feedback(strategy, latest_data, db_session)
            last_ai_feedback = datetime.now()
        
        # Generate daily summary at end of day
        now = datetime.now()
        if now.hour == 23 and now.minute >= 55:
            # Only generate summary once per day
            # More sophisticated time check could be implemented
            perf = strategy.get_performance_summary('daily')
            log_daily_summary(perf)
    
    # Clean shutdown
    await shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BTC/USDT Intra-Day Scalper')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode (default: paper trading)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial account balance for paper trading')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Run the main async function
    asyncio.run(main(args)) 
