#!/usr/bin/env python3
import os
import asyncio
import signal
import argparse
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.fetcher import DataFetcher
from src.strategy.scalping import ScalpingStrategy
from src.execution.trader import Trader
from src.db.models import init_db, Trade, FalsePositive, TradeStatistics
from src.utils.logger import logger, log_daily_summary
from src.config import (
    TIMEFRAME, HIGHER_TIMEFRAME, LOG_FALSE_POSITIVES, 
    MAX_TRADE_DURATION_MINUTES, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
)

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
        
        # Get false positives
        if LOG_FALSE_POSITIVES:
            false_positives = db_session.query(FalsePositive).order_by(
                FalsePositive.log_time.desc()
            ).limit(10).all()
            
            # Format false positives for API
            false_positive_data = []
            for fp in false_positives:
                false_positive_data.append({
                    'symbol': fp.symbol,
                    'side': fp.side,
                    'entry_time': fp.entry_time.isoformat(),
                    'entry_price': float(fp.entry_price),
                    'duration_minutes': float(fp.duration_minutes) if fp.duration_minutes else None,
                    'market_trend': fp.market_trend,
                    'higher_tf_trend': fp.higher_tf_trend,
                    'rsi_value': float(fp.rsi_value) if fp.rsi_value else None
                })
        else:
            false_positive_data = []
        
        # Format trades for API
        trades_data = []
        for trade in recent_trades:
            trades_data.append({
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_price': float(trade.entry_price),
                'exit_price': float(trade.exit_price),
                'pnl': float(trade.pnl),
                'exit_reason': trade.exit_reason,
                'rsi_value': float(trade.rsi_value) if trade.rsi_value else None,
                'market_trend': trade.market_trend,
                'higher_tf_trend': trade.higher_tf_trend,
                'duration_minutes': float(trade.duration_minutes) if trade.duration_minutes else None,
                'trailing_activated': trade.trailing_activated
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
        
        # Calculate profit factor
        winning_trades = [t for t in recent_trades if t.pnl > 0]
        losing_trades = [t for t in recent_trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = sum(abs(t.pnl) for t in losing_trades) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Prepare payload
        payload = {
            'recent_trades': trades_data,
            'false_positives': false_positive_data,
            'last_50_bars': bars_data[-50:],
            'perf': {
                'win_rate': perf['win_rate'],
                'avg_return': perf['avg_return'],
                'profit_factor': profit_factor
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

def generate_performance_report(db_session, output_dir='reports'):
    """Generate and save a performance report with visualizations."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all completed trades
        trades = db_session.query(Trade).filter(
            Trade.exit_time.isnot(None)
        ).order_by(
            Trade.entry_time
        ).all()
        
        if not trades:
            logger.warning("No completed trades found for performance report.")
            return
        
        # Convert to DataFrame for analysis
        trades_df = pd.DataFrame([{
            'id': t.id,
            'symbol': t.symbol,
            'side': t.side,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'exit_reason': t.exit_reason,
            'duration_minutes': t.duration_minutes,
            'market_trend': t.market_trend,
            'higher_tf_trend': t.higher_tf_trend,
            'micro_trend': t.micro_trend,
            'rsi_value': t.rsi_value,
            'atr_value': t.atr_value,
            'trailing_activated': t.trailing_activated
        } for t in trades])
        
        # Get false positives if available
        if LOG_FALSE_POSITIVES:
            false_positives = db_session.query(FalsePositive).all()
            fp_df = pd.DataFrame([{
                'id': fp.id,
                'symbol': fp.symbol,
                'side': fp.side,
                'entry_time': fp.entry_time,
                'entry_price': fp.entry_price,
                'duration_minutes': fp.duration_minutes,
                'market_trend': fp.market_trend,
                'higher_tf_trend': fp.higher_tf_trend,
                'rsi_value': fp.rsi_value
            } for fp in false_positives]) if false_positives else pd.DataFrame()
        else:
            fp_df = pd.DataFrame()
        
        # Calculate performance metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if not trades_df.empty else 0
        gross_loss = abs(trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if not trades_df.empty else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        balance_curve = trades_df['pnl'].cumsum()
        peak = balance_curve.expanding().max()
        drawdown = (peak - balance_curve) / peak * 100
        max_drawdown = drawdown.max()
        
        # Save metrics
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_pnl': trades_df['pnl'].sum(),
            'avg_pnl_percent': trades_df['pnl_percent'].mean() * 100,
            'max_drawdown': max_drawdown,
            'avg_duration': trades_df['duration_minutes'].mean()
        }
        
        # Save metrics to database
        stats = TradeStatistics(
            date=datetime.now(),
            period='daily',
            total_trades=metrics['total_trades'],
            winning_trades=metrics['winning_trades'],
            losing_trades=metrics['losing_trades'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_pnl=metrics['total_pnl'],
            avg_pnl_percent=metrics['avg_pnl_percent'],
            max_drawdown=metrics['max_drawdown'],
            avg_duration_minutes=metrics['avg_duration']
        )
        db_session.add(stats)
        db_session.commit()
        
        # Generate visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. PnL over time
        plt.subplot(2, 2, 1)
        plt.plot(trades_df['exit_time'], trades_df['pnl'].cumsum())
        plt.title('Cumulative PnL Over Time')
        plt.xlabel('Time')
        plt.ylabel('PnL (USDT)')
        plt.grid(True)
        
        # 2. Win/Loss distribution
        plt.subplot(2, 2, 2)
        sns.histplot(trades_df['pnl'], bins=20, kde=True)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('PnL Distribution')
        plt.xlabel('PnL (USDT)')
        plt.ylabel('Frequency')
        
        # 3. Exit reasons
        plt.subplot(2, 2, 3)
        exit_counts = trades_df['exit_reason'].value_counts()
        exit_counts.plot(kind='bar')
        plt.title('Exit Reasons')
        plt.xlabel('Reason')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # 4. Trade duration vs PnL
        plt.subplot(2, 2, 4)
        plt.scatter(trades_df['duration_minutes'], trades_df['pnl_percent'] * 100, 
                   c=trades_df['pnl'] > 0, cmap='coolwarm', alpha=0.7)
        plt.title('Trade Duration vs PnL%')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('PnL (%)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_report_{datetime.now().strftime('%Y%m%d_%H%M')}.png")
        
        # Save trades to CSV for further analysis
        trades_df.to_csv(f"{output_dir}/trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
        
        if not fp_df.empty:
            fp_df.to_csv(f"{output_dir}/false_positives_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", index=False)
        
        # Log performance metrics
        logger.info("=== PERFORMANCE REPORT ===")
        logger.info(f"Total Trades: {metrics['total_trades']}")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
        logger.info(f"Total PnL: {metrics['total_pnl']:.2f} USDT")
        logger.info(f"Average PnL%: {metrics['avg_pnl_percent']:.2f}%")
        logger.info(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        logger.info(f"Average Trade Duration: {metrics['avg_duration']:.2f} minutes")
        
        if not fp_df.empty:
            logger.info(f"False Positives: {len(fp_df)}")
        
        logger.info(f"Report saved to {output_dir}/")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {str(e)}")

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
    
    # Initialize performance report timer
    last_report_time = datetime.now()
    
    # Generate initial performance report if requested
    if args.report:
        generate_performance_report(db_session)
    
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
        
        # Log with expanded information
        logger.debug(f"Market conditions: Price={latest_bar['close']:.2f}, "
                    f"Signal={strategy_update['signal']['signal']}, "
                    f"EMA Trend={'Up' if strategy_update['signal']['ema_trend'] > 0 else 'Down'}, "
                    f"Market Trend={'Up' if strategy_update['signal']['market_trend'] > 0 else 'Down'}, "
                    f"Micro Trend={'Up' if strategy_update['signal'].get('micro_trend', 0) > 0 else 'Down'}, "
                    f"RSI={strategy_update['signal']['rsi']:.1f}, "
                    f"ATR%={strategy_update['signal'].get('atr_pct', 0):.2f}%, "
                    f"SL Range={latest_bar['close'] - (latest_bar.get('atr', 0) * ATR_SL_MULTIPLIER):.2f}, "
                    f"TP Range={latest_bar['close'] + (latest_bar.get('atr', 0) * ATR_TP_MULTIPLIER):.2f}, "
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
        
        # Generate daily summary and performance report at end of day
        now = datetime.now()
        if now.hour == 23 and now.minute >= 55:
            # Generate daily summary
            perf = strategy.get_performance_summary('daily')
            log_daily_summary(perf)
            
            # Generate performance report daily
            if args.report or datetime.now() - last_report_time > timedelta(hours=12):
                generate_performance_report(db_session)
                last_report_time = datetime.now()
    
    # Clean shutdown
    await shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BTC/USDT Intra-Day Scalper')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode (default: paper trading)')
    parser.add_argument('--balance', type=float, default=10000, help='Initial account balance for paper trading')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--report', action='store_true', help='Generate performance reports')
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        import logging
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Make sure matplotlib doesn't try to display plots on headless systems
    plt.switch_backend('agg')
    
    # Run the main async function
    asyncio.run(main(args)) 
