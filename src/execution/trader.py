import asyncio
import ccxt.pro as ccxtpro
import ccxt
from datetime import datetime, timedelta
import json

from src.config import (
    EXCHANGE, SYMBOL, BINANCE_API_KEY, BINANCE_SECRET_KEY,
    FUTURES, STOP_LOSS_PCT, TAKE_PROFIT_PCT
)
from src.utils.logger import logger, log_trade, log_alert
from src.db.models import Trade

class Trader:
    """
    Handles order execution and trade management.
    In paper trading mode it simulates orders without real execution.
    """
    
    def __init__(self, db_session, paper_trading=True):
        """Initialize the trader."""
        self.db_session = db_session
        self.paper_trading = paper_trading
        self.symbol = SYMBOL
        self.exchange_id = EXCHANGE
        
        # Setup REST client for order execution
        self.rest_exchange = getattr(ccxt, self.exchange_id)({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future' if FUTURES else 'spot'
            }
        })
        
        # Active trade tracking
        self.active_orders = {}
        self.active_trades = {}
        
        logger.info(f"Trader initialized in {'paper trading' if paper_trading else 'live trading'} mode")
    
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
    
    async def execute_trade(self, signal, account_balance):
        """
        Execute a trade based on a signal.
        
        Args:
            signal: Signal dictionary with 'signal', 'timestamp', 'close', etc.
            account_balance: Current account balance
            
        Returns:
            dict: Trade information
        """
        if signal['signal'] not in ['buy', 'sell']:
            return None
        
        side = signal['signal']
        current_price = signal['close']
        
        # Calculate stop loss and take profit
        stop_loss_price = (
            current_price * (1 - STOP_LOSS_PCT) if side == 'buy' 
            else current_price * (1 + STOP_LOSS_PCT)
        )
        take_profit_price = (
            current_price * (1 + TAKE_PROFIT_PCT) if side == 'buy' 
            else current_price * (1 - TAKE_PROFIT_PCT)
        )
        
        # Calculate position size (1% risk)
        risk_amount = account_balance * 0.01
        risk_per_unit = abs(current_price - stop_loss_price)
        position_size = risk_amount / risk_per_unit
        
        # Create trade info
        trade_info = {
            'symbol': self.symbol,
            'side': side,
            'entry_time': datetime.now(),
            'entry_price': current_price,
            'amount': position_size,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'status': 'open'
        }
        
        # Execute the trade
        if self.paper_trading:
            # Simulate execution in paper trading mode
            trade_id = f"paper_{datetime.now().timestamp()}"
            trade_info['id'] = trade_id
            self.active_trades[trade_id] = trade_info
            
            logger.info(f"Paper trade executed: {side} {position_size} {self.symbol} at {current_price}")
        else:
            try:
                # Live trading execution using CCXT
                if not hasattr(self, 'ws_exchange'):
                    await self.initialize_ws_exchange()
                
                # Market order to enter the position
                order = await self.ws_exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=side,
                    amount=position_size
                )
                
                # Store order info
                trade_info['order_id'] = order['id']
                trade_info['actual_entry_price'] = order['price'] or current_price
                trade_id = order['id']
                
                # Place stop loss and take profit orders
                if side == 'buy':
                    sl_order = await self.ws_exchange.create_order(
                        symbol=self.symbol,
                        type='stop',
                        side='sell',
                        amount=position_size,
                        price=stop_loss_price,
                        stopPrice=stop_loss_price
                    )
                    tp_order = await self.ws_exchange.create_order(
                        symbol=self.symbol,
                        type='limit',
                        side='sell',
                        amount=position_size,
                        price=take_profit_price
                    )
                else:
                    sl_order = await self.ws_exchange.create_order(
                        symbol=self.symbol,
                        type='stop',
                        side='buy',
                        amount=position_size,
                        price=stop_loss_price,
                        stopPrice=stop_loss_price
                    )
                    tp_order = await self.ws_exchange.create_order(
                        symbol=self.symbol,
                        type='limit',
                        side='buy',
                        amount=position_size,
                        price=take_profit_price
                    )
                
                # Store order IDs
                trade_info['sl_order_id'] = sl_order['id']
                trade_info['tp_order_id'] = tp_order['id']
                
                # Add to active orders
                self.active_orders[sl_order['id']] = 'stop_loss'
                self.active_orders[tp_order['id']] = 'take_profit'
                
                logger.info(f"Live trade executed: {side} {position_size} {self.symbol} at {order['price']}")
            
            except Exception as e:
                logger.error(f"Error executing trade: {str(e)}")
                return None
        
        # Log the trade
        log_trade(trade_info)
        
        # Store in database
        db_trade = Trade(
            symbol=self.symbol,
            side=side,
            entry_time=trade_info['entry_time'],
            entry_price=trade_info.get('actual_entry_price', current_price),
            amount=position_size,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price
        )
        self.db_session.add(db_trade)
        self.db_session.commit()
        
        # Update trade with ID from database
        trade_info['db_id'] = db_trade.id
        
        return trade_info
    
    async def update_trades(self, current_price):
        """
        Update active trades based on current price.
        For paper trading, simulates stop loss and take profit triggers.
        For live trading, check order status updates.
        
        Args:
            current_price: Current price of the symbol
            
        Returns:
            list: Closed trades
        """
        closed_trades = []
        
        if self.paper_trading:
            # Paper trading mode
            for trade_id, trade in list(self.active_trades.items()):
                if trade['status'] != 'open':
                    continue
                
                side = trade['side']
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                
                if side == 'buy':
                    # Check if stop loss triggered
                    if current_price <= stop_loss:
                        trade['exit_price'] = stop_loss
                        trade['exit_time'] = datetime.now()
                        trade['exit_reason'] = 'stop_loss'
                        trade['status'] = 'closed'
                        trade['pnl'] = (stop_loss - trade['entry_price']) * trade['amount']
                        closed_trades.append(trade)
                        
                        logger.info(f"Paper trade stop loss triggered: {side} at {stop_loss}")
                    
                    # Check if take profit triggered
                    elif current_price >= take_profit:
                        trade['exit_price'] = take_profit
                        trade['exit_time'] = datetime.now()
                        trade['exit_reason'] = 'take_profit'
                        trade['status'] = 'closed'
                        trade['pnl'] = (take_profit - trade['entry_price']) * trade['amount']
                        closed_trades.append(trade)
                        
                        logger.info(f"Paper trade take profit triggered: {side} at {take_profit}")
                
                else:  # sell
                    # Check if stop loss triggered
                    if current_price >= stop_loss:
                        trade['exit_price'] = stop_loss
                        trade['exit_time'] = datetime.now()
                        trade['exit_reason'] = 'stop_loss'
                        trade['status'] = 'closed'
                        trade['pnl'] = (trade['entry_price'] - stop_loss) * trade['amount']
                        closed_trades.append(trade)
                        
                        logger.info(f"Paper trade stop loss triggered: {side} at {stop_loss}")
                    
                    # Check if take profit triggered
                    elif current_price <= take_profit:
                        trade['exit_price'] = take_profit
                        trade['exit_time'] = datetime.now()
                        trade['exit_reason'] = 'take_profit'
                        trade['status'] = 'closed'
                        trade['pnl'] = (trade['entry_price'] - take_profit) * trade['amount']
                        closed_trades.append(trade)
                        
                        logger.info(f"Paper trade take profit triggered: {side} at {take_profit}")
                
                # Update database for closed trades
                if trade['status'] == 'closed':
                    try:
                        db_trade = self.db_session.query(Trade).filter_by(id=trade['db_id']).first()
                        if db_trade:
                            db_trade.exit_time = trade['exit_time']
                            db_trade.exit_price = trade['exit_price']
                            db_trade.exit_reason = trade['exit_reason']
                            db_trade.pnl = trade['pnl']
                            db_trade.pnl_percent = trade['pnl'] / (trade['entry_price'] * trade['amount'])
                            self.db_session.commit()
                    except Exception as e:
                        logger.error(f"Error updating trade in database: {str(e)}")
                        self.db_session.rollback()
                    
                    # Remove from active trades
                    del self.active_trades[trade_id]
                    
        else:
            # Live trading mode
            if not hasattr(self, 'ws_exchange'):
                await self.initialize_ws_exchange()
            
            try:
                # Check open orders
                open_orders = await self.ws_exchange.fetch_open_orders(symbol=self.symbol)
                
                # Map order IDs for easy lookup
                open_order_ids = {order['id']: order for order in open_orders}
                
                # Check for filled orders
                for order_id, order_type in list(self.active_orders.items()):
                    if order_id not in open_order_ids:
                        # Order was filled or canceled
                        try:
                            order = await self.ws_exchange.fetch_order(order_id, symbol=self.symbol)
                            
                            if order['status'] == 'filled':
                                # Find the trade
                                for trade_id, trade in self.active_trades.items():
                                    if (order_type == 'stop_loss' and trade.get('sl_order_id') == order_id) or \
                                       (order_type == 'take_profit' and trade.get('tp_order_id') == order_id):
                                        
                                        # Update trade
                                        trade['exit_price'] = order['price']
                                        trade['exit_time'] = datetime.fromtimestamp(order['timestamp'] / 1000)
                                        trade['exit_reason'] = order_type
                                        trade['status'] = 'closed'
                                        
                                        # Calculate PnL
                                        if trade['side'] == 'buy':
                                            trade['pnl'] = (order['price'] - trade['entry_price']) * trade['amount']
                                        else:
                                            trade['pnl'] = (trade['entry_price'] - order['price']) * trade['amount']
                                        
                                        closed_trades.append(trade)
                                        
                                        logger.info(f"Live trade {order_type} executed: {trade['side']} at {order['price']}")
                                        
                                        # Update database
                                        try:
                                            db_trade = self.db_session.query(Trade).filter_by(id=trade['db_id']).first()
                                            if db_trade:
                                                db_trade.exit_time = trade['exit_time']
                                                db_trade.exit_price = order['price']
                                                db_trade.exit_reason = order_type
                                                db_trade.pnl = trade['pnl']
                                                db_trade.pnl_percent = trade['pnl'] / (trade['entry_price'] * trade['amount'])
                                                self.db_session.commit()
                                        except Exception as e:
                                            logger.error(f"Error updating trade in database: {str(e)}")
                                            self.db_session.rollback()
                                        
                                        # Cancel other orders
                                        other_order_id = trade.get('tp_order_id') if order_type == 'stop_loss' else trade.get('sl_order_id')
                                        if other_order_id:
                                            try:
                                                await self.ws_exchange.cancel_order(other_order_id, symbol=self.symbol)
                                                del self.active_orders[other_order_id]
                                            except Exception as e:
                                                logger.error(f"Error canceling order: {str(e)}")
                                        
                                        # Remove from active trades and orders
                                        del self.active_trades[trade_id]
                                        del self.active_orders[order_id]
                                        break
                        
                        except Exception as e:
                            logger.error(f"Error checking order: {str(e)}")
            
            except Exception as e:
                logger.error(f"Error updating live trades: {str(e)}")
        
        return closed_trades
    
    async def close_all_positions(self):
        """Close all open positions."""
        if self.paper_trading:
            # In paper trading, just mark trades as closed
            for trade_id, trade in list(self.active_trades.items()):
                if trade['status'] == 'open':
                    trade['exit_price'] = trade['entry_price']  # Assume exit at entry for simplicity
                    trade['exit_time'] = datetime.now()
                    trade['exit_reason'] = 'manual_close'
                    trade['status'] = 'closed'
                    trade['pnl'] = 0  # No profit or loss
                    
                    # Update database
                    try:
                        db_trade = self.db_session.query(Trade).filter_by(id=trade['db_id']).first()
                        if db_trade:
                            db_trade.exit_time = trade['exit_time']
                            db_trade.exit_price = trade['exit_price']
                            db_trade.exit_reason = 'manual_close'
                            db_trade.pnl = 0
                            db_trade.pnl_percent = 0
                            self.db_session.commit()
                    except Exception as e:
                        logger.error(f"Error updating trade in database: {str(e)}")
                        self.db_session.rollback()
                    
                    logger.info(f"Paper trade manually closed: {trade['side']} at {trade['exit_price']}")
                    
                    # Remove from active trades
                    del self.active_trades[trade_id]
        else:
            # Live trading
            if not hasattr(self, 'ws_exchange'):
                await self.initialize_ws_exchange()
            
            try:
                # Cancel all open orders
                for order_id in list(self.active_orders.keys()):
                    try:
                        await self.ws_exchange.cancel_order(order_id, symbol=self.symbol)
                    except Exception as e:
                        logger.error(f"Error canceling order: {str(e)}")
                
                # Close all positions
                positions = await self.ws_exchange.fetch_positions([self.symbol])
                for position in positions:
                    if abs(float(position['size'])) > 0:
                        side = 'sell' if position['side'] == 'long' else 'buy'
                        
                        await self.ws_exchange.create_order(
                            symbol=self.symbol,
                            type='market',
                            side=side,
                            amount=abs(float(position['size']))
                        )
                
                logger.info("All live positions closed")
                
                # Update database for all active trades
                for trade_id, trade in list(self.active_trades.items()):
                    if trade['status'] == 'open':
                        # Get current price
                        ticker = await self.ws_exchange.fetch_ticker(self.symbol)
                        current_price = ticker['last']
                        
                        trade['exit_price'] = current_price
                        trade['exit_time'] = datetime.now()
                        trade['exit_reason'] = 'manual_close'
                        trade['status'] = 'closed'
                        
                        # Calculate PnL
                        if trade['side'] == 'buy':
                            trade['pnl'] = (current_price - trade['entry_price']) * trade['amount']
                        else:
                            trade['pnl'] = (trade['entry_price'] - current_price) * trade['amount']
                        
                        # Update database
                        try:
                            db_trade = self.db_session.query(Trade).filter_by(id=trade['db_id']).first()
                            if db_trade:
                                db_trade.exit_time = trade['exit_time']
                                db_trade.exit_price = current_price
                                db_trade.exit_reason = 'manual_close'
                                db_trade.pnl = trade['pnl']
                                db_trade.pnl_percent = trade['pnl'] / (trade['entry_price'] * trade['amount'])
                                self.db_session.commit()
                        except Exception as e:
                            logger.error(f"Error updating trade in database: {str(e)}")
                            self.db_session.rollback()
                        
                        # Remove from active trades
                        del self.active_trades[trade_id]
            
            except Exception as e:
                logger.error(f"Error closing all positions: {str(e)}")
                log_alert(f"Failed to close all positions: {str(e)}", error=True)
    
    async def close(self):
        """Close connections."""
        if hasattr(self, 'ws_exchange'):
            await self.ws_exchange.close()
            
    def get_active_trade(self):
        """Get the currently active trade if any."""
        for trade_id, trade in self.active_trades.items():
            if trade['status'] == 'open':
                return trade
        return None 