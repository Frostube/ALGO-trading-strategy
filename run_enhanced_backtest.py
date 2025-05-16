#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import argparse

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from src.data.fetcher import DataFetcher
from src.backtest.backtest import Backtester
from src.indicators.technical import apply_indicators
from src.utils.logger import logger
from src.strategy.high_leverage_strategy import HighLeverageStrategy

class EnhancedBacktester(Backtester):
    """
    Extended backtester that works with our enhanced high leverage strategy.
    This class overrides the necessary methods to handle the updated strategy API.
    """
    
    def __init__(self, data=None, initial_balance=10000, params=None, open_positions=None, portfolio_manager=None):
        super().__init__(data, initial_balance, params, open_positions, portfolio_manager)
        
    def _backtest_strategy(self, strategy, data, higher_tf_df=None):
        """
        Override the backtest method to work with our enhanced strategy features.
        
        Args:
            strategy: Strategy instance to test
            data: DataFrame with price data
            higher_tf_df: Optional DataFrame with higher timeframe data for dual-timeframe confirmation
            
        Returns:
            dict: Backtest results
        """
        # Set the symbol for position tracking
        if hasattr(strategy, 'symbol'):
            self.current_symbol = strategy.symbol
            # Initialize position tracking for this symbol if not exists
            if self.current_symbol not in self.open_positions:
                self.open_positions[self.current_symbol] = None
                
        # Make sure there's enough data
        if len(data) < 20:
            logger.warning("Insufficient data for backtest (min 20 candles required)")
            return {
                'total_return': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # Create a mock account
        symbol_name = data['symbol'].iloc[0] if 'symbol' in data.columns else "BTC/USDT"
        account = self._create_mock_account(initial_balance=self.initial_balance, symbol=symbol_name)
        logger.info(f"Created mock account with initial balance: ${self.initial_balance}")
        
        # Initialize strategy with account
        if hasattr(strategy, 'set_account'):
            strategy.set_account(account)
            
        # Generate signals and indicators
        data_with_indicators = strategy.generate_signals(data, higher_tf_df) if hasattr(strategy, 'generate_signals') else strategy.apply_indicators(data)
        
        # Print DataFrame columns to understand what we have
        print("Data columns:", list(data_with_indicators.columns))
        
        # Check if ema columns exist
        if not {'ema_fast', 'ema_slow'}.issubset(set(data_with_indicators.columns)):
            print("Adding EMA columns")
            data_with_indicators['ema_fast'] = data_with_indicators['close'].ewm(span=5, adjust=False).mean()
            data_with_indicators['ema_slow'] = data_with_indicators['close'].ewm(span=13, adjust=False).mean()
        
        # Check if RSI column exists
        if 'rsi' not in data_with_indicators.columns:
            print("Adding RSI column")
            from ta.momentum import RSIIndicator
            rsi_indicator = RSIIndicator(close=data_with_indicators['close'], window=14)
            data_with_indicators['rsi'] = rsi_indicator.rsi()
        
        # Check if trend_direction is missing and calculate it
        if 'trend_direction' not in data_with_indicators.columns:
            print("Adding trend_direction column")
            # Calculate EMA crossover based trend direction
            if 'ema_fast' in data_with_indicators.columns and 'ema_slow' in data_with_indicators.columns:
                data_with_indicators['trend_direction'] = 0
                # Set signal based on EMA crossover
                for i in range(1, len(data_with_indicators)):
                    if (data_with_indicators['ema_fast'].iloc[i] > data_with_indicators['ema_slow'].iloc[i] and 
                        data_with_indicators['ema_fast'].iloc[i-1] <= data_with_indicators['ema_slow'].iloc[i-1]):
                        data_with_indicators.loc[data_with_indicators.index[i], 'trend_direction'] = 1  # Buy signal
                    elif (data_with_indicators['ema_fast'].iloc[i] < data_with_indicators['ema_slow'].iloc[i] and 
                          data_with_indicators['ema_fast'].iloc[i-1] >= data_with_indicators['ema_slow'].iloc[i-1]):
                        data_with_indicators.loc[data_with_indicators.index[i], 'trend_direction'] = -1  # Sell signal
                    else:
                        # Continue previous trend direction for non-crossover bars
                        data_with_indicators.loc[data_with_indicators.index[i], 'trend_direction'] = data_with_indicators['trend_direction'].iloc[i-1]
        
        # Explicit signal generation override
        if 'signal' not in data_with_indicators.columns:
            print("Adding explicit signal column")
            data_with_indicators['signal'] = 0
            for i in range(20, len(data_with_indicators)):
                # Generate explicit signals based on RSI
                if 'rsi' in data_with_indicators.columns:
                    rsi = data_with_indicators['rsi'].iloc[i]
                    prev_rsi = data_with_indicators['rsi'].iloc[i-1]
                    
                    # Oversold and moving up = Buy
                    if rsi < 40 and rsi > prev_rsi:
                        data_with_indicators.loc[data_with_indicators.index[i], 'signal'] = 1
                    # Overbought and moving down = Sell
                    elif rsi > 60 and rsi < prev_rsi:
                        data_with_indicators.loc[data_with_indicators.index[i], 'signal'] = -1
        
        # Add a column for our strategy's entry decision
        data_with_indicators['should_enter'] = False
        for i in range(20, len(data_with_indicators)):
            # Simply check momentum filter (we disabled the others)
            if strategy.check_momentum_filter(data_with_indicators, i):
                # If RSI is good, set should_enter to True
                if (('signal' in data_with_indicators.columns and data_with_indicators['signal'].iloc[i] != 0) or
                    ('trend_direction' in data_with_indicators.columns and data_with_indicators['trend_direction'].iloc[i] != 0)):
                    data_with_indicators.loc[data_with_indicators.index[i], 'should_enter'] = True
        
        # Initialize tracking variables
        trades = []
        equity_curve = [self.initial_balance]
        peak_equity = self.initial_balance
        drawdown_curve = [0]
        cash = self.initial_balance
        position_qty = 0
        entry_price = 0
        
        # Track total fees paid
        total_commission = 0
        total_slippage = 0
        total_funding = 0
        
        # Store strategy states for analysis
        strategy_states = []
        
        # Debug counters
        signal_count = {1: 0, -1: 0, 0: 0}  # Buy, Sell, Neutral
        
        # Convert higher timeframe data to dict format if needed
        mtf_data = {}
        if higher_tf_df is not None:
            mtf_data['1d'] = higher_tf_df
        
        # Loop through each candle (except the first few which are needed for indicators)
        min_required_candles = 20  # Minimum candles needed for indicators
        
        logger.info(f"Starting backtest loop with {len(data_with_indicators) - min_required_candles} candles")
        
        for i in range(min_required_candles, len(data_with_indicators)):
            # Current data slice
            current_data = data_with_indicators.iloc[:i+1]
            timestamp = data_with_indicators.index[i]
            current_price = data_with_indicators['close'].iloc[i]
            
            # Check if we should place a trade
            signal = 0
            if hasattr(strategy, 'should_place_trade'):
                # Try our should_place_trade function
                should_trade = strategy.should_place_trade(data_with_indicators, i, mtf_data)
                
                # If that fails, use our simplified approach
                if not should_trade and 'should_enter' in data_with_indicators.columns:
                    should_trade = data_with_indicators['should_enter'].iloc[i]
                
                # Debug every 100 bars
                if i % 100 == 0:
                    print(f"Bar {i}, Should trade: {should_trade}")
                    print(f"Trend Direction: {data_with_indicators['trend_direction'].iloc[i] if 'trend_direction' in data_with_indicators.columns else 'N/A'}")
                    print(f"Signal: {data_with_indicators['signal'].iloc[i] if 'signal' in data_with_indicators.columns else 'N/A'}")
                    print(f"RSI: {data_with_indicators['rsi'].iloc[i] if 'rsi' in data_with_indicators.columns else 'N/A'}")
                    
                if should_trade:
                    # Determine signal direction from trend direction or pattern
                    if 'signal' in data_with_indicators.columns and data_with_indicators['signal'].iloc[i] != 0:
                        signal = data_with_indicators['signal'].iloc[i]
                    elif 'trend_direction' in data_with_indicators.columns:
                        signal = data_with_indicators['trend_direction'].iloc[i]
                    elif 'bullish_pattern' in data_with_indicators.columns and data_with_indicators['bullish_pattern'].iloc[i]:
                        signal = 1
                    elif 'bearish_pattern' in data_with_indicators.columns and data_with_indicators['bearish_pattern'].iloc[i]:
                        signal = -1
                    
                    print(f"Trade signal generated at bar {i}: {signal}")
            else:
                # Default to simple trend direction
                if 'signal' in data_with_indicators.columns:
                    signal = data_with_indicators['signal'].iloc[i]
                elif 'trend_direction' in data_with_indicators.columns:
                    signal = data_with_indicators['trend_direction'].iloc[i]
                
                if signal != 0:
                    print(f"Signal from trend: {signal} at bar {i}")
            
            signal_count[signal] += 1
            
            # Process exits first if we have an open position
            if position_qty != 0:
                # Check if we should exit the position
                exit_triggered = False
                exit_price = current_price
                exit_reason = ""
                
                # Use our enhanced exit management if available
                if hasattr(strategy, 'manage_trade_exits'):
                    # Create position object for the exit manager
                    position = {
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profits': take_profits,
                        'side': 'long' if position_qty > 0 else 'short',
                        'entry_time': entry_time,
                        'size': abs(position_qty),
                        'remaining_size': abs(position_qty)
                    }
                    
                    # Get exit decision
                    exit_result = strategy.manage_trade_exits(
                        df=data_with_indicators,
                        position=position,
                        idx=i,
                        timeframe=data.name if hasattr(data, 'name') else '1h'
                    )
                    
                    # Process exit if triggered
                    if exit_result['exit_triggered']:
                        exit_triggered = True
                        exit_price = exit_result['exit_price']
                        exit_reason = exit_result['exit_reason']
                        exit_size = exit_result['exit_size']
                        
                        # Apply commission
                        commission = abs(exit_size * exit_price) * self.params.get('commission', 0.0008)
                        
                        # Calculate P&L
                        side_multiplier = 1 if position_qty > 0 else -1
                        trade_pnl = side_multiplier * exit_size * (exit_price - entry_price)
                        trade_pnl -= commission
                        
                        # Track costs
                        total_commission += commission
                        
                        # Record trade
                        trade = {
                            'entry_time': entry_time,
                            'exit_time': timestamp,
                            'type': 'long' if position_qty > 0 else 'short',
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'size': exit_size,
                            'pnl': trade_pnl,
                            'commission': commission,
                            'exit_reason': exit_reason
                        }
                        trades.append(trade)
                        
                        # Update cash
                        cash += trade_pnl + (exit_size * exit_price)
                        
                        # Update position size
                        position_qty = position_qty - (exit_size * (1 if position_qty > 0 else -1))
                        
                        # If fully closed, reset position tracking
                        if position_qty == 0 or abs(position_qty) < 0.0001:
                            position_qty = 0
                            entry_price = 0
                            stop_loss = 0
                            take_profits = {}
                
                # Check for signal reversal exit
                if not exit_triggered and signal != 0 and ((position_qty > 0 and signal < 0) or (position_qty < 0 and signal > 0)):
                    exit_triggered = True
                    exit_reason = "Signal Reversal"
                    
                    # Apply commission
                    commission = abs(position_qty * current_price) * self.params.get('commission', 0.0008)
                    
                    # Calculate P&L
                    side_multiplier = 1 if position_qty > 0 else -1
                    trade_pnl = side_multiplier * abs(position_qty) * (current_price - entry_price)
                    trade_pnl -= commission
                    
                    # Track costs
                    total_commission += commission
                    
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'type': 'long' if position_qty > 0 else 'short',
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'size': abs(position_qty),
                        'pnl': trade_pnl,
                        'commission': commission,
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    
                    # Update cash
                    cash += trade_pnl + (abs(position_qty) * current_price)
                    
                    # Reset position
                    position_qty = 0
                    entry_price = 0
                    stop_loss = 0
                    take_profits = {}
            
            # Process entries if no position and we have a signal
            elif signal != 0 and position_qty == 0:
                # Calculate position size using our enhanced method
                position_size = 0
                try:
                    if hasattr(strategy, 'calculate_position_size'):
                        position_size = strategy.calculate_position_size(
                            df=data_with_indicators,
                            idx=i,
                            account_balance=cash,
                            risk_pct=self.params.get('risk_per_trade', 0.02)
                        )
                except Exception as e:
                    print(f"Error in position sizing: {e}")
                
                # If that fails, use our own position sizing logic
                if position_size == 0:
                    # Default position sizing using ATR for stop loss distance
                    risk_amount = cash * self.params.get('risk_per_trade', 0.02)
                    stop_distance = 0
                    
                    # Calculate stop distance using ATR if available
                    if 'atr' in data_with_indicators.columns:
                        atr = data_with_indicators['atr'].iloc[i]
                        # Use 1x ATR for stop distance
                        stop_distance = atr * 1.0
                    else:
                        # Fallback to percentage-based stop (1%)
                        stop_distance = current_price * 0.01
                    
                    # Calculate position size based on risk and stop distance
                    position_size = risk_amount / stop_distance if stop_distance > 0 else 0
                    
                    # Cap position size to avoid unrealistic values
                    max_position_value = cash * 0.2  # Max 20% of balance per trade (down from 50%)
                    position_size = min(position_size, max_position_value / current_price)
                    
                    # Additional cap to prevent exponential growth
                    if cash > self.initial_balance * 10:  # If balance grew more than 10x
                        # Scale down position size logarithmically
                        growth_factor = np.log10(cash / self.initial_balance) / 10
                        position_size *= (1 - min(growth_factor, 0.9))  # Cap at 90% reduction
                
                # Apply signal direction
                position_qty = position_size if signal > 0 else -position_size
                
                # Apply commission
                commission = abs(position_qty * current_price) * self.params.get('commission', 0.0008)
                cash -= commission
                
                # Track entry details
                entry_price = current_price
                entry_time = timestamp
                
                # Calculate stop loss using our own logic
                stop_loss = 0
                try:
                    if hasattr(strategy, 'calculate_stop_loss'):
                        stop_loss = strategy.calculate_stop_loss(
                            df=data_with_indicators,
                            idx=i,
                            entry_price=entry_price,
                            side='long' if signal > 0 else 'short'
                        )
                except Exception as e:
                    print(f"Error in stop loss calculation: {e}")
                
                # If that fails, use our own stop loss logic
                if stop_loss == 0:
                    # Calculate stop distance using ATR if available
                    if 'atr' in data_with_indicators.columns:
                        atr = data_with_indicators['atr'].iloc[i]
                        stop_distance = atr * 1.0  # 1x ATR for stop
                    else:
                        # Fallback to percentage-based stop (1%)
                        stop_distance = entry_price * 0.01
                    
                    # Apply stop loss based on position direction
                    if signal > 0:  # Long position
                        stop_loss = entry_price - stop_distance
                    else:  # Short position
                        stop_loss = entry_price + stop_distance
                
                # Calculate take profits using our own logic
                take_profits = {}
                try:
                    if hasattr(strategy, 'calculate_take_profit'):
                        take_profits = strategy.calculate_take_profit(
                            df=data_with_indicators,
                            idx=i,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            side='long' if signal > 0 else 'short'
                        )
                except Exception as e:
                    print(f"Error in take profit calculation: {e}")
                
                # If that fails, create our own take profit levels
                if not take_profits:
                    r_value = abs(entry_price - stop_loss)  # R-value (risk per share)
                    
                    # Create take profit levels
                    if signal > 0:  # Long
                        take_profits = {
                            'main': entry_price + (r_value * 2.0),  # 2R take profit
                            'partial': entry_price + r_value  # 1R partial take profit
                        }
                    else:  # Short
                        take_profits = {
                            'main': entry_price - (r_value * 2.0),  # 2R take profit
                            'partial': entry_price - r_value  # 1R partial take profit
                        }
                
                # Track costs
                total_commission += commission
                
                # Log entry
                logger.info(f"Entered {'LONG' if signal > 0 else 'SHORT'} at {timestamp}: Price ${entry_price:.2f}, Size {abs(position_qty):.6f}, Stop ${stop_loss:.2f}")
            
            # Update equity curve
            current_equity = cash
            if position_qty != 0:
                position_value = position_qty * current_price
                current_equity += position_value
            
            equity_curve.append(current_equity)
            
            # Update peak equity and drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            current_dd = 0 if peak_equity == 0 else (peak_equity - current_equity) / peak_equity
            drawdown_curve.append(current_dd)
        
        # Close any remaining position at the end
        if position_qty != 0:
            final_price = data_with_indicators['close'].iloc[-1]
            final_timestamp = data_with_indicators.index[-1]
            
            # Apply commission
            commission = abs(position_qty * final_price) * self.params.get('commission', 0.0008)
            
            # Calculate P&L
            side_multiplier = 1 if position_qty > 0 else -1
            trade_pnl = side_multiplier * abs(position_qty) * (final_price - entry_price)
            trade_pnl -= commission
            
            # Track costs
            total_commission += commission
            
            # Record trade
            trade = {
                'entry_time': entry_time,
                'exit_time': final_timestamp,
                'type': 'long' if position_qty > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': final_price,
                'size': abs(position_qty),
                'pnl': trade_pnl,
                'commission': commission,
                'exit_reason': 'End of Backtest'
            }
            trades.append(trade)
            
            # Update cash
            cash += trade_pnl + (abs(position_qty) * final_price)
            
            # Final equity update
            equity_curve[-1] = cash
        
        # Calculate performance metrics
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
        
        if len(trades) > 0:
            winners = [t['pnl'] for t in trades if t['pnl'] > 0]
            losers = [t['pnl'] for t in trades if t['pnl'] <= 0]
            
            win_rate = len(winners) / len(trades) if len(trades) > 0 else 0
            profit_factor = sum(winners) / abs(sum(losers)) if sum(losers) < 0 else float('inf')
            
            avg_win = sum(winners) / len(winners) if len(winners) > 0 else 0
            avg_loss = sum(losers) / len(losers) if len(losers) < 0 else 0
        
        # Calculate max drawdown
        max_drawdown = max(drawdown_curve) if drawdown_curve else 0
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            daily_returns = [(equity_curve[i] / equity_curve[i-1]) - 1 for i in range(1, len(equity_curve))]
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
        else:
            sharpe = 0
        
        # Compile results
        results = {
            'initial_balance': self.initial_balance,
            'final_balance': cash,
            'net_profit': cash - self.initial_balance,
            'roi': (cash / self.initial_balance - 1) * 100,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'winning_trades': len([t for t in trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in trades if t['pnl'] <= 0]),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'trades': trades,
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve
        }
        
        return results
    
    def _create_mock_account(self, initial_balance=10000, symbol=None):
        """Create a mock account for the backtest"""
        from src.backtest.backtest import MockAccount
        return MockAccount(initial_balance=initial_balance, symbol=symbol)

def run_backtest_on_timeframe(symbol, timeframe, days, initial_balance=10000):
    """Run backtest on specified timeframe.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe to test on
        days: Number of days to look back
        initial_balance: Starting account balance
        
    Returns:
        dict: Backtest results
    """
    logger.info(f"Running backtest for {symbol} on {timeframe} timeframe")
    
    # Create data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch historical data
    logger.info(f"Fetching {days} days of {timeframe} data...")
    
    # Determine higher timeframe
    higher_tf = '1d' if timeframe in ['5m', '15m', '30m', '1h', '4h'] else '1w'
    
    # Fetch data for main timeframe
    data = data_fetcher.fetch_historical_data(
        symbol=symbol, 
        days=days, 
        timeframe=timeframe
    )
    
    # Fetch data for higher timeframe
    higher_tf_data = data_fetcher.fetch_historical_data(
        symbol=symbol,
        days=days,
        timeframe=higher_tf
    )
    
    if data is None or data.empty:
        logger.error("Failed to fetch data or data is empty")
        return None
    
    logger.info(f"Received {len(data)} bars for {timeframe} and {len(higher_tf_data)} bars for {higher_tf}")
    
    # Set a name attribute on the data for timeframe tracking
    data.name = timeframe
    higher_tf_data.name = higher_tf
    
    # Create strategy instance with parameters tuned for this timeframe
    strategy = HighLeverageStrategy(
        fast_ema=8 if timeframe in ['4h', '1d'] else 5,
        slow_ema=21 if timeframe in ['4h', '1d'] else 12,
        trend_ema=50,
        risk_per_trade=0.02,  # Higher risk per trade (2%)
        use_mtf_filter=False,  # Disable MTF filter completely
        use_momentum_filter=True,
        use_volatility_sizing=False,  # Disable volatility sizing
        momentum_threshold=25,  # Super permissive momentum threshold
        mtf_timeframes=[higher_tf],
        mtf_signal_mode='any',  # Use "any" mode for more signals
        # Exit strategy parameters
        use_trailing_stop=True,
        take_profit_r=1.5,  # Lower take profit target for quick exits
        partial_exit_r=0.75,  # Lower partial profit target
        max_hold_periods=12,  # Short hold periods
        # Pattern filter parameters
        use_pattern_filter=False,  # Disable pattern filter 
        pattern_strictness='loose',
        use_volume_filter=False,  # Disable volume filter
        volume_threshold=1.0
    )
    
    # Initialize enhanced backtester
    backtester = EnhancedBacktester(
        data=data, 
        initial_balance=initial_balance,
        params={
            'commission': 0.0008,  # 0.08% fee
            'slippage': 0.0002,    # 0.02% slippage
            'risk_per_trade': 0.02,
            'use_atr_stops': True,
        }
    )
    
    # Run backtest
    logger.info("Running backtest...")
    
    # Debug the strategy properties
    print("Strategy Configuration:")
    print(f"MTF Filter: {strategy.use_mtf_filter}")
    print(f"Momentum Filter: {strategy.use_momentum_filter}")
    print(f"Momentum Threshold: {strategy.momentum_threshold}")
    print(f"Pattern Filter: {strategy.use_pattern_filter}")
    print(f"Volume Filter: {strategy.use_volume_filter}")
    
    results = backtester._backtest_strategy(strategy, data, higher_tf_data)
    
    # Add timeframe to results
    if results:
        results['timeframe'] = timeframe
        results['symbol'] = symbol
        results['days'] = days
        
        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/backtest_{symbol.replace('/', '_')}_{timeframe}_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(filename, 'w') as f:
            # Convert objects that aren't JSON serializable
            json_results = results.copy()
            if 'trades' in json_results:
                for trade in json_results['trades']:
                    if isinstance(trade.get('entry_time'), pd.Timestamp):
                        trade['entry_time'] = trade['entry_time'].strftime("%Y-%m-%dT%H:%M:%S")
                    if isinstance(trade.get('exit_time'), pd.Timestamp):
                        trade['exit_time'] = trade['exit_time'].strftime("%Y-%m-%dT%H:%M:%S")
            
            # Convert numpy values to native Python types
            def convert_to_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_to_json_serializable(obj.tolist())
                else:
                    return obj
            
            json_results = convert_to_json_serializable(json_results)
            json.dump(json_results, f, indent=4)
        
        logger.info(f"Saved results to {filename}")
    
    return results

def print_results(results):
    """Print backtest results in a readable format."""
    if not results:
        print("No results to display")
        return
    
    print("\n============================================================")
    print(f"Backtest Results for {results['symbol']} ({results['timeframe']} - {results['days']} days)")
    print("============================================================")
    print(f"Initial Balance: ${results.get('initial_balance', 0):.2f}")
    print(f"Final Balance: ${results.get('final_balance', 0):.2f}")
    print(f"Net Profit: ${results.get('net_profit', 0):.2f} ({results.get('roi', 0):.2f}%)")
    print(f"Win Rate: {results.get('win_rate', 0) * 100:.1f}% ({results.get('winning_trades', 0)}/{results.get('total_trades', 0)})")
    print(f"Max Drawdown: {results.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    
    # Print sample trades
    if 'trades' in results and results['trades']:
        print("\nSample Trades:")
        print("-" * 60)
        
        for i, trade in enumerate(results['trades'][:5]):
            direction = "LONG" if trade.get('type') == 'long' else "SHORT"
            pnl = trade.get('pnl', 0)
            pnl_emoji = "✅" if pnl > 0 else "❌"
            
            print(f"{i+1}. {direction} {pnl_emoji} Entry: ${trade.get('entry_price', 0):.2f}, "
                  f"Exit: ${trade.get('exit_price', 0):.2f}, "
                  f"P/L: ${pnl:.2f} ({pnl / (trade.get('size', 0) * trade.get('entry_price', 0)) * 100:.2f}%)")
            
            # Print exit reason if available
            if 'exit_reason' in trade:
                print(f"   Exit Reason: {trade['exit_reason']}")

def main():
    """Run backtests on multiple timeframes."""
    
    parser = argparse.ArgumentParser(description='Run backtests on multiple timeframes')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--days', type=int, default=60, help='Number of days to backtest')
    parser.add_argument('--balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--timeframe', type=str, default=None, help='Single timeframe to test (e.g., 1h)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("\tHigh Leverage Strategy Backtest with Enhanced Exit")
    print("="*60 + "\n")
    
    # Timeframes to test
    timeframes = ['1h', '4h', '1d'] if args.timeframe is None else [args.timeframe]
    
    all_results = {}
    
    # Run backtests on each timeframe
    for tf in timeframes:
        print(f"\nRunning backtest on {tf} timeframe...")
        results = run_backtest_on_timeframe(args.symbol, tf, args.days, args.balance)
        
        if results:
            all_results[tf] = results
            print_results(results)
    
    # Compare results across timeframes
    if len(timeframes) > 1:
        print("\n" + "="*60)
        print("\tTimeframe Comparison")
        print("="*60)
        
        print(f"{'Timeframe':<8} | {'ROI':<8} | {'Win Rate':<10} | {'# Trades':<8} | {'Drawdown':<10} | {'Sharpe':<8}")
        print("-" * 75)
        
        for tf, results in all_results.items():
            roi = results.get('roi', 0)
            win_rate = results.get('win_rate', 0) * 100
            trades = results.get('total_trades', 0)
            drawdown = results.get('max_drawdown', 0) * 100
            sharpe = results.get('sharpe_ratio', 0)
            
            print(f"{tf:<8} | {roi:<8.2f}% | {win_rate:<10.1f}% | {trades:<8} | {drawdown:<10.2f}% | {sharpe:<8.2f}")
    
    print("\n" + "="*60)
    print("Backtest completed across all timeframes")
    print("="*60)

if __name__ == "__main__":
    main() 