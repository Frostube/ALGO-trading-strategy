import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from tqdm import tqdm
import importlib

from src.data.fetcher import DataFetcher
from src.indicators.technical import apply_indicators, get_signal
from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, COMMISSION,
    SLIPPAGE, BACKTEST_TRAIN_SPLIT, RISK_PER_TRADE, USE_ATR_STOPS, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    HIGHER_TIMEFRAME, TIMEFRAME, EMA_TREND, EMA_SLOW, EMA_FAST,
    TRAIL_ACTIVATION_PCT, TRAIL_ATR_MULTIPLIER, USE_ML_FILTER, ML_MIN_TRADES_FOR_TRAINING,
    RSI_PERIOD, RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD, VOLUME_PERIOD, VOLUME_THRESHOLD,
    ATR_PERIOD
)
from src.utils.logger import logger
from src.strategy.ema_crossover import EMACrossoverStrategy

# Dynamic imports for factory and allocator
try:
    from src.strategy.strategy_factory import StrategyFactory
    HAS_FACTORY = True
except ImportError:
    HAS_FACTORY = False

try:
    from src.portfolio.allocator import PortfolioAllocator 
    HAS_ALLOCATOR = True
except ImportError:
    HAS_ALLOCATOR = False

try:
    from src.market.regime_detector import MarketRegimeDetector
    HAS_REGIME_DETECTOR = True
except ImportError:
    HAS_REGIME_DETECTOR = False

# Import ML filter if enabled
if USE_ML_FILTER:
    try:
        from src.ml.signal_filter import MLSignalFilter
        ml_filter = MLSignalFilter()
    except ImportError:
        logger.warning("ML filter module not found, disabling ML filtering")
        USE_ML_FILTER = False
    except Exception as e:
        logger.error(f"Error initializing ML filter: {e}")
        USE_ML_FILTER = False

class MockAccount:
    """
    Mock trading account for backtesting.
    Tracks balance, equity, positions, and P&L.
    """
    
    def __init__(self, initial_balance=10000, symbol=None):
        self.initial_balance = initial_balance
        self.balance = initial_balance  # Cash balance
        self.equity = initial_balance   # Cash + positions value
        self.symbol = symbol
        self.positions = {}
        self.position_price = 0
        self.position_qty = 0
        
    def update_position_value(self, price):
        """
        Update account equity based on current position value.
        
        Args:
            price (float): Current market price
            
        Returns:
            float: Current equity
        """
        position_value = self.position_qty * price
        self.equity = self.balance + position_value
        return self.equity
        
    def has_position(self):
        """Check if account has an open position."""
        return self.position_qty != 0
        
    def get_position_size(self, price, risk_pct=0.01):
        """
        Calculate position size based on risk percentage.
        
        Args:
            price (float): Current price
            risk_pct (float): Risk as percentage of equity
            
        Returns:
            float: Position size in units
        """
        risk_amount = self.equity * risk_pct
        return risk_amount / price
        
    def enter_position(self, price, qty, commission=0):
        """
        Enter a new position or add to existing.
        
        Args:
            price (float): Entry price
            qty (float): Position quantity (negative for short)
            commission (float): Trading fee
            
        Returns:
            bool: Success or failure
        """
        cost = abs(qty * price) + commission
        
        # Check if balance is sufficient
        if cost > self.balance:
            return False
            
        # Add position
        self.balance -= cost
        
        # Calculate new position with avg price
        if self.position_qty == 0:
            self.position_price = price
            self.position_qty = qty
        else:
            # Weighted average for position price when adding
            total_qty = self.position_qty + qty
            self.position_price = ((self.position_qty * self.position_price) + 
                                  (qty * price)) / total_qty
            self.position_qty = total_qty
        
        # Update equity
        self.equity = self.balance + (self.position_qty * price)
        return True
        
    def exit_position(self, price, qty=None, commission=0):
        """
        Exit position fully or partially.
        
        Args:
            price (float): Exit price
            qty (float): Quantity to exit (None for full exit)
            commission (float): Trading fee
            
        Returns:
            dict: Trade result with PnL
        """
        if self.position_qty == 0:
            return {'success': False, 'pnl': 0, 'error': 'No position'}
            
        # Default to full exit
        if qty is None or abs(qty) > abs(self.position_qty):
            qty = self.position_qty
            
        # Calculate P&L
        if self.position_qty > 0:
            # Long position
            pnl = qty * (price - self.position_price)
        else:
            # Short position
            pnl = -qty * (self.position_price - price)
            
        # Update position
        position_value = abs(qty * price)
        self.balance += position_value + pnl - commission
        self.position_qty -= qty
        
        # If position closed completely, reset position price
        if self.position_qty == 0:
            self.position_price = 0
            
        # Update equity
        self.equity = self.balance + (self.position_qty * price if self.position_qty != 0 else 0)
        
        return {
            'success': True,
            'pnl': pnl,
            'commission': commission,
            'remaining_qty': self.position_qty
        }

class Backtester:
    """
    Flexible backtester for trading strategies.
    
    Attributes:
        data: DataFrame with OHLC data
        initial_balance: Starting account balance
        strategies_map: Dictionary of active strategies
        params: Dictionary of backtest parameters
        results: Dictionary of backtest results
    """
    
    def __init__(self, data=None, initial_balance=10000, params=None):
        self.data = data
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.equity_curve = []
        self.drawdown_curve = []
        self.peak_equity = initial_balance
        self.strategies_map = {}
        self.params = params or {}
        self.results = {}
        
        # Trading costs
        self.COMMISSION = 0.0004   # 0.04%
        self.SLIPPAGE = 0.0002     # 0.02%
    
    def run(self, train_test_split=None, symbols=None, strategies=None):
        """
        Run backtests with the configured parameters.
        
        Args:
            train_test_split (float): Ratio to split data into training/testing sets
            symbols (list): List of symbols to backtest
            strategies (list): List of strategy names to use
            
        Returns:
            dict: Backtest results
        """
        # Set defaults
        if symbols is None:
            if hasattr(self, 'all_symbol_data') and self.all_symbol_data:
                symbols = list(self.all_symbol_data.keys())
            else:
                symbols = [SYMBOL]
        
        if strategies is None:
            strategies = ["ema_crossover"]
        
        # Main experiment, potentially split into train/test sets
        all_results = {}
        self.strategies_map = {}
        
        # Get strategies from factory
        strategy_factory = StrategyFactory(self.params)
        strategy_instances = [strategy_factory.get_strategy(name) for name in strategies]
        
        # Initialize allocator if using multiple symbols/strategies
        if len(symbols) > 1 or len(strategy_instances) > 1:
            logger.info("Initializing portfolio allocator for multi-symbol/strategy backtest")
            vol_target = self.params.get('vol_target_pct', 0.1)  # Default 10% vol target
            allocator = PortfolioAllocator(
                symbols=symbols,
                strategies=[s.name for s in strategy_instances],
                vol_target=vol_target
            )
            
            # Pass allocator to each strategy
            for strategy in strategy_instances:
                strategy.set_allocator(allocator)
        
        # Initialize regime detector if enabled
        if self.params.get('enable_regime_detection', False):
            regime_detector = MarketRegimeDetector(
                lookback_days=self.params.get('regime_lookback', 30)
            )
            
            # Add strategy-regime mapping
            regime_detector.set_regime_strategies(
                ranging_strategies=self.params.get('ranging_strategies', []),
                normal_strategies=self.params.get('normal_strategies', []),
                trending_strategies=self.params.get('trending_strategies', [])
            )
        else:
            regime_detector = None
        
        for symbol in symbols:
            # Get data for this symbol
            if hasattr(self, 'all_symbol_data') and symbol in self.all_symbol_data:
                symbol_data = self.all_symbol_data[symbol]
            else:
                symbol_data = self.data
            
            # Add symbol name to the data
            symbol_data = symbol_data.copy()
            symbol_data['symbol'] = symbol
            
            for strategy in strategy_instances:
                # Run strategy on this symbol
                key = f"{symbol}_{strategy.name}"
                result = self._run_single_strategy(strategy, symbol_data, train_test_split)
                all_results[key] = result
                
                # Store the strategy instance in the map for later reference
                self.strategies_map[key] = strategy
        
        # Aggregate results if multi-asset or multi-strategy
        if len(all_results) > 1:
            logger.info("Aggregating results for multi-symbol/strategy backtest")
            # Use allocator weights if available
            if 'allocator' in locals():
                strategy_weights = allocator.strategy_weights
                symbol_weights = allocator.symbol_weights
            else:
                # Default to equal weights
                strategy_weights = {s.name: 1.0/len(strategy_instances) for s in strategy_instances}
                symbol_weights = {s: 1.0/len(symbols) for s in symbols}
                
            portfolio_results = self._combine_results(all_results, strategy_weights, symbol_weights)
            all_results['portfolio'] = portfolio_results
            all_results['weights'] = {
                'strategies': strategy_weights,
                'symbols': symbol_weights
            }
        
        # Store results and return
        self.results = all_results
        return all_results
    
    def _run_single_strategy(self, strategy, data, train_test_split=None):
        """
        Run a single strategy against provided data.
        
        Args:
            strategy: Strategy instance
            data: DataFrame with OHLCV data
            train_test_split: Ratio to split data (None for no split)
            
        Returns:
            dict: Results dictionary or dict of train/test results
        """
        if train_test_split:
            # Split the data
            split_idx = int(len(data) * train_test_split)
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            
            # Run on train set
            logger.info(f"Running {strategy.name} on training set ({len(train_data)} candles)")
            train_results = self._backtest_strategy(strategy, train_data)
            
            # Reset strategy and run on test set
            strategy.reset()
            logger.info(f"Running {strategy.name} on testing set ({len(test_data)} candles)")
            test_results = self._backtest_strategy(strategy, test_data)
            
            # Combine results
            return {
                'train': train_results,
                'test': test_results
            }
        else:
            # Run on full dataset
            logger.info(f"Running {strategy.name} on full dataset ({len(data)} candles)")
            return self._backtest_strategy(strategy, data)
    
    def _backtest_strategy(self, strategy, data):
        """
        Run backtest for a single strategy on provided data.
        
        Args:
            strategy: Strategy instance to test
            data: DataFrame with price data
            
        Returns:
            dict: Backtest results
        """
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
        symbol_name = data['symbol'].iloc[0] if 'symbol' in data.columns else SYMBOL
        account = MockAccount(initial_balance=self.initial_balance, symbol=symbol_name)
        logger.info(f"Created mock account with initial balance: ${self.initial_balance}")
        
        # Initialize strategy with account
        strategy.set_account(account)
        data_with_indicators = strategy.apply_indicators(data)
        
        # Initialize tracking variables
        trades = []
        equity_curve = [self.initial_balance]
        peak_equity = self.initial_balance
        drawdown_curve = [0]
        cash = self.initial_balance
        position_qty = 0
        entry_price = 0
        
        # Store strategy states for analysis
        strategy_states = []
        
        # Debug counters
        signal_count = {'buy': 0, 'sell': 0, '': 0}
        
        # Loop through each candle (except the first few which are needed for indicators)
        min_required_candles = 10  # Minimum candles needed for indicators
        
        logger.info(f"Starting backtest loop with {len(data_with_indicators) - min_required_candles} candles")
        
        for i in range(min_required_candles, len(data_with_indicators)):
            # Get the current bar data
            current_data = data_with_indicators.iloc[:i+1]
            current_bar = current_data.iloc[-1]
            timestamp = current_data.index[-1]
            
            # Generate signal
            signal = strategy.get_signal(current_data)
            signal_count[signal] += 1
            
            # Track strategy state
            state = {
                'timestamp': timestamp,
                'signal': signal,
                'position': position_qty,
                'cash': cash,
                'equity': cash + (position_qty * current_bar['close'] if position_qty else 0),
                'indicators': {
                    key: current_bar[key] for key in current_bar.keys() 
                    if key not in ['open', 'high', 'low', 'close', 'volume', 'symbol']
                }
            }
            strategy_states.append(state)
            
            # Process signals - check for exit first
            if position_qty != 0:
                exit_signal = False
                
                # Check for exit conditions
                if (position_qty > 0 and signal == "sell") or (position_qty < 0 and signal == "buy"):
                    exit_signal = True
                    logger.info(f"Exit signal at {timestamp}: {signal} (position: {position_qty})")
                
                # Check trailing stop and other exit conditions if implemented in strategy
                if hasattr(strategy, 'check_exit_conditions'):
                    exit_cond = strategy.check_exit_conditions(current_bar)
                    if exit_cond:
                        exit_signal = True
                        logger.info(f"Exit condition at {timestamp}: {exit_cond}")
                
                # Execute exit if signaled
                if exit_signal:
                    # Apply slippage (worse price on exit)
                    exit_price = self._apply_slippage(current_bar['close'], 
                                                     "sell" if position_qty > 0 else "buy")
                    
                    # Calculate P&L
                    trade_pnl = position_qty * (exit_price - entry_price)
                    
                    # Apply commission
                    commission = abs(position_qty * exit_price) * self.COMMISSION
                    trade_pnl -= commission
                    
                    # Update cash and position
                    cash += position_qty * exit_price - commission
                    
                    # Record trade
                    trade = {
                        'entry_time': entry_time,
                        'exit_time': timestamp,
                        'type': 'long' if position_qty > 0 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'size': abs(position_qty),
                        'pnl': trade_pnl,
                        'return': trade_pnl / (entry_price * abs(position_qty)),
                        'equity_before': equity_curve[-1],
                        'equity_after': cash,
                        'pct_return': trade_pnl / equity_curve[-1],  # Return on account equity
                        'commission': commission
                    }
                    trades.append(trade)
                    logger.info(f"Exited {trade['type']} trade: Entry ${entry_price:.2f}, Exit ${exit_price:.2f}, PnL ${trade_pnl:.2f} ({trade['pct_return']*100:.2f}%)")
                    
                    # Reset position
                    position_qty = 0
                    entry_price = 0
                    
                    # Update strategy with the trade
                    if hasattr(strategy, 'update_trade_history'):
                        strategy.update_trade_history(trade)
            
            # Process entry signals if no position
            elif signal in ["buy", "sell"]:
                # Calculate position size based on cash available
                price = current_bar['close']
                position_value = cash * 0.95  # Use 95% of cash to allow for fees
                
                if hasattr(strategy, 'calculate_position_size'):
                    # Strategy-specific position sizing
                    position_size = strategy.calculate_position_size(price)
                    # Limit the position to available cash
                    max_size = position_value / price
                    position_qty = min(position_size, max_size)
                else:
                    # Default position sizing
                    position_qty = position_value / price
                
                # Short positions are negative quantity
                if signal == "sell":
                    position_qty = -position_qty
                
                # Apply slippage (worse price on entry)
                entry_price = self._apply_slippage(price, signal)
                
                # Apply commission
                commission = abs(position_qty * entry_price) * self.COMMISSION
                cash -= commission
                
                # Record entry time
                entry_time = timestamp
                
                logger.info(f"Entered {signal} trade at {timestamp}: Price ${entry_price:.2f}, Qty {position_qty:.6f}, Commission ${commission:.2f}")
                
                # Update strategy with the entry
                if hasattr(strategy, 'record_trade'):
                    strategy.record_trade('entry', entry_price, position_qty)
            
            # Update equity curve (mark-to-market)
            current_equity = cash
            if position_qty != 0:
                position_value = position_qty * current_bar['close']
                current_equity += position_value
            
            # Record equity
            equity_curve.append(current_equity)
            
            # Update peak equity and drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            # Calculate drawdown as percentage of peak equity
            current_dd = 0 if peak_equity == 0 else (peak_equity - current_equity) / peak_equity
            drawdown_curve.append(current_dd)
            
            # Update account equity for proper tracking
            account.equity = current_equity
        
        logger.info(f"Backtest complete. Signal counts - Buy: {signal_count['buy']}, Sell: {signal_count['sell']}, None: {signal_count['']}")
        logger.info(f"Final equity: ${equity_curve[-1]:.2f}, Initial: ${equity_curve[0]:.2f}, Return: {(equity_curve[-1]/equity_curve[0]-1)*100:.2f}%")
        logger.info(f"Trades executed: {len(trades)}")
        
        # Calculate performance metrics
        if len(trades) > 0:
            returns = [t['pct_return'] for t in trades]
            win_rate = sum(1 for t in trades if t['pnl'] > 0) / len(trades)
            profit_factor = sum(t['pnl'] for t in trades if t['pnl'] > 0) / (abs(sum(t['pnl'] for t in trades if t['pnl'] < 0)) + 1e-6)
            avg_win = sum(t['pnl'] for t in trades if t['pnl'] > 0) / (sum(1 for t in trades if t['pnl'] > 0) + 1e-6)
            avg_loss = sum(t['pnl'] for t in trades if t['pnl'] < 0) / (sum(1 for t in trades if t['pnl'] < 0) + 1e-6)
            
            # Annualized return calculation
            days = (data_with_indicators.index[-1] - data_with_indicators.index[0]).days
            if days > 0:
                annual_factor = 365 / days  # Annualization factor
                total_return = (equity_curve[-1] / self.initial_balance) - 1
                annual_return = ((1 + total_return) ** annual_factor) - 1
                # Cap annual return at a reasonable level (300% max)
                annual_return = min(3.0, annual_return)
            else:
                annual_return = 0
            
            # Calculate Sharpe ratio if there are multiple equity points
            if len(equity_curve) > 1:
                # Convert equity curve to returns
                equity_returns = [equity_curve[i] / equity_curve[i-1] - 1 for i in range(1, len(equity_curve))]
                
                # Calculate daily Sharpe
                if len(equity_returns) > 0 and np.std(equity_returns) > 0:
                    sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)
                else:
                    sharpe = 0
            else:
                sharpe = 0
                
            # Handle edge cases for sharpe
            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
            
            # Cap Sharpe at reasonable levels
            sharpe = min(10, max(-10, sharpe))
        else:
            # No trades case
            returns = []
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            annual_return = 0
            sharpe = 0
        
        # Calculate max drawdown
        max_dd = max(drawdown_curve) if drawdown_curve else 0
        
        # Results dictionary
        results = {
            'total_return': (equity_curve[-1] / equity_curve[0] - 1) if equity_curve else 0,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(trades),
            'trades': trades,
            'equity_curve': equity_curve,
            'drawdown_curve': drawdown_curve,
            'price_data': data_with_indicators,
            'strategy_states': strategy_states,
            'trade_returns': returns,
            'last_equity': equity_curve[-1] if equity_curve else self.initial_balance
        }
        
        # Store for this strategy
        strategy.results = results
        
        return results
    
    def _apply_slippage(self, price, side):
        """
        Apply slippage to execution price.
        
        Args:
            price (float): Price before slippage
            side (str): Either "buy" or "sell"
            
        Returns:
            float: Price with slippage applied
        """
        # Make the price worse based on direction
        slip_direction = 1 if side == "buy" else -1
        return price * (1 + (self.SLIPPAGE * slip_direction))
    
    def _combine_results(self, all_results, strategy_weights=None, symbol_weights=None):
        """
        Combine results from multiple strategies/symbols into a portfolio result.
        
        Args:
            all_results (dict): Dictionary of individual results
            strategy_weights (dict): Strategy weight allocation
            symbol_weights (dict): Symbol weight allocation
            
        Returns:
            dict: Combined portfolio results
        """
        # Default to equal weights if not provided
        if not strategy_weights:
            strategy_weights = {key.split('_')[1]: 1/len(set(key.split('_')[1] for key in all_results)) 
                               for key in all_results}
        if not symbol_weights:
            symbol_weights = {key.split('_')[0]: 1/len(set(key.split('_')[0] for key in all_results)) 
                             for key in all_results}
        
        # Calculate combined equity curve
        combined_equity = None
        combined_drawdown = None
        combined_returns = []
        all_trades = []
        
        # Combine equity curves with weighted allocation
        for key, result in all_results.items():
            # Skip portfolio result if present
            if key == 'portfolio' or key == 'weights':
                continue
                
            symbol, strat = key.split('_')
            weight = strategy_weights.get(strat, 0) * symbol_weights.get(symbol, 0)
            
            # Extend the combined equity curve if needed
            if 'equity_curve' in result:
                weighted_equity = [e * weight for e in result['equity_curve']]
                
                if combined_equity is None:
                    combined_equity = weighted_equity
                else:
                    # If lengths differ, extend the shorter one
                    if len(weighted_equity) < len(combined_equity):
                        weighted_equity.extend([weighted_equity[-1]] * 
                                             (len(combined_equity) - len(weighted_equity)))
                    elif len(weighted_equity) > len(combined_equity):
                        combined_equity.extend([combined_equity[-1]] * 
                                             (len(weighted_equity) - len(combined_equity)))
                    
                    # Add weighted equity curves
                    combined_equity = [a + b for a, b in zip(combined_equity, weighted_equity)]
            
            # Combine drawdown curves (take the worst drawdown at each point)
            if 'drawdown_curve' in result:
                weighted_dd = result['drawdown_curve']
                
                if combined_drawdown is None:
                    combined_drawdown = weighted_dd
                else:
                    # If lengths differ, extend the shorter one
                    if len(weighted_dd) < len(combined_drawdown):
                        weighted_dd.extend([weighted_dd[-1]] * 
                                         (len(combined_drawdown) - len(weighted_dd)))
                    elif len(weighted_dd) > len(combined_drawdown):
                        combined_drawdown.extend([combined_drawdown[-1]] * 
                                               (len(weighted_dd) - len(combined_drawdown)))
                    
                    # Take the maximum drawdown at each point
                    combined_drawdown = [max(a, b) for a, b in zip(combined_drawdown, weighted_dd)]
            
            # Combine trade returns and trades
            if 'trade_returns' in result:
                combined_returns.extend(result['trade_returns'])
            
            if 'trades' in result:
                # Add strategy and symbol info to trades
                for trade in result['trades']:
                    trade_copy = trade.copy()
                    trade_copy['strategy'] = strat
                    trade_copy['symbol'] = symbol
                    all_trades.append(trade_copy)
        
        # Calculate portfolio metrics
        if combined_equity and len(combined_equity) > 1:
            total_return = combined_equity[-1] / combined_equity[0] - 1
            max_dd = max(combined_drawdown) if combined_drawdown else 0
            
            # Calculate Sharpe ratio
            equity_returns = [combined_equity[i] / combined_equity[i-1] - 1 
                             for i in range(1, len(combined_equity))]
            
            if np.std(equity_returns) > 0:
                sharpe = np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)
            else:
                sharpe = 0
                
            # Cap values at reasonable levels
            sharpe = min(10, max(-10, sharpe))
            
            # Annualize returns
            for key, result in all_results.items():
                if key != 'portfolio' and key != 'weights' and 'price_data' in result:
                    price_data = result['price_data']
                    days = (price_data.index[-1] - price_data.index[0]).days
                    if days > 0:
                        annual_factor = 365 / days
                        annual_return = ((1 + total_return) ** annual_factor) - 1
                        # Cap annual return
                        annual_return = min(3.0, annual_return)
                        break
                    else:
                        annual_return = 0
            else:
                annual_return = 0
            
            # Calculate win rate and profit factor
            if all_trades:
                win_rate = sum(1 for t in all_trades if t['pnl'] > 0) / len(all_trades)
                profit_sum = sum(t['pnl'] for t in all_trades if t['pnl'] > 0)
                loss_sum = abs(sum(t['pnl'] for t in all_trades if t['pnl'] < 0)) + 1e-6
                profit_factor = profit_sum / loss_sum
            else:
                win_rate = 0
                profit_factor = 0
        else:
            total_return = 0
            max_dd = 0
            sharpe = 0
            annual_return = 0
            win_rate = 0
            profit_factor = 0
        
        # Create portfolio results
        portfolio_results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(all_trades),
            'trades': all_trades,
            'equity_curve': combined_equity,
            'drawdown_curve': combined_drawdown,
            'trade_returns': combined_returns
        }
        
        return portfolio_results
    
    def _calculate_portfolio_metrics(self, equity_curve):
        """
        Calculate performance metrics for the combined portfolio.
        
        Args:
            equity_curve (list): Combined equity curve
            
        Returns:
            dict: Performance metrics
        """
        # If equity_curve is empty or None, try to use strategy-level information
        if not equity_curve:
            # Calculate total return from final account balances
            initial_balance = self.initial_balance
            final_balance = 0
            
            # Sum up final balances across all accounts
            for key, strategy in self.strategies_map.items():
                if hasattr(strategy, 'account') and strategy.account:
                    final_balance += strategy.account.balance
            
            total_return = (final_balance / initial_balance) - 1 if initial_balance > 0 else 0
            
            # Return basic metrics
            return {
                'total_return': total_return,
                'sharpe_ratio': 0,  # Not enough data for Sharpe
                'max_drawdown': 0,  # Not enough data for drawdown
                'win_rate': 0       # Would need to extract from strategy trades
            }
        
        # Calculate metrics if we have equity curve
        total_return = (equity_curve[-1] / equity_curve[0]) - 1
        
        # Calculate returns
        returns = [(equity_curve[i] / equity_curve[i-1]) - 1 for i in range(1, len(equity_curve))]
        
        # Calculate Sharpe ratio
        if returns and len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            if np.isnan(sharpe) or np.isinf(sharpe):
                sharpe = 0
        else:
            sharpe = 0
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_dd = 0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }

def run_backtest(days=30, initial_balance=10000, plot=True):
    """Run a backtest with the specified parameters."""
    logger.info(f"Running backtest with {days} days of data")
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(use_testnet=True)
    
    # Fetch primary timeframe data
    primary_data = data_fetcher.fetch_historical_data(days=days, timeframe=TIMEFRAME)
    
    # Fetch higher timeframe data for multi-timeframe analysis
    higher_tf_data = None
    if HIGHER_TIMEFRAME:
        logger.info(f"Fetching higher timeframe data ({HIGHER_TIMEFRAME}) for confirmation")
        higher_tf_data = data_fetcher.fetch_historical_data(days=days, timeframe=HIGHER_TIMEFRAME)
    
    # Clean up resources
    data_fetcher.close()
    
    # Initialize backtester
    backtester = Backtester(data=primary_data, initial_balance=initial_balance)
    
    # Store higher timeframe data in params if available
    if higher_tf_data is not None:
        if backtester.params is None:
            backtester.params = {}
        backtester.params['higher_tf_data'] = higher_tf_data
    
    # Run the backtest with train/test split
    results = backtester.run(train_test_split=BACKTEST_TRAIN_SPLIT)
    
    if plot:
        backtester.plot_results(save_path="reports/backtest_results.png")
    
    return results

if __name__ == '__main__':
    run_backtest(days=30, plot=True) 