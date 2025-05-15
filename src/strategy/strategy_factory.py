#!/usr/bin/env python3
"""
Strategy factory to manage multiple trading strategies in an ensemble.
"""

import os
import sys
import importlib
import inspect
import logging
from typing import Dict, Any, Optional, Type
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import specific strategies
from .ema_crossover import EMACrossoverStrategy
from .base_strategy import BaseStrategy
from src.strategy.rsi_strategy import RSIOscillatorStrategy
from src.utils.health_monitor import StrategyHealthMonitor

logger = logging.getLogger('trading_bot')

class StrategyFactory:
    """
    Factory class to create strategy instances based on configuration.
    Supports auto-detection of available strategies.
    """
    
    def __init__(self, config=None, db_session=None):
        """
        Initialize the strategy factory.
        
        Args:
            config (dict): Configuration parameters for strategies
            db_session: Database session for health monitoring
        """
        self.config = config or {}
        self.db_session = db_session
        self.health_monitor = StrategyHealthMonitor()
        self.strategies = {}
        
        # Auto-discover strategies
        self._discover_strategies()
    
    def _discover_strategies(self):
        """Automatically discover available strategy classes"""
        # Register built-in strategies
        self.strategies = {
            'ema_crossover': EMACrossoverStrategy,
            'rsi_oscillator': RSIOscillatorStrategy,
        }
        
        # Discover other strategies in the strategy directory
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        for filename in os.listdir(strategy_dir):
            if filename.endswith('.py') and filename not in ['__init__.py', 'base_strategy.py', 'strategy_factory.py', 'ema_crossover.py', 'rsi_strategy.py']:
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(f"src.strategy.{module_name}")
                    
                    # Find classes that inherit from BaseStrategy
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BaseStrategy) and attr != BaseStrategy:
                            strategy_name = module_name if attr_name == 'Strategy' else attr_name.lower()
                            self.strategies[strategy_name] = attr
                            logger.info(f"Discovered strategy: {strategy_name}")
                            
                except Exception as e:
                    logger.warning(f"Could not import strategy from {filename}: {str(e)}")
    
    def create_strategy(self, strategy_name, **kwargs):
        """
        Create a strategy instance by name with specified parameters.
        
        Args:
            strategy_name (str): Name of the strategy to create
            **kwargs: Additional parameters for the strategy
            
        Returns:
            BaseStrategy: An instance of the requested strategy
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy_class = self.strategies[strategy_name]
        
        # Merge configuration
        merged_config = {}
        if self.config and 'strategies' in self.config and strategy_name in self.config['strategies']:
            merged_config.update(self.config['strategies'][strategy_name])
        
        # Override with explicit kwargs
        merged_config.update(kwargs)
        
        # Create the strategy
        return strategy_class(**merged_config)
    
    def create_ensemble_strategy(self, symbol, timeframe='4h', account_balance=10000.0, risk_allocation=None, **kwargs):
        """
        Create an ensemble of different strategies for the same symbol/timeframe.
        
        Args:
            symbol: The trading symbol
            timeframe: Timeframe for analysis
            account_balance: Initial account balance
            risk_allocation: Dict mapping strategy names to risk allocation percentages
            **kwargs: Additional parameters for strategies
            
        Returns:
            Dict of strategy instances
        """
        # Default risk allocation if not provided
        if risk_allocation is None:
            risk_allocation = {
                'ema_crossover': 0.5,  # 50% of risk allocated to EMA strategy
                'rsi_oscillator': 0.5,  # 50% of risk allocated to RSI strategy
            }
        
        # Validate risk allocation
        total_allocation = sum(risk_allocation.values())
        if total_allocation > 1.0:
            logger.warning(f"Total risk allocation ({total_allocation}) exceeds 1.0, normalizing...")
            for strategy_name in risk_allocation:
                risk_allocation[strategy_name] /= total_allocation
        
        # Create strategies
        strategies = {}
        for strategy_name, allocation in risk_allocation.items():
            if strategy_name not in self.strategies:
                logger.warning(f"Strategy {strategy_name} not found, skipping...")
                continue
                
            # Calculate risk per trade based on allocation
            base_risk = kwargs.get('risk_per_trade', 0.0075)  # Default 0.75%
            allocated_risk = base_risk * allocation
            
            # Create strategy with allocated risk
            strategy_kwargs = kwargs.copy()
            strategy_kwargs['risk_per_trade'] = allocated_risk
            strategy_kwargs['health_monitor'] = self.health_monitor
            strategy_kwargs['symbol'] = symbol
            strategy_kwargs['timeframe'] = timeframe
            strategy_kwargs['account_balance'] = account_balance
            
            strategy = self.create_strategy(
                strategy_name, 
                **strategy_kwargs
            )
            
            strategies[strategy_name] = strategy
        
        return strategies 