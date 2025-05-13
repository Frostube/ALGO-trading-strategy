#!/usr/bin/env python3
"""
Strategy factory to manage multiple trading strategies in an ensemble.
"""

import os
import sys
import importlib
import inspect

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import specific strategies
from .ema_crossover import EMACrossoverStrategy
from .base_strategy import BaseStrategy

class StrategyFactory:
    """
    Factory class to create and manage trading strategy instances.
    Enables running multiple strategies in parallel as an ensemble.
    """
    
    def __init__(self, config=None):
        """
        Initialize the strategy factory.
        
        Args:
            config (dict): Configuration parameters for strategies
        """
        self.config = config or {}
        self.strategies = {}
        
        # Register built-in strategies
        self._register_builtin_strategies()
        
        # Auto-discover other strategy modules
        self._discover_strategies()
    
    def _register_builtin_strategies(self):
        """Register the built-in strategies."""
        self.strategies = {
            'ema_crossover': EMACrossoverStrategy(self.config),
        }
        
        # Add more built-in strategies as they are implemented
        try:
            from .rsi_momentum import RSIMomentumStrategy
            self.strategies['rsi_momentum'] = RSIMomentumStrategy(self.config)
        except ImportError:
            pass
            
        try:
            from .donchian_breakout import DonchianBreakoutStrategy
            self.strategies['donchian_breakout'] = DonchianBreakoutStrategy(self.config)
        except ImportError:
            pass
            
        try:
            from .mean_reversion import MeanReversionStrategy
            self.strategies['mean_reversion'] = MeanReversionStrategy(self.config)
        except ImportError:
            pass
            
        try:
            from .volume_breakout import VolumeBreakoutStrategy
            self.strategies['volume_breakout'] = VolumeBreakoutStrategy(self.config)
        except ImportError:
            pass
    
    def _discover_strategies(self):
        """Auto-discover strategy classes in the strategy directory."""
        strategy_dir = os.path.dirname(os.path.abspath(__file__))
        
        for filename in os.listdir(strategy_dir):
            if filename.endswith('.py') and not filename.startswith('__') and filename != 'base_strategy.py':
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Import the module dynamically
                    module = importlib.import_module(f".{module_name}", package="src.strategy")
                    
                    # Find all classes in the module that inherit from BaseStrategy
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseStrategy) and 
                            obj != BaseStrategy):
                            
                            # Register the strategy if not already registered
                            strategy_id = module_name
                            if strategy_id not in self.strategies:
                                self.strategies[strategy_id] = obj(self.config)
                
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not import strategy from {filename}: {e}")
    
    def get_strategy(self, name):
        """
        Get a specific strategy by name.
        
        Args:
            name (str): Name of the strategy
            
        Returns:
            BaseStrategy: The strategy instance or None if not found
        """
        return self.strategies.get(name)
    
    def get_available_strategies(self):
        """
        Get list of available strategy names.
        
        Returns:
            list: List of available strategy names
        """
        return list(self.strategies.keys())
    
    def get_ensemble(self, strategy_names):
        """
        Get a list of strategy instances for ensemble trading.
        
        Args:
            strategy_names (list): List of strategy names to include
            
        Returns:
            list: List of strategy instances
        """
        return [self.strategies[name] for name in strategy_names if name in self.strategies]
    
    def create_strategy(self, name, custom_config=None):
        """
        Create a new instance of a strategy with custom config.
        
        Args:
            name (str): Name of the strategy
            custom_config (dict): Custom configuration to override defaults
            
        Returns:
            BaseStrategy: New strategy instance or None if not found
        """
        strategy_class = type(self.strategies.get(name))
        if not strategy_class:
            return None
            
        # Merge configs if custom config provided
        if custom_config:
            merged_config = {**self.config, **custom_config}
        else:
            merged_config = self.config
            
        return strategy_class(merged_config) 