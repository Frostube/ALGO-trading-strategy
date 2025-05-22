#!/usr/bin/env python3
"""
Simple Strategy Factory for TMA Overlay

This is a simplified strategy factory implementation that bypasses
the need to import the EMA crossover strategy, which has an indentation error.
It only handles registering and creating the TMA Overlay strategy.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the project root is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir  # Assuming we're in the project root
sys.path.append(project_root)

class SimpleStrategyFactory:
    """
    Simplified strategy factory that only supports the TMA Overlay strategy
    """
    
    def __init__(self):
        """Initialize the factory with an empty strategy registry"""
        self.strategies = {}
        logger.info("Simple Strategy Factory initialized")
    
    def register_strategy(self, name, strategy_class):
        """
        Register a strategy class with the factory
        
        Args:
            name (str): The name to register the strategy under
            strategy_class: The strategy class to register
        """
        self.strategies[name] = strategy_class
        logger.info(f"Strategy '{name}' registered with the factory")
    
    def create_strategy(self, name, **kwargs):
        """
        Create a strategy instance by name
        
        Args:
            name (str): The name of the strategy to create
            **kwargs: Parameters to pass to the strategy constructor
            
        Returns:
            An instance of the requested strategy
        
        Raises:
            ValueError: If the strategy name is not registered
        """
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found in registry")
        
        strategy_class = self.strategies[name]
        logger.info(f"Creating instance of '{name}' strategy")
        return strategy_class(**kwargs)
    
    def get_available_strategies(self):
        """
        Get a list of all registered strategy names
        
        Returns:
            list: Names of all registered strategies
        """
        return list(self.strategies.keys())


# Override the original StrategyFactory in the module namespace for import compatibility
StrategyFactory = SimpleStrategyFactory

if __name__ == "__main__":
    # Test the factory
    factory = SimpleStrategyFactory()
    print(f"Available strategies: {factory.get_available_strategies()}") 