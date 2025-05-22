#!/usr/bin/env python3
"""
Register TMA Overlay Strategy with Strategy Factory

This script registers the TMA Overlay strategy with the system's strategy factory,
making it available for use with the monte_carlo_validation.py framework
and other system components that use the factory pattern.

Run this script once before using test_tma_monte_carlo.py or any other script
that needs to use the TMA strategy through the factory.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the project root is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    # Import the strategy factory
    from src.strategy.strategy_factory import StrategyFactory
    
    # Import the TMA Overlay strategy
    from winning_strategies.tma_overlay_btc_strategy.tma_overlay_strategy import TMAOverlayStrategy
    
    def register_tma_strategy():
        """Register the TMA Overlay strategy with the strategy factory"""
        logger.info("Registering TMA Overlay strategy with the factory...")
        
        # Get the factory instance
        factory = StrategyFactory()
        
        # Register the TMA Overlay strategy
        factory.register_strategy("tma_overlay", TMAOverlayStrategy)
        
        logger.info("TMA Overlay strategy successfully registered as 'tma_overlay'")
        
        # List all available strategies
        available_strategies = factory.get_available_strategies()
        logger.info(f"Available strategies: {', '.join(available_strategies)}")
        
        return True
    
    if __name__ == "__main__":
        success = register_tma_strategy()
        if success:
            logger.info("TMA Overlay strategy is now ready for use with monte_carlo_validation.py")
            logger.info("You can now run: python test_tma_monte_carlo.py")
        else:
            logger.error("Failed to register TMA Overlay strategy")
            sys.exit(1)
    
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Please ensure that both the strategy factory and TMA strategy are available in the specified paths")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    import traceback
    logger.error(traceback.format_exc())
    sys.exit(1) 