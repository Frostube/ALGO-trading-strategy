#!/usr/bin/env python3
print("Starting import test")

try:
    print("Importing basic modules...")
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("Basic modules imported successfully")
    
    print("Importing from src.utils.logger...")
    from src.utils.logger import logger
    print("Logger imported successfully")
    
    print("Importing from src.backtest.backtest...")
    from src.backtest.backtest import run_backtest 
    print("Backtest imported successfully")
    
    print("Importing from src.strategy.ema_optimizer...")
    from src.strategy.ema_optimizer import find_best_ema_pair, fetch_historical_data
    print("EMA optimizer imported successfully")
    
    print("Importing from src.strategy.ema_crossover...")
    from src.strategy.ema_crossover import EMACrossoverStrategy
    print("EMA crossover strategy imported successfully")
    
    print("Importing from src.backtest.ema_backtest...")
    from src.backtest.ema_backtest import run_ema_backtest
    print("EMA backtest imported successfully")
    
    print("All imports successful")
except Exception as e:
    print(f"Error during imports: {type(e).__name__}: {str(e)}") 