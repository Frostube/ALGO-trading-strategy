#!/usr/bin/env python
"""
Simple test script to run backtest dashboard with relaxed parameters
"""
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Use streamlit main function from the dashboard
from src.dashboard.backtest_dashboard import show_backtest_dashboard
import streamlit as st

# Override config settings to be less restrictive
from src.config import (
    USE_ML_FILTER, ML_PROBABILITY_THRESHOLD, TRADING_HOURS_START, 
    TRADING_HOURS_END, USE_TIME_FILTERS, WEEKEND_TRADING, AVOID_MIDNIGHT_HOURS
)

# Make config less restrictive
import src.config
src.config.USE_ML_FILTER = False
src.config.USE_TIME_FILTERS = False
src.config.WEEKEND_TRADING = True
src.config.AVOID_MIDNIGHT_HOURS = False
src.config.TRADING_HOURS_START = 0
src.config.TRADING_HOURS_END = 24
src.config.STOP_LOSS_PCT = 0.0035
src.config.TAKE_PROFIT_PCT = 0.007
src.config.ATR_SL_MULTIPLIER = 1.3
src.config.ATR_TP_MULTIPLIER = 3.0
src.config.MIN_BARS_BETWEEN_TRADES = 2

if __name__ == "__main__":
    # Set page config
    st.set_page_config(
        page_title="Backtest Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    # Run dashboard
    show_backtest_dashboard() 