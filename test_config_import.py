#!/usr/bin/env python3
"""
Simple test script to check if RSI parameters are properly defined in config.py
"""

from src.config import (
    RSI_PERIOD,
    RSI_OVERSOLD,
    RSI_OVERBOUGHT,
    RSI_LONG_THRESHOLD,
    RSI_SHORT_THRESHOLD
)

print("RSI Parameters from config.py:")
print(f"RSI_PERIOD: {RSI_PERIOD}")
print(f"RSI_OVERSOLD: {RSI_OVERSOLD}")  
print(f"RSI_OVERBOUGHT: {RSI_OVERBOUGHT}")
print(f"RSI_LONG_THRESHOLD: {RSI_LONG_THRESHOLD}")
print(f"RSI_SHORT_THRESHOLD: {RSI_SHORT_THRESHOLD}") 