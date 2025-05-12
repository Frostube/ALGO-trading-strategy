#!/usr/bin/env python3
"""
Summary of the BTC/USDT intra-day scalping strategy settings.
This script prints the current configuration for review.
"""

import sys
from pprint import pprint

# Add current directory to path
sys.path.append('.')

from src.config import (
    SYMBOL, TIMEFRAME, HIGHER_TIMEFRAME,
    EMA_FAST, EMA_SLOW, EMA_TREND, EMA_MICRO_TREND,
    RSI_PERIOD, RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD,
    VOLUME_PERIOD, VOLUME_THRESHOLD,
    ATR_PERIOD, USE_ATR_STOPS, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER,
    USE_ADAPTIVE_THRESHOLDS, ADAPTIVE_LOOKBACK,
    MIN_BARS_BETWEEN_TRADES, MAX_TRADES_PER_HOUR, MIN_CONSECUTIVE_BARS_AGREE,
    RISK_PER_TRADE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
    USE_TWO_LEG_STOP, TRAIL_ACTIVATION_PCT, TRAIL_ATR_MULTIPLIER, USE_SOFT_STOP
)

def print_strategy_summary():
    """Print a formatted summary of the current strategy configuration."""
    
    config = {
        "General Settings": {
            "Symbol": SYMBOL,
            "Timeframe": TIMEFRAME,
            "Higher Timeframe": HIGHER_TIMEFRAME
        },
        "Indicators": {
            "EMA Settings": {
                "Fast EMA": EMA_FAST,
                "Slow EMA": EMA_SLOW,
                "Trend EMA": EMA_TREND,
                "Micro-trend EMA": EMA_MICRO_TREND
            },
            "RSI Settings": {
                "RSI Period": RSI_PERIOD,
                "Long Threshold": RSI_LONG_THRESHOLD,
                "Short Threshold": RSI_SHORT_THRESHOLD
            },
            "Volume Settings": {
                "Volume Period": VOLUME_PERIOD,
                "Volume Threshold": VOLUME_THRESHOLD
            },
            "ATR Settings": {
                "ATR Period": ATR_PERIOD
            },
            "Adaptive Thresholds": {
                "Enabled": USE_ADAPTIVE_THRESHOLDS,
                "Lookback Period": ADAPTIVE_LOOKBACK
            }
        },
        "Trade Frequency Controls": {
            "Min Bars Between Trades": MIN_BARS_BETWEEN_TRADES,
            "Max Trades Per Hour": MAX_TRADES_PER_HOUR,
            "Min Consecutive Bars Agree": MIN_CONSECUTIVE_BARS_AGREE
        },
        "Risk Management": {
            "Risk Per Trade": f"{RISK_PER_TRADE * 100}%",
            "Fixed Stop Loss": f"{STOP_LOSS_PCT * 100}%",
            "Fixed Take Profit": f"{TAKE_PROFIT_PCT * 100}%",
            "Use ATR Stops": USE_ATR_STOPS,
            "ATR Stop Loss Multiplier": ATR_SL_MULTIPLIER,
            "ATR Take Profit Multiplier": ATR_TP_MULTIPLIER,
            "Two-Leg Stop Strategy": {
                "Enabled": USE_TWO_LEG_STOP,
                "Trail Activation": f"{TRAIL_ACTIVATION_PCT}%",
                "Trail ATR Multiplier": TRAIL_ATR_MULTIPLIER,
                "Soft Stop Alerts": USE_SOFT_STOP
            }
        }
    }
    
    print("\n===== BTC/USDT Intra-Day Scalping Strategy Summary =====\n")
    
    for section, items in config.items():
        print(f"\n{section}:")
        print("-" * (len(section) + 1))
        
        if isinstance(items, dict):
            for key, value in items.items():
                if isinstance(value, dict):
                    print(f"\n  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"  {items}")
    
    print("\n\nStrategy Improvements Implemented:")
    print("--------------------------------")
    print("1. Stricter Entry Filters:")
    print("   - Added momentum confirmation requiring price to break above previous 5-min high for longs")
    print("   - Implemented micro-trend detection using EMA(50) slope")
    print("   - Added requirement for consecutive bars agreement")
    
    print("\n2. Adaptive Thresholds:")
    print("   - Created percentile-based RSI thresholds calculated over rolling periods")
    print("   - Implemented dynamic volume spike detection based on recent market conditions")
    
    print("\n3. Reduced Overtrading:")
    print(f"   - Added minimum bar gap between trades ({MIN_BARS_BETWEEN_TRADES} bars)")
    print(f"   - Implemented hourly trade limits (max {MAX_TRADES_PER_HOUR} trades per hour)")
    
    print("\n4. Enhanced Stop Logic:")
    print("   - Developed two-leg stop strategy with initial ATR-based stop loss")
    print(f"   - Added trailing stop that activates after {TRAIL_ACTIVATION_PCT}% profit at {TRAIL_ATR_MULTIPLIER}× ATR")
    print("   - Implemented soft stop alerts for manual intervention")
    
    print("\n5. Performance Logging:")
    print("   - Created FalsePositive database table to track trades that timeout")
    print("   - Added comprehensive performance reporting with visualizations")
    print("   - Implemented TradeStatistics for tracking metrics over time")

if __name__ == "__main__":
    print_strategy_summary() 