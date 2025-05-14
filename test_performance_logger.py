#!/usr/bin/env python3
"""
Test script for the performance logger.
"""

from src.utils.performance_logger import append_result
import os
from pathlib import Path

def test_logger():
    """Test the performance logger directly."""
    print("Testing performance logger...")
    
    # Create docs directory if it doesn't exist
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Try to append a test result
    append_result(
        strategy_name="Test Strategy",
        dataset="BTC/USDT 1h (30d)",
        params="SL 1.5×, TP 3.0×",
        pf=1.5,
        win=50.0,
        dd=10.0,
        net=15.0
    )
    
    # Verify the log file was created
    log_path = Path("docs/performance_log.md")
    if log_path.exists():
        print(f"Log file created successfully at {log_path}")
        with open(log_path, "r") as f:
            print(f"Log contents:\n{f.read()}")
    else:
        print(f"ERROR: Log file not created at {log_path}")

if __name__ == "__main__":
    test_logger() 