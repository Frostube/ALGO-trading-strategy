#!/usr/bin/env python3
"""
Run the Streamlit dashboard for the BTC/USDT scalping strategy.
"""
import os
import sys
import argparse
import subprocess

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the Streamlit dashboard."""
    parser = argparse.ArgumentParser(description='Run the BTC/USDT scalping strategy dashboard')
    parser.add_argument('--port', type=int, default=8501, help='Port for the Streamlit dashboard')
    args = parser.parse_args()
    
    # Path to the Streamlit app
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'dashboard', 'app.py')
    
    # Run the Streamlit app using subprocess with python -m
    cmd = [
        'python', '-m', 'streamlit', 'run', app_path,
        '--server.port', str(args.port),
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"Starting Streamlit dashboard on port {args.port}...")
    subprocess.run(cmd)

if __name__ == '__main__':
    main() 