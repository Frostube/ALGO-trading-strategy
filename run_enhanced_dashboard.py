#!/usr/bin/env python3
"""
Run the enhanced Streamlit dashboard for the BTC/USDT scalping strategy.
"""
import os
import argparse
import sys
import subprocess

def main():
    """Run the enhanced Streamlit dashboard."""
    parser = argparse.ArgumentParser(description='Run the enhanced BTC/USDT scalping strategy dashboard')
    parser.add_argument('--port', type=int, default=8501, help='Port for the Streamlit dashboard')
    parser.add_argument('--timeframe', type=str, default=None, help='Default timeframe to use (e.g., 1h, 4h)')
    
    args = parser.parse_args()
    
    # Construct the streamlit command
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'dashboard', 'enhanced_dashboard.py')
    
    cmd = [
        'python', '-m', 'streamlit', 'run', app_path,
        '--server.port', str(args.port)
    ]
    
    # Add any additional arguments
    if args.timeframe:
        cmd.extend(['--', '--timeframe', args.timeframe])
    
    print(f"Starting enhanced Streamlit dashboard on port {args.port}...")
    subprocess.run(cmd)

if __name__ == '__main__':
    main() 