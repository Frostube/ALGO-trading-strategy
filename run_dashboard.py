#!/usr/bin/env python3
"""
Dashboard runner that allows specifying a port number.
Usage: python run_dashboard.py [port]
"""
import sys
import os
import subprocess
import socket
import time

def is_port_in_use(port):
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_available_port(start_port=8501, max_attempts=10):
    """Find an available port starting from the specified port"""
    port = start_port
    attempts = 0
    
    while attempts < max_attempts:
        if not is_port_in_use(port):
            return port
        port += 1
        attempts += 1
    
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

def main():
    """Main entry point for the script."""
    # Default port
    port = 8501
    
    # Check if a port was specified
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}")
            sys.exit(1)
    
    # If the port is in use, find an available one
    if is_port_in_use(port):
        print(f"Port {port} is already in use.")
        port = find_available_port(port)
        print(f"Using port {port} instead.")
    
    # Run the dashboard
    print(f"Starting enhanced Streamlit dashboard on port {port}...")
    
    # Run the dashboard
    cmd = ["python", "-m", "streamlit", "run", "src/dashboard/enhanced_dashboard.py", "--server.port", str(port)]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    main() 