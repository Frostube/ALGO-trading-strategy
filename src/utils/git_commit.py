#!/usr/bin/env python3
"""
Utility to commit and optionally push performance log changes to Git.
"""

import subprocess
import os
import sys
from pathlib import Path

def commit_performance_log(push=False):
    """
    Commit performance_log.md to Git and optionally push to remote.
    
    Args:
        push (bool): Whether to push the commit to remote repository
        
    Returns:
        bool: True if successful, False otherwise
    """
    log_path = Path("docs/performance_log.md")
    
    if not log_path.exists():
        print(f"Performance log file not found at {log_path}")
        return False
    
    try:
        # Configure Git username and email if not set
        try:
            subprocess.run(["git", "config", "user.name"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "config", "--global", "user.name", "Performance Bot"], check=True)
            
        try:
            subprocess.run(["git", "config", "user.email"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "config", "--global", "user.email", "bot@example.com"], check=True)
        
        # Add the performance log file to Git
        subprocess.run(["git", "add", str(log_path)], check=True)
        
        # Commit the changes
        result = subprocess.run(
            ["git", "commit", "-m", "chore: update performance log with new backtest results"],
            check=True,
            capture_output=True
        )
        
        if "nothing to commit" in result.stdout.decode() or "nothing to commit" in result.stderr.decode():
            print("No changes to performance log to commit")
            return True
        
        print("Successfully committed performance log changes")
        
        # Push if requested
        if push:
            subprocess.run(["git", "push"], check=True)
            print("Successfully pushed changes to remote")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operation: {e}")
        print(f"Command output: {e.stdout.decode() if e.stdout else ''}")
        print(f"Command error: {e.stderr.decode() if e.stderr else ''}")
        return False

def auto_commit_log(push=False):
    """
    Automatically commit all performance log files to Git.
    
    Args:
        push (bool): Whether to push the commit to remote repository
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Define all log files to track
    log_files = [
        Path("docs/performance_log.md"),
        Path("docs/detailed_performance_log.json")
    ]
    
    # Filter to only existing files
    existing_files = [f for f in log_files if f.exists()]
    
    if not existing_files:
        print("No performance log files found")
        return False
    
    try:
        # Configure Git username and email if not set
        try:
            subprocess.run(["git", "config", "user.name"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "config", "--global", "user.name", "Performance Bot"], check=True)
            
        try:
            subprocess.run(["git", "config", "user.email"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            subprocess.run(["git", "config", "--global", "user.email", "bot@example.com"], check=True)
        
        # Add all log files to Git
        for log_file in existing_files:
            subprocess.run(["git", "add", str(log_file)], check=True)
        
        # Commit the changes
        result = subprocess.run(
            ["git", "commit", "-m", "chore: update performance logs with new backtest results"],
            check=True,
            capture_output=True
        )
        
        stdout = result.stdout.decode() if result.stdout else ""
        stderr = result.stderr.decode() if result.stderr else ""
        
        if "nothing to commit" in stdout or "nothing to commit" in stderr:
            print("No changes to performance logs to commit")
            return True
        
        print("Successfully committed performance log changes")
        
        # Push if requested
        if push:
            subprocess.run(["git", "push"], check=True)
            print("Successfully pushed changes to remote")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error in Git operation: {e}")
        print(f"Command output: {e.stdout.decode() if e.stdout else ''}")
        print(f"Command error: {e.stderr.decode() if e.stderr else ''}")
        return False

if __name__ == "__main__":
    # Allow command-line usage: python git_commit.py [--push]
    push = "--push" in sys.argv
    auto_commit_log(push=push) 