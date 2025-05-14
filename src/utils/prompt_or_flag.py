import os

def should_log(auto=False):
    """
    Determines whether to log the performance results.
    
    Returns True if either:
      • --log / AUTO_LOG=1 flag is set   → auto=True
      • user answers 'y' at the prompt   → auto=False
    
    Args:
        auto (bool): Whether to automatically log without prompt
        
    Returns:
        bool: True if results should be logged, False otherwise
    """
    # Debug output
    print(f"should_log called with auto={auto}, AUTO_LOG={os.getenv('AUTO_LOG')}")
    
    if auto or os.getenv("AUTO_LOG") == "1":
        print("Auto-logging enabled")
        return True

    ans = input("Log this run to performance_log.md? [y/N]: ").strip().lower()
    return ans == "y"


def check_prompt_or_flag(log_flag, no_log_flag, message):
    """
    Determines whether to log the performance results based on flags or user prompt.
    
    Args:
        log_flag (bool): Flag to always log
        no_log_flag (bool): Flag to never log
        message (str): Message to display in the prompt
        
    Returns:
        bool: True if results should be logged, False otherwise
    """
    # If AUTO_LOG environment variable is set, respect that
    if os.getenv("AUTO_LOG") == "1":
        return True
    
    # If auto-log flag is set, log without prompting
    if log_flag:
        return True
    
    # If no-log flag is set, don't log without prompting
    if no_log_flag:
        return False
    
    # Otherwise, prompt the user
    try:
        ans = input(f"{message} [y/N]: ").strip().lower()
        return ans == "y"
    except:
        # In case of issues with input (e.g., non-interactive environments)
        return False


def is_result_sane(pf, win, dd, net):
    """
    Checks if performance metrics are within reasonable bounds.
    
    Args:
        pf (float): Profit Factor
        win (float): Win Rate percentage
        dd (float): Maximum Drawdown percentage
        net (float): Net Return percentage
        
    Returns:
        bool: True if all metrics are within sane limits
    """
    # Define reasonable bounds for each metric
    return (0 < pf < 10 and 
            0 <= win <= 100 and 
            -50 < dd < 50 and 
            abs(net) < 500) 