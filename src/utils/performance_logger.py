import pathlib
import datetime
import os
import traceback
import json

LOG_F = pathlib.Path("docs/performance_log.md")
DETAILED_LOG_F = pathlib.Path("docs/detailed_performance_log.json")

HEADER = (
    "# Strategy Performance Log\n\n"
    "| Date (UTC) | Strategy | Dataset | Params | PF | Win % | DD % | Net Return % |\n"
    "|------------|----------|---------|--------|----|-------|------|--------------|\n"
)

def append_result(strategy_name: str,
                  dataset: str,
                  params: str,
                  pf: float,
                  win: float,
                  dd: float,
                  net: float):
    """
    Append a performance result to the performance log markdown file.
    
    Args:
        strategy_name (str): Name of the strategy
        dataset (str): Description of the dataset used
        params (str): Key parameters used
        pf (float): Profit Factor
        win (float): Win Rate percentage
        dd (float): Maximum Drawdown percentage
        net (float): Net Return percentage
    """
    try:
        print(f"Appending result to {LOG_F} (exists: {LOG_F.exists()})")
        
        # Create directory if it doesn't exist
        LOG_F.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with header if it doesn't exist
        if not LOG_F.exists():
            print(f"Creating new log file with header")
            with open(LOG_F, "w") as f:
                f.write(HEADER)
        
        # Format the log entry
        line = f"| {datetime.datetime.utcnow():%Y-%m-%d %H:%M} | {strategy_name} | {dataset} | {params} | {pf:.2f} | {win:.1f} | {dd:.1f} | {net:.1f} |\n"
        
        # Append to the file
        print(f"Appending line: {line.strip()}")
        with open(LOG_F, "a") as f:
            f.write(line)
        
        print(f"✅ Performance logged to {LOG_F}")
        return True
        
    except Exception as e:
        print(f"Error logging performance: {e}")
        traceback.print_exc()
        return False 

def log_performance_results(results: dict):
    """
    Log detailed performance results to both markdown and JSON files.
    
    Args:
        results (dict): Dictionary containing performance results
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        LOG_F.parent.mkdir(parents=True, exist_ok=True)
        
        # Create file with header if it doesn't exist
        if not LOG_F.exists():
            with open(LOG_F, "w") as f:
                f.write(HEADER)
        
        # Extract key metrics for markdown log
        strategy_name = results.get('strategy', 'Unknown')
        
        # Build dataset string
        symbol = results.get('symbol', results.get('symbols', 'Unknown'))
        timeframe = results.get('timeframe', 'Unknown')
        period = results.get('period', 'Unknown')
        dataset = f"{symbol} {timeframe} ({period})"
        
        # Build params string
        params = results.get('params', '')
        if not params and 'parameters' in results:
            # Convert parameters dict to string
            p = results['parameters']
            params = (f"EMA {p.get('ema_fast', 'N/A')}/{p.get('ema_slow', 'N/A')}/{p.get('ema_trend', 'N/A')} | "
                     f"Risk {p.get('risk_per_trade', 0) * 100:.2f}% | "
                     f"{'✓' if p.get('enable_pyramiding', False) else '✗'} Pyramid")
        
        # Extract metrics
        pf = results.get('profit_factor', 0)
        win_rate = results.get('win_rate', 0)
        max_dd = results.get('max_drawdown', 0)
        net_return = results.get('pnl_percentage', 0)
        
        # Format the log entry for markdown
        line = f"| {datetime.datetime.utcnow():%Y-%m-%d %H:%M} | {strategy_name} | {dataset} | {params} | {pf:.2f} | {win_rate:.1f} | {max_dd:.1f} | {net_return:.1f} |\n"
        
        # Append to the markdown file
        with open(LOG_F, "a") as f:
            f.write(line)
        
        # Save detailed results to JSON
        # First read existing data if file exists
        detailed_results = []
        if DETAILED_LOG_F.exists():
            try:
                with open(DETAILED_LOG_F, "r") as f:
                    detailed_results = json.load(f)
            except json.JSONDecodeError:
                detailed_results = []
        
        # Add timestamp if not present
        if 'timestamp' not in results:
            results['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add the new results
        detailed_results.append(results)
        
        # Write updated results
        with open(DETAILED_LOG_F, "w") as f:
            json.dump(detailed_results, f, indent=2)
        
        print(f"✅ Performance logged to {LOG_F} and {DETAILED_LOG_F}")
        return True
        
    except Exception as e:
        print(f"Error logging performance results: {e}")
        traceback.print_exc()
        return False 