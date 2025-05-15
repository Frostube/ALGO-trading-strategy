"""
Metrics Module

This module provides utility functions for calculating various trading and performance metrics.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

# Minimum loss threshold to avoid division by very small numbers when calculating profit factor
MIN_LOSS = 0.01  # USD
# Commission per trade (one-way)
COMMISSION = 0.04  # $0.04 per trade or whatever your exchange charges
# Maximum R-multiple cap to prevent outliers from skewing results
MAX_R = 20.0  # Cap extreme R values

logger = logging.getLogger(__name__)

def calculate_r_multiples(trades):
    """
    Convert trade results to R-multiples and calculate expectancy.
    
    Args:
        trades (list): List of trade dictionaries
        
    Returns:
        dict: R-multiple metrics including expectancy
    """
    for t in trades:
        if all(k in t for k in ('entry_price','exit_price','stop_loss')):
            risk = abs(t['entry_price'] - t['stop_loss'])
            if risk > 0:
                is_long = t.get('side','').lower() in ('buy','long') or t.get('type','').lower() == 'long'
                pnl = (t['exit_price'] - t['entry_price']) if is_long \
                      else (t['entry_price'] - t['exit_price'])
                # Cap extreme R values to prevent outliers from skewing results
                t['r_multiple'] = min(pnl / risk, MAX_R)
            else:
                t['r_multiple'] = 0
                logger.warning(f"Zero risk detected in trade: {t}")

    r_vals = [t.get('r_multiple', 0) for t in trades if 'r_multiple' in t]
    wins   = [r for r in r_vals if r > 0]
    losses = [r for r in r_vals if r <= 0]

    avg_r      = np.mean(r_vals)      if r_vals   else 0
    avg_win_r  = np.mean(wins)        if wins     else 0
    avg_loss_r = abs(np.mean(losses)) if losses   else 0
    win_rate   = len(wins) / len(r_vals) if r_vals else 0
    expectancy = win_rate * avg_win_r - (1-win_rate) * avg_loss_r
    
    # Calculate total R risked - each trade is 1R by definition
    total_r_risked = len(r_vals)

    return {
        'r_values':     r_vals,
        'avg_r':        avg_r,
        'avg_win_r':    avg_win_r,
        'avg_loss_r':   avg_loss_r,
        'win_rate':     win_rate,
        'expectancy':   expectancy,
        'total_r_risked': total_r_risked,
        'pct_trades_over_2r': len([r for r in r_vals if r >= 2.0]) / len(r_vals) if r_vals else 0,
        'pct_trades_over_5r': len([r for r in r_vals if r >= 5.0]) / len(r_vals) if r_vals else 0
    }

def profit_factor(wins, losses, min_loss=0.01, min_win_r=0.1):
    """
    Calculate profit factor with stabilization for small losses and minimum win threshold.
    
    Args:
        wins: List of winning trade profits
        losses: List of losing trade losses (negative values)
        min_loss: Minimum loss value to use (prevents division by zero)
        min_win_r: Minimum R-multiple for a trade to count as a win
        
    Returns:
        float: Profit Factor (gross profit / gross loss)
    """
    # Handle empty lists
    if not wins and not losses:
        return 0.0
    
    # Ensure all values are valid numbers
    valid_wins = []
    for w in wins:
        try:
            w_float = float(w)
            if not np.isnan(w_float):
                valid_wins.append(w_float)
        except (TypeError, ValueError):
            continue
    
    valid_losses = []
    for l in losses:
        try:
            l_float = float(l)
            if not np.isnan(l_float):
                valid_losses.append(l_float)
        except (TypeError, ValueError):
            continue
    
    # If we have no valid values after filtering, return 0
    if not valid_wins and not valid_losses:
        return 0.0
    
    # Filter out "scratch" trades that aren't meaningful wins
    if min_win_r > 0 and valid_losses:
        # Calculate average_risk as average of absolute loss values
        losses_abs = [abs(x) for x in valid_losses]
        avg_risk = sum(losses_abs) / len(losses_abs) if losses_abs else 1.0
        
        # If no losing trades, use mean of wins or default to 1.0
        if not avg_risk:
            avg_risk = np.mean(valid_wins) if valid_wins else 1.0
        
        # Filter wins that are at least min_win_r * avg_risk
        meaningful_wins = [w for w in valid_wins if w >= min_win_r * avg_risk]
    else:
        meaningful_wins = valid_wins
    
    # Include commission in the calculations
    gross_profit = sum(meaningful_wins) - (len(meaningful_wins) * COMMISSION) if meaningful_wins else 0.0
    gross_loss = sum(abs(x) for x in valid_losses) + (len(valid_losses) * COMMISSION) if valid_losses else 0.0
    
    # Apply minimum loss value to prevent division by zero or unrealistic PF
    if gross_loss < min_loss:
        gross_loss = min_loss
    
    # Return 0 if no profit to avoid misleading PF values
    if gross_profit <= 0:
        return 0.0
        
    return gross_profit / gross_loss


def win_rate(winners, losers):
    """
    Calculate win rate.
    
    Win rate = Number of winning trades / Total number of trades
    
    Args:
        winners (list): List of winning trades
        losers (list): List of losing trades
    
    Returns:
        float: Win rate as a decimal (0.0 to 1.0)
    """
    total_trades = len(winners) + len(losers)
    if total_trades == 0:
        return 0.0
    return len(winners) / total_trades


def expected_return(winners, losers):
    """
    Calculate expected return per trade.
    
    Expected return = (Win rate × Average win) - (Loss rate × Average loss)
    
    Args:
        winners (list): List of winning trade amounts
        losers (list): List of losing trade amounts (should be negative values)
    
    Returns:
        float: Expected return per trade
    """
    total_trades = len(winners) + len(losers)
    if total_trades == 0:
        return 0.0
    
    win_rate_val = len(winners) / total_trades
    loss_rate = 1.0 - win_rate_val
    
    avg_win = sum(winners) / len(winners) if winners else 0
    avg_loss = abs(sum(losers) / len(losers)) if losers else 0
    
    return (win_rate_val * avg_win) - (loss_rate * avg_loss)


def drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve.
    
    Args:
        equity_curve (list): List of equity values over time
    
    Returns:
        float: Maximum drawdown as a decimal (0.0 to 1.0)
    """
    if not equity_curve:
        return 0.0
    
    peak = equity_curve[0]
    max_dd = 0.0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        current_dd = (peak - value) / peak if peak > 0 else 0
        max_dd = max(max_dd, current_dd)
    
    return max_dd


def calculate_expectancy(trades):
    """
    Calculate system expectancy from a list of trade dictionaries.
    Robust to missing keys and empty trade lists.
    
    Args:
        trades (list): List of trade dictionaries with 'pnl' key
        
    Returns:
        dict: Dictionary containing expectancy metrics
    """
    if not trades:
        return {
            'expectancy': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'trades_count': 0
        }
    
    # Extract PnL values safely
    pnls = []
    for i, trade in enumerate(trades):
        # Check if trade is a valid dictionary and has a pnl key
        if not isinstance(trade, dict):
            logger.warning(f"Trade {i} is not a dictionary: {trade}")
            continue
            
        if "pnl" not in trade:
            logger.warning(f"Trade {i} is missing pnl: {trade}")
            
            # Try to calculate pnl if we have entry, exit prices and size
            if all(k in trade for k in ['entry_price', 'exit_price', 'size']):
                try:
                    # Determine if long or short
                    if trade.get('type', '').lower() == 'long' or trade.get('side', '').lower() == 'buy':
                        trade["pnl"] = (float(trade['exit_price']) - float(trade['entry_price'])) * float(trade['size'])
                    else:  # short
                        trade["pnl"] = (float(trade['entry_price']) - float(trade['exit_price'])) * float(trade['size'])
                    logger.info(f"Calculated missing pnl for trade {i}: {trade['pnl']}")
                    pnls.append(trade["pnl"])
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to calculate pnl for trade {i}: {e}")
            continue
            
        # Ensure pnl is a number
        try:
            pnl_value = float(trade["pnl"])
            if not np.isnan(pnl_value):
                pnls.append(pnl_value)
        except (ValueError, TypeError):
            logger.warning(f"Trade {i} has invalid pnl value: {trade['pnl']}")
    
    # Convert to numpy array for easier processing
    pnl_series = np.array(pnls)
    
    # If no valid PnL values, return zeros
    if pnl_series.size == 0:
        logger.warning("No valid PnL values found in trades. Returning zeros.")
        return {
            'expectancy': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'trades_count': 0
        }
    
    # Split into winners and losers
    winners = pnl_series[pnl_series > 0]
    losers = pnl_series[pnl_series <= 0]
    
    # Calculate win rate
    win_rate_val = len(winners) / len(pnl_series) if len(pnl_series) > 0 else 0
    
    # Calculate average win and loss
    avg_win = np.mean(winners) if len(winners) > 0 else 0
    avg_loss = np.mean(np.abs(losers)) if len(losers) > 0 else 0
    
    # Calculate expectancy
    if avg_loss == 0:
        expectancy = avg_win * win_rate_val
    else:
        expectancy = (avg_win * win_rate_val) - (avg_loss * (1 - win_rate_val))
    
    # Calculate profit factor
    pf = profit_factor(list(winners), list(losers))
    
    return {
        'expectancy': expectancy,
        'win_rate': win_rate_val,
        'profit_factor': pf,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'trades_count': len(pnl_series)
    }

def plot_r_multiple_distribution(trades, output_path='reports/r_multiple_dist.png'):
    """
    Create a visual distribution plot of R-multiples from a list of trades.
    
    Args:
        trades (list): List of trade dictionaries containing 'r_multiple' key
        output_path (str): Path to save the visualization
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not trades:
        logger.warning("No trades provided for R-multiple distribution plot")
        return False
    
    # Extract R-multiples
    r_values = [t.get('r_multiple', 0) for t in trades if 'r_multiple' in t]
    
    if not r_values:
        logger.warning("No R-multiple values found in trades")
        return False
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Histogram with transparency
    n, bins, patches = plt.hist(r_values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Calculate key statistics
    avg_r = np.mean(r_values)
    median_r = np.median(r_values)
    std_r = np.std(r_values)
    min_r = min(r_values)
    max_r = max(r_values)
    
    # Calculate win rate and profit factor
    wins = [r for r in r_values if r > 0]
    losses = [r for r in r_values if r <= 0]
    win_rate = len(wins) / len(r_values) if r_values else 0
    
    # Calculate percentage of trades >= 2R and >= 5R
    pct_over_2r = len([r for r in r_values if r >= 2.0]) / len(r_values) if r_values else 0
    pct_over_5r = len([r for r in r_values if r >= 5.0]) / len(r_values) if r_values else 0
    
    # Add vertical lines for key metrics
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Breakeven')
    plt.axvline(x=avg_r, color='green', linestyle='-', linewidth=2, label=f'Mean: {avg_r:.2f}R')
    plt.axvline(x=median_r, color='blue', linestyle=':', linewidth=2, label=f'Median: {median_r:.2f}R')
    
    # Customize plot
    plt.title(f'Distribution of R-Multiples ({len(r_values)} trades)', fontsize=16)
    plt.xlabel('R-Multiple (Profit/Loss in Risk Units)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = (
        f"Total Trades: {len(r_values)}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Avg R-Multiple: {avg_r:.2f}R\n"
        f"Median R: {median_r:.2f}R\n"
        f"Std Dev: {std_r:.2f}\n"
        f"Range: [{min_r:.2f}R, {max_r:.2f}R]\n"
        f"Trades ≥ 2R: {pct_over_2r:.2%}\n"
        f"Trades ≥ 5R: {pct_over_5r:.2%}"
    )
    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                 va='top', fontsize=10)
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"R-multiple distribution plot saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving R-multiple plot: {str(e)}")
        return False

def plot_equity_curve_with_r_multiples(equity_curve, trades, output_path='reports/equity_r_multiple.png'):
    """
    Plot equity curve with R-multiple annotations for major winning trades.
    
    Args:
        equity_curve (pd.Series): Equity curve with timestamps as index
        trades (list): List of trade dictionaries with r_multiple, entry_time and exit_time
        output_path (str): Path to save the visualization
        
    Returns:
        bool: True if successful, False otherwise
    """
    if equity_curve is None or len(equity_curve) == 0:
        logger.warning("No equity curve data provided")
        return False
    
    if not trades:
        logger.warning("No trades provided for R-multiple annotations")
        return False
        
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Filter trades with R-multiple info and significant winners (>=2R)
    significant_trades = [t for t in trades if t.get('r_multiple', 0) >= 2.0 and 'exit_time' in t]
    
    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Plot equity curve
    plt.plot(equity_curve.index, equity_curve.values, label='Equity', linewidth=2)
    
    # Annotate significant winning trades
    for trade in significant_trades:
        # Get the exit time and find the closest point on the equity curve
        exit_time = trade.get('exit_time')
        if isinstance(exit_time, str):
            # Try to convert string to datetime if needed
            try:
                exit_time = datetime.fromisoformat(exit_time.replace('Z', '+00:00'))
            except:
                continue
        
        # Find closest point to the exit time
        try:
            idx = equity_curve.index.get_indexer([exit_time], method='nearest')[0]
            x = equity_curve.index[idx]
            y = equity_curve.iloc[idx]
            
            # Annotate with R-multiple
            r_multiple = trade.get('r_multiple', 0)
            plt.scatter(x, y, color='green', s=100, zorder=5)
            plt.annotate(f"+{r_multiple:.1f}R", 
                         xy=(x, y),
                         xytext=(10, 10),
                         textcoords='offset points',
                         bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.8),
                         arrowprops=dict(arrowstyle="->", color="black"))
        except Exception as e:
            logger.debug(f"Error annotating trade: {str(e)}")
            continue
    
    # Calculate drawdowns
    drawdowns = []
    peak = equity_curve.iloc[0]
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100  # as percentage
        drawdowns.append(drawdown)
    
    # Plot drawdowns on secondary axis
    ax2 = plt.gca().twinx()
    ax2.fill_between(equity_curve.index, drawdowns, 0, alpha=0.3, color='red', label='Drawdown')
    ax2.set_ylabel('Drawdown %', color='red')
    ax2.set_ylim(0, max(drawdowns) * 1.2)  # Set y-axis limit with some headroom
    ax2.invert_yaxis()  # Invert so drawdowns go down
    
    # Customize plot
    plt.title('Equity Curve with Significant Winning Trades (≥2R)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Equity ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Add key metrics as text
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
    max_dd = max(drawdowns)
    win_trades = len([t for t in trades if t.get('r_multiple', 0) > 0])
    loss_trades = len([t for t in trades if t.get('r_multiple', 0) <= 0])
    win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
    
    stats_text = (
        f"Total Return: {total_return:.2f}%\n"
        f"Max Drawdown: {max_dd:.2f}%\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Trades: {win_trades + loss_trades} ({win_trades}W/{loss_trades}L)"
    )
    
    plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
                 va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    try:
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Equity curve with R-multiples plot saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving equity curve plot: {str(e)}")
        return False

def calculate_metrics(df):
    """
    Calculate basic performance metrics from signals DataFrame.
    Expects 'signal' column with -1, 0, 1 and OHLCV in index.
    
    Args:
        df: DataFrame with OHLCV data and signals
        
    Returns:
        dict: Dictionary with performance metrics
            - total_return: Total return as decimal (e.g., 0.25 for 25%)
            - profit_factor: Gross profits divided by gross losses
            - win_rate: Percentage of winning trades
            - max_drawdown: Maximum drawdown as decimal
            - n_trades: Number of trades
    """
    if df.empty or 'signal' not in df.columns:
        return {'total_return': 0, 'profit_factor': 0, 'win_rate': 0,
                'max_drawdown': 0, 'n_trades': 0}
    
    df = df.copy()
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Strategy returns (lag signals to align with next period's returns)
    df['strategy_returns'] = df['signal'].shift(1) * df['returns']
    
    # Count trades
    df['trade_entry'] = df['signal'].diff().abs() > 0
    n_trades = df['trade_entry'].sum()
    
    # Calculate total return
    total_return = (1 + df['strategy_returns'].fillna(0)).cumprod().iloc[-1] - 1
    
    # Calculate win rate and profit factor
    if n_trades > 0:
        # Extract individual trade returns
        trade_starts = df[df['trade_entry']].index
        
        win_trades = 0
        loss_trades = 0
        gross_profits = 0
        gross_losses = 0
        
        for i, start in enumerate(trade_starts):
            if i < len(trade_starts) - 1:
                end = trade_starts[i+1]
                trade_return = df.loc[start:end, 'strategy_returns'].sum()
            else:
                trade_return = df.loc[start:, 'strategy_returns'].sum()
            
            if trade_return > 0:
                win_trades += 1
                gross_profits += trade_return
            else:
                loss_trades += 1
                gross_losses += abs(trade_return)
                
        win_rate = win_trades / n_trades if n_trades > 0 else 0
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    # Calculate max drawdown
    cum_returns = (1 + df['strategy_returns'].fillna(0)).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    max_drawdown = drawdown.min()
    
    return {
        'total_return': total_return,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades
    }

def plot_equity_curve(equity_curves, output_path=None):
    """
    Plot equity curves for one or more symbols.
    
    Args:
        equity_curves: Dictionary mapping symbols to equity curves or single equity curve list
        output_path: Path to save the plot (optional)
        
    Returns:
        str: Output path if saved, None otherwise
    """
    plt.figure(figsize=(12, 6))
    
    if isinstance(equity_curves, dict):
        # Multiple symbols
        for symbol, curve in equity_curves.items():
            # Convert to percentage return
            initial = curve[0]
            pct_curve = [(value / initial - 1) * 100 for value in curve]
            plt.plot(pct_curve, label=symbol)
        
        plt.legend()
        plt.title('Equity Curves by Symbol (% Return)')
    else:
        # Single equity curve
        initial = equity_curves[0]
        pct_curve = [(value / initial - 1) * 100 for value in equity_curves]
        plt.plot(pct_curve)
        plt.title('Equity Curve (% Return)')
    
    plt.xlabel('Trades')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        return None

def plot_drawdown_curve(equity_curve, output_path=None):
    """
    Plot drawdown curve from equity curve.
    
    Args:
        equity_curve: List of equity values over time
        output_path: Path to save the plot (optional)
        
    Returns:
        str: Output path if saved, None otherwise
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate drawdowns
    peak = equity_curve[0]
    drawdowns = []
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        current_dd = (peak - value) / peak * 100 if peak > 0 else 0
        drawdowns.append(current_dd)
    
    plt.plot(drawdowns, 'r')
    plt.fill_between(range(len(drawdowns)), drawdowns, alpha=0.3, color='r')
    plt.title('Drawdown (%)')
    plt.xlabel('Trades')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
        return output_path
    else:
        plt.show()
        return None 