"""
Metrics Module

This module provides utility functions for calculating various trading and performance metrics.
"""

import numpy as np
import logging

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