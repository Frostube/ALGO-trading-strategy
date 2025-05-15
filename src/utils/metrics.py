"""
Metrics Module

This module provides utility functions for calculating various trading and performance metrics.
"""

# Minimum loss threshold to avoid division by very small numbers when calculating profit factor
MIN_LOSS = 0.01  # USD

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
    # Filter out "scratch" trades that aren't meaningful wins
    if min_win_r > 0 and losses:
        # Calculate average_risk as average of absolute loss values
        losses_abs = [abs(x) for x in losses]
        avg_risk = sum(losses_abs) / len(losses_abs) if losses_abs else 1.0
        
        # If no losing trades, use mean of wins or default to 1.0
        if not avg_risk:
            avg_risk = np.mean(wins) if wins else 1.0
        
        # Filter wins that are at least min_win_r * avg_risk
        meaningful_wins = [w for w in wins if w >= min_win_r * avg_risk]
    else:
        meaningful_wins = wins
    
    gross_profit = sum(meaningful_wins) if meaningful_wins else 0.0
    gross_loss = sum(abs(x) for x in losses) if losses else 0.0
    
    # Apply minimum loss value to prevent division by zero or unrealistic PF
    if gross_loss < min_loss:
        gross_loss = min_loss
    
    # Return 0 if no profit to avoid misleading PF values
    if gross_profit == 0:
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