"""
Metrics Module

This module provides utility functions for calculating various trading and performance metrics.
"""

# Minimum loss threshold to avoid division by very small numbers when calculating profit factor
MIN_LOSS = 0.01  # USD

def profit_factor(winners, losers):
    """
    Calculate profit factor with a minimum loss threshold to prevent extremely high values.
    
    Profit factor = Gross Profit / Gross Loss
    
    Args:
        winners (list): List of winning trade amounts
        losers (list): List of losing trade amounts (should be negative values)
    
    Returns:
        float: Profit factor, or infinity if no losses
    """
    gross_profit = sum(winners)
    gross_loss = max(abs(sum(losers)), MIN_LOSS)  # Apply minimum loss threshold
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