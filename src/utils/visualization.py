import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import os
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from src.backtest.backtest import Backtester, run_backtest
from src.data.fetcher import DataFetcher
from src.utils.logger import logger
from src.strategy.ema_crossover import EMACrossoverStrategy


def create_test_data():
    """
    Create sample backtest results for visualization testing.
    Returns a dictionary simulating backtest results.
    """
    return {
        "trades": [
            {
                "entry_time": pd.Timestamp("2023-01-01 00:00:00"),
                "exit_time": pd.Timestamp("2023-01-02 00:00:00"),
                "entry_price": 30000,
                "exit_price": 31000,
                "size": 0.1,
                "type": "long",
                "commission": 12.0,
                "slippage": 8.0,
                "funding": 5.0,
                "gross_pnl": 100.0,
                "net_pnl": 75.0
            },
            {
                "entry_time": pd.Timestamp("2023-01-03 00:00:00"),
                "exit_time": pd.Timestamp("2023-01-04 00:00:00"),
                "entry_price": 31000,
                "exit_price": 30500,
                "size": 0.1,
                "type": "short",
                "commission": 11.0,
                "slippage": 7.0,
                "funding": 4.0,
                "gross_pnl": 50.0,
                "net_pnl": 28.0
            }
        ],
        "equity_curve": [10000, 10075, 10103],
        "total_commission": 23.0,
        "total_slippage": 15.0,
        "total_funding": 9.0,
        "gross_pnl": 150.0,
        "net_pnl": 103.0,
        "fee_impact_pct": 31.33
    }


def plot_fee_impact(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 90,
    initial_balance: float = 10_000,
    risk_per_trade: float = 0.01,
    save_path: str = None,
    use_test_data: bool = False
):
    """
    Runs a backtest on `symbol` for the past `days` days, then plots:
      1) Gross vs. Net returns
      2) Fee breakdown (commission, slippage, funding)
      
    Args:
        symbol: Trading pair to backtest
        timeframe: Timeframe to use (e.g., "1h", "4h", "1d")
        days: Number of days of historical data to use
        initial_balance: Starting account balance
        risk_per_trade: Risk per trade as a decimal (e.g., 0.01 = 1%)
        save_path: If provided, save charts to this directory instead of displaying
        use_test_data: If True, use test data instead of running a backtest
        
    Returns:
        dict: Summary of fee impact metrics
    """
    logger.info(f"Running fee impact analysis for {symbol} over {days} days on {timeframe} timeframe")
    
    # Use test data if requested (for development/debugging)
    if use_test_data:
        test_results = create_test_data()
        trades = test_results["trades"]
        gross_pnl = test_results["gross_pnl"]
        net_pnl = test_results["net_pnl"]
        total_commission = test_results["total_commission"]
        total_slippage = test_results["total_slippage"]
        total_funding = test_results["total_funding"]
    else:
        # 1) Get data
        data_fetcher = DataFetcher(use_testnet=True)
        data = data_fetcher.fetch_historical_data(symbol=symbol, timeframe=timeframe, days=days)
        data_fetcher.close()
        
        if data is None or len(data) < 20:
            logger.error(f"Insufficient data for {symbol}")
            return None
        
        # 2) Set up backtester with our standardized fee model wrapper
        backtester = Backtester(
            data=data, 
            initial_balance=initial_balance,
            params={
                'risk_per_trade': risk_per_trade,
                'commission': 0.0004,  # Taker fee (0.04%)
                'slippage': 0.0005     # Slippage (0.05%)
            }
        )
        
        # 3) Create a patch for the fee model execute_order method to fix field naming
        original_execute_order = backtester.fee_model.execute_order
        
        def execute_order_wrapper(*args, **kwargs):
            """Wrapper that standardizes field names"""
            result = original_execute_order(*args, **kwargs)
            # Add actual_price field expected by backtester
            result['actual_price'] = result['executed_price']
            # Add slippage_amount field expected by backtester
            result['slippage_amount'] = result['order_value'] * (result['slippage_pct'] / 100)
            return result
            
        # Apply the patch
        backtester.fee_model.execute_order = execute_order_wrapper
        
        # 4) Run the backtest
        strategy = EMACrossoverStrategy(symbol=symbol)
        results = backtester._backtest_strategy(strategy, data)
        
        if not results or 'trades' not in results or len(results['trades']) == 0:
            logger.warning("No trades generated in backtest")
            return None
        
        # 5) Extract metrics
        trades = results['trades']
        
        # Calculate fee components from trades
        total_commission = sum(t.get('commission', 0) for t in trades)
        total_slippage = sum(t.get('slippage', 0) for t in trades)
        total_funding = sum(t.get('funding', 0) for t in trades)
        
        # Calculate gross and net PnL
        gross_pnl = sum(t.get('gross_pnl', 0) for t in trades)
        net_pnl = sum(t.get('net_pnl', 0) for t in trades)
        
        # If gross_pnl or net_pnl not directly available, calculate them
        if gross_pnl == 0:
            # Try to calculate from the individual trades
            gross_pnl = sum([
                (t['exit_price'] - t['entry_price']) * t['size'] if t['type'] == 'long' else
                (t['entry_price'] - t['exit_price']) * t['size'] for t in trades
            ])
        
        if net_pnl == 0:
            # Use the sum of individual trade PnLs
            net_pnl = sum(t.get('pnl', 0) for t in trades)
    
    # 6) Create visualization
    # Setup figure and grid
    fig = plt.figure(figsize=(12, 10))
    grid = plt.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # --- Chart 1: Gross vs. Net Returns ---
    ax1 = fig.add_subplot(grid[0, :])
    bars = ax1.bar(["Gross Return", "Net Return"], [gross_pnl, net_pnl], 
           color=['#72B6E4', '#1C6FAD'])
    
    # Add value labels to bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom' if height > 0 else 'top')
                
    # Add a line showing the fee impact percentage
    if gross_pnl != 0:
        fee_impact_pct = ((gross_pnl - net_pnl) / abs(gross_pnl)) * 100
        ax1.text(0.5, 0.9, f"Fee Impact: {fee_impact_pct:.2f}% of gross returns", 
                transform=ax1.transAxes, ha='center', 
                bbox=dict(facecolor='#FFF9C4', alpha=0.5))
    
    ax1.set_title(f"{symbol} {days}-day Backtest: Gross vs. Net P/L", fontsize=14)
    ax1.set_ylabel("P/L (USD)")
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Chart 2: Fee Breakdown ---
    ax2 = fig.add_subplot(grid[1, 0])
    fee_types = ["Commission", "Slippage", "Funding"]
    fee_values = [total_commission, total_slippage, total_funding]
    
    bars = ax2.bar(fee_types, fee_values, color=['#FF9E80', '#FF6E40', '#DD2C00'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom')
    
    ax2.set_title("Fee Breakdown by Type", fontsize=12)
    ax2.set_ylabel("Total Fees (USD)")
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # --- Chart 3: Fee Impact Per Trade ---
    ax3 = fig.add_subplot(grid[1, 1])
    
    # Calculate fee impact per trade
    trade_sizes = [abs(t.get('size', 0) * t.get('entry_price', 0)) for t in trades]
    trade_fees = [t.get('commission', 0) + t.get('slippage', 0) + t.get('funding', 0) for t in trades]
    fee_impact_per_trade = [100 * fee / size if size > 0 else 0 for fee, size in zip(trade_fees, trade_sizes)]
    
    # Create a histogram of fee impact percentages
    ax3.hist(fee_impact_per_trade, bins=10, color='#4DB6AC', alpha=0.7)
    ax3.set_title("Distribution of Fee Impact Per Trade", fontsize=12)
    ax3.set_xlabel("Fee Impact (% of trade size)")
    ax3.set_ylabel("Number of Trades")
    ax3.grid(linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(f"Fee Impact Analysis: {symbol} ({timeframe} timeframe, {days} days)", 
                fontsize=16, y=0.98)
    
    # Save or show
    if save_path:
        # Create directory if it doesn't exist
        save_dir = Path(save_path)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        # Save the figure
        fig_path = save_dir / f"fee_impact_{symbol.replace('/', '_')}_{timeframe}_{days}d.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved fee impact visualization to {fig_path}")
    else:
        plt.tight_layout()
        plt.show()
    
    # Return summary metrics
    return {
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fee_impact_pct": ((gross_pnl - net_pnl) / abs(gross_pnl)) * 100 if gross_pnl != 0 else 0,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_funding": total_funding,
        "total_fees": total_commission + total_slippage + total_funding,
        "trade_count": len(trades),
        "avg_fee_per_trade": sum(trade_fees) / len(trades) if trades else 0,
        "avg_fee_impact_pct": sum(fee_impact_per_trade) / len(fee_impact_per_trade) if fee_impact_per_trade else 0
    }


def compare_fee_tiers(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 90,
    initial_balance: float = 10_000,
    fee_tiers: list = None,
    save_path: str = None,
    use_test_data: bool = False
):
    """
    Compare different fee tiers for the same strategy to see profitability at different fee levels.
    
    Args:
        symbol: Trading pair to backtest
        timeframe: Timeframe to use (e.g., "1h", "4h", "1d")
        days: Number of days of historical data to use
        initial_balance: Starting account balance
        fee_tiers: List of dictionaries with fee parameters. If None, uses default tiers.
        save_path: If provided, save chart to this directory instead of displaying
        use_test_data: If True, use test data instead of running a backtest
        
    Returns:
        dict: Results for each fee tier
    """
    if fee_tiers is None:
        # Default fee tiers (VIP levels on most exchanges)
        fee_tiers = [
            {"name": "Retail", "taker": 0.0010, "maker": 0.0008, "slippage": 0.0010},
            {"name": "VIP 1", "taker": 0.0008, "maker": 0.0006, "slippage": 0.0008},
            {"name": "VIP 2", "taker": 0.0006, "maker": 0.0004, "slippage": 0.0008},
            {"name": "VIP 3", "taker": 0.0004, "maker": 0.0002, "slippage": 0.0005},
            {"name": "Zero Fees", "taker": 0.0000, "maker": 0.0000, "slippage": 0.0000}
        ]
    
    logger.info(f"Comparing fee tiers for {symbol} over {days} days on {timeframe} timeframe")
    
    if use_test_data:
        # Generate fake results for testing
        results = {}
        base_pnl = 150.0
        for tier in fee_tiers:
            # Simulate different results based on fee levels
            fee_cost = base_pnl * (tier['taker'] * 2 + tier['slippage'])
            results[tier['name']] = {
                "net_pnl": base_pnl - fee_cost,
                "gross_pnl": base_pnl,
                "total_fees": fee_cost,
                "roi": ((base_pnl - fee_cost) / initial_balance) * 100,
                "trades": 10
            }
    else:
        # 1) Get data once
        data_fetcher = DataFetcher(use_testnet=True)
        data = data_fetcher.fetch_historical_data(symbol=symbol, timeframe=timeframe, days=days)
        data_fetcher.close()
        
        if data is None or len(data) < 20:
            logger.error(f"Insufficient data for {symbol}")
            return None
        
        # Results for each fee tier
        results = {}
        
        for tier in fee_tiers:
            logger.info(f"Testing {tier['name']} fee tier")
            
            # Initialize backtester with this fee tier
            backtester = Backtester(
                data=data,
                initial_balance=initial_balance,
                params={
                    'commission': tier['taker'],  # Using taker fee as default
                    'slippage': tier['slippage']
                }
            )
            
            # Apply the execute_order wrapper to fix field naming
            original_execute_order = backtester.fee_model.execute_order
            
            def execute_order_wrapper(*args, **kwargs):
                """Wrapper that standardizes field names"""
                result = original_execute_order(*args, **kwargs)
                # Add actual_price field expected by backtester
                result['actual_price'] = result['executed_price']
                # Add slippage_amount field expected by backtester
                result['slippage_amount'] = result['order_value'] * (result['slippage_pct'] / 100)
                return result
                
            # Apply the patch
            backtester.fee_model.execute_order = execute_order_wrapper
            
            # Run strategy
            strategy = EMACrossoverStrategy(symbol=symbol)
            tier_result = backtester._backtest_strategy(strategy, data)
            
            if not tier_result or 'trades' not in tier_result or len(tier_result['trades']) == 0:
                logger.warning(f"No trades generated for {tier['name']} tier")
                continue
            
            # Extract key metrics for this tier
            trades = tier_result['trades']
            
            # Calculate PnL
            gross_pnl = sum(t.get('gross_pnl', 0) for t in trades)
            if gross_pnl == 0:
                # Try to calculate from the individual trades
                gross_pnl = sum([
                    (t['exit_price'] - t['entry_price']) * t['size'] if t['type'] == 'long' else
                    (t['entry_price'] - t['exit_price']) * t['size'] for t in trades
                ])
            
            net_pnl = sum(t.get('net_pnl', 0) for t in trades)
            if net_pnl == 0:
                # Use the sum of individual trade PnLs
                net_pnl = sum(t.get('pnl', 0) for t in trades)
            
            # Calculate fees from trades
            total_commission = sum(t.get('commission', 0) for t in trades)
            total_slippage = sum(t.get('slippage', 0) for t in trades)
            total_funding = sum(t.get('funding', 0) for t in trades)
            total_fees = total_commission + total_slippage + total_funding
            
            # Store results
            results[tier['name']] = {
                "net_pnl": net_pnl,
                "gross_pnl": gross_pnl,
                "total_fees": total_fees,
                "roi": (net_pnl / initial_balance) * 100,
                "trades": len(trades)
            }
    
    # Create visualization
    if results:
        # Sort tiers by net PnL
        tier_names = list(results.keys())
        net_pnls = [results[name]["net_pnl"] for name in tier_names]
        total_fees = [results[name]["total_fees"] for name in tier_names]
        
        # Create a paired bar chart
        x = np.arange(len(tier_names))
        width = 0.35
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Net PnL bars
        bars1 = ax1.bar(x - width/2, net_pnls, width, label='Net P&L', color='#1C6FAD')
        ax1.set_ylabel('Net P&L (USD)', color='#1C6FAD')
        
        # Total fees on secondary y-axis
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width/2, total_fees, width, label='Total Fees', color='#FF6E40')
        ax2.set_ylabel('Total Fees (USD)', color='#FF6E40')
        
        # Labels and formatting
        ax1.set_xticks(x)
        ax1.set_xticklabels(tier_names)
        ax1.set_title(f"Fee Tier Comparison: {symbol} ({timeframe} timeframe, {days} days)", fontsize=14)
        
        # Add breakeven line
        ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        
        # Add legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom' if height > 0 else 'top')
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom')
        
        # Add ROI percentages as text annotations
        for i, tier_name in enumerate(tier_names):
            roi = results[tier_name]["roi"]
            ax1.text(i, net_pnls[i] + (20 if net_pnls[i] > 0 else -20),
                    f"ROI: {roi:.2f}%",
                    ha='center', va='bottom' if net_pnls[i] > 0 else 'top',
                    bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            save_dir = Path(save_path)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)
            
            fig_path = save_dir / f"fee_tier_comparison_{symbol.replace('/', '_')}_{timeframe}_{days}d.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved fee tier comparison to {fig_path}")
        else:
            plt.show()
    
    return results


def plot_equity_curves(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    days: int = 90,
    initial_balance: float = 10_000,
    risk_per_trade: float = 0.01,
    save_path: str = None,
    use_test_data: bool = False
):
    """
    Plots gross vs net equity curves over time with drawdown overlay.
    
    Args:
        symbol: Trading pair to backtest
        timeframe: Timeframe to use (e.g., "1h", "4h", "1d")
        days: Number of days of historical data to use
        initial_balance: Starting account balance
        risk_per_trade: Risk per trade as a decimal (e.g., 0.01 = 1%)
        save_path: If provided, save charts to this directory instead of displaying
        use_test_data: If True, use test data instead of running a backtest
        
    Returns:
        dict: Summary of fee impact metrics
    """
    logger.info(f"Generating equity curves for {symbol} over {days} days on {timeframe} timeframe")
    
    if use_test_data:
        # Generate test data over time
        dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1D')
        gross_equity = [initial_balance]
        net_equity = [initial_balance]
        
        # Generate slightly different curves
        for i in range(1, len(dates)):
            # Random return that's typically positive
            daily_return = np.random.normal(0.003, 0.02)
            
            # Gross equity with no fees
            new_gross = gross_equity[-1] * (1 + daily_return)
            gross_equity.append(new_gross)
            
            # Net equity with fees (slightly lower than gross)
            fee_impact = 0.0005 * abs(daily_return) * net_equity[-1]  # More trading activity = more fees
            new_net = net_equity[-1] * (1 + daily_return) - fee_impact
            net_equity.append(new_net)
        
        # Calculate drawdowns
        gross_drawdowns = calculate_drawdowns(gross_equity)
        net_drawdowns = calculate_drawdowns(net_equity)
        
        # Calculate trade markers (one every 10 days on average)
        trade_dates = []
        trade_sizes = []
        trade_returns = []
        
        for i in range(10, len(dates), 10):
            # Add some randomness to trade timing
            idx = i + np.random.randint(-3, 3)
            if 0 <= idx < len(dates):
                trade_dates.append(dates[idx])
                # Random trade size (0.5-2% of equity)
                size = net_equity[idx] * np.random.uniform(0.005, 0.02)
                trade_sizes.append(size)
                # Random R-multiple (-2 to +3)
                r_multiple = np.random.normal(0.8, 1.2)
                trade_returns.append(r_multiple)
    else:
        # Get data
        data_fetcher = DataFetcher(use_testnet=True)
        data = data_fetcher.fetch_historical_data(symbol=symbol, timeframe=timeframe, days=days)
        data_fetcher.close()
        
        if data is None or len(data) < 20:
            logger.error(f"Insufficient data for {symbol}")
            return None
        
        # Set up backtester
        backtester = Backtester(
            data=data, 
            initial_balance=initial_balance,
            params={
                'risk_per_trade': risk_per_trade,
                'commission': 0.0004,  # Taker fee (0.04%)
                'slippage': 0.0005     # Slippage (0.05%)
            }
        )
        
        # Apply the field name wrapper
        original_execute_order = backtester.fee_model.execute_order
        
        def execute_order_wrapper(*args, **kwargs):
            """Wrapper that standardizes field names"""
            result = original_execute_order(*args, **kwargs)
            result['actual_price'] = result['executed_price']
            result['slippage_amount'] = result['order_value'] * (result['slippage_pct'] / 100)
            return result
            
        backtester.fee_model.execute_order = execute_order_wrapper
        
        # Run backtest
        strategy = EMACrossoverStrategy(symbol=symbol)
        results = backtester._backtest_strategy(strategy, data)
        
        if not results or 'trades' not in results or len(results['trades']) == 0:
            logger.warning("No trades generated in backtest")
            return None
        
        # Extract data for plotting
        trades = results['trades']
        dates = data.index
        
        # Build equity curves
        gross_equity = [initial_balance]
        net_equity = [initial_balance]
        
        # Starting values
        gross_cumulative = initial_balance
        net_cumulative = initial_balance
        trade_index = 0
        
        # Trade markers
        trade_dates = []
        trade_sizes = []
        trade_returns = []
        
        # Reconstruct equity curves by date
        for date in dates:
            # Check if a trade was executed on this date
            while trade_index < len(trades) and pd.Timestamp(trades[trade_index]['exit_time']) <= date:
                # Add trade to gross equity
                trade = trades[trade_index]
                gross_pnl = trade.get('gross_pnl', 0)
                if gross_pnl == 0:
                    # Calculate if not directly available
                    if trade['type'] == 'long':
                        gross_pnl = (trade['exit_price'] - trade['entry_price']) * trade['size']
                    else:
                        gross_pnl = (trade['entry_price'] - trade['exit_price']) * trade['size']
                
                # Add to cumulative
                gross_cumulative += gross_pnl
                net_cumulative += trade['pnl']  # Net PnL includes fees
                
                # Record trade info for markers
                trade_dates.append(pd.Timestamp(trade['exit_time']))
                trade_sizes.append(abs(trade['size'] * trade['entry_price']))
                
                # Calculate R-multiple (if risk_per_trade is available)
                if risk_per_trade > 0:
                    r_multiple = trade['pnl'] / (initial_balance * risk_per_trade)
                else:
                    r_multiple = trade['pnl'] / 100  # Reasonable default
                trade_returns.append(r_multiple)
                
                trade_index += 1
            
            # Update equity values for this date
            gross_equity.append(gross_cumulative)
            net_equity.append(net_cumulative)
        
        # Calculate drawdowns
        gross_drawdowns = calculate_drawdowns(gross_equity)
        net_drawdowns = calculate_drawdowns(net_equity)
    
    # Create visualization
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.1)
    
    # Equity curves plot
    ax1 = fig.add_subplot(gs[0])
    
    # Plot dates vs equity curves
    if use_test_data:
        ax1.plot(dates, gross_equity, label='Gross Equity', color='#4CAF50', linewidth=2)
        ax1.plot(dates, net_equity, label='Net Equity (after fees)', color='#2196F3', linewidth=2)
        
        # Calculate the difference at the end
        final_gross = gross_equity[-1]
        final_net = net_equity[-1]
        fee_impact = ((final_gross - final_net) / abs(final_gross - initial_balance)) * 100
        
        # Add markers for trades
        for i, trade_date in enumerate(trade_dates):
            r = trade_returns[i]
            color = '#4CAF50' if r > 0 else '#F44336'
            size = max(20, min(200, abs(r) * 50))  # Scale dot by R-multiple
            ax1.scatter(trade_date, net_equity[dates.get_loc(trade_date)], 
                      s=size, color=color, alpha=0.7, zorder=5)
    else:
        # Set the date axis - ensure it has same length as equity arrays
        # Fixing the dimension mismatch - the number of dates is one less than equity points
        # because we add the initial balance at the beginning
        if len(dates) + 1 == len(gross_equity):
            # We have one more equity point than dates, which is expected
            # (initial balance + one equity point per date)
            # Create x_range with same length as equity arrays
            x_range = np.arange(len(gross_equity))
            
            # Plot equity curves
            ax1.plot(x_range, gross_equity, label='Gross Equity', color='#4CAF50', linewidth=2)
            ax1.plot(x_range, net_equity, label='Net Equity (after fees)', color='#2196F3', linewidth=2)
            
            # Set x-ticks to dates - adjust to account for the initial point
            # We'll add the start date as the first tick
            start_date = dates[0] - pd.Timedelta(days=1)  # Estimate a day before first date
            date_ticks = [0, len(x_range)//4, len(x_range)//2, 3*len(x_range)//4, len(x_range)-1]
            date_labels = [start_date.strftime('%Y-%m-%d'),
                          dates[len(dates)//4 - 1].strftime('%Y-%m-%d'),
                          dates[len(dates)//2 - 1].strftime('%Y-%m-%d'),
                          dates[3*len(dates)//4 - 1].strftime('%Y-%m-%d'),
                          dates[-1].strftime('%Y-%m-%d')]
        else:
            # If the dimensions don't match as expected, adjust the equity arrays to match dates
            # by truncating or padding as needed
            if len(gross_equity) > len(dates):
                # We have more equity points than dates
                # Truncate equity arrays to match dates plus initial point
                gross_equity = gross_equity[:len(dates) + 1]
                net_equity = net_equity[:len(dates) + 1]
            else:
                # We have fewer equity points than expected
                # This shouldn't happen but let's handle it by extending the last value
                while len(gross_equity) < len(dates) + 1:
                    gross_equity.append(gross_equity[-1])
                    net_equity.append(net_equity[-1])
            
            # Create x_range with same length as adjusted equity arrays
            x_range = np.arange(len(gross_equity))
            
            # Plot equity curves
            ax1.plot(x_range, gross_equity, label='Gross Equity', color='#4CAF50', linewidth=2)
            ax1.plot(x_range, net_equity, label='Net Equity (after fees)', color='#2196F3', linewidth=2)
            
            # Set x-ticks to dates - with adjusted mapping
            date_ticks = [0, len(x_range)//4, len(x_range)//2, 3*len(x_range)//4, len(x_range)-1]
            
            # Create date labels, ensuring we don't go out of bounds
            date_labels = []
            for i, tick in enumerate(date_ticks):
                if i == 0:
                    # First tick is initial balance point
                    date_labels.append("Start")
                elif tick - 1 < len(dates):
                    date_labels.append(dates[tick - 1].strftime('%Y-%m-%d'))
                else:
                    # Last tick might be beyond dates array
                    date_labels.append(dates[-1].strftime('%Y-%m-%d'))
        
        # Set the tick positions and labels
        ax1.set_xticks(date_ticks)
        ax1.set_xticklabels(date_labels)
        
        # Calculate fee impact
        if gross_equity[-1] != initial_balance:
            fee_impact = ((gross_equity[-1] - net_equity[-1]) / abs(gross_equity[-1] - initial_balance)) * 100
        else:
            fee_impact = 0
            
        # Add trade markers
        for i, trade_date in enumerate(trade_dates):
            # Find the closest date index
            date_idx = np.searchsorted(dates, trade_date)
            if date_idx < len(dates):
                # Adjust index for the equity arrays (add 1 for initial balance point)
                equity_idx = date_idx + 1
                if equity_idx < len(net_equity):
                    r = trade_returns[i]
                    color = '#4CAF50' if r > 0 else '#F44336'
                    size = max(20, min(200, abs(r) * 50))
                    ax1.scatter(equity_idx, net_equity[equity_idx], 
                              s=size, color=color, alpha=0.7, zorder=5)
    
    # Add fee impact annotation
    ax1.text(0.02, 0.98, f"Fee Impact: {fee_impact:.2f}% of gross profit", 
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'),
            verticalalignment='top')
    
    # Format the plot
    ax1.set_title(f"Equity Curves: {symbol} ({timeframe} timeframe, {days} days)", fontsize=14)
    ax1.set_ylabel('Account Equity (USD)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Fill between the curves to highlight fee impact
    if use_test_data:
        ax1.fill_between(dates, gross_equity, net_equity, 
                       color='#F44336', alpha=0.2, label='Fee Impact')
    else:
        ax1.fill_between(x_range, gross_equity, net_equity, 
                       color='#F44336', alpha=0.2, label='Fee Impact')
    
    # Drawdown plots
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    # Ensure drawdowns have the same length as equity arrays
    if len(gross_drawdowns) != len(gross_equity):
        # Recalculate or adjust drawdowns to match
        gross_drawdowns = calculate_drawdowns(gross_equity)
        net_drawdowns = calculate_drawdowns(net_equity)
    
    # Plot drawdowns
    if use_test_data:
        ax2.fill_between(dates, 0, gross_drawdowns, color='#4CAF50', alpha=0.3, label='Gross DD')
        ax2.fill_between(dates, 0, net_drawdowns, color='#2196F3', alpha=0.3, label='Net DD')
    else:
        ax2.fill_between(x_range, 0, gross_drawdowns, color='#4CAF50', alpha=0.3, label='Gross DD')
        ax2.fill_between(x_range, 0, net_drawdowns, color='#2196F3', alpha=0.3, label='Net DD')
    
    # Annotate max drawdowns
    max_gross_dd = max(gross_drawdowns)
    max_net_dd = max(net_drawdowns)
    
    if use_test_data:
        max_gross_dd_idx = np.argmax(gross_drawdowns)
        max_net_dd_idx = np.argmax(net_drawdowns)
        
        ax2.annotate(f'Max DD: {max_gross_dd:.2%}', 
                    xy=(dates[max_gross_dd_idx], max_gross_dd),
                    xytext=(dates[max_gross_dd_idx], max_gross_dd + 0.02),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    else:
        max_gross_dd_idx = np.argmax(gross_drawdowns)
        max_net_dd_idx = np.argmax(net_drawdowns)
        
        ax2.annotate(f'Max DD: {max_gross_dd:.2%}', 
                    xy=(max_gross_dd_idx, max_gross_dd),
                    xytext=(max_gross_dd_idx, max_gross_dd + 0.02),
                    arrowprops=dict(facecolor='black', arrowstyle='->'),
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    ax2.set_ylabel('Drawdown')
    ax2.set_ylim(0, max(max_gross_dd, max_net_dd) * 1.5)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Calculate and plot rolling fee impact
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    
    # Calculate fee impact over time (as a percentage)
    fee_impacts = []
    window_size = min(20, len(gross_equity) // 5)  # Adjust window size based on data length
    
    for i in range(len(gross_equity)):
        if i < window_size:
            # Not enough data for rolling window
            fee_impact = 0
        else:
            # Calculate fee impact over the window
            gross_change = gross_equity[i] - gross_equity[i-window_size]
            net_change = net_equity[i] - net_equity[i-window_size]
            
            if abs(gross_change) > 0.01:  # Avoid division by small numbers
                impact = 100 * (gross_change - net_change) / abs(gross_change)
            else:
                impact = 0
                
            fee_impacts.append(impact)
    
    # Ensure fee_impacts has the same length as equity points for plotting
    if len(fee_impacts) < len(gross_equity):
        # Pad the beginning with zeros to match length
        fee_impacts = [0] * (len(gross_equity) - len(fee_impacts)) + fee_impacts
    
    # Plot fee impact over time
    if use_test_data:
        ax3.plot(dates, fee_impacts, color='#F44336', label='Rolling Fee Impact %')
    else:
        ax3.plot(x_range, fee_impacts, color='#F44336', label='Rolling Fee Impact %')
    
    ax3.set_ylabel('Fee Impact %')
    ax3.set_xlabel('Date')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add a horizontal line at 25% to indicate significant fee impact
    ax3.axhline(y=25, color='#F44336', linestyle='--', alpha=0.7)
    ax3.text(0.02, 0.85, "25% Impact Threshold", transform=ax3.transAxes, color='#F44336')
    
    # Format the overall plot
    plt.tight_layout()
    
    # Save or show the plot
    if save_path:
        # Create directory if it doesn't exist
        save_dir = Path(save_path)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
        
        # Save the figure
        fig_path = save_dir / f"equity_curves_{symbol.replace('/', '_')}_{timeframe}_{days}d.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved equity curves visualization to {fig_path}")
    else:
        plt.show()
    
    # Return summary metrics
    return {
        "initial_balance": initial_balance,
        "final_gross_equity": gross_equity[-1],
        "final_net_equity": net_equity[-1],
        "gross_return_pct": (gross_equity[-1] / initial_balance - 1) * 100,
        "net_return_pct": (net_equity[-1] / initial_balance - 1) * 100,
        "fee_impact_pct": ((gross_equity[-1] - net_equity[-1]) / abs(gross_equity[-1] - initial_balance)) * 100 
                         if gross_equity[-1] != initial_balance else 0,
        "max_gross_drawdown": max_gross_dd * 100,
        "max_net_drawdown": max_net_dd * 100,
        "trade_count": len(trade_dates)
    }

def calculate_drawdowns(equity_curve):
    """Calculate drawdown percentage at each point in the equity curve"""
    # Initialize drawdown array
    drawdowns = np.zeros(len(equity_curve))
    
    # Track the running maximum
    peak = equity_curve[0]
    
    # Calculate drawdown at each point
    for i in range(len(equity_curve)):
        # Update peak if we have a new high
        if equity_curve[i] > peak:
            peak = equity_curve[i]
        
        # Calculate drawdown if we're below peak
        if peak > 0:  # Avoid division by zero
            drawdowns[i] = (peak - equity_curve[i]) / peak
    
    return drawdowns

if __name__ == "__main__":
    # Example usage
    plot_fee_impact(symbol="BTC/USDT", timeframe="4h", days=90, save_path="results", use_test_data=True)
    compare_fee_tiers(symbol="BTC/USDT", timeframe="4h", days=90, save_path="results", use_test_data=True)
    plot_equity_curves(symbol="BTC/USDT", timeframe="4h", days=90, save_path="results", use_test_data=True) 