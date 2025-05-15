#!/usr/bin/env python3
"""
Dashboard Generator

This script generates an HTML dashboard that displays trading strategy performance metrics,
equity curves, fee analysis, and trade details from backtest results.
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Template

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate trading strategy dashboard')
    parser.add_argument('--symbol', type=str, default="BTC/USDT",
                        help='Trading symbol to analyze')
    parser.add_argument('--timeframe', type=str, default="4h",
                        help='Trading timeframe')
    parser.add_argument('--days', type=int, default=30,
                        help='Number of days to backtest')
    parser.add_argument('--results-dir', type=str, default="results",
                        help='Directory containing visualization results')
    parser.add_argument('--output', type=str, default="dashboard.html",
                        help='Output HTML file path')
    parser.add_argument('--backtest-file', type=str, default=None,
                        help='Path to JSON file with backtest results (if None, uses most recent)')
    return parser.parse_args()

def find_latest_backtest_file(directory="results"):
    """Find the most recent backtest results file in the given directory"""
    result_files = list(Path(directory).glob("backtest_results_*.json"))
    if not result_files:
        return None
    # Sort by modification time, newest first
    return str(sorted(result_files, key=lambda x: x.stat().st_mtime, reverse=True)[0])

def format_currency(value):
    """Format a value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format a value as percentage"""
    return f"{value:.2f}%"

def get_color_class(value):
    """Determine CSS class based on value (positive/negative/neutral)"""
    if value > 0:
        return "positive"
    elif value < 0:
        return "negative"
    else:
        return "neutral"

def load_backtest_results(file_path):
    """Load backtest results from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def calculate_metrics(results):
    """Calculate additional metrics from backtest results"""
    metrics = {}
    
    # Extract basic metrics
    metrics['initial_balance'] = results.get('initial_balance', 10000)
    metrics['final_equity'] = results.get('final_equity', metrics['initial_balance'])
    
    # Calculate returns
    metrics['net_return_pct'] = ((metrics['final_equity'] / metrics['initial_balance']) - 1) * 100
    
    # Get trade metrics
    trades = results.get('trades', [])
    metrics['trade_count'] = len(trades)
    
    if metrics['trade_count'] > 0:
        # Win rate
        if 'win_rate' in results:
            metrics['win_rate'] = results['win_rate']
        else:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            metrics['win_count'] = len(winning_trades)
            metrics['win_rate'] = (metrics['win_count'] / metrics['trade_count']) * 100
        
        # Profit factor
        if 'profit_factor' in results:
            metrics['profit_factor'] = results['profit_factor']
        else:
            gross_profits = sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0])
            gross_losses = abs(sum([t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0]))
            metrics['profit_factor'] = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        # Average trade metrics
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        # Calculate average wins and losses
        metrics['avg_win'] = sum([t.get('pnl', 0) for t in winning_trades]) / len(winning_trades) if winning_trades else 0
        metrics['avg_loss'] = sum([t.get('pnl', 0) for t in losing_trades]) / len(losing_trades) if losing_trades else 0
        metrics['avg_trade'] = sum([t.get('pnl', 0) for t in trades]) / metrics['trade_count']
        
        # Calculate fees
        total_commission = sum([t.get('commission', 0) for t in trades])
        total_slippage = sum([t.get('slippage', 0) for t in trades])
        total_funding = sum([t.get('funding', 0) for t in trades])
        metrics['total_fees'] = total_commission + total_slippage + total_funding
        metrics['commission'] = total_commission
        metrics['slippage'] = total_slippage
        metrics['funding'] = total_funding
        
        # Calculate PnL
        if 'pnl' in results:
            metrics['net_pnl'] = results['pnl']
        else:
            metrics['net_pnl'] = sum([t.get('pnl', 0) for t in trades])
        
        # Calculate gross PnL
        total_gross_pnl = sum([t.get('gross_pnl', 0) for t in trades])
        if total_gross_pnl == 0:
            # If gross_pnl not directly available in trades, calculate it
            metrics['gross_pnl'] = metrics['net_pnl'] + metrics['total_fees']
        else:
            metrics['gross_pnl'] = total_gross_pnl
        
        # Calculate fee impact
        if abs(metrics['gross_pnl']) > 0:
            metrics['fee_impact_pct'] = (metrics['total_fees'] / abs(metrics['gross_pnl'])) * 100
        else:
            metrics['fee_impact_pct'] = 0
            
        # Drawdown metrics
        if 'max_drawdown_pct' in results:
            metrics['max_drawdown'] = results['max_drawdown_pct']
        else:
            # Use a default if not available
            metrics['max_drawdown'] = 10.0
            
        # Sharpe ratio
        if 'sharpe_ratio' in results:
            metrics['sharpe_ratio'] = results['sharpe_ratio']
        elif 'returns' in results:
            returns = results['returns']
            metrics['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
    else:
        # Default values if no trades
        metrics['win_rate'] = 0
        metrics['profit_factor'] = 0
        metrics['avg_win'] = 0
        metrics['avg_loss'] = 0
        metrics['avg_trade'] = 0
        metrics['total_fees'] = 0
        metrics['commission'] = 0
        metrics['slippage'] = 0
        metrics['funding'] = 0
        metrics['fee_impact_pct'] = 0
        metrics['max_drawdown'] = 0
        metrics['sharpe_ratio'] = 0
        metrics['gross_pnl'] = 0
        metrics['net_pnl'] = 0
        
    # Ensure we never return NaN values
    for key, value in metrics.items():
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            metrics[key] = 0
    
    return metrics

def generate_trade_table_html(trades):
    """Generate HTML for the trade details table"""
    if not trades:
        return "<p>No trades executed during this period.</p>"
    
    rows = []
    for i, trade in enumerate(trades, 1):
        # Determine if it's a winning or losing trade
        pnl = trade.get('pnl', 0)
        trade_class = "trade-win" if pnl > 0 else "trade-loss"
        trade_type = trade.get('type', 'long').lower()
        badge_class = "bg-success" if trade_type == 'long' else "bg-danger"
        
        # Calculate fees
        commission = trade.get('commission', 0)
        slippage = trade.get('slippage', 0)
        funding = trade.get('funding', 0)
        total_fees = commission + slippage + funding
        
        # Format dates
        entry_time = pd.to_datetime(trade.get('entry_time', '')).strftime('%Y-%m-%d %H:%M')
        exit_time = pd.to_datetime(trade.get('exit_time', '')).strftime('%Y-%m-%d %H:%M')
        
        # Calculate R multiple if available
        r_multiple = trade.get('r_multiple', pnl / 100)  # Default to pnl/100 if not provided
        r_sign = '+' if r_multiple > 0 else ''
        
        row = f"""
        <tr class="{trade_class}">
            <td>{i}</td>
            <td><span class="badge {badge_class}">{trade_type.capitalize()}</span></td>
            <td>{entry_time}</td>
            <td>{exit_time}</td>
            <td>{format_currency(trade.get('entry_price', 0))}</td>
            <td>{format_currency(trade.get('exit_price', 0))}</td>
            <td>{trade.get('size', 0):.4f}</td>
            <td>{format_currency(trade.get('gross_pnl', pnl + total_fees))}</td>
            <td>{format_currency(pnl)}</td>
            <td>{format_currency(total_fees)}</td>
            <td>{r_sign}{r_multiple:.2f}</td>
        </tr>
        """
        rows.append(row)
    
    return "".join(rows)

def generate_dashboard(metrics, symbol, timeframe, days, results_dir, trades):
    """Generate the HTML dashboard content"""
    # Load template
    template_path = os.path.join(os.path.dirname(__file__), "dashboard_template.html")
    
    if os.path.exists(template_path):
        with open(template_path, 'r') as f:
            template_content = f.read()
    else:
        # Default to using the existing dashboard.html as template
        with open("dashboard.html", 'r') as f:
            template_content = f.read()
    
    template = Template(template_content)
    
    # Format metrics for display
    display_metrics = {
        'net_return_pct': format_percentage(metrics['net_return_pct']),
        'net_return_class': get_color_class(metrics['net_return_pct']),
        'trade_count': metrics['trade_count'],
        'win_rate': format_percentage(metrics['win_rate']),
        'max_drawdown': format_percentage(metrics['max_drawdown']),
        'max_drawdown_class': get_color_class(-metrics['max_drawdown']),
        'final_equity': format_currency(metrics['final_equity']),
        'profit_factor': f"{metrics['profit_factor']:.2f}",
        'fee_impact_pct': format_percentage(metrics['fee_impact_pct']),
        'fee_impact_class': get_color_class(-metrics['fee_impact_pct']),
        'sharpe_ratio': f"{metrics['sharpe_ratio']:.2f}",
        'initial_balance': format_currency(metrics['initial_balance']),
        'final_gross_equity': f"{format_currency(metrics['initial_balance'] + metrics['gross_pnl'])} ({format_percentage(metrics['gross_pnl']/metrics['initial_balance']*100)})",
        'final_net_equity': f"{format_currency(metrics['initial_balance'] + metrics['net_pnl'])} ({format_percentage(metrics['net_pnl']/metrics['initial_balance']*100)})",
        'commission': format_currency(metrics['commission']),
        'slippage': format_currency(metrics['slippage']),
        'funding': format_currency(metrics['funding']),
        'total_fees': format_currency(metrics['total_fees']),
        'avg_win': format_currency(metrics['avg_win']),
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    }
    
    # Paths to visualization files
    equity_curve_path = f"results/equity_curves_{symbol.replace('/', '_')}_{timeframe}_{days}d.png"
    fee_impact_path = f"results/fee_impact_{symbol.replace('/', '_')}_{timeframe}_{days}d.png"
    fee_tier_path = f"results/fee_tier_comparison_{symbol.replace('/', '_')}_{timeframe}_{days}d.png"
    
    # Verify if files exist
    if not os.path.exists(equity_curve_path):
        equity_curve_path = ""
    if not os.path.exists(fee_impact_path):
        fee_impact_path = ""
    if not os.path.exists(fee_tier_path):
        fee_tier_path = ""
    
    # Generate trade table
    trade_table_html = generate_trade_table_html(trades)
    
    # Render the template
    return template.render(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
        metrics=display_metrics,
        equity_curve_path=equity_curve_path,
        fee_impact_path=fee_impact_path,
        fee_tier_path=fee_tier_path,
        trade_table_html=trade_table_html
    )

def main():
    """Generate the trading strategy dashboard"""
    args = parse_args()
    
    # Find backtest results file
    backtest_file = args.backtest_file
    if not backtest_file:
        backtest_file = find_latest_backtest_file()
        if not backtest_file:
            print("No backtest results found. Please run a backtest first or specify a results file.")
            return
    
    # Load backtest results
    try:
        results = load_backtest_results(backtest_file)
        print(f"Loaded backtest results from {backtest_file}")
    except Exception as e:
        print(f"Error loading backtest results: {e}")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Extract trades
    trades = results.get('trades', [])
    
    # Generate dashboard HTML
    dashboard_html = generate_dashboard(
        metrics=metrics,
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        results_dir=args.results_dir,
        trades=trades
    )
    
    # Save dashboard to file
    with open(args.output, 'w') as f:
        f.write(dashboard_html)
    
    print(f"Dashboard generated and saved to {args.output}")
    print(f"Open {os.path.abspath(args.output)} in your browser to view it.")

if __name__ == "__main__":
    main() 