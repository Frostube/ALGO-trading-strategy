#!/usr/bin/env python3
"""
Performance Visualization Tool

This script parses the performance log and generates visualizations of strategy performance over time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize performance log data')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    parser.add_argument('--output', type=str, default='reports/', help='Directory to save plots')
    return parser.parse_args()

def parse_performance_log(log_path="docs/performance_log.md"):
    """
    Parse the performance log markdown file into a pandas DataFrame.
    
    Args:
        log_path (str): Path to the performance log file
        
    Returns:
        pandas.DataFrame: DataFrame containing performance data
    """
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Find the table rows (skipping header lines)
    rows = []
    for line in lines:
        if line.startswith('| ') and not line.startswith('|--') and not line.startswith('| Date'):
            rows.append(line.strip())
    
    # Parse each row
    data = []
    for row in rows:
        # Split the row by | and remove empty strings
        cells = [cell.strip() for cell in row.split('|') if cell.strip()]
        
        if len(cells) >= 8:  # Make sure we have enough cells
            try:
                date = pd.to_datetime(cells[0])
                strategy = cells[1]
                dataset = cells[2]
                params = cells[3]
                
                # Extract numeric values
                pf = float(cells[4])
                win_rate = float(cells[5])
                dd = float(cells[6])
                net_return = float(cells[7])
                
                # Parse out symbol and timeframe from dataset
                symbol_match = re.search(r'([A-Z]+/[A-Z]+)', dataset)
                timeframe_match = re.search(r'(\d+[mh])', dataset)
                
                symbol = symbol_match.group(1) if symbol_match else 'Unknown'
                timeframe = timeframe_match.group(1) if timeframe_match else 'Unknown'
                
                # Create record
                data.append({
                    'Date': date,
                    'Strategy': strategy,
                    'Symbol': symbol,
                    'Timeframe': timeframe,
                    'Dataset': dataset,
                    'Params': params,
                    'PF': pf,
                    'Win_Rate': win_rate,
                    'Max_DD': dd,
                    'Net_Return': net_return
                })
            except Exception as e:
                print(f"Error parsing row: {row}")
                print(f"Error details: {e}")
    
    # Create DataFrame
    if data:
        return pd.DataFrame(data)
    else:
        print("No data found in the performance log")
        return pd.DataFrame()

def create_performance_visualizations(df, show=False, output_dir='reports/'):
    """
    Create various performance visualizations.
    
    Args:
        df (pandas.DataFrame): DataFrame with performance data
        show (bool): Whether to show plots instead of saving
        output_dir (str): Directory to save plots
    """
    if df.empty:
        print("No data to visualize")
        return
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Strategy Performance Comparison
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Strategy', y='Net_Return', data=df, ci=None)
    plt.title('Strategy Performance Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Net Return %')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(output_path / 'strategy_comparison.png')
        plt.close()
    
    # 2. Performance metrics over time (if we have multiple dates)
    if len(df['Date'].unique()) > 1:
        # Group by date and strategy
        time_data = df.groupby(['Date', 'Strategy']).mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(x='Date', y='Net_Return', hue='Strategy', data=time_data, marker='o')
        plt.title('Strategy Performance Over Time')
        plt.xlabel('Date')
        plt.ylabel('Net Return %')
        plt.tight_layout()
        
        if show:
            plt.show()
        else:
            plt.savefig(output_path / 'performance_over_time.png')
            plt.close()
    
    # 3. Risk vs Return scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Max_DD', y='Net_Return', hue='Strategy', size='PF', sizes=(50, 250), data=df)
    plt.title('Risk vs Return by Strategy')
    plt.xlabel('Maximum Drawdown %')
    plt.ylabel('Net Return %')
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(output_path / 'risk_vs_return.png')
        plt.close()
    
    # 4. Win Rate vs Profit Factor
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Win_Rate', y='PF', hue='Strategy', size='Net_Return', sizes=(50, 250), data=df)
    plt.title('Win Rate vs Profit Factor')
    plt.xlabel('Win Rate %')
    plt.ylabel('Profit Factor')
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(output_path / 'win_rate_vs_pf.png')
        plt.close()
    
    # 5. Symbol Performance Comparison
    plt.figure(figsize=(12, 8))
    symbol_data = df.groupby('Symbol')['Net_Return'].mean().reset_index()
    sns.barplot(x='Symbol', y='Net_Return', data=symbol_data)
    plt.title('Average Performance by Symbol')
    plt.xlabel('Symbol')
    plt.ylabel('Average Net Return %')
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        plt.savefig(output_path / 'symbol_performance.png')
        plt.close()
    
    # Generate summary statistics
    stats = df.groupby('Strategy').agg({
        'PF': ['mean', 'std', 'max'],
        'Win_Rate': ['mean', 'std'],
        'Max_DD': ['mean', 'min'],
        'Net_Return': ['mean', 'std', 'max']
    }).reset_index()
    
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
    stats = stats.rename(columns={'Strategy_': 'Strategy'})
    
    # Save statistics to CSV
    stats.to_csv(output_path / 'performance_summary.csv', index=False)
    
    print(f"Visualizations and summary saved to {output_path}")

def main():
    args = parse_args()
    
    # Parse the performance log
    print("Parsing performance log...")
    performance_data = parse_performance_log()
    
    if not performance_data.empty:
        print(f"Found {len(performance_data)} performance log entries")
        # Create visualizations
        print("Creating visualizations...")
        create_performance_visualizations(performance_data, show=args.show, output_dir=args.output)
    else:
        print("No performance data found to visualize")

if __name__ == "__main__":
    main() 