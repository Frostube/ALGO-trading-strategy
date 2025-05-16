#!/usr/bin/env python3
"""
Backtest Summary Script

This script summarizes and compares backtest results from different walk-forward optimization
implementations, providing a consolidated view of performance metrics.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define paths to result files
RESULT_FILES = {
    'fastloose': 'results/fastloose_wfo_btc_4h_summary.csv',
    'simple_ema': 'results/simple_wfo_results.csv', 
    'real_wfo': 'results/real_wfo_results.csv'
}

def load_results():
    """Load results from CSV files"""
    results = {}
    
    for name, path in RESULT_FILES.items():
        if os.path.exists(path):
            results[name] = pd.read_csv(path)
            print(f"Loaded {name} results from {path} ({len(results[name])} windows)")
        else:
            print(f"Warning: {path} not found")
    
    return results

def standardize_columns(results):
    """Standardize column names across different result formats"""
    standardized = {}
    
    for name, df in results.items():
        std_df = pd.DataFrame()
        
        # Add window info
        std_df['window_id'] = df['window_id'] if 'window_id' in df.columns else range(1, len(df) + 1)
        
        # Add dates where available
        for col in ['train_start', 'train_end', 'test_start', 'test_end']:
            if col in df.columns:
                std_df[col] = df[col]
        
        # Map performance metrics with different naming conventions
        # ROI
        if 'test_roi' in df.columns:
            std_df['roi'] = df['test_roi']
        elif 'test_return' in df.columns:
            std_df['roi'] = df['test_return'] 
        elif 'roi' in df.columns:
            std_df['roi'] = df['roi']
        
        # Trades
        if 'test_trades' in df.columns:
            std_df['trades'] = df['test_trades']
        elif 'num_trades' in df.columns:
            std_df['trades'] = df['num_trades']
        elif 'total_trades' in df.columns:
            std_df['trades'] = df['total_trades']
            
        # Win rate
        if 'test_win_rate' in df.columns:
            std_df['win_rate'] = df['test_win_rate']
        elif 'win_rate' in df.columns:
            std_df['win_rate'] = df['win_rate']
        
        # Sharpe
        if 'test_sharpe' in df.columns:
            std_df['sharpe'] = df['test_sharpe']
        elif 'sharpe_ratio' in df.columns:
            std_df['sharpe'] = df['sharpe_ratio']
            
        # Profit factor
        if 'test_profit_factor' in df.columns:
            std_df['profit_factor'] = df['test_profit_factor']
        elif 'profit_factor' in df.columns:
            std_df['profit_factor'] = df['profit_factor']
        
        # Max drawdown
        if 'test_drawdown' in df.columns:
            std_df['max_drawdown'] = df['test_drawdown']
        elif 'max_drawdown' in df.columns:
            std_df['max_drawdown'] = df['max_drawdown']
        
        # Implementation name
        std_df['implementation'] = name
        
        standardized[name] = std_df
    
    return standardized

def generate_summary_statistics(standardized):
    """Generate summary statistics for each implementation"""
    summaries = {}
    
    for name, df in standardized.items():
        summary = {}
        
        # Calculate means
        for metric in ['roi', 'trades', 'win_rate', 'sharpe', 'profit_factor', 'max_drawdown']:
            if metric in df.columns:
                # Fix any inf values for stats
                values = df[metric].replace([np.inf, -np.inf], np.nan)
                summary[f'avg_{metric}'] = values.mean()
                summary[f'median_{metric}'] = values.median()
                summary[f'min_{metric}'] = values.min()
                summary[f'max_{metric}'] = values.max()
                
        # Calculate percentage of profitable windows
        if 'roi' in df.columns:
            summary['profitable_windows_pct'] = (df['roi'] > 0).mean() * 100
            
        # Total number of windows
        summary['windows'] = len(df)
        
        summaries[name] = summary
    
    return summaries

def plot_comparison(standardized):
    """Generate comparison plots of the different implementations"""
    if not standardized:
        print("No data to plot")
        return
    
    # Combine all implementation results
    combined = pd.concat(standardized.values())
    
    # Plot ROI comparison
    plt.figure(figsize=(15, 10))
    
    # ROI by implementation
    plt.subplot(2, 2, 1)
    implementations = combined['implementation'].unique()
    roi_means = [combined[combined['implementation'] == imp]['roi'].mean() for imp in implementations]
    plt.bar(implementations, roi_means, color=['blue', 'green', 'orange'])
    plt.axhline(y=0, color='red', linestyle='-')
    plt.title('Average Return on Investment by Implementation')
    plt.ylabel('ROI (%)')
    plt.grid(axis='y', alpha=0.3)
    
    # Plot trade count comparison
    plt.subplot(2, 2, 2)
    if 'trades' in combined.columns:
        trade_means = [combined[combined['implementation'] == imp]['trades'].mean() for imp in implementations]
        plt.bar(implementations, trade_means, color=['blue', 'green', 'orange'])
        plt.title('Average Trades per Window by Implementation')
        plt.ylabel('Number of Trades')
        plt.grid(axis='y', alpha=0.3)
    
    # Plot win rate comparison
    plt.subplot(2, 2, 3)
    if 'win_rate' in combined.columns:
        winrate_means = [combined[combined['implementation'] == imp]['win_rate'].mean() for imp in implementations]
        plt.bar(implementations, winrate_means, color=['blue', 'green', 'orange'])
        plt.title('Average Win Rate by Implementation')
        plt.ylabel('Win Rate (%)')
        plt.grid(axis='y', alpha=0.3)
    
    # Plot profit factor comparison
    plt.subplot(2, 2, 4)
    if 'profit_factor' in combined.columns:
        # Cap profit factor values for better visualization
        combined['profit_factor_capped'] = combined['profit_factor'].clip(upper=5)
        pf_means = [combined[combined['implementation'] == imp]['profit_factor_capped'].mean() for imp in implementations]
        plt.bar(implementations, pf_means, color=['blue', 'green', 'orange'])
        plt.title('Average Profit Factor by Implementation (capped at 5)')
        plt.ylabel('Profit Factor')
        plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/implementation_comparison.png')
    print("Saved implementation comparison plot to results/implementation_comparison.png")
    
    # ROI by window for each implementation
    plt.figure(figsize=(15, 6))
    for imp in implementations:
        imp_data = combined[combined['implementation'] == imp]
        plt.plot(imp_data['window_id'], imp_data['roi'], marker='o', label=imp)
    
    plt.axhline(y=0, color='red', linestyle='-')
    plt.legend()
    plt.title('ROI by Window for Each Implementation')
    plt.xlabel('Window ID')
    plt.ylabel('ROI (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/roi_by_window.png')
    print("Saved ROI by window plot to results/roi_by_window.png")

def print_summary_table(summaries):
    """Print a formatted summary table"""
    if not summaries:
        print("No summary data available")
        return
    
    # Create a DataFrame from the summaries
    summary_df = pd.DataFrame.from_dict(summaries, orient='index')
    
    # Define display format
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    # Print the table
    print("\n" + "=" * 80)
    print("WALK-FORWARD OPTIMIZATION IMPLEMENTATION COMPARISON")
    print("=" * 80)
    print(summary_df)
    print("=" * 80)
    
    # Identify overall best implementation based on average ROI
    if 'avg_roi' in summary_df.columns:
        best_imp = summary_df['avg_roi'].idxmax()
        worst_imp = summary_df['avg_roi'].idxmin()
        
        print(f"\nBest implementation by average ROI: {best_imp} ({summary_df.loc[best_imp, 'avg_roi']:.2f}%)")
        print(f"Worst implementation by average ROI: {worst_imp} ({summary_df.loc[worst_imp, 'avg_roi']:.2f}%)")
    
    # Save summary to CSV
    summary_df.to_csv('results/implementation_summary.csv')
    print("\nSaved implementation summary to results/implementation_summary.csv")

def main():
    """Main function to run the backtest summary script"""
    print("\nBacktest Summary Script")
    print("-" * 40)
    
    # Load results
    results = load_results()
    
    if not results:
        print("No results found. Please run the backtest implementations first.")
        return
    
    # Standardize columns
    standardized = standardize_columns(results)
    
    # Generate summary statistics
    summaries = generate_summary_statistics(standardized)
    
    # Print summary table
    print_summary_table(summaries)
    
    # Generate comparison plots
    plot_comparison(standardized)

if __name__ == "__main__":
    main() 