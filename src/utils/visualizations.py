import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter
from datetime import datetime

def plot_parameter_stability(best_params_df, output_dir, symbol, timestamp=None):
    """
    Create heatmap showing parameter stability across windows.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    
    # Get parameters that can vary
    param_cols = [col for col in best_params_df.columns 
                 if col not in ['window_id', 'train_return', 'train_pf', 'test_return', 'test_pf',
                                'train_start', 'train_end', 'test_start', 'test_end']]
    
    if not param_cols:
        print("No parameter columns found for stability visualization")
        return
    
    fig, axes = plt.subplots(len(param_cols), 1, figsize=(10, 3*len(param_cols)))
    if len(param_cols) == 1:
        axes = [axes]
        
    for i, param in enumerate(param_cols):
        ax = axes[i]
        data = best_params_df[['window_id', param]].set_index('window_id')
        
        # Check if the parameter is numeric for heatmap
        try:
            data = data.astype(float)
            sns.heatmap(data.T, ax=ax, cmap='viridis', annot=True, fmt='.1f', cbar_kws={'label': param})
        except (ValueError, TypeError):
            # For non-numeric data (like strings or dates), just display as text
            ax.axis('off')
            cell_text = [[str(val)] for val in data[param].values]
            window_labels = [f"Window {w}" for w in data.index]
            ax.table(cellText=cell_text, rowLabels=window_labels, 
                     colLabels=[param], loc='center', cellLoc='center')
            
        ax.set_title(f'Optimal {param} by Window')
        ax.set_xlabel('Window ID')
        ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/param_stability_{symbol.replace('/','_')}_{timestamp}.png")
    plt.close()

def plot_performance_distribution(all_results_df, output_dir, symbol, timestamp=None):
    """
    Create distribution plots of performance metrics.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Return distributions
    sns.histplot(all_results_df['train_return'], kde=True, ax=axes[0, 0], color='blue', alpha=0.5)
    sns.histplot(all_results_df['test_return'], kde=True, ax=axes[0, 0], color='red', alpha=0.5)
    axes[0, 0].set_title('Return Distribution')
    axes[0, 0].legend(['In-Sample', 'Out-of-Sample'])
    axes[0, 0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Profit Factor distributions
    sns.histplot(all_results_df['train_pf'], kde=True, ax=axes[0, 1], color='blue', alpha=0.5)
    sns.histplot(all_results_df['test_pf'], kde=True, ax=axes[0, 1], color='red', alpha=0.5)
    axes[0, 1].set_title('Profit Factor Distribution')
    axes[0, 1].legend(['In-Sample', 'Out-of-Sample'])
    
    # Scatterplot with regression line
    sns.regplot(x='train_return', y='test_return', data=all_results_df, ax=axes[1, 0], 
                scatter_kws={'alpha':0.5})
    axes[1, 0].set_title('Return Correlation')
    axes[1, 0].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    axes[1, 0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Scatterplot for profit factor
    sns.regplot(x='train_pf', y='test_pf', data=all_results_df, ax=axes[1, 1],
                scatter_kws={'alpha':0.5})
    axes[1, 1].set_title('Profit Factor Correlation')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_dist_{symbol.replace('/','_')}_{timestamp}.png")
    plt.close()

def plot_parameter_heatmap(results_df, param_x, param_y, metric, output_dir, symbol, timestamp=None):
    """
    Heatmap of a metric (e.g. test_return) across two parameters.
    
    Args:
        results_df: DataFrame with results
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        metric: Metric to plot
        output_dir: Directory to save plots
        symbol: Trading symbol
        timestamp: Optional timestamp for filename
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    
    # Make sure we have proper column names
    param_x_col = f'params.{param_x}' if not param_x.startswith('params.') else param_x
    param_y_col = f'params.{param_y}' if not param_y.startswith('params.') else param_y
    
    # Check if columns exist
    if param_x_col not in results_df.columns or param_y_col not in results_df.columns:
        print(f"Warning: Parameters {param_x_col} or {param_y_col} not found in results dataframe")
        return None
        
    if metric not in results_df.columns:
        print(f"Warning: Metric {metric} not found in results dataframe")
        return None
    
    # Check if we have enough unique values
    if results_df[param_x_col].nunique() <= 1 or results_df[param_y_col].nunique() <= 1:
        print(f"Warning: Not enough unique values for parameters {param_x} and {param_y}")
        return None
    
    try:
        # Create pivot table for heatmap
        pivot = results_df.pivot_table(
            values=metric,
            index=param_x_col,
            columns=param_y_col,
            aggfunc='mean'
        )
        
        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
        plt.title(f"{metric} by {param_x} and {param_y} - {symbol}")
        plt.tight_layout()
        
        out_path = os.path.join(output_dir, f"heatmap_{param_x}_{param_y}_{metric}_{symbol.replace('/','_')}_{timestamp}.png")
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception as e:
        print(f"Error creating heatmap for {param_x} vs {param_y}: {str(e)}")
        return None

def create_dashboard(all_results_df, best_params_df, output_dir, symbol, timestamp=None):
    """
    Generate a complete visual dashboard of walk-forward analysis results.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    
    # Get parameters that can vary
    param_cols = [col for col in best_params_df.columns 
                 if col not in ['window_id', 'train_return', 'train_pf', 'test_return', 'test_pf',
                               'train_start', 'train_end', 'test_start', 'test_end']]
    
    # Create HTML dashboard
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Walk-Forward Optimization Results - {symbol}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metric {{ font-weight: bold; }}
            .summary {{ margin-bottom: 20px; padding: 15px; background-color: #f8f8f8; }}
            .images {{ display: flex; flex-wrap: wrap; gap: 20px; margin-bottom: 20px; }}
            .images img {{ max-width: 100%; height: auto; }}
            .image-container {{ margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Walk-Forward Optimization Results - {symbol}</h1>
            <p>Analysis performed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
            
            <div class="summary">
                <h2>Summary Statistics</h2>
                <p>Windows analyzed: {len(best_params_df)}</p>
                <p>Parameter combinations tested: {len(all_results_df) // len(best_params_df) if len(best_params_df) > 0 else 0}</p>
                <p>Average in-sample return: {all_results_df['train_return'].mean():.2%}</p>
                <p>Average out-of-sample return: {all_results_df['test_return'].mean():.2%}</p>
                <p>Average in-sample profit factor: {all_results_df['train_pf'].mean():.2f}</p>
                <p>Average out-of-sample profit factor: {all_results_df['test_pf'].mean():.2f}</p>
            </div>
            
            <h2>Best Parameters By Window</h2>
            <table>
                <tr>
                    <th>Window</th>
                    <th>Training Period</th>
                    <th>Testing Period</th>
    """
    
    # Add headers for parameters and metrics
    for param in param_cols:
        html += f"<th>{param}</th>"
    html += "<th>In-Sample Return</th><th>Out-of-Sample Return</th>"
    html += "<th>In-Sample Profit Factor</th><th>Out-of-Sample Profit Factor</th></tr>"
    
    # Add rows for each window
    for _, row in best_params_df.iterrows():
        html += f"<tr><td>{int(row['window_id'])}</td>"
        
        # Add date ranges for training and testing
        train_start = row.get('train_start', '')
        train_end = row.get('train_end', '')
        test_start = row.get('test_start', '')
        test_end = row.get('test_end', '')
        
        # Format the dates if they exist
        if isinstance(train_start, (pd.Timestamp, datetime)):
            train_start = train_start.strftime('%Y-%m-%d')
        if isinstance(train_end, (pd.Timestamp, datetime)):
            train_end = train_end.strftime('%Y-%m-%d')
        if isinstance(test_start, (pd.Timestamp, datetime)):
            test_start = test_start.strftime('%Y-%m-%d')
        if isinstance(test_end, (pd.Timestamp, datetime)):
            test_end = test_end.strftime('%Y-%m-%d')
            
        html += f"<td>{train_start} to {train_end}</td>"
        html += f"<td>{test_start} to {test_end}</td>"
        
        for param in param_cols:
            html += f"<td>{row[param]}</td>"
        html += f"<td>{row['train_return']:.2%}</td>"
        html += f"<td>{row['test_return']:.2%}</td>"
        html += f"<td>{row['train_pf']:.2f}</td>"
        html += f"<td>{row['test_pf']:.2f}</td></tr>"
    
    html += """
            </table>
            
            <h2>Visualizations</h2>
            <div class="images">
    """
    
    # List all image files
    image_files = [f for f in os.listdir(output_dir) if f.endswith('.png') and timestamp in f]
    for img in image_files:
        html += f"""
                <div class="image-container">
                    <img src="{img}" alt="{img}">
                    <p>{img}</p>
                </div>
        """
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Write HTML to file
    with open(f"{output_dir}/dashboard_{symbol.replace('/','_')}_{timestamp}.html", 'w') as f:
        f.write(html)
    
    return f"{output_dir}/dashboard_{symbol.replace('/','_')}_{timestamp}.html"

# Enhanced visualization functions from user request
def plot_in_vs_out_scatter(results_df, output_dir, symbol, timestamp=None):
    """
    Scatter plot of in-sample vs out-of-sample returns.
    
    Args:
        results_df: DataFrame with results
        output_dir: Directory to save plots
        symbol: Trading symbol
        timestamp: Optional timestamp for filename
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
        
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=results_df,
        x='train_return',
        y='test_return',
        hue='window_id',
        palette='tab10',
        legend=False,
        alpha=0.6
    )
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.title(f"In-Sample vs Out-of-Sample Returns: {symbol}")
    plt.xlabel('In-Sample Return')
    plt.ylabel('Out-of-Sample Return')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"scatter_{symbol.replace('/','_')}_{timestamp}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_parameter_heatmap_custom(results_df, param_x, param_y, metric, output_dir, symbol, timestamp=None):
    """
    Heatmap of a metric (e.g. test_return) across two parameters.
    Improved version with error handling for empty data.
    
    Args:
        results_df: DataFrame with results
        param_x: Parameter for x-axis
        param_y: Parameter for y-axis
        metric: Metric to plot
        output_dir: Directory to save plots
        symbol: Trading symbol
        timestamp: Optional timestamp for filename
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    
    # Check if columns exist in the DataFrame
    if param_x not in results_df.columns or param_y not in results_df.columns or metric not in results_df.columns:
        print(f"Error creating heatmap: columns {param_x}, {param_y}, or {metric} not found in results")
        return None
    
    # Check if we have enough data points
    if len(results_df) < 4:
        print(f"Error creating heatmap: not enough data points ({len(results_df)}) for meaningful heatmap")
        return None
    
    try:
        # Get unique values for each parameter
        x_values = results_df[param_x].dropna().unique()
        y_values = results_df[param_y].dropna().unique()
        
        if len(x_values) < 2 or len(y_values) < 2:
            print(f"Error creating heatmap: not enough unique values for parameters ({len(x_values)}, {len(y_values)})")
            return None
        
        # Create a pivot table
        pivot = pd.pivot_table(
            results_df,
            values=metric,
            index=param_x,
            columns=param_y,
            aggfunc='mean'
        )
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title(f"Heatmap {metric}: {param_x.replace('params.', '')} vs {param_y.replace('params.', '')}")
        plt.tight_layout()
        
        # Save the plot
        out_path = os.path.join(output_dir, 
                               f"heatmap_{param_x.replace('params.', '')}_{param_y.replace('params.', '')}_{metric}_{symbol.replace('/','_')}_{timestamp}.png")
        plt.savefig(out_path)
        plt.close()
        return out_path
        
    except Exception as e:
        print(f"Error creating heatmap: {str(e)}")
        return None

def plot_return_distribution(results_df, metric, output_dir, symbol, timestamp=None):
    """
    Distribution plot of a chosen metric across all windows.
    
    Args:
        results_df: DataFrame with results
        metric: Metric to plot (e.g. 'test_return', 'test_pf')
        output_dir: Directory to save plots
        symbol: Trading symbol
        timestamp: Optional timestamp for filename
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
        
    plt.figure(figsize=(8, 5))
    sns.histplot(results_df[metric], kde=True, bins=20)
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    plt.title(f"Distribution of {metric}: {symbol}")
    plt.xlabel(metric)
    plt.ylabel('Frequency')
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"dist_{metric}_{symbol.replace('/','_')}_{timestamp}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_regime_performance(results_df, regime_df, output_dir, symbol, timestamp=None):
    """
    Analyze and visualize performance across different market regimes.
    
    Args:
        results_df: DataFrame with results
        regime_df: DataFrame with market regime classifications
        output_dir: Directory to save the plot
        symbol: Trading symbol
        timestamp: Optional timestamp for the filename
        
    Returns:
        str: Path to the saved file
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M')
    
    # For each parameter set, calculate performance in different regimes
    regime_performance = []
    
    # Get unique windows and parameter sets
    windows = results_df['window_id'].unique()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Define regimes
    regimes = ['bull_low_vol', 'bull_high_vol', 'bear_low_vol', 'bear_high_vol']
    colors = ['green', 'orange', 'blue', 'red']
    
    for i, regime in enumerate(regimes):
        # Get data for this regime
        regime_data = regime_df[regime_df['regime'] == regime]
        
        if regime_data.empty:
            axes[i].text(0.5, 0.5, f"No data for {regime}", 
                         ha='center', va='center', transform=axes[i].transAxes)
            continue
        
        # Get dates for this regime
        regime_dates = regime_data.index
        
        # For each window, calculate test return during this regime
        regime_returns = []
        
        for window in windows:
            window_params = results_df[results_df['window_id'] == window]
            
            # Get test period for this window
            test_start = window_params['test_start'].iloc[0] if 'test_start' in window_params.columns else None
            test_end = window_params['test_end'].iloc[0] if 'test_end' in window_params.columns else None
            
            if test_start is None or test_end is None:
                continue
                
            # Check if test period overlaps with this regime
            overlap_dates = regime_dates[(regime_dates >= test_start) & (regime_dates <= test_end)]
            
            if len(overlap_dates) > 0:
                # Calculate performance during this regime
                regime_returns.append({
                    'window_id': window,
                    'test_return': window_params['test_return'].iloc[0],
                    'test_pf': window_params['test_pf'].iloc[0] if 'test_pf' in window_params.columns else None,
                    'overlap_days': len(overlap_dates)
                })
        
        # Create DataFrame for this regime
        regime_df = pd.DataFrame(regime_returns)
        
        if regime_df.empty:
            axes[i].text(0.5, 0.5, f"No test windows in {regime}", 
                         ha='center', va='center', transform=axes[i].transAxes)
            continue
        
        # Plot distribution of returns for this regime
        sns.histplot(regime_df['test_return'], kde=True, ax=axes[i], color=colors[i])
        axes[i].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[i].set_title(f"{regime.replace('_', ' ').title()} (n={len(regime_df)})")
        axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
        
        # Add mean and median
        mean_return = regime_df['test_return'].mean()
        median_return = regime_df['test_return'].median()
        
        axes[i].annotate(f'Mean: {mean_return:.2%}', xy=(0.7, 0.85), xycoords='axes fraction', fontsize=10)
        axes[i].annotate(f'Median: {median_return:.2%}', xy=(0.7, 0.8), xycoords='axes fraction', fontsize=10)
    
    plt.suptitle(f"Performance by Market Regime: {symbol}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    out_path = f"{output_dir}/regime_performance_{symbol.replace('/','_')}_{timestamp}.png"
    plt.savefig(out_path)
    plt.close()
    return out_path 