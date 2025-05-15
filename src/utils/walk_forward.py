import os
import pandas as pd
from datetime import datetime
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.window_generator import generate_windows
from src.strategy.ema_crossover import EMACrossoverStrategy
from src.utils.metrics import calculate_metrics
from src.utils.visualizations import (
    plot_parameter_stability, 
    plot_performance_distribution,
    plot_parameter_heatmap,
    plot_parameter_heatmap_custom,
    create_dashboard,
    plot_in_vs_out_scatter,
    plot_return_distribution,
    plot_regime_performance
)

def walk_forward_test(full_df, symbol="BTC/USDT", train_days=90, test_days=30,
                      parameter_grid=None, base_params=None,
                      exit_strategy=None, exit_params=None,
                      output_dir="wfo_results", open_dashboard=False):
    """
    Perform walk-forward optimization with multiple parameter combinations.
    
    Args:
        full_df: DataFrame with historical price data
        symbol: Trading symbol
        train_days: Number of days in training window
        test_days: Number of days in test window
        parameter_grid: Dict of parameter names to lists of values
        base_params: Base parameters to use as default
        exit_strategy: Optional exit strategy type ('fixed', 'trailing', 'time', None)
        exit_params: Parameters for the exit strategy
        output_dir: Directory to save results and visualizations
        open_dashboard: Whether to open the HTML dashboard after creation
    
    Returns:
        (DataFrame, DataFrame): All results and best parameters per window
    """
    os.makedirs(output_dir, exist_ok=True)
    windows = generate_windows(full_df, train_days, test_days)
    
    if parameter_grid is None:
        parameter_grid = {'fast_ema': [3, 5, 8, 13],
                          'slow_ema': [13, 21, 34, 55],
                          'atr_sl_multiplier': [1.5, 2.0, 2.5, 3.0]}
    
    if base_params is None:
        base_params = {'fast_ema': 3, 'slow_ema': 15, 'atr_sl_multiplier': 2.0}
    
    # Set exit strategy parameters if provided
    if exit_strategy and not exit_params:
        if exit_strategy == 'fixed':
            exit_params = {'take_profit_pct': 0.05, 'stop_loss_pct': 0.03}
        elif exit_strategy == 'trailing':
            exit_params = {'trail_pct': 0.02}
        elif exit_strategy == 'time':
            exit_params = {'max_bars': 10}
    
    # Loop through parameter combinations
    all_results = []
    best_params_by_window = {}
    
    # Generate windows
    windows = generate_windows(full_df, train_days=train_days, test_days=test_days)
    
    if len(windows) == 0:
        print("No valid windows found!")
        return pd.DataFrame(), pd.DataFrame()
        
    for window_id, window in enumerate(windows, 1):
        train_df = full_df[(full_df.index >= window['train_start']) & 
                          (full_df.index <= window['train_end'])]
        test_df = full_df[(full_df.index >= window['test_start']) & 
                         (full_df.index <= window['test_end'])]
        
        # Loop through all parameter combinations
        window_results = []
        best_pf = 0
        best_pf_params = None
        best_train_return = 0
        best_test_pf = 0
        best_test_return = 0
        
        # Generate parameter combinations
        if parameter_grid:
            param_combinations = [dict(zip(parameter_grid.keys(), values)) 
                                for values in product(*parameter_grid.values())]
        else:
            # Use default parameters
            param_combinations = [{}]
            
        # Iterate through parameter combinations
        for params in param_combinations:
            # Create a copy of base parameters and update with current combination
            current_params = base_params.copy() if base_params else {}
            current_params.update(params)
            
            # Skip invalid parameter combinations
            if 'fast_ema' in params and 'slow_ema' in params:
                if params['fast_ema'] >= params['slow_ema']:
                    continue
                    
            # Create strategy with current parameters
            strategy = EMACrossoverStrategy(timeframe='4h', **current_params)
            
            # Set exit strategy if defined
            if exit_strategy and exit_params:
                strategy.set_exit_strategy(exit_strategy, exit_params)
                
            # Backtest on training data
            train_results = strategy.backtest(train_df)
            train_metrics = calculate_metrics(train_results)
            
            if train_metrics['profit_factor'] > best_pf:
                best_pf = train_metrics['profit_factor']
                best_pf_params = current_params.copy()
                best_train_return = train_metrics['total_return']
                
                # Test out-of-sample with these parameters
                test_results = strategy.backtest(test_df)
                test_metrics = calculate_metrics(test_results)
                best_test_pf = test_metrics['profit_factor']
                best_test_return = test_metrics['total_return']
                
            # Save all results
            result = {
                'window_id': window_id,
                'train_start': window['train_start'],
                'train_end': window['train_end'],
                'test_start': window['test_start'],
                'test_end': window['test_end'],
                'train_return': train_metrics['total_return'],
                'train_pf': train_metrics['profit_factor'],
                'train_win_rate': train_metrics['win_rate'],
                'train_max_dd': train_metrics['max_drawdown'],
                'train_n_trades': train_metrics['n_trades']
            }
            
            # Add parameters to results
            for k, v in current_params.items():
                result[f'params.{k}'] = v
                
            window_results.append(result)
                
        print(f"Window {window_id} best PF: in={best_pf:.2f}, out={best_test_pf:.2f}")
        
        # Save best parameters
        best_params_by_window[window_id] = {
            'window_id': window_id,
            'train_start': window['train_start'],
            'train_end': window['train_end'],
            'test_start': window['test_start'],
            'test_end': window['test_end'],
            'train_return': best_train_return,
            'train_pf': best_pf,
            'test_return': best_test_return,
            'test_pf': best_test_pf
        }
        
        # Add parameters to best params
        for k, v in best_pf_params.items():
            best_params_by_window[window_id][k] = v
            
        all_results.extend(window_results)
        
    # Format results for analysis
    out_columns = ['window_id', 'train_start', 'train_end', 'test_start', 'test_end',
                   'train_return', 'train_pf', 'test_return', 'test_pf'] 
    
    param_columns = list(parameter_grid.keys()) if parameter_grid else []
    
    # Convert to dataframe with the required columns
    for result in all_results:
        # Add parameter names as prefixed columns
        for param in param_columns:
            result[f'params.{param}'] = result.get(param, None)
    
    out_df = pd.DataFrame(all_results)
    
    # Make sure all expected columns exist
    for col in out_columns + [f'params.{param}' for param in param_columns]:
        if col not in out_df.columns:
            out_df[col] = None
    
    # Format best_params dataframe
    best_params_list = []
    for window_id, params in best_params_by_window.items():
        if params:
            row = {'window_id': window_id}
            row.update(params)
            best_params_list.append(row)
            
    best_params_df = pd.DataFrame(best_params_list)
    
    # Generate timestamp for files
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    
    # Save results to CSV
    out_df.to_csv(f"{output_dir}/wfo_all_{symbol.replace('/','_')}_{timestamp}.csv", index=False)
    best_params_df.to_csv(f"{output_dir}/wfo_best_{symbol.replace('/','_')}_{timestamp}.csv", index=False)

    # Plot parameter stability
    print(f"Creating visualizations for {symbol}...")
    plot_parameter_stability(best_params_df, output_dir, symbol, timestamp)
    
    # Plot parameter heatmaps (only if we have parameters)
    if parameter_grid and len(parameter_grid) > 1:
        # Extract param keys for heatmap
        param_keys = list(parameter_grid.keys())
        if len(param_keys) >= 2:
            # Create heatmaps for pairs of parameters
            for i in range(len(param_keys)-1):
                for j in range(i+1, len(param_keys)):
                    param_x = param_keys[i]
                    param_y = param_keys[j]
                    # Create heatmap for test return
                    plot_parameter_heatmap_custom(out_df, f'params.{param_x}', f'params.{param_y}', 
                                                'test_return', output_dir, symbol, timestamp)
                    # Create heatmap for profit factor
                    plot_parameter_heatmap_custom(out_df, f'params.{param_x}', f'params.{param_y}', 
                                                'test_pf', output_dir, symbol, timestamp)
    
    # Plot in-sample vs out-of-sample scatter
    plot_in_vs_out_scatter(out_df, output_dir, symbol, timestamp)
    
    # Plot distributions
    plot_return_distribution(out_df, 'test_return', output_dir, symbol, timestamp)
    plot_return_distribution(out_df, 'test_pf', output_dir, symbol, timestamp)
    
    # Create comprehensive dashboard
    dashboard_path = create_dashboard(out_df, best_params_df, output_dir, symbol, timestamp)
    
    print(f"Walk-forward analysis complete. Results saved to {output_dir}")
    
    if open_dashboard and dashboard_path:
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
            print(f"Dashboard opened: {dashboard_path}")
        except Exception as e:
            print(f"Could not open dashboard: {e}")
    
    return out_df, best_params_df

# helper entrypoint
def run_wfo(symbol="BTC/USDT", timeframe="4h", years=2,
            train_days=90, test_days=30, 
            exit_strategy=None, exit_params=None,
            output_dir="wfo_results", parameter_grid=None):
    """
    Run walk-forward optimization with specified parameters.
    
    Args:
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        years: Number of years of historical data to fetch
        train_days: Number of days in training window
        test_days: Number of days in test window
        exit_strategy: Optional exit strategy ('fixed', 'trailing', 'time')
        exit_params: Parameters for the exit strategy
        output_dir: Directory to save results and visualizations
        parameter_grid: Custom parameter grid for optimization
    
    Returns:
        Tuple of (all_results_df, best_params_df)
    """
    from src.data.fetcher import fetch_ohlcv
    df = fetch_ohlcv(symbol=symbol, tf=timeframe, days=years*365)
    if df.empty:
        print("No data fetched.")
        return None, None
    
    return walk_forward_test(
        df, symbol, train_days, test_days,
        parameter_grid=parameter_grid,
        exit_strategy=exit_strategy,
        exit_params=exit_params,
        output_dir=output_dir
    )

def analyze_market_regimes(df, symbol="BTC/USDT", lookback=90, vol_window=20, output_dir="wfo_results"):
    """
    Analyze market regimes based on trend and volatility.
    
    Args:
        df: DataFrame with price data
        symbol: Trading pair symbol
        lookback: Lookback window for trend calculation
        vol_window: Window for volatility calculation
        output_dir: Directory to save regime analysis results
        
    Returns:
        DataFrame with market regime classification
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    
    # Calculate trend using SMA
    df['sma'] = df['close'].rolling(lookback).mean()
    df['trend'] = (df['close'] > df['sma']).astype(int)
    
    # Calculate volatility
    df['volatility'] = df['returns'].rolling(vol_window).std() * (252 ** 0.5)  # Annualized
    
    # Classify market regimes
    median_vol = df['volatility'].median()
    df['regime'] = 'neutral'
    df.loc[(df['trend'] == 1) & (df['volatility'] <= median_vol), 'regime'] = 'bull_low_vol'
    df.loc[(df['trend'] == 1) & (df['volatility'] > median_vol), 'regime'] = 'bull_high_vol'
    df.loc[(df['trend'] == 0) & (df['volatility'] <= median_vol), 'regime'] = 'bear_low_vol'
    df.loc[(df['trend'] == 0) & (df['volatility'] > median_vol), 'regime'] = 'bear_high_vol'
    
    # Summary statistics
    print(f"Market regime analysis for {symbol}:")
    regime_counts = df['regime'].value_counts()
    total_days = len(df)
    
    for regime, count in regime_counts.items():
        pct = count / total_days * 100
        print(f"  {regime}: {count} days ({pct:.1f}%)")
    
    # Generate visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot price with colored background by regime
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], color='black', linewidth=1)
    ax1.set_title(f'Price and Market Regimes: {symbol}', fontsize=14)
    ax1.set_ylabel('Price')
    
    # Add colored background for regimes
    regimes = df['regime'].unique()
    colors = {'bull_low_vol': 'lightgreen', 'bull_high_vol': 'orange', 
              'bear_low_vol': 'lightblue', 'bear_high_vol': 'salmon', 'neutral': 'white'}
    
    regime_changes = df['regime'].ne(df['regime'].shift()).cumsum()
    for i, g in df.groupby(regime_changes):
        if not g.empty:
            regime = g['regime'].iloc[0]
            ax1.axvspan(g.index[0], g.index[-1], alpha=0.3, color=colors.get(regime, 'white'))
    
    # Add legend for regimes
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[r], alpha=0.3, label=r.replace('_', ' ').title()) 
                      for r in regimes if r in colors]
    ax1.legend(handles=legend_elements, loc='upper left')
    
    # Plot regime distribution as pie chart
    ax2 = plt.subplot(2, 1, 2)
    regime_pcts = regime_counts / total_days * 100
    wedges, texts, autotexts = ax2.pie(
        regime_pcts, 
        labels=[r.replace('_', ' ').title() for r in regime_counts.index], 
        autopct='%1.1f%%',
        colors=[colors.get(r, 'gray') for r in regime_counts.index],
        startangle=90
    )
    ax2.set_title('Regime Distribution')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d%H%M')
    plt.savefig(f"{output_dir}/regime_analysis_{symbol.replace('/','_')}_{timestamp}.png")
    plt.close()
    
    # Save regime data to CSV for later use in visualizations
    df[['close', 'returns', 'volatility', 'trend', 'regime']].to_csv(
        f"{output_dir}/regime_analysis_{symbol.replace('/','_')}.csv"
    )
    print(f"Regime analysis saved to {output_dir}/regime_analysis_{symbol.replace('/','_')}.csv")
    
    return df

if __name__ == '__main__':
    run_wfo() 