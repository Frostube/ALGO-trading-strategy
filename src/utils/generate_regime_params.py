import pandas as pd
import numpy as np
import os
import glob
import argparse

def extract_best_params_by_regime(wfo_dir, symbol='BTC/USDT', timestamp=None):
    """
    Extract the best parameters for each market regime from walk-forward results.
    
    Args:
        wfo_dir: Directory containing walk-forward analysis results
        symbol: Trading symbol
        timestamp: Optional specific timestamp folder to use
        
    Returns:
        DataFrame with best parameters for each regime
    """
    # Find the latest regime analysis file if timestamp not specified
    if timestamp:
        regime_file = os.path.join(wfo_dir, f"regime_analysis_{symbol.replace('/', '_')}_{timestamp}.csv")
    else:
        pattern = os.path.join(wfo_dir, f"regime_analysis_{symbol.replace('/', '_')}*.csv")
        files = glob.glob(pattern)
        if not files:
            raise FileNotFoundError(f"No regime analysis files found in {wfo_dir}")
        # Sort by modification time to get the latest file
        files.sort(key=os.path.getmtime, reverse=True)
        regime_file = files[0]
    
    print(f"Using regime analysis file: {regime_file}")
    
    # Load regime analysis results
    regime_df = pd.read_csv(regime_file)
    
    # Extract unique market regimes
    regimes = regime_df['regime'].unique()
    
    # Find the best parameters for each regime based on test returns
    best_params_by_regime = []
    
    for regime in regimes:
        regime_data = regime_df[regime_df['regime'] == regime]
        
        if regime_data.empty:
            print(f"No data found for regime: {regime}")
            continue
            
        # Find the parameters with the best test return
        best_row = regime_data.loc[regime_data['test_return'].idxmax()]
        
        # Extract parameters
        best_params = {
            'regime': regime,
            'fast_ema': int(best_row.get('fast_ema', 3)), 
            'slow_ema': int(best_row.get('slow_ema', 15)),
            'atr_multiplier': float(best_row.get('atr_multiplier', 2.0)),
            'test_return': float(best_row.get('test_return', 0)),
            'test_pf': float(best_row.get('test_pf', 0)),
            'n_trades': int(best_row.get('n_trades', 0))
        }
        
        best_params_by_regime.append(best_params)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(best_params_by_regime)
    
    # Sort by regime name
    result_df = result_df.sort_values('regime')
    
    return result_df

def save_params(params_df, output_file):
    """
    Save parameters to CSV file.
    
    Args:
        params_df: DataFrame with parameters
        output_file: Output file path
    """
    params_df.to_csv(output_file, index=False)
    print(f"Parameters saved to {output_file}")
    
    # Also print the parameters
    print("\nOptimal parameters by regime:")
    for _, row in params_df.iterrows():
        print(f"{row['regime']}:  EMA {row['fast_ema']}/{row['slow_ema']}, ATR×{row['atr_multiplier']:.1f}  —  Return: {row['test_return']:.2%}, PF: {row['test_pf']:.2f}, Trades: {row['n_trades']}")

def main():
    parser = argparse.ArgumentParser(description="Generate regime-specific parameters from walk-forward analysis")
    parser.add_argument('--wfo_dir', default='wfo_results', help='Directory with walk-forward results')
    parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    parser.add_argument('--timestamp', help='Specific timestamp folder to use')
    parser.add_argument('--output', default='regime_params.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    try:
        params_df = extract_best_params_by_regime(args.wfo_dir, args.symbol, args.timestamp)
        save_params(params_df, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main() 