#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import traceback

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our test script
from test_enhanced_strategy import test_enhanced_strategy

# Modified configuration for more realistic returns
CONFIG = {
    # Strategy parameters - more conservative settings for target 20-30% returns
    'ema_fast': 8,
    'ema_slow': 21,
    'ema_trend': 50,  # Shorter trend period for more signals
    'rsi_period': 9,  # Shorter RSI for more responsive signals
    'rsi_long_threshold': 40,  # Less strict entry for long positions
    'rsi_short_threshold': 60,  # Less strict entry for short positions
    'atr_period': 10,
    'atr_sl_multiplier': 2.0,  # Wider stop losses for fewer trades
    'atr_tp_multiplier': 3.0,  # More conservative take profits
    
    # Portfolio parameters - more conservative allocation
    'vol_target_pct': 0.10,  # Target 10% volatility (down from 20%)
    'max_allocation': 0.25,  # Max 25% to any strategy (down from 40%)
    'ensemble_method': 'Sharpe-Weighted',
    
    # Position sizing - more conservative
    'pos_sizing_method': 'Volatility Targeting',
    'risk_per_trade': 0.015,  # 1.5% risk per trade (down from 3%)
    
    # Pyramiding - more conservative
    'enable_pyramiding': True,
    'max_pyramid_units': 3,  # Fewer pyramid units (down from 4)
    
    # Exit strategy - more conservative trailing stops
    'exit_strategy': 'ATR-Based Trailing Stop',
    'trail_activation_pct': 0.01,  # Activate trailing stop later (up from 0.005)
    'trail_atr_multiplier': 1.5,  # Wider trailing stop (up from 1.2)
    
    # Market regime parameters
    'regime_lookback': 25,  # More stable regime detection
    'vol_ranging_threshold': 0.025,
    'vol_trending_threshold': 0.07
}

def annualize_return(total_return, days):
    """Convert a total return over a period to annualized return"""
    return (1 + total_return) ** (365 / days) - 1

def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    """Calculate Sortino ratio (Sharpe that only penalizes downside volatility)"""
    mean_return = np.mean(returns)
    negative_returns = [r for r in returns if r < 0]
    downside_deviation = np.std(negative_returns) if negative_returns else 0.0001
    return (mean_return - risk_free_rate) / downside_deviation * np.sqrt(252)

def calculate_calmar_ratio(total_return, max_drawdown, days):
    """Calculate Calmar ratio (annualized return / max drawdown)"""
    annual_return = annualize_return(total_return, days)
    return annual_return / (max_drawdown or 0.0001)  # Avoid division by zero

def calculate_drawdowns(equity_curve):
    """Calculate drawdown statistics"""
    if not equity_curve:
        return {
            'max_drawdown': 0,
            'avg_drawdown': 0,
            'top_drawdowns': [],
            'total_drawdowns': 0
        }
        
    equity = [point['equity'] for point in equity_curve]
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    
    # Find drawdown periods
    in_drawdown = False
    drawdown_periods = []
    current_period = {}
    
    for i, dd in enumerate(drawdown):
        if not in_drawdown and dd < 0:
            # Start of drawdown
            in_drawdown = True
            current_period = {
                'start': equity_curve[i]['timestamp'],
                'start_equity': equity[i],
                'depth': dd
            }
        elif in_drawdown:
            if dd < current_period['depth']:
                # Deeper drawdown
                current_period['depth'] = dd
            
            if dd == 0 or i == len(drawdown) - 1:
                # End of drawdown
                current_period['end'] = equity_curve[i]['timestamp']
                current_period['end_equity'] = equity[i]
                current_period['duration'] = (current_period['end'] - current_period['start']).days
                drawdown_periods.append(current_period)
                in_drawdown = False
    
    # Find top 5 drawdowns
    if drawdown_periods:
        drawdown_periods.sort(key=lambda x: x['depth'])
        top_drawdowns = drawdown_periods[:5]
    else:
        top_drawdowns = []
    
    return {
        'max_drawdown': min(drawdown) if drawdown else 0,
        'avg_drawdown': np.mean([d['depth'] for d in drawdown_periods]) if drawdown_periods else 0,
        'top_drawdowns': top_drawdowns,
        'total_drawdowns': len(drawdown_periods)
    }

def generate_report():
    """Generate a comprehensive performance report for the enhanced strategy"""
    print("Generating performance report...")
    print("Running backtests, this may take a few minutes...\n")
    
    # Create results directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    # Determine timestamp for report
    timestamp_format = "%Y%m%d_%H%M%S"
    report_time = datetime.now().strftime(timestamp_format)
    
    # Prepare report data
    report_data = {
        'timestamp': report_time,
        'portfolio': {},
        'strategies': {},
        'symbols': {},
        'config': {}
    }
    
    # Run the test with error handling
    try:
        results, backtester = test_enhanced_strategy()
        if backtester:
            report_data['config'] = backtester.params
    except Exception as e:
        error_msg = f"Error running backtest: {str(e)}\n{traceback.format_exc()}"
        print(f"\nERROR: {error_msg}")
        
        # Save error report
        error_file = f"reports/error_report_{report_time}.txt"
        with open(error_file, 'w') as f:
            f.write(error_msg)
            
        print(f"Error details saved to {error_file}")
        return None, None
    
    # Portfolio-level metrics
    if results and 'portfolio' in results:
        portfolio = results['portfolio']
        equity_curve = portfolio.get('equity_curve', [])
        returns = portfolio.get('returns', [])
        
        # Calculate days in backtest
        if equity_curve and len(equity_curve) > 1:
            start_date = equity_curve[0]['timestamp']
            end_date = equity_curve[-1]['timestamp']
            days_in_test = (end_date - start_date).days
            if days_in_test < 1:
                days_in_test = 1  # Minimum 1 day
        else:
            days_in_test = 90  # Default if can't determine
        
        # Calculate additional metrics
        total_return = portfolio.get('total_return', 0)
        annual_return = annualize_return(total_return, days_in_test)
        
        # Monthly equivalent return
        monthly_return = (1 + annual_return) ** (1/12) - 1
        
        # Sortino ratio
        sortino_ratio = calculate_sortino_ratio(returns) if returns else 0
        
        # Calmar ratio
        max_drawdown = portfolio.get('max_drawdown', 0)
        calmar_ratio = calculate_calmar_ratio(total_return, max_drawdown, days_in_test)
        
        # Calculate win/loss metrics
        win_rate = portfolio.get('win_rate', 0)
        
        # Drawdown analysis
        drawdown_stats = calculate_drawdowns(equity_curve)
        
        report_data['portfolio'] = {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return': monthly_return,
            'sharpe_ratio': portfolio.get('sharpe_ratio', 0),
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'days_in_test': days_in_test,
            'drawdown_stats': drawdown_stats
        }
    
    # Strategy-level metrics
    if results and 'individual' in results:
        for key, result in results['individual'].items():
            if 'test' not in result:
                continue
                
            test = result['test']
            parts = key.split('_', 1)
            if len(parts) != 2:
                continue
                
            symbol, strategy = parts
            
            # Track strategy performance
            if strategy not in report_data['strategies']:
                report_data['strategies'][strategy] = {
                    'return': [],
                    'sharpe': [],
                    'trades': 0,
                    'symbols': []
                }
            
            # Add performance data
            report_data['strategies'][strategy]['return'].append(test.get('total_return', 0))
            report_data['strategies'][strategy]['sharpe'].append(test.get('sharpe_ratio', 0))
            report_data['strategies'][strategy]['trades'] += test.get('total_trades', 0)
            report_data['strategies'][strategy]['symbols'].append(symbol)
            
            # Track symbol performance
            if symbol not in report_data['symbols']:
                report_data['symbols'][symbol] = {
                    'return': [],
                    'sharpe': [],
                    'trades': 0,
                    'strategies': []
                }
            
            # Add performance data
            report_data['symbols'][symbol]['return'].append(test.get('total_return', 0))
            report_data['symbols'][symbol]['sharpe'].append(test.get('sharpe_ratio', 0))
            report_data['symbols'][symbol]['trades'] += test.get('total_trades', 0)
            report_data['symbols'][symbol]['strategies'].append(strategy)
    
    # Add allocation weights
    report_data['allocations'] = results.get('weights', {}) if results else {}
    
    # Calculate averages for strategies and symbols
    for strategy in report_data['strategies']:
        returns = report_data['strategies'][strategy]['return']
        sharpes = report_data['strategies'][strategy]['sharpe']
        
        report_data['strategies'][strategy]['avg_return'] = np.mean(returns) if returns else 0
        report_data['strategies'][strategy]['avg_sharpe'] = np.mean(sharpes) if sharpes else 0
    
    for symbol in report_data['symbols']:
        returns = report_data['symbols'][symbol]['return']
        sharpes = report_data['symbols'][symbol]['sharpe']
        
        report_data['symbols'][symbol]['avg_return'] = np.mean(returns) if returns else 0
        report_data['symbols'][symbol]['avg_sharpe'] = np.mean(sharpes) if sharpes else 0
    
    # Save report to JSON
    report_file = f"reports/performance_report_{report_time}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, default=str, indent=2)
    
    # Print summary report
    print("\n============================================================")
    print("ENHANCED STRATEGY PERFORMANCE REPORT")
    print("============================================================")
    
    portfolio_data = report_data['portfolio']
    print(f"Backtest Period: {portfolio_data.get('days_in_test', 0)} days")
    print(f"Total Return: {portfolio_data.get('total_return', 0) * 100:.2f}%")
    print(f"Annualized Return: {portfolio_data.get('annual_return', 0) * 100:.2f}%")
    print(f"Monthly Return: {portfolio_data.get('monthly_return', 0) * 100:.2f}%")
    print(f"Sharpe Ratio: {portfolio_data.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {portfolio_data.get('sortino_ratio', 0):.2f}")
    print(f"Calmar Ratio: {portfolio_data.get('calmar_ratio', 0):.2f}")
    print(f"Max Drawdown: {portfolio_data.get('max_drawdown', 0) * 100:.2f}%")
    print(f"Win Rate: {portfolio_data.get('win_rate', 0) * 100:.2f}%")
    
    print("\n--- Strategy Performance ---")
    for strategy, data in report_data['strategies'].items():
        allocation = report_data['allocations'].get(strategy, 0) * 100
        print(f"{strategy}: Return: {data['avg_return'] * 100:.2f}%, Sharpe: {data['avg_sharpe']:.2f}, Allocation: {allocation:.2f}%, Trades: {data['trades']}")
    
    print("\n--- Symbol Performance ---")
    for symbol, data in report_data['symbols'].items():
        print(f"{symbol}: Return: {data['avg_return'] * 100:.2f}%, Sharpe: {data['avg_sharpe']:.2f}, Trades: {data['trades']}")
    
    print("\n============================================================")
    print(f"Full report saved to {report_file}")
    print("============================================================")
    
    return report_data, backtester

if __name__ == "__main__":
    generate_report()
    input("\nPress Enter to exit...") 