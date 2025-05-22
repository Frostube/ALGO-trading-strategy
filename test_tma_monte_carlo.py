#!/usr/bin/env python3

"""
TMA Overlay Strategy Monte Carlo Testing

This script runs Monte Carlo simulations on the TMA Overlay strategy to evaluate
its robustness across randomized trade sequences and parameter combinations.

It leverages the monte_carlo_validation.py framework to:
1. Run backtests with various TMA parameter combinations
2. Perform Monte Carlo simulations by randomizing trade order
3. Assess the strategy's robustness ratio and performance stability
4. Visualize results across different parameter sets

Usage examples:
- Basic test: python test_tma_monte_carlo.py
- Increase simulations: python test_tma_monte_carlo.py --mc_sims 1000
- Custom timeframe: python test_tma_monte_carlo.py --timeframe 1h --symbol BTC/USDT
- Full parameter sweep: python test_tma_monte_carlo.py --enable-sweep
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure the project root is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import the monte_carlo_validation module
try:
    from monte_carlo_validation import (
        run_monte_carlo_validation, 
        run_strategy_backtest, 
        parse_range_string,
        generate_parameter_combinations,
        create_parameter_sweep_visualization
    )
    logger.info("Successfully imported monte_carlo_validation module")
except ImportError as e:
    logger.error(f"Failed to import monte_carlo_validation module: {e}")
    sys.exit(1)

def run_tma_monte_carlo(args):
    """
    Run Monte Carlo validation on the TMA Overlay strategy
    
    Args:
        args: Command line arguments
    """
    # Default TMA parameter ranges
    default_param_ranges = {
        'tma_period': [8, 14, 20],
        'atr_multiplier': [1.0, 1.5, 2.0],
        'fast_period': [5, 8, 13, 21],
        'slow_period': [20, 50, 100],
        'stop_loss_pct': [0.01, 0.02, 0.03],
        'take_profit_pct': [0.02, 0.04, 0.06]
    }
    
    # Use parameter sweeping if enabled
    if args.enable_sweep:
        # Parse custom parameter ranges if provided
        param_ranges = {}
        
        # TMA specific parameters
        if args.tma_period_range:
            param_ranges['tma_period'] = parse_range_string(args.tma_period_range)
        else:
            param_ranges['tma_period'] = default_param_ranges['tma_period']
            
        if args.atr_multiplier_range:
            param_ranges['atr_multiplier'] = parse_range_string(args.atr_multiplier_range)
        else:
            param_ranges['atr_multiplier'] = default_param_ranges['atr_multiplier']
            
        # EMA parameters
        if args.fast_ema_range:
            param_ranges['fast_period'] = parse_range_string(args.fast_ema_range)
        else:
            param_ranges['fast_period'] = default_param_ranges['fast_period']
            
        if args.slow_ema_range:
            param_ranges['slow_period'] = parse_range_string(args.slow_ema_range)
        else:
            param_ranges['slow_period'] = default_param_ranges['slow_period']
            
        # Risk parameters
        if args.stop_loss_range:
            param_ranges['stop_loss_pct'] = parse_range_string(args.stop_loss_range)
        else:
            param_ranges['stop_loss_pct'] = default_param_ranges['stop_loss_pct']
            
        if args.take_profit_range:
            param_ranges['take_profit_pct'] = parse_range_string(args.take_profit_range)
        else:
            param_ranges['take_profit_pct'] = default_param_ranges['take_profit_pct']
            
        # Generate all parameter combinations
        param_combinations = generate_parameter_combinations(param_ranges)
        
        if not param_combinations:
            logger.error("No valid parameter combinations generated")
            return
            
        logger.info(f"Running parameter sweep with {len(param_combinations)} combinations")
        
        # Run Monte Carlo validation for each parameter combination
        sweep_results = []
        
        for i, params in enumerate(param_combinations):
            logger.info(f"Testing combination {i+1}/{len(param_combinations)}: {params}")
            
            # Run backtest with these parameters for the TMA strategy
            trades = run_strategy_backtest(
                strategy_name="tma_overlay",  # Use the TMA strategy
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                strategy_params=params
            )
            
            if not trades or len(trades) < 5:
                logger.warning(f"Not enough trades (minimum 5 required) for parameter set {params}. Skipping.")
                continue
            
            # Run Monte Carlo validation
            result = run_monte_carlo_validation(
                trades=trades,
                num_simulations=args.mc_sims,
                confidence_level=args.confidence,
                initial_capital=args.initial_capital
            )
            
            if result:
                # Add parameters to result
                result['parameters'] = params
                result['num_trades'] = len(trades)
                
                # Calculate win rate
                winning_trades = sum(1 for t in trades if 
                                   (t.get('profit', 0) > 0 if 'profit' in t else t.get('pnl', 0) > 0))
                result['win_rate'] = winning_trades / len(trades)
                
                # Calculate net profit
                result['net_profit'] = sum(t.get('profit', 0) if 'profit' in t else t.get('pnl', 0) for t in trades)
                
                sweep_results.append(result)
                
                # Log result for this combination
                logger.info(f"Result for {params}: Robustness={result['robustness_assessment']}, "
                          f"Ratio={result['robustness_ratio']:.2f}, Trades={len(trades)}, "
                          f"Win Rate={result['win_rate']*100:.1f}%, Net Profit=${result['net_profit']:.2f}")
        
        # Save sweep results and create visualization
        if sweep_results:
            # Create visualization comparing all parameter sets
            create_parameter_sweep_visualization(sweep_results, args.output_dir)
            
            # Save the results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(args.output_dir, f"tma_parameter_sweep_results_{timestamp}.json")
            
            import json
            # Create a simplified version for JSON serialization
            json_results = []
            for result in sweep_results:
                # Convert keys that are not JSON serializable
                simple_result = {k: v for k, v in result.items() 
                              if k not in ['mean_equity_curve', 'report_path']}
                json_results.append(simple_result)
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            logger.info(f"Parameter sweep results saved to {results_file}")
        else:
            logger.warning("No valid results from parameter sweep")
            
    else:
        # Single run with default or specified parameters
        params = {
            'tma_period': args.tma_period,
            'atr_multiplier': args.atr_multiplier,
            'fast_period': args.fast_ema,
            'slow_period': args.slow_ema,
            'stop_loss_pct': args.stop_loss_pct,
            'take_profit_pct': args.take_profit_pct,
            'use_tma_direction_filter': args.use_tma_direction_filter,
            'use_engulfing_filter': args.use_engulfing_filter,
            'use_ema_confirmation': args.use_ema_confirmation,
        }
        
        # Remove None values
        params = {k: v for k, v in params.items() if v is not None}
        
        logger.info(f"Running TMA strategy with parameters: {params}")
        
        # Run backtest to get trades
        trades = run_strategy_backtest(
            strategy_name="tma_overlay",
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            strategy_params=params
        )
        
        if not trades or len(trades) < 5:
            logger.error("Not enough trades (minimum 5 required) for Monte Carlo validation")
            return
            
        # Log trade statistics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if 
                          (t.get('profit', 0) > 0 if 'profit' in t else t.get('pnl', 0) > 0))
        
        profit_sum = sum(t.get('profit', 0) if 'profit' in t else t.get('pnl', 0) for t in trades)
        
        logger.info(f"Analyzing {total_trades} trades")
        logger.info(f"Win Rate: {winning_trades/total_trades*100:.1f}%")
        logger.info(f"Net Profit: ${profit_sum:.2f}")
        
        # Run Monte Carlo simulation
        logger.info(f"Running Monte Carlo simulation with {args.mc_sims} iterations")
        mc_results = run_monte_carlo_validation(
            trades=trades,
            num_simulations=args.mc_sims,
            confidence_level=args.confidence,
            initial_capital=args.initial_capital
        )
        
        if mc_results:
            logger.info(f"Monte Carlo simulation results:")
            logger.info(f"Mean Final Equity: ${mc_results['mean_final_equity']:.2f}")
            logger.info(f"Mean Max Drawdown: {mc_results['mean_max_drawdown']*100:.1f}%")
            logger.info(f"{args.confidence*100}% Confidence Interval: [${mc_results['lower_bound_equity']:.2f}, ${mc_results['upper_bound_equity']:.2f}]")
            logger.info(f"Worst-case Max Drawdown ({args.confidence*100}%): {mc_results['max_drawdown_worst_case']*100:.1f}%")
            logger.info(f"Mean Win Rate: {mc_results['mean_win_rate']*100:.1f}%")
            
            # Output summary of the strategy's robustness
            robustness = mc_results['robustness_assessment']
            robustness_ratio = mc_results['robustness_ratio']
            
            logger.info(f"Robustness assessment: {robustness} (Ratio: {robustness_ratio:.2f})")
            
            if robustness == "STRONG":
                logger.info("The TMA Overlay strategy shows strong statistical robustness across randomized trade sequences.")
                logger.info("It consistently maintains profitability within the confidence interval.")
            elif robustness == "MODERATE":
                logger.info("The TMA Overlay strategy shows moderate statistical robustness.")
                logger.info("It maintains most of its capital in worst-case scenarios but may need further optimization.")
            else:
                logger.info("The TMA Overlay strategy lacks statistical robustness and needs significant improvement.")
                logger.info("Consider adjusting parameters or exploring different approaches.")
            
            logger.info(f"Monte Carlo report saved to: {mc_results['report_path']}")

def main():
    parser = argparse.ArgumentParser(description='Run Monte Carlo validation on the TMA Overlay strategy')
    
    # Monte Carlo parameters
    mc_group = parser.add_argument_group('Monte Carlo Parameters')
    mc_group.add_argument('--mc_sims', type=int, default=500, help='Number of Monte Carlo simulations')
    mc_group.add_argument('--confidence', type=float, default=0.95, help='Confidence level (0-1)')
    mc_group.add_argument('--initial_capital', type=float, default=10000, help='Initial capital amount')
    
    # Data selection
    data_group = parser.add_argument_group('Data Selection')
    data_group.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair to analyze')
    data_group.add_argument('--timeframe', type=str, default='4h', help='Timeframe for analysis')
    data_group.add_argument('--start_date', type=str, default='2020-01-01', help='Start date for backtest')
    data_group.add_argument('--end_date', type=str, default=None, help='End date for backtest')
    
    # TMA Strategy parameters
    tma_group = parser.add_argument_group('TMA Strategy Parameters')
    tma_group.add_argument('--tma_period', type=int, default=14, help='TMA period')
    tma_group.add_argument('--atr_multiplier', type=float, default=1.5, help='ATR multiplier for bands')
    tma_group.add_argument('--fast_ema', type=int, default=8, help='Fast EMA period')
    tma_group.add_argument('--slow_ema', type=int, default=21, help='Slow EMA period')
    tma_group.add_argument('--stop_loss_pct', type=float, default=0.02, help='Stop loss percentage')
    tma_group.add_argument('--take_profit_pct', type=float, default=0.04, help='Take profit percentage')
    
    # TMA Filters
    filter_group = parser.add_argument_group('TMA Filter Options')
    filter_group.add_argument('--use_tma_direction_filter', action='store_true', help='Enable TMA direction filter')
    filter_group.add_argument('--use_engulfing_filter', action='store_true', help='Enable engulfing pattern filter')
    filter_group.add_argument('--use_ema_confirmation', action='store_true', help='Enable EMA confirmation filter')
    
    # Parameter sweep options
    sweep_group = parser.add_argument_group('Parameter Sweep Options')
    sweep_group.add_argument('--enable-sweep', action='store_true', help='Enable parameter sweeping')
    sweep_group.add_argument('--tma_period_range', type=str, help='Range of TMA periods (e.g., "8,14,20" or "8:20:3")')
    sweep_group.add_argument('--atr_multiplier_range', type=str, help='Range of ATR multiplier values (e.g., "1.0,1.5,2.0")')
    sweep_group.add_argument('--fast_ema_range', type=str, help='Range of fast EMA periods (e.g., "5,8,13,21")')
    sweep_group.add_argument('--slow_ema_range', type=str, help='Range of slow EMA periods (e.g., "20,50,100")')
    sweep_group.add_argument('--stop_loss_range', type=str, help='Range of stop loss percentages (e.g., "0.01,0.02,0.03")')
    sweep_group.add_argument('--take_profit_range', type=str, help='Range of take profit percentages (e.g., "0.02,0.04,0.06")')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output_dir', type=str, default='reports', help='Directory to save output files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run TMA Monte Carlo validation
    run_tma_monte_carlo(args)

if __name__ == "__main__":
    main() 