#!/bin/bash
# TMA Overlay Strategy Robustness Testing Script
#
# This script runs comprehensive Monte Carlo simulations to test the robustness
# of the TMA Overlay strategy across different parameter combinations and timeframes.
#
# It will:
# 1. Register the TMA strategy with the strategy factory
# 2. Run parameter sweeps across key variables
# 3. Test on multiple timeframes (1h, 4h, 1d)
# 4. Save results to the reports directory

echo "=== TMA Overlay Strategy Robustness Testing ==="
echo "This will run multiple tests to evaluate the TMA strategy's robustness"
echo "Results will be saved to the reports directory"
echo ""

# Create reports directory if it doesn't exist
mkdir -p reports

# Register the TMA strategy with the factory system
echo "Registering TMA strategy with the strategy factory..."
python register_tma_strategy.py

# Check if registration was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to register TMA strategy. Aborting tests."
    exit 1
fi

echo "TMA strategy registered successfully!"
echo ""

# Function to run a parameter sweep with specific settings
run_parameter_sweep() {
    local timeframe=$1
    local symbol=$2
    local period=$3
    local start_date=$4
    local num_sims=$5
    
    echo "Starting parameter sweep for $symbol on $timeframe timeframe..."
    
    # Run the Monte Carlo validation with parameter sweep
    python test_tma_monte_carlo.py \
        --symbol "$symbol" \
        --timeframe "$timeframe" \
        --start_date "$start_date" \
        --mc_sims "$num_sims" \
        --enable-sweep \
        --tma_period_range "8,14,20" \
        --atr_multiplier_range "1.0,1.5,2.0" \
        --fast_ema_range "5,8,13,21" \
        --slow_ema_range "20,50,100" \
        --stop_loss_range "0.015,0.02,0.03" \
        --take_profit_range "0.03,0.04,0.06" \
        --output_dir "reports/tma_${timeframe}_${period}"
    
    echo "Completed parameter sweep for $symbol on $timeframe timeframe"
    echo ""
}

# Function to run a single focused test with specific parameters
run_focused_test() {
    local timeframe=$1
    local symbol=$2
    local tma_period=$3
    local atr_mult=$4
    local fast_ema=$5
    local slow_ema=$6
    local start_date=$7
    local num_sims=$8
    
    echo "Running focused test with optimal parameters on $timeframe timeframe..."
    
    # Run the Monte Carlo validation with specific parameters
    python test_tma_monte_carlo.py \
        --symbol "$symbol" \
        --timeframe "$timeframe" \
        --start_date "$start_date" \
        --mc_sims "$num_sims" \
        --tma_period "$tma_period" \
        --atr_multiplier "$atr_mult" \
        --fast_ema "$fast_ema" \
        --slow_ema "$slow_ema" \
        --use_ema_confirmation \
        --output_dir "reports/tma_${timeframe}_optimal"
    
    echo "Completed focused test on $timeframe timeframe"
    echo ""
}

# Run parameter sweeps for different timeframes
echo "=== Running Parameter Sweeps ==="

# 4-hour timeframe, 250 simulations, 2-year period
run_parameter_sweep "4h" "BTC/USDT" "2y" "2022-01-01" "250"

# 1-hour timeframe, 250 simulations, 1-year period
run_parameter_sweep "1h" "BTC/USDT" "1y" "2023-01-01" "250"

# Daily timeframe, 250 simulations, 3-year period
run_parameter_sweep "1d" "BTC/USDT" "3y" "2021-01-01" "250"

echo "All parameter sweeps completed!"
echo ""

# Run focused tests with optimal parameters
echo "=== Running Focused Tests with Optimal Parameters ==="

# 4-hour timeframe with optimal parameters (from previous testing)
run_focused_test "4h" "BTC/USDT" "14" "1.5" "8" "21" "2022-01-01" "500"

# 1-hour timeframe with optimal parameters
run_focused_test "1h" "BTC/USDT" "14" "1.5" "8" "21" "2023-01-01" "500"

echo "All focused tests completed!"
echo ""

echo "=== TMA Strategy Robustness Testing Complete ==="
echo "Check the reports directory for detailed results and visualizations"
echo "Parameter sweep results are saved in JSON format for further analysis" 