@echo off
REM TMA Overlay Strategy Robustness Testing Script for Windows
REM
REM This script runs comprehensive Monte Carlo simulations to test the robustness
REM of the TMA Overlay strategy across different parameter combinations and timeframes.
REM
REM It will:
REM 1. Register the TMA strategy with the strategy factory
REM 2. Run parameter sweeps across key variables
REM 3. Test on multiple timeframes (1h, 4h, 1d)
REM 4. Save results to the reports directory

echo === TMA Overlay Strategy Robustness Testing ===
echo This will run multiple tests to evaluate the TMA strategy's robustness
echo Results will be saved to the reports directory
echo.

REM Create reports directory if it doesn't exist
if not exist reports mkdir reports

REM Register the TMA strategy with the factory system
echo Registering TMA strategy with the strategy factory...
python register_tma_strategy.py

REM Check if registration was successful
if %ERRORLEVEL% NEQ 0 (
    echo Error: Failed to register TMA strategy. Aborting tests.
    exit /b 1
)

echo TMA strategy registered successfully!
echo.

REM Run parameter sweep for 4-hour timeframe
echo === Running 4-hour Timeframe Parameter Sweep ===
echo Starting parameter sweep for BTC/USDT on 4h timeframe...

python test_tma_monte_carlo.py ^
    --symbol "BTC/USDT" ^
    --timeframe "4h" ^
    --start_date "2022-01-01" ^
    --mc_sims 250 ^
    --enable-sweep ^
    --tma_period_range "8,14,20" ^
    --atr_multiplier_range "1.0,1.5,2.0" ^
    --fast_ema_range "5,8,13,21" ^
    --slow_ema_range "20,50,100" ^
    --stop_loss_range "0.015,0.02,0.03" ^
    --take_profit_range "0.03,0.04,0.06" ^
    --output_dir "reports/tma_4h_2y"

echo Completed parameter sweep for BTC/USDT on 4h timeframe
echo.

REM Run parameter sweep for 1-hour timeframe
echo === Running 1-hour Timeframe Parameter Sweep ===
echo Starting parameter sweep for BTC/USDT on 1h timeframe...

python test_tma_monte_carlo.py ^
    --symbol "BTC/USDT" ^
    --timeframe "1h" ^
    --start_date "2023-01-01" ^
    --mc_sims 250 ^
    --enable-sweep ^
    --tma_period_range "8,14,20" ^
    --atr_multiplier_range "1.0,1.5,2.0" ^
    --fast_ema_range "5,8,13,21" ^
    --slow_ema_range "20,50,100" ^
    --stop_loss_range "0.015,0.02,0.03" ^
    --take_profit_range "0.03,0.04,0.06" ^
    --output_dir "reports/tma_1h_1y"

echo Completed parameter sweep for BTC/USDT on 1h timeframe
echo.

REM Run parameter sweep for daily timeframe
echo === Running Daily Timeframe Parameter Sweep ===
echo Starting parameter sweep for BTC/USDT on 1d timeframe...

python test_tma_monte_carlo.py ^
    --symbol "BTC/USDT" ^
    --timeframe "1d" ^
    --start_date "2021-01-01" ^
    --mc_sims 250 ^
    --enable-sweep ^
    --tma_period_range "8,14,20" ^
    --atr_multiplier_range "1.0,1.5,2.0" ^
    --fast_ema_range "5,8,13,21" ^
    --slow_ema_range "20,50,100" ^
    --stop_loss_range "0.015,0.02,0.03" ^
    --take_profit_range "0.03,0.04,0.06" ^
    --output_dir "reports/tma_1d_3y"

echo All parameter sweeps completed!
echo.

REM Run focused tests with optimal parameters
echo === Running Focused Tests with Optimal Parameters ===

REM 4-hour timeframe with optimal parameters (from previous testing)
echo Running focused test with optimal parameters on 4h timeframe...

python test_tma_monte_carlo.py ^
    --symbol "BTC/USDT" ^
    --timeframe "4h" ^
    --start_date "2022-01-01" ^
    --mc_sims 500 ^
    --tma_period 14 ^
    --atr_multiplier 1.5 ^
    --fast_ema 8 ^
    --slow_ema 21 ^
    --use_ema_confirmation ^
    --output_dir "reports/tma_4h_optimal"

echo Completed focused test on 4h timeframe
echo.

REM 1-hour timeframe with optimal parameters
echo Running focused test with optimal parameters on 1h timeframe...

python test_tma_monte_carlo.py ^
    --symbol "BTC/USDT" ^
    --timeframe "1h" ^
    --start_date "2023-01-01" ^
    --mc_sims 500 ^
    --tma_period 14 ^
    --atr_multiplier 1.5 ^
    --fast_ema 8 ^
    --slow_ema 21 ^
    --use_ema_confirmation ^
    --output_dir "reports/tma_1h_optimal"

echo All focused tests completed!
echo.

echo === TMA Strategy Robustness Testing Complete ===
echo Check the reports directory for detailed results and visualizations
echo Parameter sweep results are saved in JSON format for further analysis

pause 