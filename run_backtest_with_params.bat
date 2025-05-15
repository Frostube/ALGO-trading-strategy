@echo off
echo Running backtest with optimized parameters
echo ==========================================

python run_adaptive_strategy.py ^
  --mode backtest ^
  --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" ^
  --timeframe 2h ^
  --days 120 ^
  --initial-balance 10000 ^
  --use_cached_params ^
  --log

echo.
echo Backtest complete! Check the reports directory for results. 