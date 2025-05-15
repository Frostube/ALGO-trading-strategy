@echo off
echo Starting Sequential EMA Grid Search at %date% %time%
echo ========================================================

:: Create params directory if it doesn't exist
if not exist params mkdir params

REM === ETH ===
echo Running optimization for ETH/USDT...
python run_ema_grid_search.py ^
  --symbols "ETH/USDT" ^
  --timeframe 2h ^
  --workers 2 ^
  --days 180

REM === SOL ===
echo Running optimization for SOL/USDT...
python run_ema_grid_search.py ^
  --symbols "SOL/USDT" ^
  --timeframe 2h ^
  --workers 2 ^
  --days 180

REM === BNB ===
echo Running optimization for BNB/USDT...
python run_ema_grid_search.py ^
  --symbols "BNB/USDT" ^
  --timeframe 2h ^
  --workers 2 ^
  --days 180

echo Grid search completed at %date% %time%
echo Results saved to params/ directory

:: Ping when done (optional)
echo.
echo Grid optimization complete! 