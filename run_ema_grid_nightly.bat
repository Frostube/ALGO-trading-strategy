@echo off
echo Starting EMA Grid Search at %date% %time%
echo ========================================

:: Create params directory if it doesn't exist
if not exist params mkdir params

:: Run the grid search with default parameters (2h timeframe)
python run_ema_grid_search.py --timeframe 2h --symbols BTC/USDT,ETH/USDT,SOL/USDT,AVAX/USDT,BNB/USDT,ADA/USDT

echo Grid search completed at %date% %time%
echo Results saved to params/ directory 