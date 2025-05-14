@echo off
echo Running scheduled optimization tasks...
cd /d %~dp0

REM Run daily parameter optimization at 00:15 UTC
python -m src.optimization.daily_optimizer --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" --auto-commit

echo Optimization complete at %DATE% %TIME% 