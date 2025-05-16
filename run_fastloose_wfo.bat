@echo off
echo Running Walk-Forward Optimization for Fast ^& Loose High-Leverage Strategy
echo ----------------------------------------------------------------------

python run_fastloose_wfo.py --symbol "BTC/USDT" --timeframe "4h" --days 366 --train-days 60 --test-days 30 --output-csv "results/fastloose_wfo_btc_4h_summary.csv"

echo.
echo If you want to test with different parameters, use:
echo run_fastloose_wfo.bat --symbol "ETH/USDT" --timeframe "1h" --days 180 --train-days 45 --test-days 15

pause 