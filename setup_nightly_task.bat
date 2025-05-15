@echo off
echo Setting up Windows Task Scheduler for nightly EMA grid optimization
echo =================================================================

:: Set paths and parameters
set TASK_NAME="Nightly_EMA_Grid"
set TASK_PATH=%CD%\schedule_optimiser.bat
set START_TIME=00:20

:: Create the task
echo Creating scheduled task to run at %START_TIME% UTC daily
schtasks /Create /TN %TASK_NAME% /TR "%TASK_PATH%" /SC DAILY /ST %START_TIME% /RU SYSTEM /RL HIGHEST

echo.
echo Task created! The optimizer will run nightly at %START_TIME% UTC
echo This time allows the exchange data to finish printing the last 2-hour candle
echo You can view and modify the task in Windows Task Scheduler
echo. 