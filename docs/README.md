# Performance Logging System

This directory contains performance logs for the algorithmic trading strategies in this project. The performance logs are automatically updated after each backtest run, providing a historical record of strategy performance.

## Performance Log Format

The performance log is stored in Markdown format in the `performance_log.md` file. Each entry includes:

- **Date (UTC)**: When the test was run
- **Strategy**: Name of the strategy or strategy ensemble used
- **Dataset**: Symbol, timeframe, and period tested
- **Params**: Key parameters used in the strategy
- **PF**: Profit Factor (gross profits / gross losses)
- **Win %**: Win Rate percentage
- **DD %**: Maximum Drawdown percentage
- **Net Return %**: Net Return percentage

## How to Log Performance

Performance is logged automatically when running backtests with the `--log` flag:

```bash
# Log performance of a single strategy backtest
python log_performance.py --symbol BTC/USDT --timeframe 1h --days 90 --strategy ema_crossover --log

# Log performance of a different strategy
python log_performance.py --symbol ETH/USDT --timeframe 4h --days 60 --strategy rsi_momentum --log
```

## Git Integration

Performance logs can be automatically committed to Git and optionally pushed to remote repositories:

```bash
# Commit performance log to Git
python log_performance.py --symbol BTC/USDT --timeframe 1h --strategy ema_crossover --log --git-commit

# Commit and push performance log
python log_performance.py --symbol BTC/USDT --timeframe 1h --strategy ema_crossover --log --git-commit --git-push
```

## Automated Performance Logging

A GitHub Actions workflow is configured to automatically run backtests and log performance on a weekly basis. This ensures that the performance log is consistently updated and provides historical tracking of strategy performance over time.

The workflow is defined in `.github/workflows/performance_log.yml` and runs backtests with different configurations to provide comprehensive performance tracking. 