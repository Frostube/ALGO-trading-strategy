# Trading Strategy Performance Logging System

This document describes the performance logging system implemented for tracking the performance of algorithmic trading strategies. The system automatically records key performance metrics after each backtest/forward-test run, providing a historical record of trading strategy performance.

## System Components

The performance logging system consists of the following components:

1. **Performance Logger** (`src/utils/performance_logger.py`): Core utility that writes performance metrics to a markdown file.
2. **Performance Query Helper** (`src/utils/prompt_or_flag.py`): Utility that helps decide whether to log performance based on user input or command-line flags.
3. **Git Integration** (`src/utils/git_commit.py`): Tool for automatically committing and pushing performance logs to git.
4. **Performance Log Runner** (`log_performance.py`): Script for running backtests and logging performance metrics.
5. **Performance Visualization Tool** (`visualize_performance.py`): Tool for parsing the performance log and generating visualizations.
6. **GitHub Actions Workflow** (`.github/workflows/performance_log.yml`): Automated workflow for running performance tests and logging results on a schedule.

## Performance Log Format

The performance log is stored in Markdown format at `docs/performance_log.md`. Each entry includes:

| Field | Description |
|-------|-------------|
| Date (UTC) | When the test was run |
| Strategy | Name of the strategy or strategy ensemble used |
| Dataset | Symbol, timeframe, and period tested |
| Params | Key parameters used in the strategy |
| PF | Profit Factor (gross profits / gross losses) |
| Win % | Win Rate percentage |
| DD % | Maximum Drawdown percentage |
| Net Return % | Net Return percentage |

## Usage

### Running Backtests with Performance Logging

```bash
# Basic usage - prompts before logging
python test_enhanced_strategy.py --symbol BTC/USDT --timeframe 1h --days 90

# Auto-log results without prompting
python test_enhanced_strategy.py --symbol BTC/USDT --timeframe 1h --days 90 --log

# Using the log_performance.py wrapper script
python log_performance.py --symbol BTC/USDT --timeframe 1h --days 90 --strategy ema_crossover --log
```

### Automatic Git Integration

```bash
# Commit performance log to Git
python log_performance.py --symbol BTC/USDT --strategy ema_crossover --log --git-commit

# Commit and push performance log
python log_performance.py --symbol BTC/USDT --strategy ema_crossover --log --git-commit --git-push
```

### Visualizing Performance Data

```bash
# Generate visualizations to reports/ directory
python visualize_performance.py

# Show visualizations instead of saving
python visualize_performance.py --show

# Save visualizations to custom directory
python visualize_performance.py --output my_reports/
```

## Benefits

1. **Historical Record**: All test results are captured in one version-controlled file.
2. **Diff-Friendly**: Allows comparing performance changes when strategy code changes.
3. **Lightweight**: Pure Markdown format with no external dependencies for storage.
4. **Extensible**: Performance data can be parsed and analyzed for deeper insights.
5. **Automation-Friendly**: Integrates with CI/CD pipelines for automated testing and logging.

## Future Enhancements

1. **Advanced Visualizations**: Add more visualization options and interactive dashboards.
2. **Statistical Analysis**: Implement statistical tests for strategy comparison.
3. **Parameter Sweeps**: Automatically log results from parameter optimization.
4. **Alert System**: Send notifications when performance metrics fall below thresholds.

## Implementation Details

The system uses a modular approach where the core logging logic is separated from the backtest execution. This allows for easy integration with different backtest frameworks and strategies.

The performance logger saves results in a human-readable Markdown format, which is both easy to read directly and parseable by tools for further analysis. 