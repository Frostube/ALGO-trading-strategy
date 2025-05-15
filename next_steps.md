# Next Steps for Strategy Enhancement

## Current Status

The EMA grid optimization is now running in the background for:
1. ETH/USDT (in progress)
2. SOL/USDT (queued)  
3. BNB/USDT (queued)

This process will generate optimized parameter files in the `params/` directory that will be automatically used by the trading strategy.

## ✅ Implemented: Volatility Regime Switch

The volatility regime switch has been successfully implemented. This feature:

- Classifies markets into QUIET, NORMAL, and EXPLOSIVE volatility regimes
- Adjusts risk parameters based on current market conditions
- Enables pyramiding only in high-volatility environments
- Adapts strategy parameters to different market conditions

You can test the volatility regime monitor by running:
```
python check_volatility_regimes.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT" --plot
```

## Testing the Optimized Parameters

Once the grid search completes, run:

```
.\run_backtest_with_params.bat
```

This will execute a backtest using the optimized parameters across all symbols.

## Next Edge Upgrade: Fee & Funding Cost Model

The fee and funding model is ready for implementation:

1. Copy the code from `src/utils/fee_model.py` to your project
2. Integrate it with your trading engine:

```python
# --- In Account class ---
self.fee_model = FeeModel(exchange_name='binance')

# --- In place_order method ---
execution_details = self.fee_model.execute_order(
    symbol=symbol,
    side=side,
    quantity=quantity,
    price=price,
    order_type=order_type
)

# Store fees in trade record
trade['fee_amount'] = execution_details['fee_amount']
trade['slippage_pct'] = execution_details['slippage_pct']

# --- In update method ---
# Apply funding payments every 8 hours
positions, funding_applied = self.fee_model.update_positions_with_funding(
    self.positions, 
    current_time=datetime.now()
)
```

3. Update the backtest reporting to show fee and funding impact:

```python
# In performance report
print(f"Gross P&L: ${backtest_results['gross_pnl']:.2f}")
print(f"Fee Cost: ${backtest_results['fee_cost']:.2f} ({backtest_results['fee_impact']:.2f}%)")
print(f"Funding Cost: ${backtest_results['funding_cost']:.2f} ({backtest_results['funding_impact']:.2f}%)")
print(f"Net P&L: ${backtest_results['net_pnl']:.2f}")
```

## Commit Changes

After all implementations are complete:

```
git add .
git commit -m "Added volatility regime switch and EMA grid optimization"
git push
``` 