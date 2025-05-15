# Exchange Fee & Funding Model

## Overview

This feature implements a realistic exchange fee and funding rate model to accurately simulate trading costs in both backtests and live trading. By incorporating actual market fees, funding rates, and slippage, the system can provide more realistic performance estimates.

## Fee Structure Implementation

### Exchange Fees

```python
# Standard fee structure
TAKER_FEE = 0.0004  # 0.04% for market orders/taker fills
MAKER_REBATE = -0.0001  # -0.01% rebate for limit orders that provide liquidity

# VIP tiers can be configured based on volume
def get_fee_rate(account_volume_30d, is_taker=True):
    """Dynamic fee calculation based on 30-day volume"""
    if is_taker:
        return TAKER_FEE
    else:
        return MAKER_REBATE
```

### Funding Rates

```python
# Fetch historical funding rates from exchange
def fetch_funding_history(symbol, start_time, end_time):
    """Fetch 8-hour funding rate payments history"""
    # Implementation using CCXT to fetch actual funding rate history
    return funding_rates_df  # DataFrame with timestamp and rate columns

# Apply funding costs to open positions
def calculate_funding_cost(position_size, funding_rate, position_value):
    """
    Calculate 8-hour funding payment/charge
    
    Args:
        position_size: Position size in base currency (negative for shorts)
        funding_rate: Current funding rate (negative means shorts pay longs)
        position_value: USD value of position
        
    Returns:
        Cost in USD (negative is payment, positive is charge)
    """
    # Long positions pay positive funding, receive negative funding
    # Short positions pay negative funding, receive positive funding
    return position_value * funding_rate * np.sign(position_size)
```

### Execution Slippage

```python
# Default slippage model
LIMIT_SLIPPAGE = 0.0002  # 0.02% for limit orders
MARKET_SLIPPAGE = 0.0005  # 0.05% for market orders/fallbacks

# Apply slippage to order execution
def apply_slippage(price, order_type, side):
    """
    Apply realistic slippage to order execution
    
    Args:
        price: Base execution price
        order_type: 'limit' or 'market'
        side: 'buy' or 'sell'
        
    Returns:
        Adjusted execution price
    """
    slippage = LIMIT_SLIPPAGE if order_type == 'limit' else MARKET_SLIPPAGE
    multiplier = (1 + slippage) if side == 'buy' else (1 - slippage)
    return price * multiplier
```

## Integration with Trading Engine

### 1. Order Execution Hooks

```python
def execute_order(symbol, side, quantity, price=None, order_type='limit'):
    """
    Execute order with fees and slippage
    
    Returns:
        dict: Execution details including fees
    """
    # 1. Determine order type and apply slippage
    actual_price = apply_slippage(price, order_type, side)
    
    # 2. Calculate order value
    order_value = quantity * actual_price
    
    # 3. Apply exchange fees
    is_taker = (order_type == 'market')
    fee_rate = get_fee_rate(account_volume_30d, is_taker=is_taker)
    fee_amount = order_value * fee_rate
    
    # 4. Return execution details
    return {
        'symbol': symbol,
        'side': side,
        'executed_price': actual_price,
        'executed_quantity': quantity,
        'fee_rate': fee_rate,
        'fee_amount': fee_amount,
        'order_value': order_value,
        'net_value': order_value + fee_amount
    }
```

### 2. Position Maintenance

```python
def update_positions(positions):
    """
    Update positions with funding rate costs
    Called every 8 hours in backtest, on funding timestamp in live
    """
    for symbol, position in positions.items():
        if position['size'] != 0:
            # Fetch current funding rate
            current_funding_rate = get_current_funding_rate(symbol)
            
            # Calculate funding cost
            position_value = abs(position['size'] * position['mark_price'])
            funding_cost = calculate_funding_cost(
                position['size'], 
                current_funding_rate,
                position_value
            )
            
            # Apply to position P&L
            position['realized_pnl'] -= funding_cost
            position['funding_payments'] += funding_cost
```

## Integration with Backtesting Engine

1. Modify backtest engine to apply fees on every trade
2. Add 8-hour funding payment events to backtest timeline
3. Track and separate fee and funding costs from raw P&L
4. Include fee and funding metrics in performance reports

## Performance Reporting

The performance reports will be enhanced to include:

```
Total P&L:           $XXX.XX
   Raw P&L:          $XXX.XX
   Fee Costs:        $XX.XX
   Funding Costs:    $XX.XX
   
Net Return:          XX.XX%
Gross Return:        XX.XX%
Fee Impact:          -X.XX%
Funding Impact:      -X.XX%
```

## Expected Impact

Based on typical fee structures and funding rates:

| Cost Type | Typical Impact | Notes |
|-----------|---------------|-------|
| Exchange Fees | -1.0% to -1.5% quarterly | Higher for market orders/high frequency |
| Funding Rates | -0.2% to -0.8% quarterly | Varies by market conditions |
| Slippage | -0.1% to -0.3% quarterly | Higher during volatility |

Total expected cost impact: **-1.3% to -2.6% quarterly**

## Next Steps

1. Implement the fee and funding model modules
2. Fetch historical funding rates for backtesting
3. Update backtesting engine to incorporate fees and funding
4. Re-run previous backtests to measure impact 