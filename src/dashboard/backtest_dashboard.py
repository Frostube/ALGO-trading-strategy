#!/usr/bin/env python3
"""
Interactive backtesting dashboard for BTC/USDT scalping strategy.
This module provides a Streamlit-based interface for running backtests
with customizable parameters and visualizing the results.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import directly from the backtest module
from src.backtest.backtest import Backtester
from src.data.fetcher import DataFetcher
from src.config import (
    SYMBOL, STOP_LOSS_PCT, TAKE_PROFIT_PCT, RSI_PERIOD,
    RSI_LONG_THRESHOLD, RSI_SHORT_THRESHOLD, EMA_FAST, 
    EMA_SLOW, VOLUME_PERIOD, VOLUME_THRESHOLD,
    ATR_PERIOD, ATR_SL_MULTIPLIER, ATR_TP_MULTIPLIER
)

def show_backtest_dashboard():
    """Display the interactive backtesting dashboard."""
    st.header("Interactive Strategy Backtesting")
    
    # Connection settings
    st.sidebar.subheader("Connection Settings")
    use_testnet = st.sidebar.checkbox("Use Binance Testnet", value=True)
    
    if use_testnet:
        st.sidebar.info("⚠️ Using Binance Testnet - If testnet data is unavailable, mock data will be generated for testing.")
    
    # Sidebar for parameters
    st.sidebar.subheader("Backtest Parameters")
    
    # Data parameters
    st.sidebar.subheader("Data Settings")
    days = st.sidebar.slider("Days of Historical Data", min_value=7, max_value=90, value=30, step=1)
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h"], index=1)
    
    # Strategy parameters
    st.sidebar.subheader("Strategy Settings")
    
    # EMA parameters
    ema_col1, ema_col2 = st.sidebar.columns(2)
    ema_fast = ema_col1.number_input("EMA Fast Period", min_value=3, max_value=50, value=EMA_FAST)
    ema_slow = ema_col2.number_input("EMA Slow Period", min_value=5, max_value=200, value=EMA_SLOW)
    
    # RSI parameters
    rsi_col1, rsi_col2, rsi_col3 = st.sidebar.columns(3)
    rsi_period = rsi_col1.number_input("RSI Period", min_value=2, max_value=14, value=RSI_PERIOD)
    rsi_long = rsi_col2.number_input("RSI Long Threshold", min_value=1, max_value=40, value=RSI_LONG_THRESHOLD)
    rsi_short = rsi_col3.number_input("RSI Short Threshold", min_value=60, max_value=99, value=RSI_SHORT_THRESHOLD)
    
    # Volume parameters
    vol_col1, vol_col2 = st.sidebar.columns(2)
    volume_period = vol_col1.number_input("Volume Period", min_value=5, max_value=50, value=VOLUME_PERIOD)
    volume_threshold = vol_col2.number_input("Volume Threshold", min_value=1.1, max_value=3.0, value=VOLUME_THRESHOLD, format="%.1f")
    
    # Risk parameters
    st.sidebar.subheader("Risk Settings")
    risk_col1, risk_col2 = st.sidebar.columns(2)
    stop_loss = risk_col1.number_input("Stop Loss %", min_value=0.1, max_value=2.0, value=STOP_LOSS_PCT*100, format="%.1f") / 100
    take_profit = risk_col2.number_input("Take Profit %", min_value=0.1, max_value=5.0, value=TAKE_PROFIT_PCT*100, format="%.1f") / 100
    
    # ATR parameters
    atr_col1, atr_col2, atr_col3 = st.sidebar.columns(3)
    atr_period = atr_col1.number_input("ATR Period", min_value=5, max_value=20, value=ATR_PERIOD)
    atr_sl = atr_col2.number_input("ATR SL Multiplier", min_value=0.5, max_value=3.0, value=ATR_SL_MULTIPLIER, format="%.1f")
    atr_tp = atr_col3.number_input("ATR TP Multiplier", min_value=0.5, max_value=5.0, value=ATR_TP_MULTIPLIER, format="%.1f")
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    use_train_test = st.sidebar.checkbox("Split into Train/Test", value=True)
    
    if use_train_test:
        train_size = st.sidebar.slider("Training Set Size", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
    
    initial_balance = st.sidebar.number_input("Initial Balance ($)", min_value=100, max_value=100000, value=10000, step=1000)
    
    # Run backtest button
    if st.sidebar.button("Run Backtest", use_container_width=True):
        # Set up the progress bar
        progress_bar = st.progress(0)
        
        # Status message
        status_msg = st.empty()
        status_msg.info("Loading historical data...")
        
        # Collect parameters
        params = {
            "ema_fast_period": ema_fast,
            "ema_slow_period": ema_slow,
            "rsi_period": rsi_period,
            "rsi_long_threshold": rsi_long,
            "rsi_short_threshold": rsi_short,
            "volume_period": volume_period,
            "volume_threshold": volume_threshold,
            "stop_loss_pct": stop_loss,
            "take_profit_pct": take_profit,
            "atr_period": atr_period,
            "atr_sl_multiplier": atr_sl,
            "atr_tp_multiplier": atr_tp
        }
        
        try:
            # Load data
            data_fetcher = DataFetcher(use_testnet=use_testnet)
            progress_bar.progress(10)
            
            data = data_fetcher.fetch_historical_data(days=days, timeframe=timeframe)
            progress_bar.progress(40)
            
            # Check if data is empty
            if data.empty:
                progress_bar.empty()
                status_msg.error(f"No historical data available for {SYMBOL} with timeframe {timeframe} over the last {days} days.")
                st.error("""
                Could not fetch data. Possible reasons:
                1. Connection issues with the exchange
                2. API rate limits exceeded
                3. The selected symbol or timeframe is not available
                4. The selected date range has no data
                
                Try with a different timeframe or date range, or check your API connection.
                """)
                return
            
            status_msg.info("Running backtest...")
            
            # Initialize backtester
            backtester = Backtester(data=data, initial_balance=initial_balance)
            
            # Run backtest with train/test split if selected
            if use_train_test:
                results = backtester.run(train_test_split=train_size)
                progress_bar.progress(80)
            else:
                results = backtester.run()
                progress_bar.progress(80)
            
            # Display results
            progress_bar.progress(100)
            status_msg.success("Backtest completed successfully!")
            
            # Clear progress elements after completion
            progress_bar.empty()
            
            # Display the results
            display_backtest_results(results, use_train_test, backtester)
            
        except Exception as e:
            status_msg.error(f"Error in backtest: {str(e)}")
            st.exception(e)
    
    # If no backtest has been run yet
    else:
        st.info("👈 Adjust the parameters in the sidebar and click 'Run Backtest' to start.")
        st.markdown("""
        ### How to use this dashboard
        
        1. Set the data parameters (days of history, timeframe)
        2. Adjust strategy parameters to test different configurations
        3. Configure risk settings
        4. Click 'Run Backtest' to execute with these parameters
        5. Review performance metrics and trade analysis
        6. Iterate and refine your strategy
        
        The backtest will run the BTC/USDT scalping strategy with your custom parameters and
        display comprehensive results including equity curve, drawdown analysis, trade distribution,
        and key performance metrics.
        """)

def display_backtest_results(results, is_split, backtester):
    """
    Display the backtest results in a structured format.
    
    Args:
        results: Dict containing backtest results
        is_split: Boolean indicating if results are split into train/test
        backtester: Backtester instance with equity curve data
    """
    if is_split:
        # Create tabs for Train/Test results
        train_tab, test_tab, combined_tab = st.tabs(["Training Set", "Testing Set", "Combined Analysis"])
        
        # Convert equity_curve to list if it's not already
        equity_curve = backtester.equity_curve
        if not isinstance(equity_curve, list):
            equity_curve = list(equity_curve)
        
        # Calculate the midpoint index for splitting
        mid_point = len(equity_curve) // 2
        
        with train_tab:
            display_result_set(results["train"], "Training Set", equity_curve[:mid_point])
        
        with test_tab:
            display_result_set(results["test"], "Testing Set", equity_curve[mid_point:])
        
        with combined_tab:
            display_combined_results(results)
    else:
        # Display full results
        display_result_set(results, "Full Backtest", backtester.equity_curve)

def display_result_set(result, title, equity_data):
    """Display a set of backtest results."""
    st.subheader(f"{title} Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Return", f"{result['total_return']*100:.2f}%")
    col2.metric("Win Rate", f"{result['win_rate']*100:.2f}%")
    col3.metric("Profit Factor", f"{result.get('profit_factor', 0):.2f}")
    col4.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Max Drawdown", f"{result['max_drawdown']*100:.2f}%")
    col2.metric("Total Trades", f"{result['total_trades']}")
    col3.metric("Avg. Trade", f"${result.get('avg_trade_pnl', 0):.2f}")
    col4.metric("Avg. Duration", f"{result.get('avg_trade_duration', '0')}") 
    
    # Equity curve
    st.subheader("Equity Curve")
    
    # Create equity curve DataFrame
    equity_df = pd.DataFrame(equity_data)
    
    # Plot equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[entry["timestamp"] for entry in equity_data],
        y=[entry["equity"] for entry in equity_data],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title="Account Equity Over Time",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True, key=f"equity_curve_{title}")
    
    # Drawdown analysis
    if "drawdowns" in result:
        st.subheader("Drawdown Analysis")
        
        # Create drawdown DataFrame
        drawdowns = result["drawdowns"]
        
        if isinstance(drawdowns, list) and len(drawdowns) > 0:
            drawdown_df = pd.DataFrame(drawdowns)
            
            # Plot drawdown
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=drawdown_df["start"],
                y=drawdown_df["depth"] * 100,  # Convert to percentage
                mode='markers',
                name='Drawdowns',
                marker=dict(
                    size=10,
                    color=drawdown_df["depth"] * 100,
                    colorscale='Reds',
                    colorbar=dict(title="Depth (%)"),
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Drawdown Analysis",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"drawdown_{title}")
    
    # Trade analysis
    if "trades" in result and len(result["trades"]) > 0:
        st.subheader("Trade Analysis")
        
        trades_df = pd.DataFrame(result["trades"])
        
        # Trade distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Trade outcome distribution
            outcome_counts = trades_df["pnl"].apply(lambda x: "Win" if x > 0 else "Loss").value_counts()
            
            fig = px.pie(
                values=outcome_counts.values,
                names=outcome_counts.index,
                title="Trade Outcomes",
                color=outcome_counts.index,
                color_discrete_map={"Win": "green", "Loss": "red"}
            )
            
            st.plotly_chart(fig, use_container_width=True, key=f"outcome_pie_{title}")
        
        with col2:
            # Trade side distribution
            # Note: Check if 'side' exists in columns, otherwise use 'position'
            side_column = 'side' if 'side' in trades_df.columns else 'position'
            if side_column in trades_df.columns:
                side_counts = trades_df[side_column].value_counts()
                
                fig = px.pie(
                    values=side_counts.values,
                    names=side_counts.index,
                    title="Trade Direction",
                    color=side_counts.index,
                    color_discrete_map={"long": "blue", "short": "orange"}
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"side_pie_{title}")
        
        # Trade PnL distribution
        st.subheader("Trade PnL Distribution")
        
        fig = px.histogram(
            trades_df,
            x="pnl",
            nbins=20,
            title="Trade PnL Distribution",
            color_discrete_sequence=["blue"]
        )
        
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            xaxis_title="PnL ($)",
            yaxis_title="Count",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"pnl_hist_{title}")
        
        # Trade table
        st.subheader("Trade List")
        
        # Clean up the trades DataFrame for display
        if "duration" in trades_df.columns:
            trades_df["duration"] = trades_df["duration"].apply(lambda x: str(x).split(".")[0])
        
        display_df = trades_df.copy()
        if "entry_time" in display_df.columns:
            display_df["entry_time"] = pd.to_datetime(display_df["entry_time"]).dt.strftime('%Y-%m-%d %H:%M')
        if "exit_time" in display_df.columns:
            display_df["exit_time"] = pd.to_datetime(display_df["exit_time"]).dt.strftime('%Y-%m-%d %H:%M')
        
        # Round numeric columns
        numeric_cols = ["entry_price", "exit_price", "pnl", "stop_loss", "take_profit"]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        st.dataframe(display_df.tail(50), use_container_width=True, key=f"trades_df_{title}")

def display_combined_results(results):
    """Display combined analysis of train/test results."""
    st.subheader("Train vs. Test Performance")
    
    # Create comparison dataframe
    comparison = {
        "Metric": ["Total Return", "Win Rate", "Sharpe Ratio", "Max Drawdown", "Total Trades"],
        "Training Set": [
            f"{results['train']['total_return']*100:.2f}%",
            f"{results['train']['win_rate']*100:.2f}%",
            f"{results['train']['sharpe_ratio']:.2f}",
            f"{results['train']['max_drawdown']*100:.2f}%",
            f"{results['train']['total_trades']}"
        ],
        "Testing Set": [
            f"{results['test']['total_return']*100:.2f}%",
            f"{results['test']['win_rate']*100:.2f}%",
            f"{results['test']['sharpe_ratio']:.2f}",
            f"{results['test']['max_drawdown']*100:.2f}%",
            f"{results['test']['total_trades']}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison)
    st.table(comparison_df)
    
    # Consistency analysis
    st.subheader("Consistency Analysis")
    
    # Calculate numeric values for plotting
    train_values = {
        "Return": results['train']['total_return'] * 100,
        "Win Rate": results['train']['win_rate'] * 100,
        "Sharpe": results['train']['sharpe_ratio'],
        "Drawdown": -results['train']['max_drawdown'] * 100  # Negate to show as negative
    }
    
    test_values = {
        "Return": results['test']['total_return'] * 100,
        "Win Rate": results['test']['win_rate'] * 100,
        "Sharpe": results['test']['sharpe_ratio'],
        "Drawdown": -results['test']['max_drawdown'] * 100  # Negate to show as negative
    }
    
    # Create DataFrame for plotting
    plot_data = {
        "Metric": list(train_values.keys()),
        "Training": list(train_values.values()),
        "Testing": list(test_values.values())
    }
    
    plot_df = pd.DataFrame(plot_data)
    
    # Calculate overfitting score
    # Simple overfitting score based on ratio of train to test performance
    # Higher score means more overfitting
    overfitting_score = (train_values["Return"] / max(0.1, test_values["Return"])) * 0.4 + \
                        (train_values["Win Rate"] / max(0.1, test_values["Win Rate"])) * 0.3 + \
                        (train_values["Sharpe"] / max(0.1, test_values["Sharpe"])) * 0.3
    
    overfitting_normalized = min(10, max(1, overfitting_score)) / 10
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart comparing train vs test metrics
        fig = px.bar(
            plot_df, 
            x="Metric", 
            y=["Training", "Testing"],
            barmode="group",
            title="Train vs. Test Metrics",
            color_discrete_map={"Training": "blue", "Testing": "green"}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="train_test_comparison")
    
    with col2:
        # Overfitting gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = overfitting_normalized * 10,
            title = {'text': "Overfitting Risk"},
            gauge = {
                'axis': {'range': [0, 10], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 3], 'color': "green"},
                    {'range': [3, 7], 'color': "yellow"},
                    {'range': [7, 10], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': overfitting_normalized * 10
                }
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True, key="overfitting_gauge")
    
    # Interpretation
    if overfitting_normalized < 0.3:
        st.success("✅ Low risk of overfitting. The strategy performs consistently across both training and testing datasets.")
    elif overfitting_normalized < 0.7:
        st.warning("⚠️ Moderate risk of overfitting. Some divergence between training and testing performance.")
    else:
        st.error("❌ High risk of overfitting. Strategy performs much better on training data than testing data.")
    
    st.markdown("""
    **Interpreting the overfitting risk:**
    - **Low risk (1-3)**: The strategy generalizes well to new data
    - **Moderate risk (3-7)**: Some parameter adjustments may be needed for better generalization
    - **High risk (7-10)**: Significant overfitting detected, strategy needs fundamental changes
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="Backtest Dashboard",
        page_icon="📈",
        layout="wide"
    )
    show_backtest_dashboard() 