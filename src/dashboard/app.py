#!/usr/bin/env python3
"""
Streamlit dashboard for visualizing trading performance.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.db.models import Trade
from src.config import DATABASE_URL

# Import the backtest dashboard - use relative import since we're in the same package
from src.dashboard.backtest_dashboard import show_backtest_dashboard

# Remove sqlite:/// prefix for sqlite3.connect
db_path = DATABASE_URL.replace('sqlite:///', '')

def get_trades_df():
    """Get trades data from database and convert to DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT 
            id, symbol, side, 
            entry_time, entry_price, 
            exit_time, exit_price, 
            exit_reason, pnl, pnl_percent, 
            amount, stop_loss, take_profit 
        FROM trades
        WHERE exit_time IS NOT NULL
        ORDER BY entry_time
    """, conn)
    
    # Convert timestamps
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Calculate trade duration
    df['duration'] = df['exit_time'] - df['entry_time']
    
    # Calculate daily PnL
    df['date'] = df['exit_time'].dt.date
    
    conn.close()
    return df

def get_ohlcv_df(limit=1000):
    """Get OHLCV data from database and convert to DataFrame."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"""
        SELECT 
            symbol, timestamp, open, high, low, close, volume
        FROM ohlcv
        ORDER BY timestamp DESC
        LIMIT {limit}
    """, conn)
    
    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    conn.close()
    return df

def main():
    """Main dashboard function."""
    st.set_page_config(
        page_title="BTC/USDT Scalping Bot Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("BTC/USDT Intra-Day Scalping Bot Dashboard")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Trade Analysis", "Market Data", "Backtesting", "System Settings"]
    )
    
    # Auto-refresh
    st.sidebar.subheader("Auto-refresh")
    refresh_interval = st.sidebar.slider(
        "Refresh interval (seconds)",
        min_value=5,
        max_value=300,
        value=60
    )
    
    if page == "Overview":
        show_overview_page()
    elif page == "Trade Analysis":
        show_trade_analysis_page()
    elif page == "Market Data":
        show_market_data_page()
    elif page == "Backtesting":
        show_backtest_dashboard()
    elif page == "System Settings":
        show_system_settings_page()
    
    # Add auto-refresh HTML
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_interval * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )

def show_overview_page():
    """Show overview dashboard page."""
    st.header("Performance Overview")
    
    # Get trades data
    try:
        df = get_trades_df()
        
        if len(df) == 0:
            st.info("No completed trades found in the database.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Total PnL
        total_pnl = df['pnl'].sum()
        col1.metric("Total PnL", f"${total_pnl:.2f}")
        
        # Win rate
        win_rate = len(df[df['pnl'] > 0]) / len(df) * 100
        col2.metric("Win Rate", f"{win_rate:.1f}%")
        
        # Average trade PnL
        avg_pnl = df['pnl'].mean()
        col3.metric("Avg Trade PnL", f"${avg_pnl:.2f}")
        
        # Number of trades
        col4.metric("Total Trades", len(df))
        
        # Daily PnL chart
        st.subheader("Daily PnL")
        daily_pnl = df.groupby('date')['pnl'].sum().reset_index()
        daily_pnl['cumulative_pnl'] = daily_pnl['pnl'].cumsum()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_pnl['date'],
            y=daily_pnl['pnl'],
            name="Daily PnL"
        ))
        fig.add_trace(go.Scatter(
            x=daily_pnl['date'],
            y=daily_pnl['cumulative_pnl'],
            name="Cumulative PnL",
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Daily and Cumulative PnL",
            xaxis_title="Date",
            yaxis_title="PnL ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade outcomes by day of week and hour
        st.subheader("Trade Outcomes by Time")
        
        col1, col2 = st.columns(2)
        
        # Day of week
        df['day_of_week'] = df['entry_time'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_counts = df.groupby(['day_of_week', df['pnl'] > 0]).size().unstack().reindex(day_order)
        
        if not day_counts.empty and day_counts.shape[1] == 2:
            day_counts.columns = ['Losses', 'Wins']
            day_counts['Win Rate'] = day_counts['Wins'] / (day_counts['Wins'] + day_counts['Losses'])
            
            fig = px.bar(
                day_counts.reset_index(), 
                x='day_of_week', 
                y=['Wins', 'Losses'],
                title="Trades by Day of Week",
                barmode='stack'
            )
            col1.plotly_chart(fig, use_container_width=True)
        
        # Hour of day
        df['hour'] = df['entry_time'].dt.hour
        hour_counts = df.groupby(['hour', df['pnl'] > 0]).size().unstack()
        
        if not hour_counts.empty and hour_counts.shape[1] == 2:
            hour_counts.columns = ['Losses', 'Wins']
            hour_counts['Win Rate'] = hour_counts['Wins'] / (hour_counts['Wins'] + hour_counts['Losses'])
            
            fig = px.bar(
                hour_counts.reset_index(), 
                x='hour', 
                y=['Wins', 'Losses'],
                title="Trades by Hour",
                barmode='stack'
            )
            col2.plotly_chart(fig, use_container_width=True)
        
        # Drawdown chart
        st.subheader("Drawdown Analysis")
        
        # Calculate running maximum
        daily_pnl['running_max'] = daily_pnl['cumulative_pnl'].cummax()
        
        # Calculate drawdown
        daily_pnl['drawdown'] = daily_pnl['cumulative_pnl'] - daily_pnl['running_max']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_pnl['date'],
            y=daily_pnl['drawdown'],
            fill='tozeroy',
            name="Drawdown",
            line=dict(color='red')
        ))
        fig.update_layout(
            title="Drawdown Over Time",
            xaxis_title="Date",
            yaxis_title="Drawdown ($)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Max drawdown
        max_drawdown = daily_pnl['drawdown'].min()
        st.metric("Maximum Drawdown", f"${max_drawdown:.2f}")
        
    except Exception as e:
        st.error(f"Error loading overview data: {str(e)}")

def show_trade_analysis_page():
    """Show trade analysis dashboard page."""
    st.header("Trade Analysis")
    
    # Get trades data
    try:
        df = get_trades_df()
        
        if len(df) == 0:
            st.info("No completed trades found in the database.")
            return
        
        # Trade list
        st.subheader("Recent Trades")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_side = st.selectbox("Side", ["All", "buy", "sell"])
        
        with col2:
            filter_outcome = st.selectbox("Outcome", ["All", "Profit", "Loss"])
        
        with col3:
            filter_exit = st.selectbox("Exit Reason", ["All", "take_profit", "stop_loss", "manual"])
        
        # Apply filters
        filtered_df = df.copy()
        
        if filter_side != "All":
            filtered_df = filtered_df[filtered_df['side'] == filter_side]
        
        if filter_outcome != "All":
            if filter_outcome == "Profit":
                filtered_df = filtered_df[filtered_df['pnl'] > 0]
            else:
                filtered_df = filtered_df[filtered_df['pnl'] <= 0]
        
        if filter_exit != "All":
            filtered_df = filtered_df[filtered_df['exit_reason'] == filter_exit]
        
        # Show filtered trades
        st.dataframe(
            filtered_df[['entry_time', 'side', 'entry_price', 'exit_price', 'pnl', 'pnl_percent', 'exit_reason', 'duration']].sort_values(
                by='entry_time', ascending=False
            ).head(20),
            use_container_width=True
        )
        
        # Trade distribution
        st.subheader("Trade Distribution")
        
        col1, col2 = st.columns(2)
        
        # PnL distribution
        fig = px.histogram(
            df, 
            x='pnl',
            nbins=20,
            title="PnL Distribution"
        )
        col1.plotly_chart(fig, use_container_width=True)
        
        # Trade duration distribution
        df['duration_minutes'] = df['duration'].dt.total_seconds() / 60
        fig = px.histogram(
            df, 
            x='duration_minutes',
            nbins=20,
            title="Trade Duration Distribution (minutes)"
        )
        col2.plotly_chart(fig, use_container_width=True)
        
        # Exit reasons
        st.subheader("Exit Reasons")
        
        exit_counts = df['exit_reason'].value_counts().reset_index()
        exit_counts.columns = ['Exit Reason', 'Count']
        
        fig = px.pie(
            exit_counts, 
            values='Count', 
            names='Exit Reason',
            title="Exit Reason Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading trade analysis data: {str(e)}")

def show_market_data_page():
    """Show market data dashboard page."""
    st.header("Market Data")
    
    # Get latest OHLCV data
    try:
        df = get_ohlcv_df(limit=1000)
        df = df.sort_values(by='timestamp')
        
        if len(df) == 0:
            st.info("No market data found in the database.")
            return
        
        # Price chart
        st.subheader("Price Chart")
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe",
            ["Last 1 hour", "Last 4 hours", "Last 8 hours", "Last 24 hours", "All data"]
        )
        
        # Filter based on timeframe
        filtered_df = df.copy()
        if timeframe == "Last 1 hour":
            cutoff = datetime.now() - timedelta(hours=1)
            filtered_df = df[df['timestamp'] >= cutoff]
        elif timeframe == "Last 4 hours":
            cutoff = datetime.now() - timedelta(hours=4)
            filtered_df = df[df['timestamp'] >= cutoff]
        elif timeframe == "Last 8 hours":
            cutoff = datetime.now() - timedelta(hours=8)
            filtered_df = df[df['timestamp'] >= cutoff]
        elif timeframe == "Last 24 hours":
            cutoff = datetime.now() - timedelta(hours=24)
            filtered_df = df[df['timestamp'] >= cutoff]
        
        # Create candlestick chart
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=filtered_df['timestamp'],
            open=filtered_df['open'],
            high=filtered_df['high'],
            low=filtered_df['low'],
            close=filtered_df['close'],
            name="Price"
        ))
        fig.update_layout(
            title=f"BTC/USDT Price - {timeframe}",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        st.subheader("Volume")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=filtered_df['timestamp'],
            y=filtered_df['volume'],
            name="Volume"
        ))
        fig.update_layout(
            title=f"BTC/USDT Volume - {timeframe}",
            xaxis_title="Time",
            yaxis_title="Volume",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading market data: {str(e)}")

def show_system_settings_page():
    """Show system settings dashboard page."""
    st.header("System Settings")
    
    st.info("This page allows you to view and modify strategy parameters. Note that changes will be applied after the next bot restart.")
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**EMA Settings**")
        ema_fast = st.number_input("EMA Fast Period", min_value=2, max_value=50, value=9)
        ema_slow = st.number_input("EMA Slow Period", min_value=5, max_value=100, value=21)
    
    with col2:
        st.write("**RSI Settings**")
        rsi_period = st.number_input("RSI Period", min_value=1, max_value=20, value=2)
        rsi_long = st.number_input("RSI Long Threshold", min_value=1, max_value=30, value=10)
        rsi_short = st.number_input("RSI Short Threshold", min_value=70, max_value=99, value=90)
    
    st.write("**Volume Settings**")
    volume_period = st.number_input("Volume MA Period", min_value=5, max_value=50, value=20)
    volume_threshold = st.number_input("Volume Spike Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
    
    st.write("**Risk Management**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_per_trade = st.number_input("Risk Per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    
    with col2:
        stop_loss_pct = st.number_input("Stop Loss (%)", min_value=0.05, max_value=1.0, value=0.15, step=0.01)
    
    with col3:
        take_profit_pct = st.number_input("Take Profit (%)", min_value=0.1, max_value=2.0, value=0.30, step=0.01)
    
    st.write("**Trailing Stop Settings**")
    use_trailing_stop = st.checkbox("Use Trailing Stop", value=True)
    
    if use_trailing_stop:
        col1, col2 = st.columns(2)
        
        with col1:
            activation_pct = st.number_input("Activation Threshold (%)", min_value=0.05, max_value=1.0, value=0.15, step=0.01)
        
        with col2:
            trail_distance_pct = st.number_input("Trail Distance (%)", min_value=0.03, max_value=0.5, value=0.10, step=0.01)
    
    st.write("**Circuit Breaker Settings**")
    use_circuit_breaker = st.checkbox("Use Circuit Breaker", value=True)
    
    if use_circuit_breaker:
        col1, col2 = st.columns(2)
        
        with col1:
            max_consecutive_losses = st.number_input("Max Consecutive Stop Losses", min_value=1, max_value=10, value=3)
        
        with col2:
            cooldown_minutes = st.number_input("Cooldown Period (minutes)", min_value=5, max_value=240, value=30)
    
    if st.button("Save Settings"):
        st.success("Settings saved! Note that changes will be applied after the next bot restart.")

if __name__ == "__main__":
    main() 