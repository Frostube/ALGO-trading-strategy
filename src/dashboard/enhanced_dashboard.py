#!/usr/bin/env python3
"""
Enhanced backtesting dashboard for BTC/USDT scalping strategy.
Combines features from our current implementation with additional visualization options.
"""
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

def show_enhanced_dashboard():
    """Display the enhanced interactive backtesting dashboard."""
    st.set_page_config(
        page_title="Enhanced Backtest Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    st.title("📊 BTC/USDT Scalping Strategy - Enhanced Dashboard")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Connection settings
    st.sidebar.subheader("Connection Settings")
    use_testnet = st.sidebar.checkbox("Use Binance Testnet", value=True)
    
    if use_testnet:
        st.sidebar.info("⚠️ Using Binance Testnet - If testnet data is unavailable, mock data will be generated for testing.")
    
    # Data parameters
    st.sidebar.subheader("Data Settings")
    days = st.sidebar.slider("Days of Historical Data", min_value=7, max_value=90, value=30, step=1)
    timeframe = st.sidebar.selectbox("Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h"], index=1)
    
    # Strategy parameters section with collapsible UI
    with st.sidebar.expander("Strategy Settings", expanded=True):
        # EMA parameters
        st.subheader("EMA Parameters")
        ema_col1, ema_col2 = st.columns(2)
        ema_fast = ema_col1.number_input("Fast Period", min_value=3, max_value=50, value=EMA_FAST)
        ema_slow = ema_col2.number_input("Slow Period", min_value=5, max_value=200, value=EMA_SLOW)
        
        # RSI parameters
        st.subheader("RSI Parameters")
        rsi_col1, rsi_col2, rsi_col3 = st.columns(3)
        rsi_period = rsi_col1.number_input("Period", min_value=2, max_value=14, value=RSI_PERIOD)
        rsi_long = rsi_col2.number_input("Long Threshold", min_value=1, max_value=40, value=RSI_LONG_THRESHOLD)
        rsi_short = rsi_col3.number_input("Short Threshold", min_value=60, max_value=99, value=RSI_SHORT_THRESHOLD)
        
        # Volume parameters
        st.subheader("Volume Parameters")
        vol_col1, vol_col2 = st.columns(2)
        volume_period = vol_col1.number_input("Period", min_value=5, max_value=50, value=VOLUME_PERIOD)
        volume_threshold = vol_col2.number_input("Threshold", min_value=1.1, max_value=3.0, value=VOLUME_THRESHOLD, format="%.1f")
    
    # Risk parameters section
    with st.sidebar.expander("Risk Management", expanded=True):
        st.subheader("Stop Loss & Take Profit")
        risk_col1, risk_col2 = st.columns(2)
        stop_loss = risk_col1.number_input("Stop Loss %", min_value=0.1, max_value=2.0, value=STOP_LOSS_PCT*100, format="%.1f") / 100
        take_profit = risk_col2.number_input("Take Profit %", min_value=0.1, max_value=5.0, value=TAKE_PROFIT_PCT*100, format="%.1f") / 100
        
        st.subheader("ATR Parameters")
        atr_col1, atr_col2, atr_col3 = st.columns(3)
        atr_period = atr_col1.number_input("Period", min_value=5, max_value=20, value=ATR_PERIOD)
        atr_sl = atr_col2.number_input("SL Multiplier", min_value=0.5, max_value=3.0, value=ATR_SL_MULTIPLIER, format="%.1f")
        atr_tp = atr_col3.number_input("TP Multiplier", min_value=0.5, max_value=5.0, value=ATR_TP_MULTIPLIER, format="%.1f")
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings", expanded=False):
        use_train_test = st.checkbox("Split into Train/Test", value=True)
        
        if use_train_test:
            train_size = st.slider("Training Set Size", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
        
        initial_balance = st.number_input("Initial Balance ($)", min_value=100, max_value=100000, value=10000, step=1000)
    
    # Visualization options section
    with st.sidebar.expander("Visualization Options", expanded=False):
        show_candles = st.checkbox("Show Candlesticks", value=True)
        show_volume = st.checkbox("Show Volume", value=True)
        show_trades = st.checkbox("Show Trades", value=True)
        show_indicators = st.checkbox("Show Indicators", value=True)
        
        if show_indicators:
            indicator_options = st.multiselect(
                "Select Indicators",
                ["EMA Fast", "EMA Slow", "RSI", "Volume Trend", "ATR"],
                default=["EMA Fast", "EMA Slow", "RSI"]
            )
    
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
            display_enhanced_results(results, use_train_test, backtester, params, 
                                  show_candles, show_volume, show_trades, show_indicators, 
                                  indicator_options if show_indicators else [])
            
        except Exception as e:
            status_msg.error(f"Error in backtest: {str(e)}")
            st.exception(e)
    
    # If no backtest has been run yet
    else:
        st.info("👈 Adjust the parameters in the sidebar and click 'Run Backtest' to start.")
        
        # Show the introduction with columns layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enhanced Backtesting Dashboard")
            st.markdown("""
            This dashboard provides advanced visualization and analysis tools for your BTC/USDT trading strategy.
            
            **Key Features:**
            - Interactive parameter adjustment
            - Detailed performance metrics
            - Advanced chart visualization
            - Trade distribution analysis
            - Risk/reward analytics
            
            Adjust the parameters in the sidebar and run a backtest to see detailed performance analysis.
            """)
        
        with col2:
            st.image("https://www.pngitem.com/pimgs/m/547-5479644_algorithmic-trading-strategy-png-download-algorithmic-trading-logo.png", 
                    width=200, caption="Algorithmic Trading")

def display_enhanced_results(results, is_split, backtester, params, 
                          show_candles, show_volume, show_trades, show_indicators, indicator_options):
    """
    Display enhanced backtest results with advanced visualization.
    
    Args:
        results: Dict containing backtest results
        is_split: Boolean indicating if results are split into train/test
        backtester: Backtester instance with equity curve data
        params: Strategy parameters used for the backtest
        show_candles: Boolean to show candlestick chart
        show_volume: Boolean to show volume
        show_trades: Boolean to show trade markers
        show_indicators: Boolean to show technical indicators
        indicator_options: List of selected indicators to display
    """
    if is_split:
        # Create tabs for Train/Test results
        train_tab, test_tab, combined_tab = st.tabs(["Training Set", "Testing Set", "Combined Analysis"])
        
        # Get the equity curve and split it if train/test specific curves aren't available
        equity_curve = pd.DataFrame(backtester.equity_curve)
        
        # Calculate the split point based on the data size
        if hasattr(backtester, 'train_equity_curve') and hasattr(backtester, 'test_equity_curve'):
            # Use the backtester's own split curves if available
            train_equity = backtester.train_equity_curve
            test_equity = backtester.test_equity_curve
        else:
            # Manually split the equity curve
            split_idx = int(len(equity_curve) * 0.7)  # Default 70% split
            train_equity = equity_curve.iloc[:split_idx]
            test_equity = equity_curve.iloc[split_idx:]
        
        with train_tab:
            display_result_set(results['train'], "Training Set Results", 
                            train_equity, params,
                            show_candles, show_volume, show_trades, show_indicators, indicator_options)
        
        with test_tab:
            display_result_set(results['test'], "Testing Set Results (Out-of-Sample)", 
                            test_equity, params,
                            show_candles, show_volume, show_trades, show_indicators, indicator_options)
        
        with combined_tab:
            display_combined_analysis(results, backtester)
    else:
        # Single result set
        display_result_set(results, "Backtest Results", 
                        pd.DataFrame(backtester.equity_curve), params,
                        show_candles, show_volume, show_trades, show_indicators, indicator_options)

def display_result_set(result, title, equity_data, params, 
                    show_candles, show_volume, show_trades, show_indicators, indicator_options):
    """
    Display a single result set with enhanced visualization.
    
    Args:
        result: Dict containing backtest results for this set
        title: Title for this result set
        equity_data: Equity curve data
        params: Strategy parameters
        show_candles: Boolean to show candlestick chart
        show_volume: Boolean to show volume
        show_trades: Boolean to show trade markers
        show_indicators: Boolean to show technical indicators
        indicator_options: List of selected indicators to display
    """
    st.header(title)
    
    # Summary metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    # Helper function to safely get values with defaults
    def safe_get(key, default=0.0, format_as_pct=False):
        """Safely get a value from result dict with default"""
        value = result.get(key, default)
        if format_as_pct:
            return f"{value*100:.2f}%" if value is not None else f"{default*100:.2f}%"
        return f"{value:.2f}" if value is not None else f"{default:.2f}"
    
    with col1:
        st.metric("Total Return", safe_get('total_return', format_as_pct=True))
        st.metric("Sharpe Ratio", safe_get('sharpe_ratio'))
    
    with col2:
        st.metric("Win Rate", safe_get('win_rate', format_as_pct=True))
        st.metric("Profit Factor", safe_get('profit_factor', default=1.0))
    
    with col3:
        st.metric("Max Drawdown", safe_get('max_drawdown', format_as_pct=True))
        st.metric("Recovery Factor", safe_get('recovery_factor', default=0.0))
    
    with col4:
        st.metric("Total Trades", str(result.get('total_trades', 0)))
        # Handle both 'avg_trade' and 'avg_trade_pnl' keys for compatibility
        if 'avg_trade' in result:
            avg_trade = f"{result['avg_trade']*100:.2f}%"
        elif 'avg_trade_pnl' in result:
            avg_trade = f"${result['avg_trade_pnl']:.2f}"
        else:
            avg_trade = "0.00%"
        st.metric("Avg Trade", avg_trade)
    
    # Create enhanced visualization
    create_enhanced_charts(equity_data, result, show_candles, show_volume, show_trades, show_indicators, indicator_options)
    
    # Trade analysis section
    st.subheader("Trade Analysis")
    
    # Show trade distribution
    # Check for both 'trade_returns' and 'returns' keys for compatibility
    trade_returns = result.get('trade_returns', result.get('returns', []))
    if trade_returns and len(trade_returns) > 0:
        trade_returns = [r * 100 for r in trade_returns]  # Convert to percentage
        
        fig = px.histogram(
            trade_returns, 
            nbins=20,
            title="Trade Return Distribution (%)",
            labels={"value": "Return %"},
            color_discrete_sequence=['#2E86C1'],
            marginal="box"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Trade details table
    if 'trades' in result and len(result['trades']) > 0:
        st.subheader("Trade Details")
        
        try:
            # Convert trades list to DataFrame
            trades_df = pd.DataFrame(result['trades'])
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
            
            # Calculate duration if not already provided
            if 'duration' not in trades_df.columns:
                trades_df['duration'] = trades_df['exit_time'] - trades_df['entry_time']
                
            # Add return percentage if not already provided
            if 'return_pct' not in trades_df.columns and 'return' in trades_df.columns:
                trades_df['return_pct'] = trades_df['return'] * 100  # Convert to percentage
            
            # Create display columns based on available data
            display_cols = []
            for col in ['entry_time', 'exit_time', 'duration', 'type', 'entry_price', 'exit_price', 'return_pct']:
                if col in trades_df.columns:
                    display_cols.append(col)
            
            # Format DataFrame for display
            if display_cols:
                display_df = trades_df[display_cols]
                
                # Rename columns for display
                column_mapping = {
                    'entry_time': 'Entry Time',
                    'exit_time': 'Exit Time',
                    'duration': 'Duration',
                    'type': 'Type',
                    'entry_price': 'Entry Price',
                    'exit_price': 'Exit Price',
                    'return_pct': 'Return %'
                }
                
                # Only rename columns that exist
                rename_map = {k: v for k, v in column_mapping.items() if k in display_cols}
                display_df = display_df.rename(columns=rename_map)
                
                st.dataframe(display_df, use_container_width=True, height=300)
            else:
                st.info("Trade details not available in the required format.")
        except Exception as e:
            st.error(f"Error displaying trade details: {str(e)}")

def create_enhanced_charts(equity_data, result, show_candles, show_volume, show_trades, show_indicators, indicator_options):
    """
    Create enhanced interactive charts for the backtest results.
    
    Args:
        equity_data: Equity curve data (DataFrame)
        result: Dict containing backtest results
        show_candles: Boolean to show candlestick chart
        show_volume: Boolean to show volume
        show_trades: Boolean to show trade markers
        show_indicators: Boolean to show technical indicators
        indicator_options: List of selected indicators to display
    """
    # Check if we have the necessary data
    if equity_data is None or equity_data.empty:
        st.warning("No equity curve data available for visualization.")
        return
    
    # Check if price data is available
    if 'price_data' not in result:
        st.warning("No price data available for visualization.")
        # Create a simple equity chart if at least that data is available
        if not equity_data.empty and 'equity' in equity_data.columns:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=equity_data.index,
                    y=equity_data['equity'],
                    mode='lines',
                    name='Portfolio Equity',
                    line=dict(color='#2471A3', width=2)
                )
            )
            fig.update_layout(
                title="Portfolio Equity",
                xaxis_title="Date",
                yaxis_title="Equity ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        return

    try:
        # Create subplots
        num_rows = 2
        if show_indicators and "RSI" in indicator_options:
            num_rows += 1
        
        # Row heights
        row_heights = [0.6, 0.2]
        if num_rows > 2:
            row_heights = [0.5, 0.2, 0.3]  # Adjust for RSI panel
        
        # Create subplot specs
        specs = [[{"secondary_y": True}], [{}]]
        if num_rows > 2:
            specs.append([{}])
        
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            specs=specs
        )
        
        # Price data - ensure it's a DataFrame
        price_data = result['price_data']
        
        # Add candlestick chart if OHLC data available
        required_cols = ['open', 'high', 'low', 'close']
        has_ohlc = all(col in price_data.columns for col in required_cols)
        
        if show_candles and has_ohlc:
            fig.add_trace(
                go.Candlestick(
                    x=price_data.index,
                    open=price_data['open'],
                    high=price_data['high'],
                    low=price_data['low'],
                    close=price_data['close'],
                    name="OHLC"
                ),
                row=1, col=1
            )
        else:
            # Add line chart if candlesticks not shown or no OHLC data
            if 'close' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#2E86C1', width=1.5)
                    ),
                    row=1, col=1
                )
            else:
                st.warning("Close price data not available.")
        
        # Add EMA indicators if selected and available
        if show_indicators:
            if "EMA Fast" in indicator_options and 'ema_fast' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['ema_fast'],
                        mode='lines',
                        line=dict(color='#58D68D', width=1.5),
                        name=f'EMA Fast'
                    ),
                    row=1, col=1
                )
            
            if "EMA Slow" in indicator_options and 'ema_slow' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['ema_slow'],
                        mode='lines',
                        line=dict(color='#C0392B', width=1.5),
                        name=f'EMA Slow'
                    ),
                    row=1, col=1
                )
            
            # Add RSI in a separate row if selected and available
            if "RSI" in indicator_options and 'rsi' in price_data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=price_data.index,
                        y=price_data['rsi'],
                        mode='lines',
                        line=dict(color='#8E44AD', width=1.5),
                        name='RSI'
                    ),
                    row=3, col=1
                )
                
                # Add RSI thresholds
                fig.add_trace(
                    go.Scatter(
                        x=[price_data.index[0], price_data.index[-1]],
                        y=[30, 30],
                        mode='lines',
                        line=dict(color='#58D68D', width=1, dash='dash'),
                        name='RSI Oversold'
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=[price_data.index[0], price_data.index[-1]],
                        y=[70, 70],
                        mode='lines',
                        line=dict(color='#C0392B', width=1, dash='dash'),
                        name='RSI Overbought'
                    ),
                    row=3, col=1
                )
        
        # Add volume if selected and available
        if show_volume and 'volume' in price_data.columns:
            if 'open' in price_data.columns and 'close' in price_data.columns:
                colors = np.where(price_data['close'] >= price_data['open'], '#58D68D', '#C0392B')
            else:
                colors = '#58D68D'  # Default color if open/close not available
                
            fig.add_trace(
                go.Bar(
                    x=price_data.index,
                    y=price_data['volume'],
                    marker_color=colors,
                    name='Volume',
                    opacity=0.8
                ),
                row=1, col=1, secondary_y=True
            )
        
        # Add trade markers if selected and available
        if show_trades and 'trades' in result and len(result['trades']) > 0:
            trades = result['trades']
            
            # Check if trades have the required fields
            if all(key in trades[0] for key in ['type', 'entry_time', 'entry_price']):
                # Extract long and short trades
                long_entries = [t for t in trades if t['type'] == 'long']
                short_entries = [t for t in trades if t['type'] == 'short']
                
                # Add long trade entry markers
                if long_entries:
                    entry_times = [t['entry_time'] for t in long_entries]
                    entry_prices = [t['entry_price'] for t in long_entries]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=entry_times,
                            y=entry_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-up',
                                size=10,
                                color='#58D68D',
                                line=dict(width=2, color='#145A32')
                            ),
                            name='Long Entry',
                            hoverinfo='text',
                            text=[f"Long Entry: {p:.2f}" for p in entry_prices]
                        ),
                        row=1, col=1
                    )
                
                # Add short trade entry markers
                if short_entries:
                    entry_times = [t['entry_time'] for t in short_entries]
                    entry_prices = [t['entry_price'] for t in short_entries]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=entry_times,
                            y=entry_prices,
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='#C0392B',
                                line=dict(width=2, color='#641E16')
                            ),
                            name='Short Entry',
                            hoverinfo='text',
                            text=[f"Short Entry: {p:.2f}" for p in entry_prices]
                        ),
                        row=1, col=1
                    )
        
        # Add equity curve
        if 'equity' in equity_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_data.index,
                    y=equity_data['equity'],
                    mode='lines',
                    name='Portfolio Equity',
                    line=dict(color='#2471A3', width=2)
                ),
                row=2, col=1
            )
        
        # Add drawdown curve if available
        if 'drawdown' in equity_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity_data.index,
                    y=-equity_data['drawdown'] * 100,  # Negative to show downward
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='#C0392B', width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(192, 57, 43, 0.2)'
                ),
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Backtest Performance",
            xaxis_title="Date",
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode="x unified",
            template="plotly_white"
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Equity / Drawdown", row=2, col=1)
        
        if num_rows > 2:
            fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        # Show figure
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error creating charts: {str(e)}")
        st.exception(e)

def display_combined_analysis(results, backtester):
    """
    Display combined analysis of training and testing results.
    
    Args:
        results: Dict containing both training and testing results
        backtester: Backtester instance with all data
    """
    st.header("Training vs. Testing Analysis")
    
    # Validate that we have both train and test results
    if not isinstance(results, dict) or 'train' not in results or 'test' not in results:
        st.warning("Cannot display combined analysis: Missing train/test results data")
        return
    
    # Compare key metrics
    metrics = ['total_return', 'win_rate', 'sharpe_ratio', 'max_drawdown', 'profit_factor']
    metric_names = ['Total Return', 'Win Rate', 'Sharpe Ratio', 'Max Drawdown', 'Profit Factor']
    
    # Convert to the right format
    train_data = []
    test_data = []
    
    try:
        for metric, name in zip(metrics, metric_names):
            # Get values with defaults for missing keys
            train_val = results['train'].get(metric, 0.0)
            test_val = results['test'].get(metric, 0.0)
            
            # Handle percentages
            if metric in ['total_return', 'win_rate', 'max_drawdown']:
                train_data.append(train_val * 100 if train_val is not None else 0.0)
                test_data.append(test_val * 100 if test_val is not None else 0.0)
            else:
                train_data.append(train_val if train_val is not None else 0.0)
                test_data.append(test_val if test_val is not None else 0.0)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Metric': metric_names,
            'Training': train_data,
            'Testing': test_data
        })
        
        # Calculate consistency score
        train_return = results['train'].get('total_return', 0.0) * 100 if results['train'].get('total_return') is not None else 0.0
        test_return = results['test'].get('total_return', 0.0) * 100 if results['test'].get('total_return') is not None else 0.0
        train_win_rate = results['train'].get('win_rate', 0.0) * 100 if results['train'].get('win_rate') is not None else 0.0
        test_win_rate = results['test'].get('win_rate', 0.0) * 100 if results['test'].get('win_rate') is not None else 0.0
        
        # Avoid division by zero
        if test_return == 0 or test_win_rate == 0:
            consistency = 0.0
        else:
            return_ratio = min(train_return / max(0.1, test_return), 10)  # Cap at 10x difference
            win_rate_ratio = min(train_win_rate / max(0.1, test_win_rate), 10)  # Cap at 10x difference
            consistency = 10 - (return_ratio * 0.5 + win_rate_ratio * 0.5)  # Higher is better, scale 0-10
            consistency = max(0, min(10, consistency))  # Ensure 0-10 range
        
        # Create comparison bar chart
        fig = px.bar(
            comparison_df, 
            x='Metric', 
            y=['Training', 'Testing'],
            barmode='group',
            title="Key Metrics Comparison: Training vs Testing",
            color_discrete_sequence=['#3498DB', '#27AE60']
        )
        
        # Adjust y-axis range based on metric
        for i, metric in enumerate(metric_names):
            if metric == 'Max Drawdown':
                # Make drawdown bars negative for visual effect
                fig.data[0].y[i] = -comparison_df.loc[i, 'Training']
                fig.data[1].y[i] = -comparison_df.loc[i, 'Testing']
        
        fig.update_layout(
            xaxis_title="Metric",
            yaxis_title="Value",
            legend_title="Dataset",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add consistency score visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Strategy Consistency Score")
            
            # Create gauge chart for consistency
            consistency_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=consistency,
                title={'text': "Consistency Score"},
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "red"},
                        {'range': [3, 7], 'color': "orange"},
                        {'range': [7, 10], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': consistency
                    }
                }
            ))
            
            consistency_fig.update_layout(height=300)
            st.plotly_chart(consistency_fig, use_container_width=True)
        
        with col2:
            st.subheader("Interpretation")
            
            if consistency >= 7:
                st.success("""
                **High Consistency (7-10):** The strategy performs consistently across both training and testing datasets. 
                This suggests the strategy is robust and less likely to be overfitted.
                """)
            elif consistency >= 3:
                st.warning("""
                **Moderate Consistency (3-7):** There are some differences between training and testing performance.
                Some parameter adjustments may be needed for better generalization.
                """)
            else:
                st.error("""
                **Low Consistency (0-3):** Significant differences between training and testing performance.
                The strategy may be overfitted to the training data and needs fundamental changes.
                """)
            
            # Add specific recommendations
            st.markdown("### Recommendations")
            if train_return > 0 and test_return < 0:
                st.markdown("- Strategy performs well in training but loses money in testing - suggests overfitting")
            elif train_return > test_return * 2:
                st.markdown("- Returns in training are much higher than testing - consider more conservative parameters")
            
            if results['train'].get('sharpe_ratio', 0) > 1 and results['test'].get('sharpe_ratio', 0) < 0:
                st.markdown("- Sharpe ratio inconsistency indicates different risk characteristics in test period")
        
    except Exception as e:
        st.error(f"Error generating combined analysis: {str(e)}")
        st.exception(e)
    
    # Add note about interpretation
    st.info("""
    **Interpretation Guide:**
    - Similar performance between training and testing sets indicates a robust strategy
    - Large differences may suggest overfitting on the training data
    - Higher Sharpe ratio and profit factor values indicate better risk-adjusted returns
    - Lower max drawdown values are better (shown as negative bars)
    """)

if __name__ == "__main__":
    show_enhanced_dashboard() 