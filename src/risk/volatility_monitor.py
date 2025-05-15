#!/usr/bin/env python3
"""
Volatility Monitor Module

Monitors and classifies market volatility regimes for better risk management.
Provides tools to detect changes in market conditions and adjust strategy parameters accordingly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

from src.data.fetcher import fetch_ohlcv
from src.utils.logger import logger

# Volatility regime thresholds
VOL_CALM = 0.03  # 3% realized volatility threshold for calm markets
VOL_STORM = 0.06  # 6% realized volatility threshold for volatile markets

class VolatilityMonitor:
    """
    Monitors market volatility and classifies regimes for trading adaptation.
    """
    
    def __init__(self, symbols=None, timeframe="1d", lookback=30, cache_dir="data/volatility"):
        """
        Initialize the volatility monitor.
        
        Args:
            symbols: List of symbols to monitor
            timeframe: Data timeframe for volatility calculation
            lookback: Lookback period in days for volatility calculation
            cache_dir: Directory to cache volatility data
        """
        self.symbols = symbols or []
        self.timeframe = timeframe
        self.lookback = lookback
        self.cache_dir = cache_dir
        self.regime_data = {}
        
        # Ensure cache directory exists
        self._ensure_cache_dir()
        
        # Load cached data if available
        self._load_cached_data()
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        cache_path = Path(self.cache_dir)
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self):
        """Get the path to the cache file"""
        return os.path.join(self.cache_dir, f"volatility_regimes_{self.timeframe}_{self.lookback}d.json")
    
    def _load_cached_data(self):
        """Load cached volatility data if available"""
        cache_path = self._get_cache_path()
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                
                # Only use cached data if it's fresh (less than 4 hours old)
                if "last_update" in cached_data:
                    last_update = datetime.fromisoformat(cached_data["last_update"])
                    if datetime.now() - last_update < timedelta(hours=4):
                        self.regime_data = cached_data.get("regimes", {})
                        logger.info(f"Loaded cached volatility regimes for {len(self.regime_data)} symbols")
                        return
        except Exception as e:
            logger.error(f"Error loading cached volatility data: {str(e)}")
        
        # If no valid cache is available, initialize empty data
        self.regime_data = {}
    
    def _save_cached_data(self):
        """Save current volatility data to cache"""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "regimes": self.regime_data,
                    "last_update": datetime.now().isoformat()
                }, f)
            logger.info(f"Saved volatility regimes for {len(self.regime_data)} symbols to cache")
        except Exception as e:
            logger.error(f"Error saving volatility data to cache: {str(e)}")
    
    def calculate_realized_volatility(self, symbol, update_cache=True):
        """
        Calculate realized volatility for a symbol.
        
        Args:
            symbol: Trading pair symbol
            update_cache: Whether to update the cache
            
        Returns:
            float: Realized volatility as decimal (e.g., 0.05 = 5%)
        """
        try:
            # Fetch data
            df = fetch_ohlcv(symbol, self.timeframe, days=self.lookback+5)  # Add buffer days
            
            if df is None or len(df) < self.lookback:
                logger.warning(f"Insufficient data for volatility calculation: {symbol}")
                return 0.05  # Default to 5% if data insufficient
            
            # Calculate daily returns
            df['return'] = df['close'].pct_change()
            
            # Calculate annualized volatility
            daily_vol = df['return'].std()
            
            # Convert to period volatility based on timeframe
            if self.timeframe == "1d":
                period_vol = daily_vol
            elif self.timeframe == "4h":
                period_vol = daily_vol * np.sqrt(6)  # 6 4-hour periods in a day
            elif self.timeframe == "1h":
                period_vol = daily_vol * np.sqrt(24)  # 24 hours in a day
            else:
                # Default to daily
                period_vol = daily_vol
            
            # Update regime data if requested
            if update_cache:
                self._update_regime_data(symbol, period_vol)
            
            return period_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            return 0.05  # Default to 5% if calculation fails
    
    def _update_regime_data(self, symbol, volatility):
        """
        Update regime data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            volatility: Calculated volatility value
        """
        # Determine regime
        if volatility < VOL_CALM:
            regime = "calm"
        elif volatility > VOL_STORM:
            regime = "storm"
        else:
            regime = "normal"
        
        # Update regime data
        self.regime_data[symbol] = {
            "volatility": float(volatility),
            "regime": regime,
            "updated_at": datetime.now().isoformat()
        }
        
        # Save to cache
        self._save_cached_data()
    
    def get_regime(self, symbol):
        """
        Get the current volatility regime for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            str: Market regime ('calm', 'normal', or 'storm')
        """
        # Check if we have fresh data for this symbol (less than 4 hours old)
        if symbol in self.regime_data:
            last_update = datetime.fromisoformat(self.regime_data[symbol]["updated_at"])
            if datetime.now() - last_update < timedelta(hours=4):
                return self.regime_data[symbol]["regime"]
        
        # Calculate new volatility if data is stale or missing
        vol = self.calculate_realized_volatility(symbol)
        
        # Return the new regime
        return self.regime_data[symbol]["regime"]
    
    def update_all_symbols(self):
        """
        Update volatility data for all tracked symbols.
        
        Returns:
            dict: Updated regime data
        """
        for symbol in self.symbols:
            self.calculate_realized_volatility(symbol)
        
        return self.regime_data
    
    def add_symbol(self, symbol):
        """
        Add a symbol to the monitor.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.calculate_realized_volatility(symbol)
    
    def remove_symbol(self, symbol):
        """
        Remove a symbol from the monitor.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            if symbol in self.regime_data:
                del self.regime_data[symbol]
    
    def get_all_regimes(self):
        """
        Get regimes for all tracked symbols.
        
        Returns:
            dict: Regime data for all symbols
        """
        return self.regime_data
    
    def plot_volatility_history(self, symbol, days=90, save_path=None):
        """
        Plot historical volatility with regime bands.
        
        Args:
            symbol: Trading pair symbol
            days: Number of days to plot
            save_path: Path to save the plot (if None, display plot)
            
        Returns:
            matplotlib.figure.Figure: The created figure
        """
        try:
            # Fetch data
            df = fetch_ohlcv(symbol, self.timeframe, days=days+30)  # Add buffer for calculation
            
            if df is None or len(df) < 30:
                logger.warning(f"Insufficient data for volatility plotting: {symbol}")
                return None
            
            # Calculate rolling volatility
            df['return'] = df['close'].pct_change()
            df['vol_30d'] = df['return'].rolling(30).std()
            
            # Convert to percentage
            df['vol_pct'] = df['vol_30d'] * 100
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot volatility
            ax.plot(df.index[-days:], df['vol_pct'][-days:], 'b-', linewidth=2, label='30-day Realized Volatility')
            
            # Add regime bands
            ax.axhspan(0, VOL_CALM*100, color='green', alpha=0.2, label='Calm Regime')
            ax.axhspan(VOL_CALM*100, VOL_STORM*100, color='yellow', alpha=0.2, label='Normal Regime')
            ax.axhspan(VOL_STORM*100, 20, color='red', alpha=0.2, label='Storm Regime')
            
            # Add labels and title
            ax.set_title(f'{symbol} 30-Day Realized Volatility')
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility (%)')
            ax.legend()
            ax.grid(True)
            
            # Add horizontal lines at regime thresholds
            ax.axhline(y=VOL_CALM*100, color='green', linestyle='--')
            ax.axhline(y=VOL_STORM*100, color='red', linestyle='--')
            
            # Annotate current regime
            current_vol = df['vol_pct'].iloc[-1]
            current_regime = "Calm" if current_vol < VOL_CALM*100 else "Storm" if current_vol > VOL_STORM*100 else "Normal"
            ax.annotate(f'Current: {current_vol:.2f}% ({current_regime})', 
                        xy=(df.index[-1], current_vol),
                        xytext=(df.index[-int(days/10)], current_vol + 1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                        fontsize=12)
            
            # Save or display
            if save_path:
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Volatility plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting volatility for {symbol}: {str(e)}")
            return None


if __name__ == "__main__":
    """CLI for volatility monitoring"""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Monitor market volatility regimes")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT",
                        help="Comma-separated list of symbols to monitor")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe for volatility calculation")
    parser.add_argument("--lookback", type=int, default=30, help="Lookback period in days")
    parser.add_argument("--plot", action="store_true", help="Generate volatility plots")
    parser.add_argument("--plot-days", type=int, default=90, help="Days to include in plot")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create volatility monitor
    symbols = args.symbols.split(",")
    monitor = VolatilityMonitor(symbols=symbols, timeframe=args.timeframe, lookback=args.lookback)
    
    # Update all symbols
    monitor.update_all_symbols()
    
    # Print results
    print(f"\nVolatility Regimes ({args.timeframe}, {args.lookback}-day lookback):")
    print("-" * 60)
    print(f"{'Symbol':<12} {'Volatility':<12} {'Regime':<10} {'Updated'}")
    print("-" * 60)
    
    regimes = monitor.get_all_regimes()
    for symbol, data in regimes.items():
        vol_pct = data["volatility"] * 100
        regime = data["regime"].upper()
        updated = datetime.fromisoformat(data["updated_at"]).strftime("%Y-%m-%d %H:%M")
        print(f"{symbol:<12} {vol_pct:>6.2f}%      {regime:<10} {updated}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating volatility plots...")
        os.makedirs("reports/volatility", exist_ok=True)
        
        for symbol in symbols:
            save_path = f"reports/volatility/{symbol.replace('/', '_')}_vol_{args.timeframe}.png"
            monitor.plot_volatility_history(symbol, days=args.plot_days, save_path=save_path)
            print(f"  Plot saved: {save_path}")
    
    print("\nVol-Regime Switch implications:")
    print("-" * 60)
    for symbol, data in regimes.items():
        regime = data["regime"]
        vol_pct = data["volatility"] * 100
        
        if regime == "calm":
            print(f"{symbol:<12} CALM ({vol_pct:.2f}%) → halve position sizes, disable pyramiding")
        elif regime == "storm":
            print(f"{symbol:<12} STORM ({vol_pct:.2f}%) → full position sizes, enable pyramiding")
        else:
            print(f"{symbol:<12} NORMAL ({vol_pct:.2f}%) → standard position sizes") 