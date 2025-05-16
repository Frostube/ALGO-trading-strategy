import pandas as pd
import numpy as np
import logging
from src.strategy.pattern_filtered_strategy import PatternFilteredStrategy
from src.strategy.confirm import (
    add_rsi, add_macd, add_stochastic, add_vwap, add_bollinger_bands,
    is_rsi_aligned, is_macd_aligned, is_stoch_aligned,
    is_volume_spike, is_vwap_aligned, is_atr_favorable,
    is_bb_squeeze, is_near_pivot, is_momentum_confirmed
)

logger = logging.getLogger(__name__)

class EnhancedConfirmationStrategy(PatternFilteredStrategy):
    """
    Enhanced trading strategy that combines all previous filters with additional
    confirmation tools:
    
    1. Momentum Oscillators (RSI, MACD, Stochastic)
    2. Volume Filters (Volume spike, VWAP alignment)
    3. Volatility Regime Checks (ATR Percentile, Bollinger Band squeeze)
    4. Support/Resistance Pivot Zones
    
    This strategy represents the most comprehensive filtering approach, using
    multiple layers of confirmation to ensure only the highest-probability
    trades are taken.
    """
    
    def __init__(self, 
                 # Base parameters
                 fast_ema=3, slow_ema=15, trend_ema=50, atr_period=14, atr_multiplier=2.0,
                 lookback_window=20, vol_window=14, use_trend_filter=True,
                 regime_params_file=None, daily_ema_period=200, 
                 enforce_trend_alignment=True, vol_threshold_percentile=80,
                 require_pattern_confirmation=True, doji_threshold=0.1,
                 # New confirmation parameters
                 use_momentum_filter=True, use_volume_filter=True,
                 use_volatility_filter=True, use_pivot_filter=True,
                 rsi_period=14, rsi_threshold=50,
                 volume_lookback=20, volume_factor=1.5,
                 atr_percentile=80, bb_squeeze_lookback=20,
                 pivot_lookback=20, pivot_proximity=1.0,
                 min_confirmations=3,
                 # Filter weights (new parameter)
                 filter_weights=None,
                 # Early exit diagnostics
                 track_exit_conditions=True):
        """
        Initialize the enhanced confirmation strategy.
        
        Args:
            fast_ema, slow_ema, etc.: Parameters inherited from parent strategies
            use_momentum_filter: Whether to use momentum oscillator confirmations
            use_volume_filter: Whether to use volume-based confirmations
            use_volatility_filter: Whether to use volatility-based confirmations
            use_pivot_filter: Whether to check for pivots before trading
            rsi_period: Period for RSI calculation
            rsi_threshold: RSI threshold for trend alignment
            volume_lookback: Periods for volume average calculation
            volume_factor: Factor for volume spike detection
            atr_percentile: Maximum percentile for acceptable ATR
            bb_squeeze_lookback: Periods for BB squeeze detection
            pivot_lookback: Periods for pivot point detection
            pivot_proximity: Percentage distance for pivot proximity
            min_confirmations: Minimum number of confirmation filters required
            filter_weights: Dictionary of weights for each filter type
            track_exit_conditions: Whether to track which filters would have signaled exits
        """
        super().__init__(
            fast_ema, slow_ema, trend_ema, atr_period, atr_multiplier,
            lookback_window, vol_window, use_trend_filter,
            regime_params_file, daily_ema_period, 
            enforce_trend_alignment, vol_threshold_percentile,
            require_pattern_confirmation, doji_threshold
        )
        
        # Store confirmation parameters
        self.use_momentum_filter = use_momentum_filter
        self.use_volume_filter = use_volume_filter
        self.use_volatility_filter = use_volatility_filter
        self.use_pivot_filter = use_pivot_filter
        
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.volume_lookback = volume_lookback
        self.volume_factor = volume_factor
        self.atr_percentile = atr_percentile
        self.bb_squeeze_lookback = bb_squeeze_lookback
        self.pivot_lookback = pivot_lookback
        self.pivot_proximity = pivot_proximity
        self.min_confirmations = min_confirmations
        self.track_exit_conditions = track_exit_conditions
        
        # Set up filter weights (default weights if none provided)
        self.filter_weights = filter_weights or {
            'momentum': 2.0,   # Higher weight for momentum
            'volume': 1.0,     # Standard weight for volume
            'volatility': 1.5, # Medium weight for volatility
            'pivot': 2.5       # Highest weight for pivot zones
        }
        
        # Track confirmation statistics
        self.confirmation_stats = {
            'total_signals': 0,
            'momentum_confirmed': 0,
            'momentum_rejected': 0,
            'volume_confirmed': 0,
            'volume_rejected': 0,
            'volatility_confirmed': 0,
            'volatility_rejected': 0,
            'pivot_confirmed': 0,
            'pivot_rejected': 0,
            'signals_passed': 0,
            'signals_rejected': 0,
            'weighted_scores': [],  # Track weighted scores for analysis
            
            # For exit diagnostics
            'exit_signals': {
                'momentum_triggered': 0,
                'volume_triggered': 0, 
                'volatility_triggered': 0,
                'pivot_triggered': 0,
                'total_exits': 0,
                'avg_exit_score': 0.0,
                'profitable_exits': 0,
                'unprofitable_exits': 0
            }
        }
        
        # Track active trades for exit diagnostics
        self.active_trades = {}
        
        # Initialize daily data holder for MTF analysis
        self.daily_df = None
    
    def _calculate_indicators(self, df):
        """
        Calculate all required indicators for confirmations.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with indicators added
        """
        # Calculate indicators based on which filters are enabled
        if self.use_momentum_filter:
            df = add_rsi(df, period=self.rsi_period)
            df = add_macd(df)
            df = add_stochastic(df)
            
        if self.use_volume_filter:
            df = add_vwap(df)
            
        if self.use_volatility_filter:
            df = add_bollinger_bands(df)
            
        return df
    
    def _check_momentum_confirmation(self, df, signal_type):
        """
        Check if momentum indicators confirm the signal.
        
        Args:
            df: DataFrame with indicators calculated
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if momentum confirms the signal
        """
        if not self.use_momentum_filter:
            return True
            
        # Check RSI alignment
        rsi_confirmed = is_rsi_aligned(df, signal_type, threshold=self.rsi_threshold)
        
        # Check MACD alignment (optional - less strict)
        macd_confirmed = is_macd_aligned(df, signal_type) if 'macd' in df.columns else True
        
        # For this strategy, we'll require RSI confirmation at minimum
        is_confirmed = rsi_confirmed
        
        self.confirmation_stats['total_signals'] += 1
        if is_confirmed:
            self.confirmation_stats['momentum_confirmed'] += 1
        else:
            self.confirmation_stats['momentum_rejected'] += 1
            
        logger.debug(f"Momentum confirmation: {is_confirmed} (RSI: {rsi_confirmed}, MACD: {macd_confirmed})")
        return is_confirmed
    
    def _check_volume_confirmation(self, df, signal_type):
        """
        Check if volume indicators confirm the signal.
        
        Args:
            df: DataFrame with indicators calculated
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if volume confirms the signal
        """
        if not self.use_volume_filter:
            return True
            
        # Check for volume spike
        volume_spike = is_volume_spike(df, lookback=self.volume_lookback, factor=self.volume_factor)
        
        # Check VWAP alignment
        vwap_aligned = is_vwap_aligned(df, signal_type) if 'vwap' in df.columns else True
        
        # For this strategy, we'll accept either volume confirmation
        is_confirmed = volume_spike or vwap_aligned
        
        if is_confirmed:
            self.confirmation_stats['volume_confirmed'] += 1
        else:
            self.confirmation_stats['volume_rejected'] += 1
            
        logger.debug(f"Volume confirmation: {is_confirmed} (Spike: {volume_spike}, VWAP: {vwap_aligned})")
        return is_confirmed
    
    def _check_volatility_confirmation(self, df, signal_type):
        """
        Check if volatility indicators confirm the signal.
        
        Args:
            df: DataFrame with indicators calculated
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if volatility conditions are favorable
        """
        if not self.use_volatility_filter:
            return True
            
        # Check if ATR is at acceptable levels
        atr_favorable = is_atr_favorable(df, period=self.atr_period, percentile=self.atr_percentile)
        
        # Check for Bollinger Band squeeze (optional)
        bb_squeeze = is_bb_squeeze(df, lookback=self.bb_squeeze_lookback) if 'bb_width' in df.columns else False
        
        # For this strategy, we primarily care about ATR levels
        is_confirmed = atr_favorable
        
        if is_confirmed:
            self.confirmation_stats['volatility_confirmed'] += 1
        else:
            self.confirmation_stats['volatility_rejected'] += 1
            
        logger.debug(f"Volatility confirmation: {is_confirmed} (ATR: {atr_favorable}, BB Squeeze: {bb_squeeze})")
        return is_confirmed
    
    def _check_pivot_confirmation(self, df, signal_type):
        """
        Check if the trade is not near a pivot point.
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if not near a pivot point (favorable for trading)
        """
        if not self.use_pivot_filter:
            return True
            
        # Check if price is near a pivot point (we want it NOT to be near pivot)
        near_pivot = is_near_pivot(df, lookback=self.pivot_lookback, proximity_pct=self.pivot_proximity)
        is_confirmed = not near_pivot
        
        if is_confirmed:
            self.confirmation_stats['pivot_confirmed'] += 1
        else:
            self.confirmation_stats['pivot_rejected'] += 1
            
        logger.debug(f"Pivot confirmation: {is_confirmed} (Near pivot: {near_pivot})")
        return is_confirmed
    
    def check_enhanced_confirmations(self, df, signal_type):
        """
        Run all confirmation checks and use weighted scoring instead of simple counting.
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 'buy' or 'sell'
            
        Returns:
            bool: True if sufficient weighted score is achieved
        """
        # Calculate all required indicators
        df_with_indicators = self._calculate_indicators(df.copy())
        
        # Initialize weighted score
        weighted_score = 0.0
        max_possible_score = 0.0
        confirmations = {}
        
        # 1. Momentum check
        if self.use_momentum_filter:
            momentum_confirmed = self._check_momentum_confirmation(df_with_indicators, signal_type)
            confirmations['momentum'] = momentum_confirmed
            weighted_score += momentum_confirmed * self.filter_weights['momentum']
            max_possible_score += self.filter_weights['momentum']
        
        # 2. Volume check
        if self.use_volume_filter:
            volume_confirmed = self._check_volume_confirmation(df_with_indicators, signal_type)
            confirmations['volume'] = volume_confirmed
            weighted_score += volume_confirmed * self.filter_weights['volume'] 
            max_possible_score += self.filter_weights['volume']
        
        # 3. Volatility check
        if self.use_volatility_filter:
            volatility_confirmed = self._check_volatility_confirmation(df_with_indicators, signal_type)
            confirmations['volatility'] = volatility_confirmed
            weighted_score += volatility_confirmed * self.filter_weights['volatility']
            max_possible_score += self.filter_weights['volatility']
        
        # 4. Pivot check
        if self.use_pivot_filter:
            pivot_confirmed = self._check_pivot_confirmation(df_with_indicators, signal_type)
            confirmations['pivot'] = pivot_confirmed
            weighted_score += pivot_confirmed * self.filter_weights['pivot']
            max_possible_score += self.filter_weights['pivot']
        
        # Calculate normalized score (0-1 range)
        normalized_score = weighted_score / max_possible_score if max_possible_score > 0 else 0
        
        # Store score for analysis
        if 'weighted_scores' not in self.confirmation_stats:
            self.confirmation_stats['weighted_scores'] = []
        elif not isinstance(self.confirmation_stats['weighted_scores'], list):
            self.confirmation_stats['weighted_scores'] = []
        
        self.confirmation_stats['weighted_scores'].append(normalized_score)
        
        # Calculate threshold score
        if isinstance(self.min_confirmations, float) and self.min_confirmations <= 1.0:
            # Use min_confirmations directly as a threshold percentage
            threshold_score = self.min_confirmations
        else:
            # Convert absolute count to a percentage of max possible
            active_filters = sum([
                self.use_momentum_filter,
                self.use_volume_filter,
                self.use_volatility_filter,
                self.use_pivot_filter
            ])
            required_confirmations = min(int(self.min_confirmations), active_filters)
            threshold_score = required_confirmations / active_filters if active_filters > 0 else 0
        
        # Determine if we have enough confirmation weight
        has_enough_confirmations = normalized_score >= threshold_score
        
        # Track statistics
        if has_enough_confirmations:
            self.confirmation_stats['signals_passed'] += 1
        else:
            self.confirmation_stats['signals_rejected'] += 1
        
        logger.info(f"Enhanced confirmations weighted score: {normalized_score:.2f} " 
                  f"(threshold: {threshold_score:.2f}): {'PASS' if has_enough_confirmations else 'FAIL'}")
        
        # Log which filters contributed positively
        logger.debug(f"Filter contributions: {', '.join([f'{k}: {v}' for k, v in confirmations.items()])}")
        
        return has_enough_confirmations
    
    def on_new_candle(self, candle_data, symbol=None):
        """
        Process new candle data with enhanced confirmations and exit diagnostics.
        
        Args:
            candle_data: Dictionary with latest candle OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            str: Signal type ('buy', 'sell', 'exit', 'none')
        """
        # Convert candle data to DataFrame if it's not already
        if not isinstance(candle_data, pd.DataFrame):
            if isinstance(candle_data, pd.Series):
                df = pd.DataFrame([candle_data.to_dict()])
            else:
                df = pd.DataFrame([candle_data])
        else:
            df = candle_data.copy()
            
        # First, check for exit signals if we're tracking them and have active trades
        if self.track_exit_conditions and bool(self.active_trades):
            self._check_early_exit_conditions(df, symbol)
            
        # Get signal from parent strategy (with all previous filters)
        signal = super().on_new_candle(candle_data, symbol)
        
        # If we have a buy or sell signal, apply enhanced confirmations
        if signal in ['buy', 'sell']:
            if not self.check_enhanced_confirmations(df, signal):
                logger.info(f"Signal {signal} rejected by enhanced confirmations")
                return 'none'
            
            # If we're tracking exits, record entry information for this trade
            if self.track_exit_conditions:
                trade_id = f"{symbol or 'default'}_{df.index[-1]}"
                self.active_trades[trade_id] = {
                    'entry_time': df.index[-1],
                    'entry_price': df['close'].iloc[-1],
                    'signal_type': signal,
                    'entry_conditions': self._get_current_conditions(df, signal)
                }
                logger.debug(f"Recorded entry conditions for trade {trade_id}")
        
        # If we get an exit signal, record which filters would have signaled early
        elif signal == 'exit' and self.track_exit_conditions:
            # Find the active trade for this symbol
            trade_key = None
            for key in self.active_trades:
                if key.startswith(f"{symbol or 'default'}_"):
                    trade_key = key
                    break
            
            if trade_key:
                # Store the exit conditions that were present at actual exit
                exit_conditions = self._get_current_conditions(df, 'exit')
                self.active_trades[trade_key]['exit_time'] = df.index[-1]
                self.active_trades[trade_key]['exit_price'] = df['close'].iloc[-1]
                self.active_trades[trade_key]['exit_conditions'] = exit_conditions
                
                # Calculate P&L for this trade
                entry_price = self.active_trades[trade_key]['entry_price']
                exit_price = df['close'].iloc[-1]
                is_long = self.active_trades[trade_key]['signal_type'] == 'buy'
                pnl = exit_price - entry_price if is_long else entry_price - exit_price
                
                # Store diagnostics about this exit
                self._record_exit_diagnostics(trade_key, exit_conditions, pnl > 0)
                
                # Clean up
                del self.active_trades[trade_key]
                logger.debug(f"Cleaned up exit diagnostics for trade {trade_key}")
                
        return signal
    
    def _get_current_conditions(self, df, signal_type):
        """
        Get the current filter conditions for diagnostics.
        
        Args:
            df: DataFrame with OHLCV data
            signal_type: 'buy', 'sell', or 'exit'
            
        Returns:
            dict: Current conditions of all filters
        """
        df_with_indicators = self._calculate_indicators(df.copy())
        
        # Check all confirmation indicators even if filters are disabled
        momentum_status = self._check_momentum_confirmation(df_with_indicators, signal_type)
        volume_status = self._check_volume_confirmation(df_with_indicators, signal_type)
        volatility_status = self._check_volatility_confirmation(df_with_indicators, signal_type)
        pivot_status = self._check_pivot_confirmation(df_with_indicators, signal_type)
        
        # Calculate weighted score
        weighted_score = 0.0
        max_score = 0.0
        
        if self.use_momentum_filter:
            weighted_score += momentum_status * self.filter_weights['momentum']
            max_score += self.filter_weights['momentum']
            
        if self.use_volume_filter:
            weighted_score += volume_status * self.filter_weights['volume']
            max_score += self.filter_weights['volume']
            
        if self.use_volatility_filter:
            weighted_score += volatility_status * self.filter_weights['volatility']
            max_score += self.filter_weights['volatility']
            
        if self.use_pivot_filter:
            weighted_score += pivot_status * self.filter_weights['pivot']
            max_score += self.filter_weights['pivot']
            
        normalized_score = weighted_score / max_score if max_score > 0 else 0
        
        # Return all conditions with details
        return {
            'time': df.index[-1],
            'price': df['close'].iloc[-1],
            'momentum': momentum_status,
            'volume': volume_status,
            'volatility': volatility_status,
            'pivot': pivot_status,
            'weighted_score': normalized_score,
            'normalized_score': normalized_score,
            # Include values of key indicators for reference
            'indicators': {
                'rsi': df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators else None,
                'macd_hist': df_with_indicators['macd_hist'].iloc[-1] if 'macd_hist' in df_with_indicators else None,
                'atr': df_with_indicators['atr'].iloc[-1] if 'atr' in df_with_indicators else None
            }
        }
    
    def _check_early_exit_conditions(self, df, symbol):
        """
        Check if any filters would trigger an early exit for active trades.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            bool: True if any early exit conditions detected
        """
        # Look for active trades for this symbol
        active_trade_keys = [k for k in self.active_trades if k.startswith(f"{symbol or 'default'}_")]
        
        if not active_trade_keys:
            return False
            
        # Calculate all indicators once
        df_with_indicators = self._calculate_indicators(df.copy())
        
        for trade_key in active_trade_keys:
            trade_data = self.active_trades[trade_key]
            signal_type = trade_data['signal_type']
            
            # We want to check for conditions that would counter the original signal
            counter_signal = 'sell' if signal_type == 'buy' else 'buy'
            
            # Check all filters for exit signals (opposite of entry)
            momentum_exit = self._check_momentum_confirmation(df_with_indicators, counter_signal)
            volume_exit = self._check_volume_confirmation(df_with_indicators, counter_signal)
            volatility_exit = self._check_volatility_confirmation(df_with_indicators, counter_signal)
            pivot_exit = self._check_pivot_confirmation(df_with_indicators, counter_signal)
            
            # Store any flag that would have signaled an exit
            if momentum_exit:
                trade_data['momentum_exit_flag'] = {
                    'time': df.index[-1],
                    'price': df['close'].iloc[-1],
                    'rsi': df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators else None
                }
            
            if volume_exit:
                trade_data['volume_exit_flag'] = {
                    'time': df.index[-1],
                    'price': df['close'].iloc[-1],
                    'volume': df['volume'].iloc[-1]
                }
            
            if volatility_exit:
                trade_data['volatility_exit_flag'] = {
                    'time': df.index[-1],
                    'price': df['close'].iloc[-1],
                    'atr': df_with_indicators['atr'].iloc[-1] if 'atr' in df_with_indicators else None
                }
            
            if pivot_exit:
                trade_data['pivot_exit_flag'] = {
                    'time': df.index[-1],
                    'price': df['close'].iloc[-1]
                }
                
            # Check if we have enough signals to suggest an early exit
            exit_score = 0.0
            max_score = 0.0
            exit_indicators = []
            
            if self.use_momentum_filter:
                exit_score += momentum_exit * self.filter_weights['momentum']
                max_score += self.filter_weights['momentum']
                if momentum_exit:
                    exit_indicators.append('momentum')
                
            if self.use_volume_filter:
                exit_score += volume_exit * self.filter_weights['volume']
                max_score += self.filter_weights['volume']
                if volume_exit:
                    exit_indicators.append('volume')
                
            if self.use_volatility_filter:
                exit_score += volatility_exit * self.filter_weights['volatility']
                max_score += self.filter_weights['volatility']
                if volatility_exit:
                    exit_indicators.append('volatility')
                
            if self.use_pivot_filter:
                exit_score += pivot_exit * self.filter_weights['pivot']
                max_score += self.filter_weights['pivot']
                if pivot_exit:
                    exit_indicators.append('pivot')
            
            normalized_exit_score = exit_score / max_score if max_score > 0 else 0
            
            # If we're above a certain threshold, this would be a strong exit signal
            if normalized_exit_score >= 0.5:  # Configurable threshold
                trade_data['early_exit_signal'] = {
                    'time': df.index[-1],
                    'price': df['close'].iloc[-1],
                    'indicators': exit_indicators,
                    'score': normalized_exit_score
                }
                logger.debug(f"Early exit signal detected for {trade_key}: "
                           f"score={normalized_exit_score:.2f}, indicators={exit_indicators}")
                
                # Calculate potential P&L if exited here
                entry_price = trade_data['entry_price']
                current_price = df['close'].iloc[-1]
                is_long = signal_type == 'buy'
                potential_pnl = current_price - entry_price if is_long else entry_price - current_price
                trade_data['potential_pnl'] = potential_pnl
                
                # For live trading, we might return True here to suggest exiting
        
        return True
    
    def _record_exit_diagnostics(self, trade_key, exit_conditions, is_profitable):
        """
        Record diagnostics about which filters triggered exits.
        
        Args:
            trade_key: Identifier for the trade
            exit_conditions: Dictionary of exit filter conditions
            is_profitable: Whether the trade was profitable
        """
        # Increment overall exit counter
        self.confirmation_stats['exit_signals']['total_exits'] += 1
        
        # Track which filters triggered the exit
        if exit_conditions['momentum']:
            self.confirmation_stats['exit_signals']['momentum_triggered'] += 1
        
        if exit_conditions['volume']:
            self.confirmation_stats['exit_signals']['volume_triggered'] += 1
            
        if exit_conditions['volatility']:
            self.confirmation_stats['exit_signals']['volatility_triggered'] += 1
            
        if exit_conditions['pivot']:
            self.confirmation_stats['exit_signals']['pivot_triggered'] += 1
            
        # Track exit score
        exit_score = exit_conditions['normalized_score']
        prev_avg = self.confirmation_stats['exit_signals']['avg_exit_score']
        prev_count = max(0, self.confirmation_stats['exit_signals']['total_exits'] - 1)
        
        # Update running average of exit scores
        if prev_count > 0:
            self.confirmation_stats['exit_signals']['avg_exit_score'] = \
                (prev_avg * prev_count + exit_score) / (prev_count + 1)
        else:
            self.confirmation_stats['exit_signals']['avg_exit_score'] = exit_score
            
        # Track if this exit was profitable or not
        if is_profitable:
            self.confirmation_stats['exit_signals']['profitable_exits'] += 1
        else:
            self.confirmation_stats['exit_signals']['unprofitable_exits'] += 1
        
        # Extract early exit flags for analysis
        trade_data = self.active_trades[trade_key]
        early_flags = {}
        
        if 'momentum_exit_flag' in trade_data:
            early_flags['momentum'] = trade_data['momentum_exit_flag']
            
        if 'volume_exit_flag' in trade_data:
            early_flags['volume'] = trade_data['volume_exit_flag']
            
        if 'volatility_exit_flag' in trade_data:
            early_flags['volatility'] = trade_data['volatility_exit_flag']
            
        if 'pivot_exit_flag' in trade_data:
            early_flags['pivot'] = trade_data['pivot_exit_flag']
        
        # Log if any early exit signals would have improved outcome
        if early_flags and 'early_exit_signal' in trade_data and not is_profitable:
            early_exit_price = trade_data['early_exit_signal']['price']
            entry_price = trade_data['entry_price']
            exit_price = trade_data['exit_conditions']['price']
            is_long = trade_data['signal_type'] == 'buy'
            
            # Calculate what P&L would have been with early exit
            actual_pnl = exit_price - entry_price if is_long else entry_price - exit_price
            early_pnl = early_exit_price - entry_price if is_long else entry_price - early_exit_price
            
            if (early_pnl > 0 and actual_pnl <= 0) or (early_pnl > actual_pnl):
                logger.info(f"Early exit would have improved outcome for {trade_key}: "
                          f"Actual P&L: {actual_pnl:.2f}, Early P&L: {early_pnl:.2f}, "
                          f"Triggered by: {trade_data['early_exit_signal']['indicators']}")
    
    def backtest(self, df, symbol=None):
        """
        Backtest with all confirmations applied.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol for fetching daily data
            
        Returns:
            DataFrame: DataFrame with signals
        """
        # Reset statistics
        self.confirmation_stats = {key: 0 for key in self.confirmation_stats}
        
        # Run backtest with parent strategy filters
        signals_df = super().backtest(df, symbol)
        
        # Make a copy to avoid modifying the original
        enhanced_df = signals_df.copy()
        
        # Calculate indicators for the entire dataset
        enhanced_df = self._calculate_indicators(enhanced_df)
        
        # Track filtered signals
        if 'enhanced_filtered' not in enhanced_df.columns:
            enhanced_df['enhanced_filtered'] = 0
        
        # Apply enhanced confirmations
        for i in range(max(5, self.lookback_window), len(enhanced_df)):
            # If we have a signal, check for confirmations
            if enhanced_df.iloc[i]['signal'] != 0:
                # Get subset of data up to current row
                subset_df = enhanced_df.iloc[:i+1]
                
                # Determine signal type
                signal_type = 'buy' if enhanced_df.iloc[i]['signal'] == 1 else 'sell'
                
                # Check enhanced confirmations
                if not self.check_enhanced_confirmations(subset_df, signal_type):
                    # Track the filtered signal
                    enhanced_df.loc[enhanced_df.index[i], 'enhanced_filtered'] = enhanced_df.iloc[i]['signal']
                    
                    # Remove the signal
                    enhanced_df.loc[enhanced_df.index[i], 'signal'] = 0
        
        # Log confirmation stats
        if self.confirmation_stats['total_signals'] > 0:
            momentum_pct = (self.confirmation_stats['momentum_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_momentum_filter else 'N/A'
            volume_pct = (self.confirmation_stats['volume_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_volume_filter else 'N/A'
            volatility_pct = (self.confirmation_stats['volatility_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_volatility_filter else 'N/A'
            pivot_pct = (self.confirmation_stats['pivot_confirmed'] / self.confirmation_stats['total_signals']) * 100 if self.use_pivot_filter else 'N/A'
            
            logger.info(f"Enhanced Confirmation Stats:")
            if self.use_momentum_filter:
                logger.info(f"Momentum: {self.confirmation_stats['momentum_confirmed']}/{self.confirmation_stats['total_signals']} ({momentum_pct:.1f}%)")
            if self.use_volume_filter:
                logger.info(f"Volume: {self.confirmation_stats['volume_confirmed']}/{self.confirmation_stats['total_signals']} ({volume_pct:.1f}%)")
            if self.use_volatility_filter:
                logger.info(f"Volatility: {self.confirmation_stats['volatility_confirmed']}/{self.confirmation_stats['total_signals']} ({volatility_pct:.1f}%)")
            if self.use_pivot_filter:
                logger.info(f"Pivot: {self.confirmation_stats['pivot_confirmed']}/{self.confirmation_stats['total_signals']} ({pivot_pct:.1f}%)")
                
            overall_pct = (self.confirmation_stats['signals_passed'] / (self.confirmation_stats['signals_passed'] + self.confirmation_stats['signals_rejected'])) * 100
            logger.info(f"Overall: {self.confirmation_stats['signals_passed']}/{self.confirmation_stats['signals_passed'] + self.confirmation_stats['signals_rejected']} ({overall_pct:.1f}%) passed all filters")
            
        return enhanced_df
        
    def get_info(self):
        """
        Get current strategy information.
        
        Returns:
            dict: Strategy parameters and state
        """
        info = super().get_info()
        info.update({
            'strategy_type': 'EnhancedConfirmationStrategy',
            'use_momentum_filter': self.use_momentum_filter,
            'use_volume_filter': self.use_volume_filter,
            'use_volatility_filter': self.use_volatility_filter,
            'use_pivot_filter': self.use_pivot_filter,
            'rsi_period': self.rsi_period,
            'rsi_threshold': self.rsi_threshold,
            'volume_lookback': self.volume_lookback,
            'volume_factor': self.volume_factor,
            'atr_percentile': self.atr_percentile,
            'min_confirmations': self.min_confirmations,
            'filter_weights': self.filter_weights,
            'track_exit_conditions': self.track_exit_conditions,
            'active_trades': len(self.active_trades),
            'confirmation_stats': self.confirmation_stats
        })
        return info 

    def generate_signals(self, df, higher_tf_data=None):
        """
        Generate trading signals with indicators. Compatibility method for timelapse simulator.
        
        Args:
            df: DataFrame with OHLCV data
            higher_tf_data: Optional dictionary with higher timeframe data
            
        Returns:
            DataFrame with signals added
        """
        # Calculate all technical indicators first
        df = self._calculate_indicators(df)
        
        # Initialize signal column
        df['signal'] = 0
        
        # EMA crossover signals
        if 'fast_ema' in df.columns and 'slow_ema' in df.columns:
            # Bullish crossover
            df.loc[(df['fast_ema'] > df['slow_ema']) & 
                   (df['fast_ema'].shift(1) <= df['slow_ema'].shift(1)), 'signal'] = 1
            
            # Bearish crossover
            df.loc[(df['fast_ema'] < df['slow_ema']) & 
                   (df['fast_ema'].shift(1) >= df['slow_ema'].shift(1)), 'signal'] = -1
            
        # Apply additional filters
        for i in range(1, len(df)):
            if df['signal'].iloc[i] != 0:
                # Get signal type
                signal_type = 'buy' if df['signal'].iloc[i] > 0 else 'sell'
                
                # Apply our filters
                if not self.check_enhanced_confirmations(df.iloc[:i+1], signal_type):
                    df.loc[df.index[i], 'signal'] = 0
        
        return df 