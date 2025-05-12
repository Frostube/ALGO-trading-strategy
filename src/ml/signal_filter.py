#!/usr/bin/env python3
"""
Machine learning-based signal filter for trading strategy.
This module provides a classifier to filter out low-probability trading setups
based on historical performance data.
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging

from src.utils.logger import logger

class MLSignalFilter:
    """Machine learning filter for trading signals."""
    
    def __init__(self, model_path='models/signal_filter.joblib'):
        """
        Initialize the ML signal filter.
        
        Args:
            model_path: Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'ema_trend', 'hma_trend', 'market_trend', 'micro_trend',
            'rsi', 'volume_spike', 'atr_pct', 'momentum_up', 'momentum_down',
            'at_support', 'at_resistance', 'bullish_engulfing', 'bearish_engulfing',
            'hammer', 'shooting_star', 'doji', 'morning_star', 'evening_star',
            'high_volatility', 'trend_strength'
        ]
        
        # Try to load an existing model
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                logger.info(f"Loaded existing ML model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                self.model = None
    
    def prepare_features(self, data):
        """
        Prepare features for the ML model.
        
        Args:
            data: DataFrame with signal data
            
        Returns:
            DataFrame with prepared features
        """
        # Create a copy to avoid modifying the original
        features = data.copy()
        
        # Convert boolean columns to int
        bool_columns = [
            'volume_spike', 'momentum_up', 'momentum_down', 
            'at_support', 'at_resistance', 'bullish_engulfing', 
            'bearish_engulfing', 'hammer', 'shooting_star', 'doji',
            'morning_star', 'evening_star', 'high_volatility'
        ]
        
        for col in bool_columns:
            if col in features.columns:
                features[col] = features[col].astype(int)
            else:
                features[col] = 0  # Default value if column doesn't exist
        
        # Ensure all required columns exist
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0
        
        # Return only the columns we need
        return features[self.feature_columns]
    
    def train(self, historical_trades, market_data):
        """
        Train the ML model on historical trade data.
        
        Args:
            historical_trades: DataFrame with historical trades
            market_data: DataFrame with market data including indicators
            
        Returns:
            dict: Training metrics
        """
        if len(historical_trades) < 20:
            logger.warning("Not enough historical trades for ML training")
            return None
        
        # Prepare training data
        X = []
        y = []
        
        for _, trade in historical_trades.iterrows():
            # Find the market data at entry time
            entry_time = trade['entry_time']
            if entry_time not in market_data.index:
                continue
                
            # Get features at entry time
            features = self.prepare_features(market_data.loc[entry_time:entry_time])
            
            if len(features) == 0:
                continue
                
            # Label: 1 for profitable trade, 0 for losing trade
            label = 1 if trade['pnl'] > 0 else 0
            
            X.append(features.iloc[0].values)
            y.append(label)
        
        if len(X) < 10:
            logger.warning("Not enough valid trades for ML training")
            return None
            
        X = np.array(X)
        y = np.array(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_split=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Save the model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        
        logger.info(f"ML model trained with accuracy: {metrics['accuracy']:.2f}, precision: {metrics['precision']:.2f}")
        
        return metrics
    
    def predict_probability(self, signal_data):
        """
        Predict the probability of a successful trade.
        
        Args:
            signal_data: DataFrame row with signal data
            
        Returns:
            float: Probability of success (0-1)
        """
        if self.model is None:
            logger.warning("ML model not trained yet")
            return 0.5  # Neutral prediction
        
        # Prepare features
        features = self.prepare_features(pd.DataFrame([signal_data]))
        
        # Scale features
        X_scaled = self.scaler.transform(features)
        
        # Predict probability
        proba = self.model.predict_proba(X_scaled)[0, 1]  # Probability of class 1 (profitable)
        
        return proba
    
    def filter_signal(self, signal_data, threshold=0.6):
        """
        Filter a trading signal based on ML prediction.
        
        Args:
            signal_data: DataFrame row with signal data
            threshold: Probability threshold for accepting a signal
            
        Returns:
            bool: True if signal should be taken, False otherwise
        """
        # If no ML model, don't filter
        if self.model is None:
            return True
            
        # Get prediction probability
        proba = self.predict_probability(signal_data)
        
        # Apply threshold
        return proba >= threshold 