from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, create_engine, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from src.config import DATABASE_URL

# Create engine and session
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

class OHLCV(Base):
    """Model for storing OHLCV (Open, High, Low, Close, Volume) data."""
    __tablename__ = 'ohlcv'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False, default='1m')
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Create index for faster queries
    __table_args__ = (
        Index('idx_ohlcv_symbol_timeframe_timestamp', 'symbol', 'timeframe', 'timestamp'),
    )
    
    def __repr__(self):
        return f"<OHLCV(symbol='{self.symbol}', timeframe='{self.timeframe}', timestamp='{self.timestamp}', close={self.close})>"
    
    @classmethod
    def from_ccxt(cls, symbol, candle, timeframe='1m'):
        """Create an OHLCV instance from a CCXT candle."""
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.fromtimestamp(candle[0] / 1000),
            open=candle[1],
            high=candle[2],
            low=candle[3],
            close=candle[4],
            volume=candle[5]
        )

class Trade(Base):
    """Model for storing trade information."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    stop_loss = Column(Float, nullable=False)
    take_profit = Column(Float, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String, nullable=True)  # 'tp', 'sl', 'manual', 'timeout', etc.
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    
    # Additional metrics for analysis
    rsi_value = Column(Float, nullable=True)
    atr_value = Column(Float, nullable=True)
    market_trend = Column(Integer, nullable=True)
    higher_tf_trend = Column(Integer, nullable=True)
    micro_trend = Column(Integer, nullable=True)
    momentum_confirmed = Column(Boolean, nullable=True)
    duration_minutes = Column(Float, nullable=True)
    trailing_activated = Column(Boolean, nullable=True, default=False)
    adaptive_threshold_used = Column(Boolean, nullable=True, default=False)
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', side='{self.side}', entry_price={self.entry_price}, pnl={self.pnl})>"
    
    def close_trade(self, exit_price, exit_reason):
        """Close the trade and calculate PnL."""
        self.exit_time = datetime.now()
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        
        # Calculate PnL
        if self.side == 'buy':
            self.pnl = (self.exit_price - self.entry_price) * self.amount
            self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price
        else:  # sell
            self.pnl = (self.entry_price - self.exit_price) * self.amount
            self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price
            
        # Calculate duration
        self.duration_minutes = (self.exit_time - self.entry_time).total_seconds() / 60
        
        return self

class FalsePositive(Base):
    """Model for storing trades that never hit TP or SL (false positives)."""
    __tablename__ = 'false_positives'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # 'buy' or 'sell'
    entry_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    duration_minutes = Column(Float, nullable=True)
    
    # Market conditions when the trade was entered
    market_trend = Column(Integer, nullable=True)  # 1: up, -1: down
    higher_tf_trend = Column(Integer, nullable=True)  # 1: up, -1: down 
    micro_trend = Column(Integer, nullable=True)  # 1: up, -1: down
    rsi_value = Column(Float, nullable=True)
    atr_value = Column(Float, nullable=True)
    momentum_confirmed = Column(Boolean, nullable=True, default=False)
    log_time = Column(DateTime, default=datetime.now)
    
    def __repr__(self):
        return f"<FalsePositive(symbol='{self.symbol}', side='{self.side}', duration_minutes={self.duration_minutes})>"

class Adjustment(Base):
    """Model for storing strategy adjustments from AI feedback."""
    __tablename__ = 'adjustments'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.now)
    adjustment_type = Column(String, nullable=False)  # 'parameter', 'rule', etc.
    parameter = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    reason = Column(String, nullable=True)
    applied = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<Adjustment(type='{self.adjustment_type}', parameter='{self.parameter}', new_value='{self.new_value}')>"

class TradeStatistics(Base):
    """Model for storing aggregated trade statistics by period."""
    __tablename__ = 'trade_statistics'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    period = Column(String, nullable=False)  # 'daily', 'weekly', 'monthly'
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, nullable=True)
    profit_factor = Column(Float, nullable=True)
    total_pnl = Column(Float, default=0.0)
    avg_pnl_percent = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    avg_duration_minutes = Column(Float, nullable=True)
    
    def __repr__(self):
        return f"<TradeStatistics(period='{self.period}', date='{self.date}', win_rate={self.win_rate}, profit_factor={self.profit_factor})>"

# Create all tables
def init_db():
    Base.metadata.create_all(engine)
    
    # Return a session
    return Session() 