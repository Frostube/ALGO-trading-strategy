from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, create_engine
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
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<OHLCV(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"
    
    @classmethod
    def from_ccxt(cls, symbol, candle):
        """Create an OHLCV instance from a CCXT candle."""
        return cls(
            symbol=symbol,
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
    exit_reason = Column(String, nullable=True)  # 'tp', 'sl', 'manual', etc.
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    
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
        
        return self

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

# Create all tables
def init_db():
    Base.metadata.create_all(engine)
    
    # Return a session
    return Session() 