from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from app.db.base import Base

class StockAnalysis(Base):
    __tablename__ = "stock_analyses"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, index=True)
    price = Column(Float)
    score = Column(Float) 
    signal = Column(String) 
    
    # --- НОВЫЕ ПОЛЯ V2 ---
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    rr_ratio = Column(Float, nullable=True)  # Risk/Reward
    # ---------------------

    created_at = Column(DateTime, default=datetime.utcnow)