from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import select  
from sqlalchemy.ext.asyncio import AsyncSession      
import traceback
from app.db.base import engine, get_db, Base # Добавили Base в импорт
from app.db import models
from app.services.moex_client import MoexService
from app.services.analyzer import PatternAnalyzer

app = FastAPI(title="Sirius Orderlog Pipeline")

# ПРАВИЛЬНЫЙ СПОСОБ СОЗДАНИЯ ТАБЛИЦ ДЛЯ ASYNC
@app.on_event("startup")
async def startup():
    async with engine.begin() as conn:
        # Это "мост", который позволяет асинхронному движку создать таблицы
        await conn.run_sync(models.Base.metadata.create_all)

@app.get("/analyze/{symbol}")
async def analyze(symbol: str, db: AsyncSession = Depends(get_db)):
    try:
        # 1. Получаем данные (теперь 5 значений!)
        candles, d_sup, d_res, ma200_glob, trades = await MoexService.get_data(symbol)
        
        if candles is None:
            raise HTTPException(status_code=404, detail="Ticker not found")

        # 2. Анализируем
        res = PatternAnalyzer.analyze_all(
            candles=candles,
            d_sup=d_sup,
            d_res=d_res,
            ma200_glob=ma200_glob,
            trades=trades,
            ticker=symbol
        )
        # 3. Сохраняем (с новыми полями SL/TP/RR)
        analysis_entry = models.StockAnalysis(
            ticker=symbol.upper(),
            price=res["price"],
            score=res["score"],
            signal=res["signal"],
            stop_loss=res["stop_loss"],
            take_profit=res["take_profit"],
            rr_ratio=res["rr_ratio"]
        )
        db.add(analysis_entry)
        await db.commit()
        
        return {
            "ticker": symbol.upper(), 
            "analysis": res,
            "verdict": {
                "signal": res["signal"],
                "entry": res["price"],
                "stop": res["stop_loss"],
                "target": res["take_profit"],
                "is_good_deal": (
                    res["rr_ratio"] is not None and res["rr_ratio"] >= 1.5
                )
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(traceback.format_exc()) 
        return {"status": "error", "message": str(e)}
    
@app.get("/history")
async def history(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(models.StockAnalysis))
    return result.scalars().all()