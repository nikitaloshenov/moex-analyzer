from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Прописываем напрямую для теста, чтобы исключить ошибку парсинга
DATABASE_URL = "sqlite+aiosqlite:///./moex_database.db"

engine = create_async_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}
)
SessionLocal = async_sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine, 
    class_=AsyncSession
)
Base = declarative_base()

async def get_db():
    async with SessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()