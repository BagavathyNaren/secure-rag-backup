# db/connection.py

import os
import logging
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool
from models.database import Base

logger = logging.getLogger(__name__)

# Neon requires asyncpg driver
DATABASE_URL = os.environ.get("DATABASE_URL", "").replace(
    "postgresql://", 
    "postgresql+asyncpg://"
)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,            # Set True to see SQL queries in logs
    pool_pre_ping=True,    # Verify connection before use
    poolclass=NullPool,    # Required for serverless (Neon)
    connect_args={
        "ssl": "require"   # Force SSL — production security
    }
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():
    """Dependency injection for FastAPI routes"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables on startup"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")