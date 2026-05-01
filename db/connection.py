# db/connection.py

import os
import logging
import ssl
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool
from models.database import Base

logger = logging.getLogger(__name__)

# ============================================================
# CONNECTION STRING
# ============================================================

# Neon gives: postgresql://user:pass@host/db?sslmode=require
# asyncpg needs: postgresql+asyncpg://user:pass@host/db
# We strip ?sslmode=require and handle SSL via connect_args

_raw_url = os.environ.get("DATABASE_URL", "")

# Step 1 — replace scheme for asyncpg driver
_async_url = _raw_url.replace(
    "postgresql://",
    "postgresql+asyncpg://"
).replace(
    "postgres://",          # some providers use postgres:// shorthand
    "postgresql+asyncpg://"
)

# Step 2 — strip ?sslmode=require from URL
#           (we pass SSL properly via connect_args instead)
if "?" in _async_url:
    _async_url = _async_url.split("?")[0]

# ============================================================
# SSL CONTEXT
# ============================================================

# asyncpg expects an ssl.SSLContext object, not a string
_ssl_context = ssl.create_default_context()
_ssl_context.check_hostname = True
_ssl_context.verify_mode    = ssl.CERT_REQUIRED

# ============================================================
# ENGINE
# ============================================================

engine = create_async_engine(
    _async_url,
    echo=False,          # Set True to debug SQL queries
    pool_pre_ping=True,
    poolclass=NullPool,  # Required for serverless (Neon)
    connect_args={
        "ssl": _ssl_context   # ✅ correct way for asyncpg
    },
)

# ============================================================
# SESSION FACTORY
# ============================================================

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# ============================================================
# DEPENDENCY — FastAPI route injection
# ============================================================

async def get_db():
    """Dependency injection for FastAPI routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

# ============================================================
# TABLE CREATION — called on startup
# ============================================================

async def init_db():
    """Create all tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified")