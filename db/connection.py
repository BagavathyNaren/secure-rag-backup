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

from db.url_utils import validate_database_url, redact_database_url

logger = logging.getLogger(__name__)

# ============================================================
# CONNECTION STRING (validated + redacted logs)
# ============================================================

_raw_url = os.environ.get("DATABASE_URL", "").strip()
validate_database_url(_raw_url)

# Never print secrets. This is safe.
logger.info("DATABASE_URL_OK: %s", redact_database_url(_raw_url))

# Convert to asyncpg driver URL
_async_url = (
    _raw_url.replace("postgresql://", "postgresql+asyncpg://")
            .replace("postgres://", "postgresql+asyncpg://")
)

# Keep a flag for SSL requirement based on original URL
_ssl_required = "sslmode=require" in _raw_url.lower()

# Strip query params (sslmode, etc.). We'll handle SSL via connect_args.
if "?" in _async_url:
    _async_url = _async_url.split("?", 1)[0]

# ============================================================
# SSL CONTEXT (only if required)
# ============================================================

connect_args = {}

if _ssl_required:
    _ssl_context = ssl.create_default_context()
    _ssl_context.check_hostname = True
    _ssl_context.verify_mode = ssl.CERT_REQUIRED
    connect_args = {"ssl": _ssl_context}

# ============================================================
# ENGINE
# ============================================================

engine = create_async_engine(
    _async_url,
    echo=False,
    pool_pre_ping=True,
    poolclass=NullPool,
    connect_args=connect_args,
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
# TABLE CREATION — legacy (you now use Alembic in startup)
# ============================================================

async def init_db():
    """Create all tables if they don't exist (legacy)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created/verified (create_all)")