# db/connection.py
import logging
import os
import re

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import event

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")


def _mask_url(url: str) -> str:
    return _SECRET_RE.sub("***", url)


def _normalize_asyncpg_url(url: str) -> str:
    """
    Convert DATABASE_URL to asyncpg format:
    1. postgresql:// → postgresql+asyncpg://
    2. Strip ALL query parameters (sslmode, channel_binding, etc.)
    """
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    # Strip query params — asyncpg handles SSL via connect_args
    if "?" in url:
        url = url.split("?")[0]

    return url


def _get_async_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return _normalize_asyncpg_url(raw)


_async_url = _get_async_database_url()

# ── engine ────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    _async_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"ssl": "require"},  # ← asyncpg SSL parameter
)

# ── safe connect-event log ────────────────────────────────────────────────────
@event.listens_for(engine.sync_engine, "connect")
def _on_connect(dbapi_conn, connection_record):
    logger.debug("New DB connection → %s", _mask_url(_async_url))


AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session