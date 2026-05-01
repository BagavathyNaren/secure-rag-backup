# db/connection.py
import logging
import re
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import event

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")


def _mask_url(url: str) -> str:
    return _SECRET_RE.sub("***", url)


def _get_async_database_url() -> str:
    """
    Return an asyncpg-compatible URL.
    Never logs the raw value — callers receive the string directly.
    """
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL environment variable is not set.")

    # Neon / standard postgres → asyncpg driver
    if raw.startswith("postgresql://"):
        raw = raw.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif raw.startswith("postgres://"):
        raw = raw.replace("postgres://", "postgresql+asyncpg://", 1)

    return raw


_async_url = _get_async_database_url()

# ── engine ────────────────────────────────────────────────────────────────────
# echo=False is mandatory — SQLAlchemy logs the full connection string
# (including credentials) when echo=True or when the root logger is DEBUG.
engine = create_async_engine(
    _async_url,
    echo=False,        # ← MUST stay False; credentials leak at echo=True
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

# ── safe connect-event log (fired per new physical connection) ────────────────
@event.listens_for(engine.sync_engine, "connect")
def _on_connect(dbapi_conn, connection_record):
    # Only the masked URL is ever written to the log
    logger.debug("New DB connection → %s", _mask_url(_async_url))


AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db():
    """FastAPI dependency — yields a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        yield session