# db/connection.py
import logging
import os
import re
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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
    2. Strip ?sslmode= (asyncpg doesn't recognize it)
    """
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    query_params.pop("sslmode", None)  # Remove sslmode

    new_query = urlencode(query_params, doseq=True)
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment,
    ))


def _get_async_database_url() -> str:
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return _normalize_asyncpg_url(raw)


_async_url = _get_async_database_url()

# ── engine ────────────────────────────────────────────────────────────────────
engine = create_async_engine(
    _async_url,
    echo=False,        # ← must stay False to prevent credential leaks
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"ssl": "require"},  # ← asyncpg SSL param
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
    """FastAPI dependency — yields a scoped async DB session."""
    async with AsyncSessionLocal() as session:
        yield session