# alembic/env.py
import asyncio
import logging
import os
import re
import sys
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from alembic import context
from logging.config import fileConfig
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger("alembic.env")

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")


def _mask_url(url: str) -> str:
    """Mask credentials in DATABASE_URL for safe logging."""
    return _SECRET_RE.sub("***", url)


def _normalize_asyncpg_url(url: str) -> str:
    """
    Convert DATABASE_URL to asyncpg-compatible format:
    1. postgresql:// → postgresql+asyncpg://
    2. Remove ?sslmode= (asyncpg uses connect_args={'ssl': ...} instead)
    """
    # Fix scheme
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+asyncpg://", 1)

    # Parse and strip sslmode from query params
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)
    query_params.pop("sslmode", None)  # Remove sslmode if present

    # Rebuild URL without sslmode
    new_query = urlencode(query_params, doseq=True)
    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        new_query,
        parsed.fragment,
    ))
    return normalized


# ── Make sure /app is importable ──────────────────────────────────────────────
_APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_ROOT not in sys.path:
    sys.path.insert(0, _APP_ROOT)

# ── Alembic config ────────────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Get raw URL and normalize for asyncpg
_raw_url = os.getenv("DATABASE_URL", "").strip()
if not _raw_url:
    raise RuntimeError("DATABASE_URL is not set — Alembic cannot continue.")

_async_url = _normalize_asyncpg_url(_raw_url)

config.set_main_option("sqlalchemy.url", _async_url)

# ✅ Only masked URL is ever logged
logger.info("Alembic targeting: %s", _mask_url(_async_url))

# ── Metadata ──────────────────────────────────────────────────────────────────
from models.database import Base
target_metadata = Base.metadata


# ── Migration runners ─────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    logger.info("Offline mode → %s", _mask_url(url))

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations using async engine."""
    url = config.get_main_option("sqlalchemy.url")

    # asyncpg expects ssl=True or ssl=False instead of sslmode
    # Neon requires SSL, so we force it here
    connectable = create_async_engine(
        url,
        poolclass=NullPool,
        echo=False,
        connect_args={"ssl": "require"},  # ← asyncpg SSL parameter
    )

    def do_run_migrations(connection):
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())