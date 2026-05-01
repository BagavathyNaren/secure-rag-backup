# alembic/env.py
import io
import logging
import os
import re

from alembic import context
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool

logger = logging.getLogger("alembic.env")

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")


def _mask_url(url: str) -> str:
    return _SECRET_RE.sub("***", url)


# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject the real URL from the environment — Alembic never logs this value
# because sqlalchemy.engine is kept at WARN in alembic.ini
_raw_url = os.getenv("DATABASE_URL", "").strip()
if not _raw_url:
    raise RuntimeError("DATABASE_URL is not set — Alembic cannot continue.")

config.set_main_option("sqlalchemy.url", _raw_url)

# Only the masked form appears in any log record
logger.info("Alembic targeting: %s", _mask_url(_raw_url))

# ── metadata ──────────────────────────────────────────────────────────────────
from db.base import Base          # noqa: E402
target_metadata = Base.metadata


# ── migration runners ─────────────────────────────────────────────────────────
def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    logger.info("Offline mode → %s", _mask_url(url))   # masked
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        echo=False,          # ← credentials never echoed to logs
    )
    logger.info("Online migration engine ready.")   # no URL here

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()