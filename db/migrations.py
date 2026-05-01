# db/migrations.py
import logging
import os
import re
from alembic.config import Config
from alembic import command

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")


def _mask_url(url: str) -> str:
    return _SECRET_RE.sub("***", url)


def upgrade_head() -> None:
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        raise RuntimeError("DATABASE_URL is not set — cannot run migrations.")

    logger.info("Running Alembic upgrade to head against: %s", _mask_url(raw_url))

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "alembic")
    alembic_cfg.set_main_option("sqlalchemy.url", raw_url)

    # Silence Alembic output
    alembic_cfg.stdout = io.StringIO()

    # Run synchronously (the async handling is inside alembic/env.py)
    command.upgrade(alembic_cfg, "head")

    logger.info("Alembic upgrade complete.")