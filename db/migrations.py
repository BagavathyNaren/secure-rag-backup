# db/migrations.py
import logging
import re
import os
from alembic.config import Config
from alembic import command

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(?<=://)[^:]+:[^@]+(?=@)")


def _mask_url(url: str) -> str:
    """Replace user:password in a DSN with ***."""
    return _SECRET_RE.sub("***", url)


def upgrade_head() -> None:
    """
    Run Alembic migrations programmatically.
    The raw DATABASE_URL is passed to Alembic internally but is NEVER
    written to any log record here — only the masked form is emitted.
    """
    raw_url = os.getenv("DATABASE_URL", "").strip()
    if not raw_url:
        raise RuntimeError("DATABASE_URL is not set — cannot run migrations.")

    # Log only the masked URL before handing it to Alembic
    logger.info(
        "Running Alembic upgrade to head against: %s",
        _mask_url(raw_url),
    )

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", "alembic")
    # Inject URL into Alembic config — Alembic itself never logs this value
    # because we do NOT call alembic_cfg.print_stdout and keep logging
    # at WARN for sqlalchemy.engine (see alembic.ini)
    alembic_cfg.set_main_option("sqlalchemy.url", raw_url)

    # Silence Alembic's own progress output to stdout
    import io
    alembic_cfg.stdout = io.StringIO()

    command.upgrade(alembic_cfg, "head")

    logger.info("Alembic upgrade complete.")