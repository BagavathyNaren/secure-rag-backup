# db/migrations.py

import os
from alembic import command
from alembic.config import Config

def upgrade_head() -> None:
    """
    Run: alembic upgrade head
    Reads DATABASE_URL inside alembic/env.py.
    """
    # Ensure we can find alembic.ini regardless of cwd
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
    alembic_ini_path = os.path.join(base_dir, "alembic.ini")

    cfg = Config(alembic_ini_path)
    command.upgrade(cfg, "head")