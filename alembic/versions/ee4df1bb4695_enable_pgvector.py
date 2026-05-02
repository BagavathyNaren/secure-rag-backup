"""enable_pgvector

Revision ID: ee4df1bb4695
Revises: f29275ad909b
Create Date: 2026-05-02

"""
from alembic import op

# revision identifiers, used by Alembic.
revision = "ee4df1bb4695"
down_revision = "f29275ad909b"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade():
    # Generally leave as no-op in managed Postgres; dropping extension can break other deps
    pass