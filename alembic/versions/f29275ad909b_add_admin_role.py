# alembic/versions/f29275ad909b_add_admin_role.py

"""add_admin_role

Revision ID: f29275ad909b
Revises: 89cc6acc0ddc
Create Date: 2026-05-01 17:25:00

"""
from alembic import op

# revision identifiers
revision = 'f29275ad909b'
down_revision = '89cc6acc0ddc'
branch_labels = None
depends_on = None


def upgrade():
    # Add 'admin' to the existing userrole enum in PostgreSQL
    # IF NOT EXISTS prevents failure if the value already exists
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'admin'")


def downgrade():
    # PostgreSQL does not support removing enum values directly.
    # A full enum recreation would be required — left as no-op intentionally.
    pass