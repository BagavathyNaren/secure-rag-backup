"""add_security_role

Revision ID: a8c1d2e3f4b5
Revises: ee4df1bb4695
Create Date: 2026-05-20

"""
from alembic import op


revision = "a8c1d2e3f4b5"
down_revision = "ee4df1bb4695"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("ALTER TYPE userrole ADD VALUE IF NOT EXISTS 'security'")


def downgrade():
    pass

