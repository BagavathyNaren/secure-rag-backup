"""0001_baseline

Revision ID: db43ef050032
Revises: 
Create Date: 2026-05-01 20:20:09.484515
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'db43ef050032'
down_revision = None
branch_labels = None
depends_on = None


# Match SQLAlchemy Enum(UserRole) default naming.
userrole_enum = sa.Enum(
    "employee",
    "manager",
    "hr",
    "finance",
    "executive",
    name="userrole",
)


def upgrade() -> None:
    # Ensure enum type exists before table creation
    userrole_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),

        sa.Column("user_id", sa.String(length=50), nullable=False),
        sa.Column("username", sa.String(length=100), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),

        sa.Column("hashed_password", sa.Text(), nullable=False),

        sa.Column("role", userrole_enum, nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),

        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),

        sa.Column("failed_login_attempts", sa.String(length=10), nullable=True),
        sa.Column("is_locked", sa.Boolean(), nullable=True),

        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )

    # Indexes from index=True in model
    op.create_index("ix_users_user_id", "users", ["user_id"], unique=False)
    op.create_index("ix_users_username", "users", ["username"], unique=False)
    op.create_index("ix_users_email", "users", ["email"], unique=False)

    # Indexes from __table_args__
    op.create_index("ix_users_email_active", "users", ["email", "is_active"], unique=False)
    op.create_index("ix_users_role", "users", ["role"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_users_role", table_name="users")
    op.drop_index("ix_users_email_active", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_index("ix_users_user_id", table_name="users")

    op.drop_table("users")

    # Drop enum type after dropping table
    userrole_enum.drop(op.get_bind(), checkfirst=True)