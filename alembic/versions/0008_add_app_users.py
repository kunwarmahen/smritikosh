"""add app_users table for UI authentication

Creates the app_users table with username/password auth and role-based
access control. The username column is also the user_id used throughout
the memory system, linking authentication directly to stored memories.

Revision ID: 0008
Revises: 0007
Create Date: 2026-03-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("username", sa.String(255), nullable=False, unique=True),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="user"),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )
    op.create_index("ix_app_users_username", "app_users", ["username"], unique=True)
    op.create_index("ix_app_users_email", "app_users", ["email"])
    op.create_index("ix_app_users_role", "app_users", ["role"])


def downgrade() -> None:
    op.drop_index("ix_app_users_role", table_name="app_users")
    op.drop_index("ix_app_users_email", table_name="app_users")
    op.drop_index("ix_app_users_username", table_name="app_users")
    op.drop_table("app_users")
