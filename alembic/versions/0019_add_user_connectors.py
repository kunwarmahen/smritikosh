"""add user_connectors table for OAuth2 connector credentials.

Revision ID: 0019
Revises: 0018_add_user_voice_profiles
Create Date: 2026-04-26
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, ARRAY

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_connectors",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default=sa.text("'default'")),
        sa.Column("provider", sa.String(32), nullable=False),  # gmail, gcal, etc.
        sa.Column("status", sa.String(16), nullable=False, server_default=sa.text("'active'")),
        # Encrypted token dict JSON string
        sa.Column("encrypted_tokens", sa.Text, nullable=True),
        # When the current access token expires
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        # OAuth scopes granted
        sa.Column("scopes", ARRAY(sa.Text), nullable=False, server_default=sa.text("'{}'::text[]")),
        sa.Column("connected_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("user_id", "app_id", "provider", name="uq_user_connector"),
    )
    op.create_index("ix_user_connectors_user_app", "user_connectors", ["user_id", "app_id"])


def downgrade() -> None:
    op.drop_index("ix_user_connectors_user_app", table_name="user_connectors")
    op.drop_table("user_connectors")
