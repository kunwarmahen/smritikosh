"""add user_voice_profiles table for Phase 12 voice enrollment.

Revision ID: 0018
Revises: 0017_add_media_ingests
Create Date: 2026-04-24
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0018"
down_revision = "0017_add_media_ingests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_voice_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default=sa.text("'default'")),
        # Speaker d-vector embedding stored as float array (JSON). NULL if resemblyzer not installed.
        sa.Column("embedding", JSONB, nullable=True),
        sa.Column("embedding_dim", sa.Integer, nullable=True),
        sa.Column("enrolled_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.UniqueConstraint("user_id", "app_id", name="uq_user_voice_profile"),
    )
    op.create_index("ix_user_voice_profiles_user_id", "user_voice_profiles", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_user_voice_profiles_user_id", table_name="user_voice_profiles")
    op.drop_table("user_voice_profiles")
