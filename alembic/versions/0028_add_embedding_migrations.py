"""add embedding_migrations table for resumable bulk re-embed.

Item H1: POST /admin/re-embed becomes a chunked, resumable queue job with a
progress row per run — keyset cursor, processed/error counters, and a status
endpoint (GET /admin/re-embed/status).

Revision ID: 0028
Revises: 0027
Create Date: 2026-07-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0028"
down_revision = "0027"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "embedding_migrations",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("target_model", sa.String(255), nullable=False),
        sa.Column("target_dim", sa.Integer(), nullable=False),
        sa.Column("total", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("processed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("errors", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cursor_created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cursor_id", UUID(as_uuid=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_embedding_migrations_status", "embedding_migrations", ["status"])
    op.create_index("ix_embedding_migrations_started", "embedding_migrations", ["started_at"])


def downgrade() -> None:
    op.drop_index("ix_embedding_migrations_started", table_name="embedding_migrations")
    op.drop_index("ix_embedding_migrations_status", table_name="embedding_migrations")
    op.drop_table("embedding_migrations")
