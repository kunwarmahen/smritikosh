"""Add processed_sessions table for session ingest idempotency

Revision ID: 0015
Revises: 0014
Create Date: 2026-04-21

Adds processed_sessions to track which conversation sessions have already been
processed by POST /ingest/session. Prevents double-extraction and supports
streaming (partial) extraction via last_turn_index tracking.
"""

import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "processed_sessions",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("turns_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("facts_extracted", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_turn_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_partial", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("processed_at", sa.DateTime(timezone=True),
                  nullable=False, server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "app_id", "session_id", name="uq_processed_session"),
    )
    op.create_index("ix_processed_sessions_user_app", "processed_sessions",
                    ["user_id", "app_id"])


def downgrade() -> None:
    op.drop_index("ix_processed_sessions_user_app", table_name="processed_sessions")
    op.drop_table("processed_sessions")
