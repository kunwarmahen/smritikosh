"""Add source_type, source_meta to events + user_facts; add status to user_facts

Revision ID: 0014
Revises: 0013
Create Date: 2026-04-21

Adds the source provenance layer (Phase 1 of passive memory extraction):
  - events.source_type    VARCHAR(32)  which ingestion path created this event
  - events.source_meta    JSONB        extra context (session_id, trigger_phrases, …)
  - user_facts.source_type VARCHAR(32) which ingestion path created this fact
  - user_facts.source_meta JSONB       extra context
  - user_facts.status      VARCHAR(16) 'active' | 'pending' | 'rejected'
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "events",
        sa.Column("source_type", sa.String(32), nullable=False, server_default="api_explicit"),
    )
    op.add_column(
        "events",
        sa.Column("source_meta", JSONB, nullable=False, server_default="{}"),
    )
    op.create_index("ix_events_source_type", "events", ["source_type"])

    op.add_column(
        "user_facts",
        sa.Column("source_type", sa.String(32), nullable=False, server_default="api_explicit"),
    )
    op.add_column(
        "user_facts",
        sa.Column("source_meta", JSONB, nullable=False, server_default="{}"),
    )
    op.add_column(
        "user_facts",
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
    )
    op.create_index("ix_user_facts_status", "user_facts", ["status"])
    op.create_index("ix_user_facts_source_type", "user_facts", ["source_type"])


def downgrade() -> None:
    op.drop_index("ix_user_facts_source_type", table_name="user_facts")
    op.drop_index("ix_user_facts_status", table_name="user_facts")
    op.drop_column("user_facts", "status")
    op.drop_column("user_facts", "source_meta")
    op.drop_column("user_facts", "source_type")

    op.drop_index("ix_events_source_type", table_name="events")
    op.drop_column("events", "source_meta")
    op.drop_column("events", "source_type")
