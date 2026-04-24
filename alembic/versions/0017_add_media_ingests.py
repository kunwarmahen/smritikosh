"""add media_ingests table

Revision ID: 0017_add_media_ingests
Revises: 0016_add_fact_contradictions
Create Date: 2026-04-24 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "0017_add_media_ingests"
down_revision = "0016_add_fact_contradictions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "media_ingests",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column(
            "app_id",
            sa.String(255),
            nullable=False,
            server_default="default",
        ),
        sa.Column("content_type", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.String(16),
            nullable=False,
            server_default="processing",
        ),
        sa.Column("source_type", sa.String(32), nullable=True),
        sa.Column(
            "facts_extracted",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "facts_pending_review",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "pending_facts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="[]",
        ),
        sa.Column("event_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
            name="fk_media_ingests_event_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "app_id",
            "idempotency_key",
            name="uq_media_ingests_idempotency",
            postgresql_where="idempotency_key IS NOT NULL",
        ),
    )
    op.create_index(
        "ix_media_ingests_user_app",
        "media_ingests",
        ["user_id", "app_id"],
    )
    op.create_index("ix_media_ingests_status", "media_ingests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_media_ingests_status", table_name="media_ingests")
    op.drop_index("ix_media_ingests_user_app", table_name="media_ingests")
    op.drop_table("media_ingests")
