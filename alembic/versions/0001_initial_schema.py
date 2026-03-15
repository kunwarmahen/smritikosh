"""initial schema — events, user_facts, memory_links

Revision ID: 0001
Revises:
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID

from smritikosh.config import settings

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── pgvector extension (must exist before Vector columns) ──────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── events ─────────────────────────────────────────────────────────────
    op.create_table(
        "events",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("raw_text", sa.Text, nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("embedding", Vector(settings.embedding_dimensions), nullable=True),
        sa.Column("importance_score", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("consolidated", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("event_metadata", JSONB, nullable=False, server_default="{}"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_events_user_app", "events", ["user_id", "app_id"])
    op.create_index("ix_events_consolidated", "events", ["consolidated"])
    op.create_index("ix_events_created_at", "events", ["created_at"])

    # IVFFlat index for fast approximate nearest-neighbour vector search.
    # lists=100 is a good starting point; tune upward as data grows.
    op.execute(
        "CREATE INDEX ix_events_embedding ON events "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )

    # ── user_facts ─────────────────────────────────────────────────────────
    op.create_table(
        "user_facts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("key", sa.String(255), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("frequency_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "user_id", "app_id", "category", "key", name="uq_user_fact"
        ),
    )
    op.create_index("ix_user_facts_user_app", "user_facts", ["user_id", "app_id"])

    # ── memory_links ────────────────────────────────────────────────────────
    op.create_table(
        "memory_links",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "from_event_id",
            UUID(as_uuid=True),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "to_event_id",
            UUID(as_uuid=True),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("relation_type", sa.String(50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_memory_links_from", "memory_links", ["from_event_id"])
    op.create_index("ix_memory_links_to", "memory_links", ["to_event_id"])


def downgrade() -> None:
    op.drop_table("memory_links")
    op.drop_table("user_facts")
    op.drop_index("ix_events_embedding", table_name="events")
    op.drop_table("events")
    op.execute("DROP EXTENSION IF EXISTS vector")
