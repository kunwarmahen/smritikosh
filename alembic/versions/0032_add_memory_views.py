"""add memory_views and user_beliefs.last_used_at.

An admin decides what a client's prompt is given about a user (facts,
beliefs, memories, each on or off) and what keeps a memory from ageing
(an answer that used it, or being shown at all). One row per user and
app; user_id "*" is everyone's default, NULL inherits. A belief records
when it was last used, as an event does.

Revision ID: 0032
Revises: 0031
Create Date: 2026-10-31
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0032"
down_revision = "0031"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_views",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("show_facts", sa.Boolean(), nullable=True),
        sa.Column("show_beliefs", sa.Boolean(), nullable=True),
        sa.Column("show_memories", sa.Boolean(), nullable=True),
        sa.Column("keep_by", sa.String(16), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("user_id", "app_id", name="uq_memory_view"),
    )
    op.add_column(
        "user_beliefs",
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_beliefs", "last_used_at")
    op.drop_table("memory_views")
