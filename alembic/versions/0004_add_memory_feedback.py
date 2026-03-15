"""add memory_feedback table

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "event_id",
            UUID(as_uuid=True),
            sa.ForeignKey("events.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("feedback_type", sa.String(50), nullable=False),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_memory_feedback_event", "memory_feedback", ["event_id"])
    op.create_index(
        "ix_memory_feedback_user_app", "memory_feedback", ["user_id", "app_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_memory_feedback_user_app", table_name="memory_feedback")
    op.drop_index("ix_memory_feedback_event", table_name="memory_feedback")
    op.drop_table("memory_feedback")
