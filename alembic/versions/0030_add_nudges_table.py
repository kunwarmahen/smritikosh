"""add nudges table + nudge watermark (Proactive Life OS).

Item E4 remainder (FUTURE.md #5):
  nudges                      — one proactive digest of reflection insights
                                per user per cycle (in-app feed + optional
                                webhook delivery)
  user_activity.last_nudged_at — scheduler staleness watermark for the
                                 lifeos nudge job

Revision ID: 0030
Revises: 0029
Create Date: 2026-07-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0030"
down_revision = "0029"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nudges",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("digest", sa.Text(), nullable=False),
        sa.Column("reflection_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("severity", sa.String(10), nullable=False, server_default="notice"),
        sa.Column("channel", sa.String(16), nullable=False, server_default="feed"),
        sa.Column("status", sa.String(12), nullable=False, server_default="delivered"),
        sa.Column("acknowledged", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_nudges_user_app_created", "nudges", ["user_id", "app_id", "created_at"]
    )
    op.create_index("ix_nudges_acknowledged", "nudges", ["acknowledged"])

    op.add_column(
        "user_activity",
        sa.Column("last_nudged_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_activity", "last_nudged_at")
    op.drop_index("ix_nudges_acknowledged", table_name="nudges")
    op.drop_index("ix_nudges_user_app_created", table_name="nudges")
    op.drop_table("nudges")
