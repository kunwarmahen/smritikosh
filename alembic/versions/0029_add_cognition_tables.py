"""add cognition tables: memory_predictions, reflections, reflection watermark.

Item E4 (cognitive agent layer):
  memory_predictions — one predict-observe-learn cycle per /context call
  reflections        — insights from periodic ReflectionAgent cycles
  user_activity.last_reflected_at — scheduler staleness watermark for the
                                    reflection job

Revision ID: 0029
Revises: 0028
Create Date: 2026-07-05
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0029"
down_revision = "0028"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "memory_predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("query_preview", sa.String(300), nullable=False, server_default=""),
        sa.Column("intent", sa.String(32), nullable=False, server_default="general"),
        sa.Column("predicted_event_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("predicted_cluster_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("actual_event_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("hit_rate", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_memory_predictions_user_app_created",
        "memory_predictions",
        ["user_id", "app_id", "created_at"],
    )

    op.create_table(
        "reflections",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("kind", sa.String(20), nullable=False, server_default="observation"),
        sa.Column("insight", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False, server_default="info"),
        sa.Column("evidence", JSONB, nullable=False, server_default="{}"),
        sa.Column("acknowledged", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_reflections_user_app_created",
        "reflections",
        ["user_id", "app_id", "created_at"],
    )
    op.create_index("ix_reflections_acknowledged", "reflections", ["acknowledged"])

    op.add_column(
        "user_activity",
        sa.Column("last_reflected_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("user_activity", "last_reflected_at")
    op.drop_index("ix_reflections_acknowledged", table_name="reflections")
    op.drop_index("ix_reflections_user_app_created", table_name="reflections")
    op.drop_table("reflections")
    op.drop_index("ix_memory_predictions_user_app_created", table_name="memory_predictions")
    op.drop_table("memory_predictions")
