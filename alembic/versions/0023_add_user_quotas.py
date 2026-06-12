"""add user_quotas table for per-tenant usage quotas.

Per-(user_id, app_id) caps on event count and LLM token spend per UTC day /
month (item D2). NULL limit = fall back to the QUOTA_DEFAULT_* config value
(0 there = unlimited). Enforced at encode / ingest / context entry points;
token windows are computed from the llm_usage table (migration 0022).

Revision ID: 0023
Revises: 0022
Create Date: 2026-06-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0023"
down_revision = "0022"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_quotas",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("daily_event_limit", sa.Integer(), nullable=True),
        sa.Column("monthly_event_limit", sa.Integer(), nullable=True),
        sa.Column("daily_token_limit", sa.BigInteger(), nullable=True),
        sa.Column("monthly_token_limit", sa.BigInteger(), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("user_id", "app_id", name="uq_user_quota"),
    )
    op.create_index("ix_user_quotas_user_app", "user_quotas", ["user_id", "app_id"])


def downgrade() -> None:
    op.drop_index("ix_user_quotas_user_app", table_name="user_quotas")
    op.drop_table("user_quotas")
