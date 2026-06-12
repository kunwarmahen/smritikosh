"""add llm_usage table for per-call token/cost accounting.

One row per billed LLM API call, attributed to (user_id, app_id, source) via
the ambient llm_context (item D1 — LLM token/cost accounting). Written
fire-and-forget by smritikosh.llm.usage; aggregated by the admin
GET /admin/llm-usage endpoint.

Revision ID: 0022
Revises: 0021
Create Date: 2026-06-11
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0022"
down_revision = "0021"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=True),
        sa.Column("app_id", sa.String(255), nullable=True),
        sa.Column("source", sa.String(50), nullable=False, server_default="unknown"),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_llm_usage_user_app_created", "llm_usage", ["user_id", "app_id", "created_at"]
    )
    op.create_index("ix_llm_usage_created", "llm_usage", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_llm_usage_created", table_name="llm_usage")
    op.drop_index("ix_llm_usage_user_app_created", table_name="llm_usage")
    op.drop_table("llm_usage")
