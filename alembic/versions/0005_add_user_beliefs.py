"""add user_beliefs table

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-15
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_beliefs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("statement", sa.Text, nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("evidence_count", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "first_inferred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("user_id", "app_id", "statement", name="uq_user_belief"),
    )
    op.create_index("ix_user_beliefs_user_app", "user_beliefs", ["user_id", "app_id"])
    op.create_index("ix_user_beliefs_confidence", "user_beliefs", ["confidence"])


def downgrade() -> None:
    op.drop_index("ix_user_beliefs_confidence", table_name="user_beliefs")
    op.drop_index("ix_user_beliefs_user_app", table_name="user_beliefs")
    op.drop_table("user_beliefs")
