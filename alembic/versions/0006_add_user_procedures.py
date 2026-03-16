"""add user_procedures table

Revision ID: 0006
Revises: 0005
Create Date: 2026-03-16
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_procedures",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False, server_default="default"),
        sa.Column("trigger", sa.Text, nullable=False),
        sa.Column("instruction", sa.Text, nullable=False),
        sa.Column("category", sa.String(50), nullable=False, server_default="topic_response"),
        sa.Column("priority", sa.Integer, nullable=False, server_default="5"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("hit_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("confidence", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("source", sa.String(50), nullable=False, server_default="manual"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_user_procedures_user_app", "user_procedures", ["user_id", "app_id"])
    op.create_index("ix_user_procedures_active", "user_procedures", ["is_active"])
    op.create_index("ix_user_procedures_priority", "user_procedures", ["priority"])


def downgrade() -> None:
    op.drop_index("ix_user_procedures_priority", table_name="user_procedures")
    op.drop_index("ix_user_procedures_active", table_name="user_procedures")
    op.drop_index("ix_user_procedures_user_app", table_name="user_procedures")
    op.drop_table("user_procedures")
