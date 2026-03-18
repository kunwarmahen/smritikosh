"""multi-app access: app_ids array on app_users and api_keys

Revision ID: 0009
Revises: 0008
Create Date: 2026-03-18

Handles two deployment scenarios:
  - Existing deployment: api_keys already exists with a single app_id column → migrate it.
  - Fresh deployment:    api_keys does not exist yet → create it with app_ids directly.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    existing_tables = inspect(conn).get_table_names()

    # ── app_users: app_id (str) → app_ids (text[]) ────────────────────────────
    op.add_column("app_users", sa.Column("app_ids", ARRAY(sa.String(255)), nullable=True))
    op.execute("UPDATE app_users SET app_ids = ARRAY[app_id]")
    op.alter_column("app_users", "app_ids", nullable=False)
    op.drop_column("app_users", "app_id")

    # ── api_keys ───────────────────────────────────────────────────────────────
    if "api_keys" in existing_tables:
        # Existing deployment: table was created outside migrations with a single app_id.
        # Migrate it to the array form.
        op.add_column("api_keys", sa.Column("app_ids", ARRAY(sa.String(255)), nullable=True))
        op.execute("UPDATE api_keys SET app_ids = ARRAY[app_id]")
        op.alter_column("api_keys", "app_ids", nullable=False)
        op.drop_column("api_keys", "app_id")
    else:
        # Fresh deployment: create the table directly with app_ids.
        op.create_table(
            "api_keys",
            sa.Column("id", UUID(as_uuid=True), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(255),
                sa.ForeignKey("app_users.username", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("app_ids", ARRAY(sa.String(255)), nullable=False),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("key_prefix", sa.String(16), nullable=False),
            sa.Column("key_hash", sa.String(64), nullable=False, unique=True),
            sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("NOW()"),
            ),
            sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_api_keys_user_id", "api_keys", ["user_id"])
        op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)


def downgrade() -> None:
    conn = op.get_bind()
    existing_tables = inspect(conn).get_table_names()

    # Check whether api_keys was created by this migration (fresh) or pre-existed (existing)
    # We detect "fresh" by checking if the table has no app_id column (only app_ids)
    if "api_keys" in existing_tables:
        cols = {c["name"] for c in inspect(conn).get_columns("api_keys")}
        if "app_ids" in cols and "app_id" not in cols:
            # Could be either case — safest is to restore app_id from app_ids[1]
            op.add_column("api_keys", sa.Column("app_id", sa.String(255), nullable=True))
            op.execute("UPDATE api_keys SET app_id = app_ids[1]")
            op.alter_column("api_keys", "app_id", nullable=False)
            op.drop_column("api_keys", "app_ids")

    # ── app_users: restore single app_id ──────────────────────────────────────
    op.add_column("app_users", sa.Column("app_id", sa.String(255), nullable=True))
    op.execute("UPDATE app_users SET app_id = app_ids[1]")
    op.alter_column("app_users", "app_id", nullable=False)
    op.drop_column("app_users", "app_ids")
