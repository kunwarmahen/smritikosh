"""Remove fixed dimension from embedding column — accepts any vector size

Revision ID: 0013
Revises: 0012
Create Date: 2026-04-19

Removes the hard-coded dimension from events.embedding so the column accepts
any vector length. This lets you switch embedding models (768, 1536, 2048, …)
by updating EMBEDDING_DIMENSIONS in .env without writing a new migration.

Existing embeddings are cleared because the stored vectors are 2048-dim while
the active model now produces a different dimension — they will be re-generated
on next encode.

HNSW / IVFFlat indexes are not created here because:
  - pgvector < 0.7 caps indexed vectors at 2000 dims
  - A dimension-less column cannot carry an index; you must specify the cast:
      CREATE INDEX … USING hnsw ((embedding::vector(768)) vector_cosine_ops)
  Search falls back to exact cosine scan (sequential), which is fine for
  development and small datasets. Run migration 0014 once you settle on a
  fixed embedding model to restore indexed ANN search.
"""

import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    # Drop any vector indexes before altering the column type
    for idx in ("ix_events_embedding_hnsw", "ix_events_embedding", "events_embedding_idx"):
        conn.execute(sa.text(f"DROP INDEX IF EXISTS {idx}"))
    conn.execute(sa.text("COMMIT"))

    conn.execute(sa.text("UPDATE events SET embedding = NULL"))
    conn.execute(sa.text("ALTER TABLE events ALTER COLUMN embedding TYPE vector"))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text("UPDATE events SET embedding = NULL"))
    conn.execute(sa.text("ALTER TABLE events ALTER COLUMN embedding TYPE vector(2048)"))
