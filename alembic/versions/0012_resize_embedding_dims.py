"""Resize embedding vector column from 768 to 2048 dimensions

Revision ID: 0012
Revises: 0011
Create Date: 2026-04-19

Switches from nomic-embed-text (768 dims) to Qwen3 via llama.cpp native
/embedding endpoint (2048 dims). Existing embeddings are cleared because
vector dimensions are incompatible — they will be re-generated on next access.

Note: pgvector's HNSW and IVFFlat indexes cap at 2000 dimensions, so no
vector index is created here. Similarity search falls back to a sequential
scan (exact cosine), which is fine for development / small datasets.
To re-enable indexing, either switch to a ≤2000-dim embedding model or
upgrade to pgvector >= 0.7 and use the halfvec type (supports up to 4000 dims).
"""

import sqlalchemy as sa
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop ALL vector indexes on events.embedding — pgvector validates dimension
    # limits against existing indexes even within the same transaction, so they
    # must be gone before ALTER TABLE runs. Covers both the migration-tracked HNSW
    # index and any manually-created IVFFlat index.
    conn = op.get_bind()
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_events_embedding_hnsw"))
    conn.execute(sa.text("DROP INDEX IF EXISTS ix_events_embedding"))
    conn.execute(sa.text("DROP INDEX IF EXISTS events_embedding_idx"))
    conn.execute(sa.text("COMMIT"))  # flush drops before ALTER TABLE

    conn.execute(sa.text("UPDATE events SET embedding = NULL"))
    conn.execute(sa.text("ALTER TABLE events ALTER COLUMN embedding TYPE vector(2048)"))
    # No HNSW index: pgvector caps indexed vectors at 2000 dims; 2048 exceeds this.
    # Sequential scan (exact cosine) is used instead.


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_events_embedding_hnsw")
    op.execute("UPDATE events SET embedding = NULL")
    op.execute("ALTER TABLE events ALTER COLUMN embedding TYPE vector(768)")
    op.execute(
        "CREATE INDEX ix_events_embedding_hnsw ON events "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64) "
        "WHERE embedding IS NOT NULL"
    )
