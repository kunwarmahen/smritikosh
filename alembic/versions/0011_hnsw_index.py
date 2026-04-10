"""Replace IVFFlat embedding index with HNSW

Revision ID: 0011
Revises: 0010
Create Date: 2026-04-10

The original IVFFlat index (ix_events_embedding) was created in migration 0001
on an empty table. IVFFlat uses k-means clustering to partition vectors into
lists; built on an empty table it produces empty lists and provides no speedup
— every query falls back to a sequential scan.

HNSW (Hierarchical Navigable Small World) is the preferred alternative:
  - No training data required — works correctly from an empty table.
  - Better recall/speed trade-off than IVFFlat at typical dataset sizes.
  - Supports concurrent inserts without an index rebuild.
  - Available in pgvector >= 0.5.0 (shipped with pgvector/pgvector:pg17).

Index parameters:
  m = 16              connections per layer; higher → better recall, more memory
  ef_construction=64  candidate list size at build time; higher → better recall,
                      slower initial build.  64 is a safe default.

Partial WHERE clause (embedding IS NOT NULL) excludes events stored before
their embedding was generated, keeping the index small and accurate.

At query time, set hnsw.ef_search (default 40) to tune recall:
  SET hnsw.ef_search = 100;   -- better recall, slightly slower
The EpisodicMemory.hybrid_search method does this automatically.
"""

from alembic import op

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the IVFFlat index (built on empty table, effectively a no-op)
    op.execute("DROP INDEX IF EXISTS ix_events_embedding")

    # Create HNSW index for approximate nearest-neighbour cosine search.
    # The partial WHERE clause skips NULL embeddings so they don't bloat the
    # index and cause type errors on search.
    op.execute(
        "CREATE INDEX ix_events_embedding_hnsw ON events "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64) "
        "WHERE embedding IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_events_embedding_hnsw")

    # Restore the original IVFFlat index
    op.execute(
        "CREATE INDEX ix_events_embedding ON events "
        "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
    )
