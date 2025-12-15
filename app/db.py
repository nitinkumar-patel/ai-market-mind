from typing import Any, Iterable, List, Optional, Sequence, Tuple

import psycopg
from pgvector.psycopg import register_vector

from app.core.config import settings


def _get_dsn() -> str:
    return (
        f"postgresql://{settings.postgres_user}:{settings.postgres_password}"
        f"@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"
    )


def get_sync_connection() -> psycopg.Connection[Any]:
    """
    Get a synchronous psycopg connection.
    Note: pgvector type registration is done lazily in functions that need it,
    after we are sure the extension exists.
    """
    return psycopg.connect(_get_dsn())


def ensure_schema() -> None:
    """
    Initialize the pgvector extension and the basic table used for RAG memory.
    This is intentionally simple and synchronous; call it at startup.
    """
    # Create the extension before any pgvector type registration
    with get_sync_connection() as conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS research_chunks (
                id BIGSERIAL PRIMARY KEY,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                source_url TEXT,
                embedding vector(1536) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding
            ON research_chunks
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
        conn.commit()


def upsert_chunks(
    topic: str,
    chunks: Sequence[Tuple[str, Optional[str], List[float]]],
) -> None:
    """
    Store chunks into Postgres with their embeddings.
    Each chunk: (content, source_url, embedding_vector)
    """
    if not chunks:
        return

    with get_sync_connection() as conn, conn.cursor() as cur:
        # Ensure pgvector type is registered for this connection (extension already created in ensure_schema)
        register_vector(conn)
        cur.executemany(
            """
            INSERT INTO research_chunks (topic, content, source_url, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            [(topic, content, source_url, embedding) for content, source_url, embedding in chunks],
        )
        conn.commit()


def query_similar_chunks(
    topic: str,
    embedding: List[float],
    max_distance: float = 0.2,
    limit: int = 5,
) -> List[Tuple[str, Optional[str]]]:
    """
    Simple cosine-distance search over stored chunks for the topic.
    Returns list of (content, source_url).
    """
    with get_sync_connection() as conn, conn.cursor() as cur:
        # Ensure pgvector type is registered for this connection (extension already created in ensure_schema)
        register_vector(conn)
        cur.execute(
            """
            SELECT content, source_url, (embedding <=> %s::vector) AS distance
            FROM research_chunks
            WHERE topic ILIKE %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
            """,
            (embedding, f"%{topic}%", embedding, limit),
        )
        rows = cur.fetchall()

    return [(row[0], row[1]) for row in rows if row[2] <= max_distance]


