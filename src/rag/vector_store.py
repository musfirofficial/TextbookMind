import os
from typing import Any

from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient, models

from .core.config import DEFAULT_COLLECTION_NAME, QDRANT_STORAGE_PATH
from .embedding import (
    embed_query_dense,
    embed_query_sparse,
    prepare_qdrant_points,
)

# ==============================================================================
# Create or Recreate Qdrant Client
# ==============================================================================


def get_qdrant_client(
    path: str | None = QDRANT_STORAGE_PATH,
    url: str | None = None,
    api_key: str | None = None,
) -> QdrantClient:
    """
    Returns an initialized QdrantClient.
    Priority:
    1. Explicit url/path parameter
    2. QDRANT_URL from environment (e.g. Docker container or cloud)
    3. Local persistent storage path on disk (data/qdrant_db)
    """
    env_url = os.getenv("QDRANT_URL")
    env_api_key = os.getenv("QDRANT_API_KEY")

    if url or env_url:
        return QdrantClient(url=url or env_url, api_key=api_key or env_api_key)

    if path:
        path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


# ==============================================================================
# Create a collection in Qdrant Vector base
# ==============================================================================


def init_textbook_collection(
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """
    Initializes a Qdrant collection configured for dual-vector hybrid search:
    - Dense vectors: 384 dimensions, Cosine distance (BAAI/bge-small-en-v1.5)
    - Sparse vectors: BM25 index (Qdrant/bm25)
    - Payload field indexes for fast filtering
    """
    exists = client.collection_exists(collection_name)
    if exists and recreate:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        print(f"[QDRANT] Collection '{collection_name}' created.")

        # Create payload field indexes for rapid filtered retrieval
        filter_fields = [
            ("grade", models.PayloadSchemaType.INTEGER),
            ("subject", models.PayloadSchemaType.KEYWORD),
            ("medium", models.PayloadSchemaType.KEYWORD),
            ("edition", models.PayloadSchemaType.KEYWORD),
            ("chapter_number", models.PayloadSchemaType.INTEGER),
            ("book_page", models.PayloadSchemaType.INTEGER),
            ("document_name", models.PayloadSchemaType.KEYWORD),
        ]
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for field_name, field_type in filter_fields:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
                except Exception:
                    pass
        print(f"[QDRANT] Payload indexes configured for: {[f[0] for f in filter_fields]}")


# ==============================================================================
# Embed and store chunks
# ==============================================================================


def upsert_chunks(
    chunks: list[TextNode],
    client: QdrantClient | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    batch_size: int = 64,
) -> int:
    """
    Embeds chunks (dense + sparse BM25) and upserts them into Qdrant in batches.
    """
    if not chunks:
        print("[QDRANT] No chunks provided for upsert.")
        return 0

    if client is None:
        client = get_qdrant_client()

    init_textbook_collection(client, collection_name=collection_name)

    total_chunks = len(chunks)
    print(f"[QDRANT] Embedding and preparing {total_chunks} points...")
    points = prepare_qdrant_points(chunks, batch_size=batch_size)

    # Upsert in batches of 100 points
    upsert_batch_size = 100
    for i in range(0, len(points), upsert_batch_size):
        batch = points[i : i + upsert_batch_size]
        client.upsert(collection_name=collection_name, points=batch)
        print(
            f"[QDRANT UPSERT] Indexed {min(i + upsert_batch_size, total_chunks)}/{total_chunks} points."
        )

    print(
        f"[QDRANT SUCCESS] Successfully indexed {len(points)} points into '{collection_name}'."
    )
    return len(points)


# ==============================================================================
# Perform Hybrid search for given query
# ==============================================================================


def hybrid_search(
    query: str,
    client: QdrantClient | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 5,
    filter_conditions: dict[str, Any] | None = None,
) -> list[models.ScoredPoint]:
    """
    Executes hybrid search combining Dense (semantic) + Sparse (BM25 keyword)
    using Reciprocal Rank Fusion (RRF).
    """
    if client is None:
        client = get_qdrant_client()

    dense_query = embed_query_dense(query)
    sparse_query = embed_query_sparse(query)

    query_filter = None
    if filter_conditions:
        must_conditions = []
        for key, val in filter_conditions.items():
            must_conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=val),
                )
            )
        query_filter = models.Filter(must=must_conditions)

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=sparse_query,
                using="sparse",
                limit=top_k * 2,
                filter=query_filter,
            ),
            models.Prefetch(
                query=dense_query,
                using="dense",
                limit=top_k * 2,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
    )
    return results.points
