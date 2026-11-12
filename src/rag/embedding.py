import uuid

from fastembed import SparseTextEmbedding, TextEmbedding
from llama_index.core.schema import TextNode
from qdrant_client import models

from .core.config import DENSE_MODEL_NAME, SPARSE_MODEL_NAME, model_cache_path

# ==============================================================================
# Model Singletons (Lazy Loaded)
# ==============================================================================

_dense_embedder: TextEmbedding | None = None
_sparse_embedder: SparseTextEmbedding | None = None


# Lazy Loading


def get_dense_embedder() -> TextEmbedding:
    """Returns the cached dense embedding model (BAAI/bge-small-en-v1.5)."""
    global _dense_embedder
    if _dense_embedder is None:
        _dense_embedder = TextEmbedding(
            model_name=DENSE_MODEL_NAME,
            cache_dir=str(model_cache_path),
            threads=4,
        )
    return _dense_embedder


def get_sparse_embedder() -> SparseTextEmbedding:
    """Returns the cached sparse BM25 embedding model (Qdrant/bm25)."""
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = SparseTextEmbedding(
            model_name=SPARSE_MODEL_NAME,
            cache_dir=str(model_cache_path),
            threads=4,
        )
    return _sparse_embedder


# ==============================================================================
# Batch Embedding Functions
# ==============================================================================


def embed_dense(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    64 chunks become a batch and all batch runs simultaneously in RAM
    Computes dense 384-dimensional embeddings for a list of texts using FastEmbed.
    Returns: list of float vectors.
    """
    if not texts:
        return []
    embedder = get_dense_embedder()
    return [vec.tolist() for vec in embedder.embed(texts, batch_size=batch_size)]


def embed_sparse(texts: list[str], batch_size: int = 64) -> list[models.SparseVector]:
    """
    Computes sparse BM25 token weight embeddings for a list of texts using FastEmbed.
    Returns: list of Qdrant SparseVector objects (indices + values).
    """
    if not texts:
        return []
    embedder = get_sparse_embedder()
    sparse_vectors: list[models.SparseVector] = []
    for s in embedder.embed(texts, batch_size=batch_size):
        sparse_vectors.append(
            models.SparseVector(
                indices=s.indices.tolist(),
                values=s.values.tolist(),
            )
        )
    return sparse_vectors


def embed_query_dense(query: str) -> list[float]:
    """Generates a dense vector for a single search query."""
    return embed_dense([query], batch_size=1)[0]


def embed_query_sparse(query: str) -> models.SparseVector:
    """Generates a sparse BM25 vector for a single search query."""
    return embed_sparse([query], batch_size=1)[0]


# ==============================================================================
# Chunk-to-Point Conversion
# ==============================================================================


def prepare_qdrant_points(
    chunks: list[TextNode],
    batch_size: int = 64,
) -> list[models.PointStruct]:
    """
    Takes TextNode chunks from chunking.py and produces Qdrant PointStructs:
    1. Embeds chunk texts in batches with Dense (bge-small) and Sparse (bm25).
    2. Generates deterministic UUID5 from chunk.id_ (preserving identity across re-runs).
    3. Retains pure textbook text in payload along with full metadata.
    """
    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    dense_vectors = embed_dense(texts, batch_size=batch_size)
    sparse_vectors = embed_sparse(texts, batch_size=batch_size)

    points: list[models.PointStruct] = []
    for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
        # Qdrant requires UUID string or int ID. We derive a deterministic UUID5 from chunk.id_
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id_))

        payload = {
            "text": chunk.text,
            "chunk_id": chunk.id_,
            **chunk.metadata,
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec,
                },
                payload=payload,
            )
        )

    return points
