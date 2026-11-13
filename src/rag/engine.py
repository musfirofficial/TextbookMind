from typing import Any

from src.rag.core.config import (
    DEFAULT_COLLECTION_NAME,
    GROQ_MAIN_MODEL,
    get_groq_client,
)
from src.rag.core.prompts import main_prompt
from src.rag.vector_store import get_qdrant_client, hybrid_search


def _format_context(points: list) -> tuple[str, list[dict]]:
    """Formats retrieved chunks into clean prompt context and citations."""
    blocks, citations = [], []
    for idx, p in enumerate(points, 1):
        meta = p.payload or {}
        chunk_id = meta.get("chunk_id", "")
        doc = (
            chunk_id.rsplit("_p", 1)[0]
            if "_p" in chunk_id
            else meta.get("document_name", "Textbook")
        )
        ch_num = meta.get("chapter_number")
        ch_title = meta.get("chapter_title", "General")
        page = meta.get("book_page")

        blocks.append(
            f"[Source {idx} | {doc} | Chapter {ch_num}: {ch_title} | Book Page {page}]\n{meta.get('text', '')}"
        )
        citations.append(
            {
                "source": idx,
                "document": doc,
                "chapter_number": ch_num,
                "chapter_title": ch_title,
                "book_page": page,
                "chunk_id": chunk_id,
            }
        )
    return "\n\n---------------------\n\n".join(blocks), citations


def query(
    question: str,
    top_k: int = 5,
    filter_conditions: dict[str, Any] | None = None,
    stream: bool = False,
    collection_name: str = DEFAULT_COLLECTION_NAME,
):
    """
    RAG Query:
    - stream=False -> returns: (answer_string, citations_list)
    - stream=True  -> returns: (token_generator, citations_list)
    """
    client = get_qdrant_client()
    points = hybrid_search(
        query=question,
        client=client,
        collection_name=collection_name,
        top_k=top_k,
        filter_conditions=filter_conditions,
    )

    if not points:
        fallback = "Based on the provided textbook excerpt, there is not enough information to answer this question."
        return (iter([fallback]), []) if stream else (fallback, [])

    context, citations = _format_context(points)
    prompt = (
        main_prompt().replace("{context_str}", context).replace("{query_str}", question)
    )

    groq = get_groq_client()
    response = groq.chat.completions.create(
        model=GROQ_MAIN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        stream=stream,
    )

    if stream:

        def stream_gen():
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        return stream_gen(), citations

    return response.choices[0].message.content, citations
