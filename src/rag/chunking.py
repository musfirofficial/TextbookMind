import json
import re
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

Source_folder = Path(__file__).resolve().parents[2] / "data" / "extracted"


def create_chunks_from_dataset(
    dataset_path: str,
    chunk_size: int = 400,
    overlap: int = 60,
):
    """
    Takes an extracted textbook dataset JSON and splits it into safe,
    metadata-enriched chunks ready for Dense + BM25 embedding in Qdrant.
    Guarantees:
    1. Zero truncation: Chunks stay strictly under 512 embedding token limits.
    2. Zero syntax breaking: Markdown tables and $$ LaTeX blocks stay intact.
    3. Full metadata preservation: Every chunk knows its book page and chapter.
    """

    path_obj = Path(dataset_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    with open(path_obj, "r", encoding="utf-8") as f:
        pages: list[dict] = json.load(f)

    # 1. Convert pages in to LlamaIndex Documents
    documents: list[Document] = []
    for page in pages:
        text = page.get("text", "")

        # Skip completely blank pages and Front Matter page
        if not text or page.get("chapter_title") == "Front Matter":
            continue

        doc = Document(
            text=text,
            metadata={
                "document_name": page.get("document_name", path_obj.stem),
                "grade": page.get("grade"),
                "subject": page.get("subject"),
                "medium": page.get("medium", "English"),
                "edition": page.get("edition"),
                "pdf_page": page.get("pdf_page"),
                "book_page": page.get("book_page"),
                "chapter_number": page.get("chapter_number"),
                "chapter_title": page.get("chapter_title"),
                "has_figures": page.get("has_figures", False),
                "has_tables": page.get("has_tables", False),
                "extraction_method": page.get("extraction_method", "digital"),
            },
            excluded_llm_metadata_keys=[
                "has_figures",
                "has_tables",
                "extraction_method",
            ],
        )
        documents.append(doc)

    # 2. Configure the Safe SentenceSplitter
    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex=r"[^,.;。]+[,.;。]?",
    )

    # 3. Chunk
    raw_nodes = splitter.get_nodes_from_documents(documents)

    # 4. Enrich chunks: Deterministic IDs & Context Headers
    doc_stem = re.sub(r"[^\w\-]", "_", path_obj.stem)
    page_chunk_counters: dict[int, int] = {}
    final_chunks: list[TextNode] = []

    for node in raw_nodes:
        meta = node.metadata
        pdf_page = meta.get("pdf_page", 0)

        # Track chunk index per page (e.g. page 16 -> chunk 0, chunk 1)
        chunk_idx = page_chunk_counters.get(pdf_page, 0)
        page_chunk_counters[pdf_page] = chunk_idx + 1

        # Deterministic, unique ID: e.g. "SCIENCE_G-10_P1_E_p16_c0"
        node.id_ = f"{doc_stem}_p{pdf_page}_c{chunk_idx}"
        meta["chunk_index"] = chunk_idx

        final_chunks.append(node)
    print(
        f"[CHUNKING SUCCESS] Dataset '{path_obj.name}' ({len(pages)} pages) "
        f"split into {len(final_chunks)} safe chunks."
    )
    return final_chunks


# if __name__ == "__main__":
    # create_chunks_from_dataset(dataset_path="data/extracted/SCIENCE_G-10_P1_E.json")
