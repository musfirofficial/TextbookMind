from pathlib import Path

import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from llamaindex_config import init_llama_index
from util import extract_student_materials

init_llama_index()


def index_and_persist_to_chroma(file: str, persist_dir: str = "./chroma_db"):

    # Extract File
    pages = extract_student_materials(file)
    file_name = Path(file)

    # Initalize ChromaDB
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(
        name=file_name.stem.replace(" ", "-")
    )

    # Wire ChromaDB to LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Wrap Pages in to LlamaIndex
    documents = [
        Document(
            text=page["text"],
            metadata={
                "file_name": page["document_name"],
                "page_number": page["page_number"],
                "has_figure": page["has_figures"],
            },
            excluded_embd_metadata_keys=["has_figure"],
            exclude_llm_metadata_keys=["has_figure"],
        )
        for page in pages
    ]

    # Chunking
    node_parser = SentenceSplitter(
        chunk_size=700, chunk_overlap=100, paragraph_separator="\n\n"
    )

    # Build Vector
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        transformations=[node_parser],
        show_progress=True,
    )

    print(
        f"Successfully indexed and persisted {len(documents)} pages to '{persist_dir}'."
    )
    return index


if __name__ == "__main__":
    index_and_persist_to_chroma("history G-11 E.pdf")
