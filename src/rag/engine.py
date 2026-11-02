from pathlib import Path

import chromadb
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import init_llama_index

DEFAULT_CHROMA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "chroma_db")


def load_query_engine(
    collection_name: str = "history-G-11-E",
    persist_dir: str = DEFAULT_CHROMA_DIR,
    similarity_top_k: int = 4,
):
    init_llama_index()

    # connect to exisiting chroma client
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_collection(name=collection_name)

    # Reconstruct the existing vectore store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Prompt
    prompt_tmpl = (
        "You are an academic tutor helping high school students.\n"
        "Answer the student's question strictly using the provided context blocks.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Rules:\n"
        "1. Base your answer solely on the facts provided above; do not speculate.\n"
        "2. format your response with clean Markdown tables only if the context contains tabular data, \n"
        "3. If multiple sources discuss the topic, synthesize them chronologically.\n\n"
        "Question: {query_str}\n"
    )

    qa_prompt = PromptTemplate(prompt_tmpl)

    # Build Query engine

    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, response_mode="compact"
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

    return query_engine
