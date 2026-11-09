from pathlib import Path

import chromadb
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import get_condense_llm, init_llama_index
from .prompts import condense_prompt, main_prompt

DEFAULT_CHROMA_DIR = str(Path(__file__).resolve().parents[2] / "data" / "chroma_db")


def load_query_engine(collection_name: str, persist_dir: str, similarity_top_k: int):
    init_llama_index()

    # connect to exisiting chroma client
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_collection(name=collection_name)

    # Reconstruct the existing vectore store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    # Prompt
    prompt_tmpl = main_prompt()

    qa_prompt = PromptTemplate(prompt_tmpl)

    # Build Query engine

    query_engine = index.as_query_engine(
        similarity_top_k=similarity_top_k, response_mode="compact"
    )
    query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})

    return query_engine


def load_chat_engine(
    collection_name: str = "Full",
    persist_dir: str = DEFAULT_CHROMA_DIR,
    similarity_top_k: int = 6,
):

    # Base Query Engine
    query_engine = load_query_engine(
        collection_name=collection_name,
        persist_dir=persist_dir,
        similarity_top_k=similarity_top_k,
    )

    # Get the condense LLM
    condense_llm = get_condense_llm()

    # Create a memory buffer for the chat engine
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    # Condense Prompt
    condense_prompt_tmpl = PromptTemplate(condense_prompt())

    # Define chat engine
    chat_engine = CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        llm=condense_llm,
        memory=memory,
        condense_question_prompt=condense_prompt_tmpl,
        verbose=True,  # Prints rewritten query in terminal so you can see it working!
    )

    return chat_engine
