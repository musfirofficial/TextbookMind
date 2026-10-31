import chromadb
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from llamaindex_config import init_llama_index

init_llama_index()

is_running = True


def load_query_engine(
    collection_name: str = "history-G-11-E",
    persist_dir: str = "./chroma_db",
    similarity_top_k: int = 4,
):
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


def ask_text_book(query_engine, question: str):
    response = query_engine.query(question)

    print(f"\nAnswer:{str(response).strip()}")

    print("Sources:\n")
    seen_pages = set()
    for index, node in enumerate(response.source_nodes, start=1):
        meta = node.metadata
        page = meta.get("page_number", "Unknown")
        score = node.score if node.score is not None else 0.0

        seen_pages.add(page)
        print(f"[{index}] Page {page}| Similarity Score: {score:.3f}")

    print(f"Cited Pages: {sorted(seen_pages)}")


if __name__ == "__main__":
    engine = load_query_engine()
    while is_running:
        question = input("Question:")
        if question.lower() == "exit":
            break

        ask_text_book(engine, question=question)
