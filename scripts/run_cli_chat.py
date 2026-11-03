from rag.engine import load_chat_engine

is_running = True


def ask_text_book(chat_engine, question: str):
    response = chat_engine.chat(question)

    print(f"\nAnswer:\n{str(response).strip()}\n")

    if hasattr(response, "source_nodes") and response.source_nodes:
        print("Sources:")
        for idx, node in enumerate(response.source_nodes, start=1):
            page = node.metadata.get("page_number", "Unknown")
            score = node.score if node.score is not None else 0.0
            print(f"[{idx}] Page {page} | Score: {score:.3f}")


if __name__ == "__main__":
    engine = load_chat_engine()
    print("Academic Tutor Ready! Type 'exit' to quit.\n")
    while is_running:
        question = input("Question: ")
        if question.lower().strip() == "exit":
            break
        if not question.strip():
            continue
        ask_text_book(engine, question=question)
