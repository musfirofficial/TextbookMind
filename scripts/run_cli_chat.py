from rag.engine import load_query_engine

is_running = True


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
