from rag import query

filter_conditions = {"grade": 10, "subject": "Science P1"}

while True:
    user_query = input("\nAsk something (type exit to close): ")
    if user_query.strip().lower() == "exit":
        break

    if not user_query.strip():
        continue

    print("\nSearching textbooks & thinking...\n")

    # 👉 1. Receive the answer and citations
    answer, citations = query(question=user_query, filter_conditions=filter_conditions)

    # 👉 2. Print the answer!
    print("=" * 60)
    print(answer)
    print("=" * 60)

    # 👉 3. Print the source pages
    if citations:
        print("\n📚 Sources:")
        for c in citations:
            print(
                f"  • {c['document']} | Ch.{c['chapter_number']}: {c['chapter_title']}"
                f" | Page {c['book_page']}"
            )
