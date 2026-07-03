from vector_store import load_vector_store

def get_retriever(k=3):
    """
    Loads the persisted vector store and returns it as a Retriever object.
    k = number of top relevant chunks to fetch per query.
    """
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever


# Quick test
if __name__ == "__main__":
    retriever = get_retriever(k=3)

    test_queries = [
        "What internship experience do you have?",
        "What programming languages do you know?",
        "Tell me about your machine learning projects"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"QUERY: {query}")
        print('='*60)

        results = retriever.invoke(query)

        for i, doc in enumerate(results):
            print(f"\n[Chunk {i+1}]")
            print(doc.page_content[:200])