# src/vector_store.py

from langchain_chroma import Chroma
from embedder import get_embedding_model

PERSIST_DIRECTORY = "chroma_db"  # folder where the database will be saved on disk

def create_vector_store(chunks):
    """
    Creates a new Chroma vector store from document chunks,
    embeds them, and persists to disk.
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )

    print(f"Vector store created and saved to '{PERSIST_DIRECTORY}/'")
    return vector_store


def load_vector_store():
    """
    Loads an existing Chroma vector store from disk
    (use this once the store has already been created).
    """
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )

    print(f"Vector store loaded from '{PERSIST_DIRECTORY}/'")
    return vector_store


# Quick test
if __name__ == "__main__":
    from data_loader import load_resume
    from chunker import chunk_documents

    docs = load_resume("data/resume.pdf")
    chunks = chunk_documents(docs)

    # Build and save the vector store (only needs to run once per resume version)
    vector_store = create_vector_store(chunks)

    # Test a similarity search
    query = "What internship experience do you have?"
    results = vector_store.similarity_search(query, k=2)  # top 2 matches

    print(f"\n--- Top matches for query: '{query}' ---")
    for i, result in enumerate(results):
        print(f"\nMatch {i+1}:")
        print(result.page_content)
        print("-" * 50)