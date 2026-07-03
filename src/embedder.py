# src/embedder.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Loads and returns the sentence-transformer embedding model.
    This model converts text into 384-dimensional vectors.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model


# Quick test
if __name__ == "__main__":
    from data_loader import load_resume
    from chunker import chunk_documents

    # Reuse Part 3 + Part 4 pipeline
    docs = load_resume("data/resume.pdf")
    chunks = chunk_documents(docs)

    # Load embedding model
    embeddings = get_embedding_model()

    # Embed just the first chunk as a test
    sample_text = chunks[0].page_content
    vector = embeddings.embed_query(sample_text)

    print(f"\nSample chunk text:\n{sample_text[:150]}...")
    print(f"\nEmbedding vector length: {len(vector)}")
    print(f"First 10 values of vector:\n{vector[:10]}")