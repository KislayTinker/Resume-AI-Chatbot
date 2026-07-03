# src/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=400, chunk_overlap=50):
    """
    Splits loaded documents into smaller overlapping chunks.
    Returns a list of Document objects (same structure as loader output).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]  # order of preference when splitting
    )
    
    chunks = splitter.split_documents(documents)
    
    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


# Quick test
if __name__ == "__main__":
    from data_loader import load_resume
    
    docs = load_resume("data/resume.pdf")
    chunks = chunk_documents(docs)
    
    print("\n--- Preview of first 3 chunks ---")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\nChunk {i+1} (length: {len(chunk.page_content)} chars):")
        print(chunk.page_content)
        print("-" * 50)