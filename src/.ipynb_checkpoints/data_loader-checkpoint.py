from langchain_community.document_loaders import PyPDFLoader

def load_resume(file_path: str):
    """
    Loads a PDF file and returns a list of LangChain Document objects.
    Each Document contains the text of one page + metadata (like page number).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    print(f"Loaded {len(documents)} page(s) from {file_path}")
    return documents


# Quick test — only runs when you execute this file directly
if __name__ == "__main__":
    docs = load_resume("data/resume.pdf")
    
    # Let's inspect the first page to confirm it worked
    print("\n--- Preview of Page 1 ---")
    print(docs[0].page_content[:500])  # first 500 characters
    print("\n--- Metadata ---")
    print(docs[0].metadata)