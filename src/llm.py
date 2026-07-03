import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

def get_llm():
    """
    Returns a configured Groq LLM instance.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Check your .env file.")

    llm = ChatGroq(
        api_key=api_key,
        model="llama-3.3-70b-versatile",  # strong, free, general-purpose model
        temperature=0.3  # low temperature = more focused/factual, less "creative"
    )

    return llm


# Quick test
if __name__ == "__main__":
    llm = get_llm()

    response = llm.invoke("Say hello and confirm you're working correctly in one sentence.")
    print(response.content)