import streamlit as st
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loader import load_resume
from chunker import chunk_documents
from vector_store import create_vector_store, PERSIST_DIRECTORY
from rag_chain import build_rag_chain

st.set_page_config(page_title="Kislay's Resume Chatbot", page_icon="💼")

st.title("💼 Ask Me About Kislay's Resume")
st.caption("A RAG-powered chatbot answering questions grounded in Kislay Tinker's actual resume.")


@st.cache_resource
def initialize_vector_store():
    """
    Builds the vector store if it doesn't already exist (e.g., fresh deployment).
    Skips rebuilding if it's already been created (e.g., local dev after first run).
    """
    if not os.path.exists(PERSIST_DIRECTORY):
        docs = load_resume("data/resume.pdf")
        chunks = chunk_documents(docs)
        create_vector_store(chunks)


@st.cache_resource
def load_chain():
    initialize_vector_store()
    return build_rag_chain()

chain = load_chain()


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history_tuples" not in st.session_state:
    st.session_state.chat_history_tuples = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_question = st.chat_input("Ask a question about Kislay's background...")

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = chain.invoke({
                "question": user_question,
                "chat_history": st.session_state.chat_history_tuples
            })
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history_tuples.append((user_question, answer))