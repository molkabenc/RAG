import streamlit as st
import requests
import os
import sys

# Add current directory to path for relative imports if needed
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from indexing.loader import load_documents
from indexing.splitter import split_documents
from indexing.embedder import get_embeddings
from indexing.vector_store import create_vector_store

st.set_page_config(page_title="RAG AI Assistant", layout="wide")

st.title("🚀 RAG AI Knowledge Assistant")
st.markdown("---")

# Sidebar for indexing
with st.sidebar:
    st.header("📂 Knowledge Management")
    doc_dir = st.text_input("Document Directory", value="data/documents")
    
    if st.button("🔄 Index Documents"):
        with st.spinner("Indexing..."):
            try:
                if not os.path.exists(doc_dir):
                    os.makedirs(doc_dir)
                
                docs = load_documents(doc_dir)
                if not docs:
                    st.warning("No documents found in directory.")
                else:
                    chunks = split_documents(docs)
                    embeddings = get_embeddings()
                    create_vector_store(chunks, embeddings, persist_directory="./faiss_db")
                    st.success(f"Successfully indexed {len(docs)} documents into {len(chunks)} chunks!")
            except Exception as e:
                st.error(f"Error: {e}")

st.header("💬 Ask your documents")
query = st.text_input("Enter your question:")

if st.button("Send") or query:
    if query:
        with st.spinner("Generating answer..."):
            try:
                # In a real app, this should call the FastAPI backend
                # For this demo, we'll implement the logic directly if needed or use the backend
                
                # Option A: Call Local API (uncomment if uvicorn is running)
                # response = requests.post("http://localhost:8000/ask", json={"question": query})
                # if response.status_code == 200:
                #     data = response.json()
                #     st.write("### Answer")
                #     st.info(data["answer"])
                #     st.write("### Sources")
                #     for s in data["sources"]:
                #         st.write(f"- {s}")
                
                # Option B: Direct call for faster evaluation
                from indexing.embedder import get_embeddings
                from indexing.vector_store import load_vector_store
                from retrieval.retriever import retrieve_context
                from retrieval.prompt import get_rag_prompt
                from retrieval.generator import get_llm, generate_response
                
                embeddings = get_embeddings()
                vs = load_vector_store(embeddings, persist_directory="./faiss_db")
                if vs:
                    context, docs = retrieve_context(query, vs)
                    llm = get_llm(model_name="llama-3.3-70b-versatile")
                    prompt = get_rag_prompt()
                    answer = generate_response(llm, prompt, context, query)
                    
                    st.write("### Answer")
                    st.info(answer)
                    
                    with st.expander("📚 Sources & References"):
                        for doc in docs:
                            st.markdown(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                            st.write(doc.page_content[:200] + "...")
                            st.markdown("---")
                else:
                    st.error("Vector store not found. Please index documents first.")
                    
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question.")
