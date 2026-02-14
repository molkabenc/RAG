from langchain_community.vectorstores import FAISS
import os

def create_vector_store(chunks, embeddings, persist_directory="./faiss_db"):
    """
    Creates and saves a FAISS vector store from document chunks.
    """
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vector_store.save_local(persist_directory)
    return vector_store

def load_vector_store(embeddings, persist_directory="./faiss_db"):
    """
    Loads an existing FAISS vector store.
    """
    if not os.path.exists(persist_directory):
        return None
    
    vector_store = FAISS.load_local(
        persist_directory, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return vector_store

if __name__ == "__main__":
    from loader import load_documents
    from splitter import split_documents
    from embedder import get_embeddings
    
    docs = load_documents("data/documents")
    if docs:
        chunks = split_documents(docs)
        embeddings = get_embeddings("huggingface") # Local for test
        
        store = create_vector_store(chunks, embeddings)
        print("FAISS vector store created and saved.")
    else:
        print("No documents found to index.")
