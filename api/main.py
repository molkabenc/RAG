from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indexing.embedder import get_embeddings
from indexing.vector_store import load_vector_store
from retrieval.retriever import retrieve_context
from retrieval.prompt import get_rag_prompt
from retrieval.generator import get_llm, generate_response

app = FastAPI(title="RAG AI Backend")

class QueryRequest(BaseModel):
    question: str
    model: str = "llama-3.3-70b-versatile"

# Global state for components (should be properly managed in production)
embeddings = None
vector_store = None
llm = None
prompt_template = None

@app.on_event("startup")
async def startup_event():
    global embeddings, vector_store, llm, prompt_template
    try:
        embeddings = get_embeddings()
        vector_store = load_vector_store(embeddings, persist_directory="./faiss_db")
        llm = get_llm()
        prompt_template = get_rag_prompt()
        print("Backend components loaded.")
    except Exception as e:
        print(f"Error during startup: {e}")

@app.get("/")
def read_root():
    return {"status": "online", "model": "RAG System"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    if not vector_store:
        raise HTTPException(status_code=500, detail="Vector store not initialized. Please index documents first.")
    
    try:
        context, docs = retrieve_context(request.question, vector_store)
        answer = generate_response(llm, prompt_template, context, request.question)
        
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        
        return {
            "answer": answer,
            "sources": list(set(sources))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
