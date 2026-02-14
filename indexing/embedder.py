import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

def get_embeddings(model_type=None):
    """
    Returns the embedding model based on selection.
    Defaults to huggingface (local) if no OpenAI key is found or if a Groq key is used.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Auto-detect key type
    if model_type is None:
        if api_key and api_key.startswith("sk-"):
            model_type = "openai"
        else:
            model_type = "huggingface"

    if model_type == "openai":
        return OpenAIEmbeddings()
    else:
        # Local model (no API key needed)
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    # Test
    try:
        embedder = get_embeddings("huggingface") # local test
        print("Embedding model initialized.")
    except Exception as e:
        print(f"Error: {e}")
