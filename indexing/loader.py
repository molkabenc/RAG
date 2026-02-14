import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader

def load_documents(directory_path: str):
    """
    Loads documents from a specified directory.
    Supports PDF and TXT files.
    """
    loaders = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }

    documents = []
    for file in os.listdir(directory_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in loaders:
            loader_path = os.path.join(directory_path, file)
            loader = loaders[ext](loader_path)
            documents.extend(loader.load())
    
    return documents

if __name__ == "__main__":
    # Test loading
    sample_dir = "data/documents"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    
    # Create a dummy txt for testing
    test_file = os.path.join(sample_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("This is a test document for the RAG system.")
    
    docs = load_documents(sample_dir)
    print(f"Loaded {len(docs)} documents.")
