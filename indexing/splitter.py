from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    """
    Splits documents into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

if __name__ == "__main__":
    from loader import load_documents
    docs = load_documents("data/documents")
    chunks = split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")
