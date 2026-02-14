def get_retriever(vector_store, k=5):
    """
    Returns a retriever object from the vector store.
    """
    return vector_store.as_retriever(search_kwargs={"k": k})

def retrieve_context(query, vector_store, k=5):
    """
    Performs similarity search and returns concatenated context.
    """
    docs = vector_store.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs
