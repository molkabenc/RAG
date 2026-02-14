from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt():
    """
    Returns the standard RAG prompt template.
    """
    template = """
    You are an AI assistant helping with questions based on specific context.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and professional.

    Context:
    {context}

    Question: {question}

    Answer:
    """
    return ChatPromptTemplate.from_template(template)
