import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_llm(model_name="llama-3.3-70b-versatile", temperature=0):
    """
    Returns the Groq LLM instance.
    Uses GROQ_API_KEY or OPENAI_API_KEY (as user currently has it there) from .env
    """
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    return ChatGroq(
        model_name=model_name, 
        temperature=temperature,
        groq_api_key=api_key
    )

def generate_response(llm, prompt, context, question):
    """
    Generates a response using the LLM, prompt, context, and question.
    """
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})

if __name__ == "__main__":
    # Mock test
    from prompt import get_rag_prompt
    prompt = get_rag_prompt()
    # llm = get_llm() # Needs API key
    print("Generator components initialized.")
