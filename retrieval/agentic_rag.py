from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.tools import Tool
from .retriever import retrieve_context

def get_agentic_rag_executor(llm, vector_store):
    """
    Creates an agent executor for RAG.
    """
    
    # Define the tool
    def search_docs(query):
        context, _ = retrieve_context(query, vector_store)
        return context

    tools = [
        Tool(
            name="Search_Knowledge_Base",
            func=search_docs,
            description="Useful for when you need to answer questions about the provided documents. Input should be a search query."
        )
    ]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/react")

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # Create an agent executor by passing in the agent and tools
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("Agentic RAG components defined.")
