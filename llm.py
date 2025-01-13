from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

#Using Llama3.1 as the LLM
model = OllamaLLM(model="llama3.1")

#Template for querying the LLM
template = (
    "You are tasked with answering the following query based on the provided context, set in the domain of Indian Institute of Technology Hyderabad (IITH):\n\n"
    "Query: {query}\n\n"
    "Context:\n{combined_content}\n\n"
    "Guidelines:\n"
    "1. Provide a direct answer providing the exact details asked in the query.\n"
    "2. Use a professional tone.\n"
    "3. Address all relevant aspects of the query.\n"
)

def prompt_llm(relevant_sections, query):
    combined_content = "\n\n".join(relevant_sections).strip()
    formatted_prompt = template.format(
        query=query,
        combined_content=combined_content
    )
    response = model.invoke(formatted_prompt)
    
    return response.strip()