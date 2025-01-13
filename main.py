import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from scraper import parse_content, build_index
from llm import prompt_llm

# Load and parse multiple JSON files
def load_json_files(data_dir):
    """
    Load all JSON files from the specified directory and combine their contents.
    """
    combined_data = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    for entry in data:
                        entry["source_file"] = file_name  # Add the file name as context
                    combined_data.extend(data)
                except json.JSONDecodeError:
                    print(f"Error decoding {file_path}. Skipping...")
    return combined_data

# Build a unified FAISS index for all data
def build_combined_index(combined_data):
    """
    Build a FAISS index from the combined dataset, including file name context.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    embeddings = []

    for group in combined_data:
        source_file = group.get("source_file", "Unknown File")
        title = group.get("title", "")
        content = " ".join(group.get("content", []))
        
        # Combine file name, title, and content
        combined_text = f"Source: {source_file} | Title: {title} | Content: {content}"
        
        # Generate separate embeddings for title, content, and file name
        title_embedding = model.encode(title, normalize_embeddings=True)
        content_embedding = model.encode(content, normalize_embeddings=True)
        file_embedding = model.encode(source_file, normalize_embeddings=True)
        
        # Weighted combination of embeddings
        combined_embedding = (1.2 * file_embedding + 1.2 * title_embedding + content_embedding) / 3.4
        embeddings.append(combined_embedding)
        
        # Store combined text for retrieval
        texts.append(combined_text)

    # Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return {"index": index, "texts": texts}

# Query the FAISS index
def retrieve_relevant_sections(query, indexed_data, top_k=3):
    """
    Retrieve the most relevant sections for a given query.
    """
    index = indexed_data["index"]
    texts = indexed_data["texts"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Generate query embedding
    query_embedding = model.encode(query, normalize_embeddings=True)
    
    # Search in the FAISS index
    distances, indices = index.search(np.array([query_embedding]), k=top_k)
    relevant_sections = [texts[i] for i in indices[0] if i < len(texts)]
    return relevant_sections

# Main terminal program
def main():
    print("Welcome to the Terminal Query-Response Program using LLM-RAG.")
    
    # Specify the directory containing JSON files
    data_dir = input("Enter the directory path containing JSON files: ").strip()
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        print("Invalid directory path. Exiting.")
        return

    # Load and combine data
    print("Loading JSON files...")
    combined_data = load_json_files(data_dir)
    if not combined_data:
        print("No valid data found in the directory. Exiting.")
        return

    print("Building FAISS index...")
    indexed_data = build_combined_index(combined_data)
    print("Index built successfully. Ready to handle queries.")

    # Query loop
    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break

        print("Retrieving relevant sections...")
        relevant_sections = retrieve_relevant_sections(query, indexed_data)

        print("Generating response...")
        response = prompt_llm(relevant_sections, query)
        print("\nResponse:\n", response)

if __name__ == "__main__":
    main()