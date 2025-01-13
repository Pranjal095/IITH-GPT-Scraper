import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import numpy as np
import time
import json
import re
from sentence_transformers import SentenceTransformer
import faiss
import json

def perform_scraping(url):
    chrome_driver_path = "./chromedriver"
    chrome_binary_path = "./chrome-headless-shell"
    options = webdriver.ChromeOptions()
    options.binary_location = chrome_binary_path
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(chrome_driver_path), options=options)

    try:
        driver.get(url)
        print("Page loaded")
        time.sleep(10)
        return driver.page_source
    finally:
        driver.quit()

def clean_text(text):
    return re.sub(r'\s+',' ',text).strip()

def parse_content(html):
    inline_tags = ["span","a","em","strong","i","b","u"]
    for tag in inline_tags:
        html = re.sub(r'<{0}[^>]*>(.*?)</{0}>'.format(tag),r'\1',html, flags=re.DOTALL)

    html = clean_text(html)

    soup = BeautifulSoup(html,"html.parser")
    for tag in soup(["script","style","footer","nav","header"]):
        tag.decompose()


    grouped_data = []
    current_group = None

    for element in soup.descendants:
        if element.name and element.name in ["h1","h2","h3","h4","h5"]:
            if current_group:
                grouped_data.append(current_group)
            current_group = {
                "title": clean_text(element.get_text()),
                "content": []
            }
        elif current_group and element.string and clean_text(element.string):
            cleaned_text = clean_text(element.string)
            if cleaned_text not in current_group["content"] and cleaned_text not in current_group["title"]:
                current_group["content"].append(cleaned_text)

    if current_group and current_group["content"]:
        grouped_data.append(current_group)

    return grouped_data

def save_data(grouped_data,output_file="output.json"):
    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(grouped_data,f,ensure_ascii=False,indent=4)

#Using a sentence transformer as the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

faiss_index = None

def build_index(grouped_data):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = []
    embeddings = []

    for group in grouped_data:
        title = group['title']
        content = " ".join(group['content'])
        
        #Generate separate embeddings for title and content
        title_embedding = model.encode(title,normalize_embeddings=True)
        content_embedding = model.encode(content,normalize_embeddings=True)
        
        #Combine embeddings while giving higher weightage to the title
        combined_embedding = (1.2*title_embedding+content_embedding)/2.2
        embeddings.append(combined_embedding)
        
        texts.append(f"{title} | {content}")

    #Create FAISS index
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return {"index": index, "texts": texts}

def retrieve_relevant_sections(query, indexed_data,top_k=5):
    index = indexed_data["index"]
    texts = indexed_data["texts"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_embedding = model.encode(query,normalize_embeddings=True)    
    distances,indices = index.search(np.array([query_embedding]),k=top_k)    
    relevant_sections = [texts[i] for i in indices[0] if i < len(texts)]
    print(relevant_sections)

    return relevant_sections