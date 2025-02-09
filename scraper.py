from bs4 import BeautifulSoup
import numpy as np
import json
import re
from sentence_transformers import SentenceTransformer
import faiss
import json
import time
import logging
import os
from DrissionPage import ChromiumPage, ChromiumOptions
from urllib.parse import urljoin
from cloudfare_bypasser import CloudflareBypasser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cloudflare_bypass.log', mode='w')
    ]
)

def get_chromium_options(browser_path: str,arguments: list) -> ChromiumOptions:
    options = ChromiumOptions().auto_port()
    options.set_paths(browser_path=browser_path)
    for argument in arguments:
        options.set_argument(argument)
    return options

def scrape_page(url):
    browser_path = os.getenv('CHROME_PATH',"/usr/bin/google-chrome")

    arguments = [
        "-no-first-run",
        "-force-color-profile=srgb",
        "-metrics-recording-only",
        "-password-store=basic",
        "-use-mock-keychain",
        "-export-tagged-pdf",
        "-no-default-browser-check",
        "-disable-background-mode",
        "-enable-features=NetworkService,NetworkServiceInProcess,LoadCryptoTokenExtension,PermuteTLSExtensions",
        "-disable-features=FlashDeprecationWarning,EnablePasswordsAccountStorage",
        "-deny-permission-prompts",
        "-disable-gpu",
        "-accept-lang=en-US",
    ]

    options = get_chromium_options(browser_path,arguments)

    driver = ChromiumPage(addr_or_opts=options)
    try:
        logging.info('Navigating to the page.')
        driver.get(url)

        logging.info('Starting Cloudflare bypass.')
        cf_bypasser = CloudflareBypasser(driver)
        cf_bypasser.bypass()

        #Wait for JavaScript content to load
        time.sleep(5)

        page_html = driver.html

        #Extract iframe contents
        soup = BeautifulSoup(page_html, 'html.parser')
        for iframe in soup.find_all("iframe"):
            iframe_src = iframe.get("src")

            if iframe_src:
                #Convert relative iframe URLs to absolute URLs
                absolute_iframe_url = urljoin(url, iframe_src)
                logging.info(f"Switching to iframe: {absolute_iframe_url}")

                driver.get(absolute_iframe_url)
                time.sleep(2)
                iframe_html = driver.html

                #Insert the iframe content into the main page
                iframe.insert_after(BeautifulSoup(
                    f"<!-- IFRAME CONTENT START -->{iframe_html}<!-- IFRAME CONTENT END -->",
                    "html.parser"
                ))

        page_html = str(soup)

    except Exception as e:
        logging.error("An error occurred: %s", str(e))
        page_html = ""
    finally:
        logging.info('Closing the browser.')
        driver.quit()

    return page_html

def clean_text(text):
    return re.sub(r'\s+',' ',text).strip()

def parse_content(html):
    inline_tags = ["span","a","em","strong","i","b","u"]
    for tag in inline_tags:
        html = re.sub(r'<{0}[^>]*>(.*?)</{0}>'.format(tag),r'\1',html, flags=re.DOTALL)

    html = clean_text(html)

    soup = BeautifulSoup(html,"html.parser")
    for tag in soup(["footer","nav","header"]):
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

    if(grouped_data == []):
        return []

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

def retrieve_relevant_sections(query,indexed_data,top_k=5):
    index = indexed_data["index"]
    texts = indexed_data["texts"]
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    query_embedding = model.encode(query,normalize_embeddings=True)    
    distances,indices = index.search(np.array([query_embedding]),k=top_k)    
    relevant_sections = [texts[i] for i in indices[0] if i < len(texts)]

    return relevant_sections