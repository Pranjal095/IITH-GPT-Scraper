import streamlit as st
from scraper import scrape_page, parse_content, save_data, build_index, retrieve_relevant_sections
from llm import prompt_llm

st.title("Universal Web Scraper Tool")
st.markdown("""
This tool allows you to scrape website content, index it, and query relevant information using an LLM.
- **Step 1:** Enter a URL to scrape.
- **Step 2:** Submit queries to extract relevant information interactively.
""")

st.subheader("Step 1: Scrape Website")
url = st.text_input("Enter the website's URL")

if st.button("Scrape Site"):
    if url.strip():
        st.write("Scraping the website. This may take a moment...")
        html = scrape_page(url)
        grouped_data = parse_content(html)
        indexed_data = build_index(grouped_data)
        save_data(grouped_data,output_file="website_data.json")
        st.session_state.indexed_data = indexed_data
        st.session_state.grouped_data = grouped_data
        st.session_state.scraped = True
        st.success("Content scraped, indexed, and ready for queries!")
    else:
        st.error("Please enter a valid URL.")

if "grouped_data" in st.session_state:
    st.subheader("View Scraped Content")
    with st.expander("Grouped Content"):
        st.json(st.session_state.grouped_data)

if "indexed_data" in st.session_state:
    st.subheader("Step 2: Query the Website Content")
    query = st.text_area("Enter your query", key="query_input")

    if st.button("Submit Query", key="submit_query"):
        if query.strip():
            st.write("Retrieving relevant content...")
            relevant_sections = retrieve_relevant_sections(query,st.session_state.indexed_data)
            st.write("Generating response...")
            result = prompt_llm(relevant_sections,query)
            st.write(result)

            if "queries" not in st.session_state:
                st.session_state.queries = []
            st.session_state.queries.append({"query": query, "response": result})
            
            st.success("Response generated!")
        else:
            st.error("Please enter a valid query.")

    if "queries" in st.session_state and st.session_state.queries:
        st.subheader("Query History")
        for i, entry in enumerate(st.session_state.queries):
            with st.expander(f"Query {i+1}: {entry['query']}"):
                st.markdown(f"**Response:**\n\n{entry['response']}")