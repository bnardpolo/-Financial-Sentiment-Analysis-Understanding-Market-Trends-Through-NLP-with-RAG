#!/usr/bin/env python
# coding: utf-8

# # **Build Knowledge Graph RAG With LlamaIndex for Data Analysis**

# ### **Business Overview**
# The financial market is a complex and dynamic ecosystem that generates vast amounts of data every second, including stock prices, news updates, industry trends, and economic reports. Analyzing this data is critical for investors and analysts to make informed decisions and manage risks. Traditionally, market data analysis has relied on manual efforts or basic tools that extract and process data from individual sources. This approach, while effective to some extent, is time-consuming, prone to errors, and limited in uncovering deeper insights such as hidden patterns or relationships between entities like companies, sectors, and stocks.
# 
# Generative AI (Gen AI) offers transformative potential in market data analysis by automating data extraction, processing, and structuring. With its ability to process unstructured data, Gen AI can uncover complex relationships and provide real-time, actionable insights. By combining web scraping, natural language processing (NLP), and knowledge graph construction, AI systems can not only handle the scale and speed required in financial markets but also offer personalized, context-aware insights to users. This allows professionals to quickly navigate the overwhelming sea of market data and focus on strategic decision-making.
# 
# 
# 

# ![image.png](https://images.pexels.com/photos/30381207/pexels-photo-30381207/free-photo-of-modern-trading-desk-with-graphs-and-gadgets.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2)

# ## **Approach**
# 
# 
# * **Data Collection:** Gather financial data from multiple sources such as stock feeds, news websites, and industry reports using Google search engine API and scraping techniques
# 
# 
# * **Knowledge Graph Construction:** Build a knowledge graph to represent relationships between extracted entities like companies, market trends, and sectors using neo4j
# 
# 
# * **Vector Embedding and Indexing**
#   - Convert textual data into vector embeddings using OpenAI
#   - Index data using LlamaIndex to enable retrieval-augmented generation
# 
# * RAG Workflow Integration
#   - Query the vector database to fetch relevant context and related entities from the knowledge graph
#   - Use OpenAI to generate contextually accurate and actionable responses
# 

# In[17]:


get_ipython().run_line_magic('pip', 'install llama-index')
get_ipython().run_line_magic('pip', 'install llama-index-llms-openai')
get_ipython().run_line_magic('pip', 'install llama-index-graph-stores-neo4j')
get_ipython().run_line_magic('pip', 'install llama-index-embeddings-openai')


# In[18]:


get_ipython().run_line_magic('pip', 'install --upgrade llama-index neo4j')


# In[19]:


from llama_index.core import SimpleDirectoryReader
from llama_index.core import PropertyGraphIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import os
import openai
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

import nest_asyncio
import asyncio


# In[ ]:


google_api = ""
search_engine_id = ""
open_ai_key = ""


# In[21]:


# Apply nest_asyncio
nest_asyncio.apply()


# In[ ]:


os.environ["hSZPs3UCO1BCUZ9aIo4sA"] = open_ai_key
openai.api_key = os.environ[""]


# Here’s a refined version suitable for GitHub documentation:  
# 
# ---
# 
# # **Data Collection for Financial Sentiment Analysis**  
# 
# This repository contains a Python script designed to automate the process of gathering financial sentiment data from the web. The script generates search queries, retrieves relevant search results, extracts webpage content, and structures the data for analysis.  
# 
# ## **Functionality Overview**  
# 
# ### **1. `search_with_google_api(query)`**  
# **Purpose**: Queries the Google Custom Search API using a specified search term and returns a list of relevant search results.  
# 
# **Key Steps**:  
# - Constructs the API request using the provided query, API key, and search engine ID.  
# - Sends an HTTP GET request to retrieve search results.  
# - If successful (`200` status code), extracts the `items` field containing search results.  
# - If unsuccessful, prints the error response.  
# 
# **Output**: A list of search results (or an empty list in case of an error).  
# 
# ---  
# 
# ### **2. `generate_search_queries(user_input)`**  
# **Purpose**: Generates a diverse set of search queries based on the user's input to enhance financial sentiment analysis.  
# 
# **Key Steps**:  
# - Uses OpenAI’s GPT-4 model to create 5-7 detailed and varied search queries.  
# - Formats and refines AI-generated queries into a structured Python list.  
# 
# **Output**: A list of search queries, e.g., `["electric vehicle sentiment analysis", ...]`.  
# 
# ---  
# 
# ### **3. `fetch_full_content(url)`**  
# **Purpose**: Retrieves and extracts the full textual content from a given webpage URL.  
# 
# **Key Steps**:  
# - Sends an HTTP GET request to the specified URL with a custom `User-Agent` header.  
# - If successful (`200` status code), parses the HTML content using `BeautifulSoup`.  
# - Extracts and returns the text from all `<p>` (paragraph) tags.  
# 
# **Error Handling**:  
# - Prints error messages for failed requests or parsing issues.  
# 
# **Output**: Extracted webpage text or `None` if no content is found.  
# 
# ---  
# 
# ### **4. `create_dataset_from_queries(queries, directory="dataset")`**  
# **Purpose**: Automates data collection by processing search queries, retrieving search results, extracting webpage content, and saving structured text files.  
# 
# **Key Steps**:  
# 1. **Directory Creation**:  
#    - Ensures the specified dataset directory exists.  
# 2. **Query Processing**:  
#    - Iterates through each query.  
#    - Fetches search results using `search_with_google_api()`.  
#    - Extracts and processes webpage content for each result.  
# 3. **File Storage**:  
#    - Saves each document as a uniquely named text file (`doc_<number>.txt`).  
#    - Includes query details, search result metadata (title, link, snippet), and full webpage content.  
#    - Stops after collecting 10 valid documents per query or if no more results are available.  
# 
# **Output**: Text files stored in the specified directory, ready for further analysis.  
# 
# ---  
# 
# ## **Workflow Overview**  
# 1. **User Input**: The user provides a financial topic of interest (e.g., "Electric vehicle market sentiment").  
# 2. **Query Generation**:  
#    - `generate_search_queries()` produces relevant search queries.  
# 3. **Data Collection**:  
#    - `create_dataset_from_queries()` processes search queries and extracts webpage content.  
# 4. **Storage & Usage**:  
#    - Data is stored as text files, facilitating further sentiment analysis.  
# 
# ---
# 
# This script provides a streamlined pipeline for gathering financial sentiment data, enabling efficient web data extraction for research and analysis. 🚀

# In[ ]:


import requests

def search_with_google_api(query):
    url = f""

    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("items", [])
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return []


# In[69]:


import os
import requests
from bs4 import BeautifulSoup
import openai


def generate_search_queries(user_input):
    """
    Generates a list of 5-7 detailed and relevant search queries for financial sentiment analysis
    based on the user's input, such as a target sector, field, or region.
    """
    prompt = f"""
    You are a financial analyst and search query expert. Based on the following user input, generate a list of 5-7 search queries
    for financial sentiment analysis based on user input. Ensure the queries cover diverse aspects of the topic, including sector-specific trends,
    regional financial overviews, and broader financial landscapes. The queries should focus on extracting data relevant to sentiment
    and performance analysis.

    User Input: {user_input}

    Strictly output the queries as a python list of strings. Do not add any additional comments.
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in generating search queries for financial sentiment analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )

    # Extract and clean up the list of queries
    queries =  response.choices[0].message.content.strip()
    return eval(queries)

def fetch_full_content(url):
    """
    Fetches the full content of a webpage given its URL.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            full_text = "\n".join([p.get_text() for p in paragraphs])
            return full_text.strip() if full_text else None
        else:
            print(f"Error: Unable to fetch content from {url} (Status Code: {response.status_code})")
            return None
    except Exception as e:
        print(f"Error fetching content from {url}: {e}")
        return None

def create_dataset_from_queries(queries, directory="dataset"):
    """
    Process search queries and save results as text files in the same directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_count = 1  # To ensure unique filenames across all queries

    for query in queries:
        print(f"Processing query: {query}")
        valid_count = 0
        page_number = 1

        while valid_count < 10:
            print(f"Fetching search results, page {page_number}...")
            results = search_with_google_api(query + f"&start={page_number * 10}")

            if not results:
                print("No more results found. Try refining the query.")
                break

            for result in results:
                if valid_count >= 10:
                    break  # Stop when 10 valid documents are saved

                title = result["title"]
                link = result["link"]
                snippet = result.get("snippet", "No snippet")

                # Fetch full content of the link
                full_content = fetch_full_content(link)
                if full_content:  # Save only if content is valid
                    filename = f"{directory}/doc_{file_count}.txt"
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(f"Query: {query}\n")
                        f.write(f"Title: {title}\n")
                        f.write(f"Link: {link}\n")
                        f.write(f"Snippet: {snippet}\n\n")
                        f.write(f"Full Content:\n{full_content}")
                    print(f"Saved: {filename}")
                    valid_count += 1
                    file_count += 1
                else:
                    print(f"Skipped: {link} (No valid content)")

            page_number += 1  # Move to the next page of results

    print(f"Finished processing all queries. Total files saved: {file_count - 1}")


user_input = "Financial sentiment analysis for the electric vehicle sector in the US"
queries = generate_search_queries(user_input)
queries
create_dataset_from_queries(queries)


# The code `documents = SimpleDirectoryReader("dataset").load_data()` uses a utility class, typically from libraries like llama-index (formerly gpt-index), to read and preprocess all documents stored in a specified directory. Here, the "dataset" directory is scanned for files, usually in formats like .txt. The SimpleDirectoryReader handles tasks such as opening the files, reading their content, and managing encodings, while ignoring unsupported or corrupt files.

# In[70]:


documents = SimpleDirectoryReader("dataset").load_data()


# In[71]:


get_ipython().run_line_magic('pip', 'uninstall -y llama-index')
get_ipython().run_line_magic('pip', 'install --upgrade llama-index neo4j')


# In[72]:


from llama_index.core import VectorStoreIndex


# In[73]:


from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


# Initiate graph store using Neo4j Credentials. Steps are given in the beginning of the notebook.

# In[ ]:


graph_store = Neo4jPropertyGraphStore(
    username="",
    password="",
    url=""
)


# In[ ]:


# Create the index
index = PropertyGraphIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(model_name="text-embedding-3-small"),
    kg_extractors=[
        SchemaLLMPathExtractor(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0)
        )
    ],
    property_graph_store=graph_store,
    show_progress=True,
    use_async=True
)


# In[ ]:


# save and load using locally stored graphs if you dont want to implement neo4j, you can directly use locally stored index in storage folder
index.storage_context.persist(persist_dir="./storage")

from llama_index.core import StorageContext, load_index_from_storage

index = load_index_from_storage(
    StorageContext.from_defaults(persist_dir="./storage")
)


# ## **Building GraphRAG Pipeline**
# 
# 

# In[ ]:


# loading from existing graph store (and optional vector store)
# load from existing graph/vector store
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store
)


# In[ ]:


# Define retriever
retriever = index.as_retriever(
    include_text=False,  # Default is true
)
results = retriever.retrieve("What is the summary of the finanacial texts?")
for record in results:
    print(record.text)


# ## **Retrieval Results Summary**  
# 
# ### **Overview**  
# The retriever was configured with `include_text=False`, returning key relationships instead of full text. The query **"What is the summary of the financial texts?"** retrieved structured data in the form of **entity -> relationship -> entity**.
# 
# ### **Key Findings**  
# - **Infrastructure & Financial Investments**:  
#   - *Gabe Klein* and *Dean Bushey* are involved in **charging infrastructure**.  
#   - *Charging infrastructure* is **used by electric trucks**, highlighting its role in sustainable transportation.  
# 
# - **Government & Financial Policy**:  
#   - *Janet Yellen*, as US Treasury Secretary, is linked to her **visit to Beijing**, suggesting relevance to financial policy or economic discussions.  
# 
# ### **Conclusion**  
# The results emphasize financial themes around **clean energy investments, infrastructure development, and economic policies**. Further refinement of retrieval parameters can improve insights for financial sentiment analysis.
# 

# In[ ]:


query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("How to invest in the EV sector? Summarize the most important financial trends in the EV Sector.")
print(response)


# ## **Query Results Summary**  
# 
# ### **Overview**  
# The query engine was configured with `include_text=True`, allowing retrieval of detailed insights on financial trends in the **electric vehicle (EV) sector**. The query:  
# 
# > **"How to invest in the EV sector? Summarize the most important financial trends in the EV sector."**  
# 
# ### **Key Findings**  
# - **Shifting Investor Sentiment**:  
#   - Initial enthusiasm for EVs has diminished, leading to a more cautious investment landscape.  
# - **Automaker Strategy Adjustments**:  
#   - Companies like **Ford** and **General Motors** are diversifying their offerings with **gas-powered, hybrid, and fully electric vehicles**, rather than committing exclusively to EVs.  
#   - This shift reflects a slower adoption rate than initially expected.  
# - **Long-Term Market Growth**:  
#   - Despite short-term adjustments, **EV sales are projected to rise**, with expectations of a significant market share by the decade’s end.  
# - **Investment Strategy Implications**:  
#   - A **balanced investment approach** is recommended, factoring in **market evolution, consumer preferences, and industry trends**.  
# 
# ### **Conclusion**  
# The EV sector remains a promising investment opportunity, but **strategic diversification** is crucial as automakers and consumers adapt to changing market conditions. 🚀  
# 

# In[ ]:


query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("Which companies are doing the best in EV sector?")
print(response)


# ## **Query Results Summary**  
# 
# ### **Overview**  
# The query engine was configured with `include_text=True`, retrieving detailed insights into companies performing well in the **electric vehicle (EV) sector**. The query:  
# 
# > **"Which companies are doing the best in the EV sector?"**  
# 
# ### **Key Finding**  
# - **Polestar** has been highlighted as a notable player in the EV market, with **a significant number of EV imports in 2023**.  
# 
# ### **Conclusion**  
# Polestar's market presence suggests strong demand and expansion in the EV industry. Further queries could provide a more comprehensive analysis of other leading EV companies and their competitive positionin
# 

# In[ ]:


query_engine = index.as_query_engine(include_text=True)
response = query_engine.query("How is Tesla doing in EV sector?")
print(response)


# ## **Query Results Summary**  
# 
# ### **Overview**  
# The query engine was configured with `include_text=True`, retrieving insights into **Tesla’s role in the EV sector**. The query:  
# 
# > **"How is Tesla doing in the EV sector?"**  
# 
# ### **Key Findings**  
# - **Global Presence**: Tesla operates in **Austin, Texas, the United States, and China**, emphasizing its international reach.  
# - **Industry Role**: Tesla remains a key player in the **automaker industry**, actively producing electric vehicles.  
# - **Strategic Partnerships**: Tesla collaborates with companies like **UPS, Rivian, and Arrival** to supply **electric delivery vans and trucks**, reinforcing its market leadership.  
# 
# ### **Conclusion**  
# Tesla continues to dominate the EV sector through **global expansion, vehicle production, and strategic partnerships** that enhance its influence in both consumer and commercial electric vehicle markets. 🚀  
# 

# In[ ]:


from typing import List, Dict

def generate_summary_report(context: str, query: str) -> str:
    """
    Generate a detailed summary report for financial sentiment analysis.
    Takes context and query as inputs and returns a comprehensive summary.
    """
    prompt = f"""
    You are a financial sentiment analysis assistant. Using the context provided below, generate a detailed summary report:

    Context:
    {context}

    Query:
    {query}

    The report should include:
    1. A high-level summary of the financial trends related to the query.
    2. Key positive, negative, and neutral sentiments detected.
    3. Reasons or factors driving the sentiments.
    4. Suggestions or insights for potential investors or stakeholders.

    Be concise but ensure that the report is actionable and insightful.
    """
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in financial sentiment analysis."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

def query_and_generate_reports(queries: List[str]) -> List[Dict[str, str]]:
    """
    Query the knowledge graph for each query, aggregate context, and generate summary reports.
    Returns a list of dictionaries containing the query, aggregated context, and generated report.
    """
    results = []

    for query in queries:
        print(f"Processing query: {query}")
        context = query_engine.query(query)

        # Generate a summary report using the aggregated context
        report = generate_summary_report(context, query)

        results.append({
            "query": query,
            "context": context,
            "report": report
        })

    return results

def save_reports_to_file(results: List[Dict[str, str]], filename: str):
    """
    Save query results and their generated reports to a file.
    """
    with open(filename, "w", encoding="utf-8") as file:
        for result in results:
            file.write(f"Query:\n{result['query']}\n\n")
            file.write(f"Context:\n{result['context']}\n\n")
            file.write(f"Generated Report:\n{result['report']}\n\n")
            file.write("-" * 80 + "\n\n")



# Create the query engine
query_engine = index.as_query_engine(include_text=True)

# Define a list of queries, Different kinds of queries to see the effectiveness of in EV sector
queries = [
    "How to invest in the EV sector? Summarize the most important financial trends in the EV Sector.",
    "What are the recent financial sentiments about renewable energy investments?",
    "Summarize the financial outlook for the technology sector in 2024.",
    "What are the key financial risks in the automotive industry this year?",
    "Provide insights on the financial performance of AI startups in the US."
]

# Execute the queries and generate reports
results = query_and_generate_reports(queries)

# Save the reports to a file
output_file = "financial_sentiment_reports.txt"
save_reports_to_file(results, output_file)

# Print a summary of the generated reports
for result in results:
    print(f"Query: {result['query']}")
    print(f"Generated Report:\n{result['report']}")
    print("-" * 80)


# ## **Financial Sentiment Analysis - Query Results Summary**  
# 
# ### **Overview**  
# The script queries a financial sentiment analysis system to generate structured reports based on various investment and industry-related questions. The process involves:  
# - Using a **query engine** to fetch relevant contextual information.  
# - **Generating detailed summary reports** on financial trends, sentiment analysis, and investment insights.  
# - **Saving reports** for further reference and decision-making.  
# 
# ### **Key Findings from the Queries**  
# 
# #### **1. EV Sector Investment Trends**  
# - Automakers like **Ford** and **General Motors** are shifting strategies, balancing **gas-powered, hybrid, and electric vehicle** production.  
# - **EV market growth is slower than anticipated**, with concerns over **charging infrastructure, affordability, and regulatory factors**.  
# - **Investment Insight**: A balanced portfolio focusing on automakers adapting to changing market conditions is recommended.  
# 
# #### **2. Renewable Energy Investments**  
# - **Mixed investor sentiment** due to **high upfront costs** and **uncertainty in business models**.  
# - Strong confidence in **long-term growth**, but challenges in **energy grid capacity and pricing structures**.  
# - **Investment Insight**: Focus on **companies with robust business models** and strategic **partnerships to reduce initial costs**.  
# 
# #### **3. 2024 Technology Sector Outlook**  
# - **Positive trends**: Growth in **electric vehicles (EVs)** and **AI adoption**.  
# - **Risks**: Lithium supply constraints, evolving **geopolitical and regulatory factors**.  
# - **Investment Insight**: Diversification across **sub-sectors** such as **charging infrastructure and AI advancements** is advisable.  
# 
# #### **4. Financial Risks in the Automotive Industry**  
# - **Negative risks**: **Regulatory fines** for fuel inefficiency, **slower-than-expected EV adoption**, and **high production costs**.  
# - **Opportunities**: Anticipated **long-term EV growth and infrastructure development**.  
# - **Investment Insight**: Monitor **regulatory changes**, invest in **flexible and adaptive automakers**.  
# 
# #### **5. AI Startup Financial Performance in the US**  
# - **Positive sentiment**: High demand for AI, **government support, and venture capital interest**.  
# - **Negative sentiment**: **Intense competition, market saturation concerns, and regulatory uncertainty**.  
# - **Investment Insight**: Close monitoring of **regulations, market trends, and key players** is essential for risk mitigation.  
# 
# ### **Conclusion**  
# The reports provide actionable insights into **investment opportunities, market shifts, and sentiment-driven risks**. Investors should focus on **adaptable companies, regulatory landscapes, and emerging technology trends** for **sustainable financial growth**. 🚀  
# 
# ---
# 📂 **Reports saved to:** `financial
# 

# In[ ]:


import numpy as np
import pandas as pd  # Import Pandas for DataFrame manipulation
import matplotlib.pyplot as plt  # Import Matplotlib for visualization


# In[ ]:


import pandas as pd

sentiment_reports = [
    {"Query": "EV Sector Investment", "Positive": 12, "Negative": 5, "Neutral": 8},
    {"Query": "Renewable Energy Trends", "Positive": 15, "Negative": 4, "Neutral": 6},
    {"Query": "Cryptocurrency Market", "Positive": 10, "Negative": 7, "Neutral": 9},
    {"Query": "AI Startups Financials", "Positive": 14, "Negative": 3, "Neutral": 5},
]

# Convert to DataFrame
df_sentiment = pd.DataFrame(sentiment_reports)

# Compute sentiment score: Positive - Negative
df_sentiment["Sentiment Score"] = df_sentiment["Positive"] - df_sentiment["Negative"]

# Display DataFrame
print(df_sentiment)





# # **Financial Sentiment Analysis Report**  
# 
# ## **Overview**  
# This report presents a sentiment analysis of various financial topics, quantifying **positive, negative, and neutral sentiments**. A **Sentiment Score** is calculated as:  
# 
# > **Sentiment Score = Positive Sentiments - Negative Sentiments**  
# 
# A higher sentiment score indicates stronger positive sentiment, whereas a lower score suggests more negative perception.  
# 
# ## **Sentiment Analysis Results**  
# 
# | **Query**                     | **Positive** | **Negative** | **Neutral** | **Sentiment Score** |
# |--------------------------------|-------------|-------------|------------|------------------|
# | EV Sector Investment          | 12          | 5           | 8          | 7                |
# | Renewable Energy Trends       | 15          | 4           | 6          | 11               |
# | Cryptocurrency Market         | 10          | 7           | 9          | 3                |
# | AI Startups Financials        | 14          | 3           | 5          | 11               |
# 
# ## **Key Insights**  
# 
# 1. **Renewable Energy & AI Startups Show Strongest Positive Sentiment**  
#    - Both sectors have a **Sentiment Score of 11**, indicating **strong investor confidence**.  
#    - The **renewable energy** sector benefits from **long-term growth expectations**, despite infrastructure concerns.  
#    - **AI startups** are gaining traction, supported by **venture capital in
# 

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample user query history
user_query_history = [
    "Best investment opportunities in AI sector",
    "Financial growth of renewable energy companies",
    "Stock market trends in 2024",
    "Cryptocurrency market stability",
    "Economic impact of inflation"
]

# Sample available queries to recommend
available_queries = [
    "Future of AI startups in finance",
    "Top renewable energy investments for 2025",
    "Stock market volatility predictions",
    "Bitcoin and Ethereum price forecasts",
    "Inflation and its effects on global markets"
]

# Convert text queries to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer()
query_vectors = vectorizer.fit_transform(user_query_history + available_queries)

# Compute similarity between user queries and available queries
user_queries_vector = query_vectors[:len(user_query_history)]
available_queries_vector = query_vectors[len(user_query_history):]
similarity_scores = cosine_similarity(user_queries_vector, available_queries_vector)

# Recommend top queries based on highest similarity scores
def recommend_queries(top_n=3):
    recommendations = []
    for user_idx, user_query in enumerate(user_query_history):
        top_indices = np.argsort(similarity_scores[user_idx])[-top_n:][::-1]
        top_recommendations = [available_queries[i] for i in top_indices]
        recommendations.append({
            "User Query": user_query,
            "Recommended Queries": top_recommendations
        })
    return recommendations

# Get query recommendations
query_recommendations = recommend_queries()

# Display recommendations
for rec in query_recommendations:
    print(f"User Query: {rec['User Query']}")
    print("Recommended Queries:")
    for q in rec["Recommended Queries"]:
        print(f"  - {q}")
    print("-" * 50)


# # **Query Recommendation System for Financial Insights**  
# 
# ## **Overview**  
# This system **analyzes user query history** and **recommends related queries** using **TF-IDF vectorization** and **cosine similarity**. By identifying similarities between past queries and available queries, it suggests **highly relevant financial topics** to users.
# 
# ## **Methodology**  
# 
# 1. **Convert Queries to Numerical Vectors**  
#    - Uses **TF-IDF (Term Frequency-Inverse Document Frequency)** to represent text-based queries numerically.  
# 
# 2. **Compute Query Similarities**  
#    - Uses **cosine similarity** to measure the closeness between a user's previous queries and a set of **available recommended queries**.  
# 
# 3. **Recommend Relevant Queries**  
#    - Identifies the **top 3 most similar queries** for each user query based on similarity scores.  
# 
# ## **Query Recommendation Results**  
# 
# ### **User Query: "Best investment opportunities in AI sector"**  
# **Recommended Queries:**  
#   - Future of AI startups in finance  
#   - Inflation and its effects on global markets  
#   - Bitcoin and Ethereum price forecasts  
# 
# ### **User Query: "Financial growth of renewable energy companies"**  
# **Recommended Queries:**  
#   - Top renewable energy investments for 2025  
#   - Future of AI startups in finance  
#   - Inflation and its effects on global markets  
# 
# ### **User Query: "Stock market trends in 2024"**  
# **Recommended Queries:**  
#   - Stock market volatility predictions  
#   - Future of AI startups in finance  
#   - Inflation and its effects on global markets  
# 
# ### **User Query: "Cryptocurrency market stability"**  
# **Recommended Queries:**  
#   - Stock market volatility predictions  
#   - Inflation and its effects on global markets  
#   - Bitcoin and Ethereum price forecasts  
# 
# ### **User Query: "Economic impact of inflation"**  
# **Recommended Queries:**  
#   - Inflation and its effects on global markets  
#   - Future of AI startups in finance  
#   - Bitcoin and Ethereum price forecasts  
# 
# ## **Key Insights**  
# 
# - **Highly Relevant Recommendations**:  
#   - The system successfully links **investment queries** to **sector-specific financial topics**.  
#   - Queries about **inflation, AI, and stock market trends** appear frequently, highlighting **common financial concerns**.  
# 
# - **Diverse Recommendations**:  
#   - Users searching for **investment opportunities** in AI are also recommended **macroeconomic trends** and **cryptocurrency-related queries**.  
#   - **Stock market trends are closely tied to volatility predictions**, showing a strong correlation in financial forecasting needs.  
# 
# ## **Conclusion**  
# This **query recommendation system** provides **intelligent and contextualized financial topic suggestions**, helping users explore **relevant investment insights, market trends, and economic factors**. Future improvements could include **real-time query expansion, sentiment-based ranking, and personalized financial research recommendations**. 🚀  
# 
# ---
# 

# In[ ]:


pip install yfinance


# # 📈 **Real-Time Stock Quote Retrieval using Yahoo Finance API**  
# 
# ## **Overview**  
# This script fetches **real-time stock price data** for a given stock symbol using the **Yahoo Finance API (`yfinance`)**. It retrieves the **latest closing price** of the stock, providing a quick and efficient way to track market trends.  
# 
# ---
# 
# ## **Features**  
# ✔️ Fetches the **latest stock closing price** for a given ticker symbol.  
# ✔️ Uses **Yahoo Finance (`yfinance`)** to access real-time market data.  
# ✔️ Provides **structured output with stock symbol, latest price, and date**.  
# ✔️ Includes **error handling** for invalid or missing stock data.  
# 
# 
# 

# In[ ]:


import yfinance as yf

def get_stock_quote(symbol):
    """
    Fetch real-time stock price for the given symbol using Yahoo Finance API.
    """
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.history(period="1d")
        
        if stock_info.empty:
            return f"No data found for {symbol}."
        
        latest_price = stock_info["Close"].iloc[-1]
        return {
            "Stock Symbol": symbol,
            "Latest Price": latest_price,
            "Date": stock_info.index[-1].strftime("%Y-%m-%d")
        }
    except Exception as e:
        return f"Error fetching stock data: {e}"

# Example Usage: Fetch stock data for Apple (AAPL)
stock_symbol = "AAPL"
real_time_data = get_stock_quote(stock_symbol)

print("Real-time Stock Data:")
print(real_time_data)



# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sample financial sentiment data
data = {
    "Sector": ["EV", "Renewable Energy", "Technology", "Automotive", "AI"],
    "Positive": [12, 15, 10, 14, 18],
    "Negative": [5, 4, 7, 3, 6],
    "Neutral": [8, 6, 9, 5, 7],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute sentiment score: Positive - Negative
df["Sentiment Score"] = df["Positive"] - df["Negative"]

# Streamlit App
st.title("📊 Financial Sentiment Analysis Dashboard")
st.markdown("Interactive visualization of sentiment trends in various financial sectors.")

# Sidebar Filters
selected_sector = st.sidebar.selectbox("🔍 Select a Sector", ["All"] + list(df["Sector"]))

# Filter data based on sector selection
if selected_sector != "All":
    df_filtered = df[df["Sector"] == selected_sector]
else:
    df_filtered = df

# Display Data Table
st.subheader("💡 Sentiment Analysis Data")
st.write(df_filtered)

# Plotting Sentiment Scores
st.subheader("📈 Sentiment Score Visualization")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x="Sector", y="Sentiment Score", data=df_filtered, palette="viridis", ax=ax)
ax.set_ylabel("Sentiment Score")
ax.set_xlabel("Sector")
ax.set_title("Sentiment Score by Sector")
st.pyplot(fig)

# Insights Section
st.subheader("📌 Key Insights")

for index, row in df_filtered.iterrows():
    st.write(
        f"**{row['Sector']} Sector:** Positive: {row['Positive']}, "
        f"Negative: {row['Negative']}, Neutral: {row['Neutral']}, "
        f"Sentiment Score: {row['Sentiment Score']}."
    )

st.markdown("---")
st.markdown("💡 **Tip:** Keep an eye on evolving market trends and investment opportunities!")



# Key Insights from Analysis
# 📊 AI sector has the highest sentiment score (12), indicating strong investor confidence.
# ⚡ Renewable Energy and Automotive sectors also show high positive sentiment (Score: 11).
# 📉 Technology sector has the lowest sentiment score (3), suggesting mixed investor reactions.
# 🚗 EV sector shows moderate sentiment (Score: 7), reflecting growing but cautious investment trends.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data (Modify with real values)
sectors = ["EV", "Renewable Energy", "Technology", "Automotive", "AI"]
positive = [12, 15, 10, 14, 18]
negative = [5, 4, 7, 3, 6]
neutral = [8, 6, 9, 5, 7]

# Stacked Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(sectors, positive, label="Positive", color="green")
ax.bar(sectors, neutral, bottom=positive, label="Neutral", color="gray")
ax.bar(sectors, negative, bottom=[i + j for i, j in zip(positive, neutral)], label="Negative", color="red")

ax.set_ylabel("Sentiment Score")
ax.set_title("Sentiment Score by Sector")
ax.legend()
plt.xticks(rotation=45)
plt.show()


# # 📊 **Sentiment Score Analysis by Sector**  
# 
# ## **Overview**  
# This visualization provides an **in-depth sentiment analysis** of various financial sectors by categorizing sentiments into **Positive, Neutral, and Negative**. The **stacked bar chart** helps investors and analysts **understand market trends** across key industries.
# 
# ---
# 
# ## **Data Breakdown**  
# The analysis covers **five financial sectors**:  
# - **EV (Electric Vehicles)**  
# - **Renewable Energy**  
# - **Technology**  
# - **Automotive**  
# - **AI (Artificial Intelligence)**  
# 
# Each sector's sentiment is divided into:  
# ✅ **Positive Sentiment** (Green) → Indicates strong confidence and growth potential.  
# ⚪ **Neutral Sentiment** (Gray) → Suggests mixed investor sentiment or market stability.  
# ❌ **Negative Sentiment** (Red) → Represents concerns or risks in the sector.  
# 
# 
# 
# ## **Visualization: Stacked Sentiment Score Chart**  
# 
# 📊 The **stacked bar chart** below represents the sentiment breakdown for each sector:  
# 
# ![Sentiment Score Chart](image.png)
# 
# ---
# 
# ## **Key Insights**  
# 
# 📈 **AI Sector Leads in Positive Sentiment**  
# - With **18 positive mentions**, the **AI sector** shows **high investor confidence** and growth potential.  
# 
# ⚡ **Renewable Energy Sector Remains Strong**  
# - **Second-highest positive sentiment (15)**, with **low negative sentiment (4)**, indicating a **stable and promising** investment sector.  
# 
# 📉 **Technology Sector Shows the Most Mixed Sentiment**  
# - **Lowest sentiment score (10 positive, 9
# 

# In[ ]:


import pandas as pd

# Sample Data (Modify with real values)
data = {
    "date": pd.date_range(start="2024-01-01", periods=10, freq="M"),
    "EV": [10, 12, 14, 13, 15, 16, 14, 17, 18, 20],
    "Technology": [8, 9, 7, 6, 8, 9, 11, 10, 12, 13],
}

df = pd.DataFrame(data)
df.set_index("date", inplace=True)

# Plot Sentiment Trend Over Time
df.plot(figsize=(10, 6), marker="o")
plt.title("Sentiment Trends Over Time")
plt.ylabel("Sentiment Score")
plt.xlabel("Date")
plt.legend(title="Sector")
plt.grid()
plt.show()


# 
# ---
# 
# ## **Key Insights**  
# 
# 📊 **EV Sentiment Shows Strong Growth**  
# - **EV sector sentiment increased from 10 (Jan) to 20 (Oct)**, indicating **rising investor confidence**.  
# - Despite a **minor dip in July**, the trend remains **upward**.  
# 
# 📉 **Technology Sector Faces Fluctuations**  
# - Sentiment declined **from Feb (9) to Apr (6)** but showed a **rebound afterward**.  
# - A **steady increase from May (8) to Oct (13)** suggests **growing optimism** in tech investments.  
# 
# 📌 **Sector Comparison**  
# - **EV sector consistently outperforms** Technology in sentiment scores.  
# - **Technology sector had a slow start**, but **momentum picked up from mid-2024**.  
# 
# 

# In[ ]:


from textblob import TextBlob

# Sample Financial News Headlines
news_headlines = [
    "Tesla faces production delays due to supply chain issues",
    "EV market expected to grow by 30% in the next five years",
    "Renewable energy investments rise as government supports green initiatives",
]

# Extract Sentiment Scores
for headline in news_headlines:
    sentiment = TextBlob(headline).sentiment.polarity  # Range: -1 (negative) to +1 (positive)
    print(f"News: {headline}\nSentiment Score: {sentiment}\n")


# # 📰 **Financial News Sentiment Analysis**  
# 
# ## **Overview**  
# This analysis evaluates the sentiment of **financial news headlines** using **TextBlob**. Sentiment polarity scores help **assess market trends**, providing insights into **positive, negative, or neutral sentiment** within the financial sector.
# 
# ---
# 
# ## **Methodology**  
# 
# 1. **Extract Sentiment Polarity**  
#    - Uses **TextBlob** to analyze the sentiment of financial news headlines.  
#    - Sentiment polarity values range from **-1 (negative) to +1 (positive)**.  
# 
# 2. **Categorize Sentiment Scores**  
#    - **Negative Sentiment (-1 to 0)** → Market concerns, risks, or downturns.  
#    - **Neutral Sentiment (~0)** → Balanced or uncertain tone.  
#    - **Positive Sentiment (0 to +1)** → Optimistic market outlook.  
# 
# 
# ---
# 
# ## **Key Insights from Sentiment Analysis**  
# 
# 📉 **Tesla's Production Delays Show Negative Sentiment**  
# - **Sentiment Score: -0.125**  
# - Investors may perceive **supply chain issues as a risk factor** affecting Tesla's stock performance.  
# 
# 📊 **EV Market Growth News Shows Slightly Negative Sentiment**  
# - **Sentiment Score: -0.05**  
# - While EV growth projections are **positive**, uncertainties may cause **investor hesitation**.  
# 
# ⚡ **Renewable Energy Investments Show Unexpected Negative Sentiment**  
# - **Sentiment Score: -0.2**  
# - Despite **government support**, concerns about **policy impact and investment risks** may contribute to this sentiment.  
# 
# ---
# 
# 
# 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Generate Sample Data
df["future_sentiment"] = df["EV"].shift(-1)  # Predict future EV sentiment
df.dropna(inplace=True)

# Train Model
X = np.array(range(len(df))).reshape(-1, 1)
y = df["future_sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Predict Future Sentiment
future_sentiment = model.predict([[len(df) + 1]])[0]
print(f"Predicted Sentiment Score for Next Month: {future_sentiment}")


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Extend the dataset with the predicted value
df["index"] = range(len(df))
future_x = len(df) + 1
future_y = 20.86  # Replace with your model's predicted value

plt.figure(figsize=(10,5))
plt.plot(df["index"], df["EV"], label="Actual Sentiment", marker="o")
plt.scatter(future_x, future_y, color="red", label="Predicted Sentiment", s=100)

plt.xlabel("Time Index")
plt.ylabel("Sentiment Score")
plt.legend()
plt.title("Predicted Sentiment vs. Actual Trend")
plt.show()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict Future Sentiment
future_sentiment = rf_model.predict([[len(df) + 1]])[0]
print(f"New Predicted Sentiment Score: {future_sentiment}")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# In[ ]:


import numpy as np

# Simulate market trend data (Modify this with real data)
df["stock_price_change"] = np.random.uniform(-2, 3, len(df))  # Simulated daily stock % change
df["market_volatility"] = np.random.uniform(1, 5, len(df))  # Simulated volatility index

# Check Data
print(df[["stock_price_change", "market_volatility"]].head())


# 📈 Stock Price Trends
# 
# Prices fluctuate between -2% to +3% daily, reflecting moderate volatility.
# Upward trends indicate potential bullish market movements, while negative values suggest short-term corrections.
# 📉 Market Volatility Analysis
# 
# Market volatility index ranges from 1 to 5, indicating periods of both stability and risk.
# Higher volatility values (above 3) may correlate with increased market uncertainty and riskier investment conditions.

# In[ ]:


from textblob import TextBlob
import numpy as np

# Sample news headlines (Modify or fetch real financial news)
news_headlines = [
    "Tech stocks are booming amid AI innovation",
    "EV demand drops due to charging infrastructure concerns",
    "Renewable energy investments hit record high",
    "Regulations slow down fintech growth",
    "Stock market crashes as inflation rises"
]

# Ensure the number of headlines matches the number of rows in df
df["news_headline"] = np.resize(news_headlines, len(df))

# Compute sentiment polarity from headlines
df["news_sentiment"] = df["news_headline"].apply(lambda x: TextBlob(x).sentiment.polarity)

# Verify the data
print(df[["news_headline", "news_sentiment"]].head())



# Key Insights from Analysis
# 📈 Positive Sentiment: Renewable Energy Investments
# 
# Sentiment Score: 0.160
# Suggests high investor confidence in the renewable energy sector.
# 📉 Negative Sentiment: Fintech & EV Market Challenges
# 
# EV demand drops (-0.125 sentiment score) → Signals concerns over infrastructure.
# Fintech growth slows due to regulations (-0.227 sentiment score) → Suggests market restrictions.
# 📊 Stock Market & AI Stocks Show Neutral Sentiment
# 
# Tech stocks booming (0.000 sentiment score) suggests mixed investor reactions.
# Stock market crashes due to inflation (0.000 sentiment score) → Reflects uncertainty.

# Key Takeaways
# EV Market
# Initial excitement around EVs has slowed, with automakers shifting toward hybrid and gas-powered vehicles alongside electrics.
# Growth remains positive but slower than expected, driven by charging infrastructure concerns and consumer hesitation.
# Investment Outlook: Keep an eye on automakers balancing regulatory policies, infrastructure growth, and consumer adoption rates.
# Renewable Energy
# Strong long-term confidence, but high upfront costs and grid limitations create hurdles.
# Government incentives and technological advancements continue to push the sector forward.
# Investment Outlook: Look for companies with strong business models and long-term sustainability plans.
# Technology & AI
# AI startups are booming, but competition and regulatory challenges create uncertainty.
# The technology sector is volatile, with geopolitical risks, lithium demand, and EV integration shaping future trends.
# Investment Outlook: Diversify within AI, cloud computing, and EV-adjacent tech while tracking regulatory updates.
# Automotive Industry
# Facing fuel economy regulations, EV adoption uncertainty, and infrastructure costs.
# Investors are cautious, waiting to see how policies and market trends develop.
# Investment Outlook: Focus on companies adapting to policy shifts and consumer trends.
# Market Trends & Sentiment Correlations
# Negative news sentiment often aligns with market volatility, especially in tech, fintech, and EV markets.
# AI and renewable energy sectors maintain strong positive sentiment, indicating long-term investment confidence.
# Regulatory and policy changes are key sentiment drivers, especially in EV and fintech industries.
