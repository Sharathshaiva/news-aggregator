import validators
import os
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")

load_dotenv()
api_key  = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="Llama3-8b-8192",groq_api_key=api_key)



import requests
from bs4 import BeautifulSoup
import re

def extract_ndtv_latest_news_urls():
  
    url = "https://www.ndtv.com/latest"
    
   
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  
        
       
        soup = BeautifulSoup(response.text, 'html.parser')
        
        
        news_links = []
        
        # Look for news story links - based on NDTV's structure
        news_elements = soup.select('.news_Itm-cont a, .news-card a, .newsHdng a, .item-title a, .lstng_Hdng a')
        
        # If the above selectors don't find links, try a more general approach
        if not news_elements:
            # Find all links that likely contain news articles (filtering out navigation, etc.)
            all_links = soup.find_all('a', href=True)
            for link in all_links:
                href = link.get('href', '')
                # Filter news article links (typically they contain patterns like /india-news/, /world-news/, etc.)
                if (re.search(r'\/[a-z\-]+\/[a-z\-]+\/\d+', href) or 
                    '/news/' in href or 
                    '/india-news/' in href or 
                    '/world-news/' in href or
                    '/opinion/' in href):
                    
                    # Make sure the URL is absolute
                    if href.startswith('/'):
                        href = 'https://www.ndtv.com' + href
                    
                    # Add to our list if it's not already there
                    if href not in news_links:
                        news_links.append(href)
        else:
            # Process the links we found with the specific selectors
            for element in news_elements:
                href = element.get('href', '')
                
                # Make sure the URL is absolute
                if href.startswith('/'):
                    href = 'https://www.ndtv.com' + href
                
                # Add to our list if it's not already there
                if href and href not in news_links:
                    news_links.append(href)
        
        return news_links
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the webpage: {e}")
        return []
    except Exception as e:
        print(f"Error processing the webpage: {e}")
        return []





st.set_page_config(page_title="NEW BOT AGGREGATOR")
st.title("WRITES NEWS IN STORY MODE")
col1,col2 = st.columns(2,gap="large", border=True)
col1.subheader("Summarize NEWS")



# if st.button("Start"):
#     global url
#     url = extract_ndtv_latest_news_urls()
#     st.success("News extraction is Completed")

if "url" not in st.session_state:
    st.session_state.url = []  # Initialize an empty list

if st.button("Start"):
    st.session_state.url = extract_ndtv_latest_news_urls()  # Store URLs in session state
    st.success("News extraction is Completed")

# if st.session_state.url:
#     st.write("Extracted URLs:", st.session_state.url)

# with st.sidebar:
    # api_key = st.text_input("Groq API KEY",value="",type="password")



# url = st.text_input("URL",label_visibility="collapsed")

Prompt_template = """
Provide summary of the following content
content:{text}

"""
# Prompt_template1 = """
# Provide summary of the following content which consits of list of summaries
# content:{text}

# """
from langchain.chains import LLMChain
from langchain import PromptTemplate

generic_template = """
Write a meaningfull story of the following summary provided in the list, if needed you can make seperate sections for story:
content:{text}
"""

prompt1 = PromptTemplate(
    input_variables = ["text"],
    template = generic_template
)
output_summary=[]
prompt = PromptTemplate(template=Prompt_template,input_variables=["text"])
# prompt1 = PromptTemplate(template=Prompt_template1,input_variables=["text"])

# if col1.button("Summarize the content"):
#     try:
#         with st.spinner("Waiting......"):
#             for i in range(5):
#                 loader = UnstructuredURLLoader(urls=[url[i]],ssl_verify=False)
                
#                 data = loader.load()

#                 chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
#                 output_summary.append(str(chain.run(data)))
#             llm_chain = LLMChain(llm=llm,prompt=prompt1)
#             summary = llm_chain.run({"text":output_summary})

#             col1.write(summary)
#     except Exception as e:
#         col1.exception(e)

if "summary" not in st.session_state:
    st.session_state.summary = ""  # Initialize empty summary

if col1.button("Summarize the content"):
    try:
        with st.spinner("Waiting......"):
            output_summary = []  

            for i in range(5):
                loader = UnstructuredURLLoader(urls=[st.session_state.url[i]], ssl_verify=False)
                data = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary.append(str(chain.run(data)))

            llm_chain = LLMChain(llm=llm, prompt=prompt1)
            st.session_state.summary = llm_chain.run({"text": output_summary})  

    except Exception as e:
        col1.exception(e)

# Display persisted summary
col1.write(st.session_state.summary)


prompt_doc= ChatPromptTemplate.from_template(
    
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question:{input}

    """
)
def create_vectors():
    if "vetors" not in st.session_state:
        st.session_state.loader1= UnstructuredURLLoader(urls=st.session_state.url[:5],ssl_verify=False)
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.documents = st.session_state.loader1.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

if col2.button("Q&A FROM the above NEWS"):
    try:
        create_vectors()
        col2.write("Vector Database is ready")
    except Exception as e:
        col2.exception(e)

user_prompt = col2.text_input("Enter your query from the document")
if col2.button("QUERY") and user_prompt:
    doc_chain = create_stuff_documents_chain(llm,prompt_doc)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever,doc_chain)
    response = retriever_chain.invoke({"input":user_prompt})

    col2.write(response['answer'])
