import os
import streamlit as st
import openai
# from langchain.llms import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = 'key-to-be-kept-private'

# creating an instance of the openai language model
# llm = openai.OpenAI(temperature=0.1, verbose=True)
llm = openai.OpenAI()
embeddings = OpenAIEmbeddings()

loader = PyPDFLoader('Report.pdf')
# splitting the pages in the pdf 
pages = loader.load_and_split()
# loading the documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')


vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)

# the document is converted into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title("Investment Banker using Langchain")
prompt = st.text_input("Enter your prompt here:")

if prompt:
    response = agent_executor(prompt)
    st.write(response)

    with st.expander("Document Similarity Search"):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)