from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
import os
import re
import csv
from langchain_community.vectorstores import Neo4jVector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_groq import ChatGroq  # Import Groq's Chat model
from logging_setup import logger  # Import the logger from logging_setup.py
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_google_genai import GoogleGenerativeAIEmbeddings

GROQ_API_KEY = "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"
GOOGLE_API_KEY="AIzaSyDQM51t_NqqH5Vc00IqPtxWzkmdtDCYgjI"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=GROQ_API_KEY,
    # other params...
)  # Initialize ChatGroq with specified parameters

def loading_graph(graph, final_documents):

    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(final_documents)
    logger.info("Received Chunked documents and converted to Graph Documents")

    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    
    logger.info("Graph loading completed.")
    
# if __name__ == "__main__":
#     file_path = "/home/krishhindocha/Downloads/HR Policy Manual.pdf"
#     answer = loading_graph(file_path)
#     print(answer)
