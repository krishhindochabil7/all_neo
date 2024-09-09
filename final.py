from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from typing import Tuple, List, Optional
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import ConfigurableField
import os
import re
import csv
from langchain_community.vectorstores import Neo4jVector
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from typing import List, Dict
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.text_splitter import TokenTextSplitter
from unstructured_example import k_main_2
from text_query import k_cypher
from extraction import full_graph,sub_graph
from logging_setup import setup_logger, getLogfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Set up the logger
logger = setup_logger(getLogfile())

GROQ_API_KEY = "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"
GOOGLE_API_KEY="AIzaSyDQM51t_NqqH5Vc00IqPtxWzkmdtDCYgjI"
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

graph = None
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "kh27042001"

os.environ["NEO4J_URI"] = NEO4J_URI
os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD
os. environ["GROQ_API_KEY"] = GROQ_API_KEY
# Confirm environment variables are set
logger.info("Environment variables loaded and Neo4jGraph initialized.")
graph = Neo4jGraph()

def process_question(Unstructured, question: str):
    global graph
    GROQ_API_KEY = "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"
    llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key = GROQ_API_KEY,
    # other params...
)  # Initialize ChatGroq with specified parameters
    
    logger.info("Starting to process question.")
    try:
        logger.info("Creating full-text index.")
        graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

        class Entities(BaseModel):
            """Identifying information about entities."""
            names: List[str] = Field(
                ...,
                description="All the person, organization, or business entities that appear in the text",
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are extracting organization and person entities from the text.",
                ),
                (
                    "human",
                    "Use the given format to extract information from the following input: {question}",
                ),
            ]
        )

        entity_chain = prompt | llm.with_structured_output(Entities)

        def remove_lucene_chars(text: str) -> str:
            lucene_chars = r'[+\-&|!(){}[\]^"~*?:\\]'
            return re.sub(lucene_chars, '', text)

        def generate_full_text_query(input: str) -> str:
            full_text_query = ""
            words = [el for el in remove_lucene_chars(input).split() if el]
            for word in words[:-1]:
                full_text_query += f" {word}~2 AND"
            full_text_query += f" {words[-1]}~2"
            return full_text_query.strip()

        def structured_retriever(question: str) -> str:
            result = ""
            try:
                entities = entity_chain.invoke({"question": question})
                for entity in entities.names:
                    response = graph.query(
                        """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                        YIELD node,score
                        CALL {
                          WITH node
                          MATCH (node)-[r:!MENTIONS]->(neighbor)
                          RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                          UNION ALL
                          WITH node
                          MATCH (node)<-[r:!MENTIONS]-(neighbor)
                          RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                        }
                        RETURN output
                        """,
                        {"query": generate_full_text_query(entity)},
                    )
                    result += "\n".join([el['output'] for el in response])
            except Exception as e:
                logger.error(f"Error occurred during structured retrieval: {e}")
            return result

        def s_retriever(question: str):
            S_data = structured_retriever(question)
            return S_data

        def retriever(question: str):
            final_data = s_retriever(question)
            logger.info("Printing Structured Data in terminal")
            print(f"\nStructured Data: \n{final_data}")
            unstructured_strings = []

            for item in Unstructured:
                if isinstance(item, tuple):
                    document = item[0] 
                    if isinstance(document, Document):
                        unstructured_strings.append(document.page_content)  
                    elif isinstance(document, dict):
                        # Handle dictionary case if needed
                        unstructured_strings.append(str(document.get('page_content', 'No page_content key')))
                    else:
                        unstructured_strings.append(str(document))
                else:
                    unstructured_strings.append(str(item))

            final_data += "\n".join(unstructured_strings)
            return final_data

        _template = """You are a helpful assistant for question-answering tasks.
        Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        If the follow-up question is unclear or lacks context, state that you do not know and provide any available context.
        Ensure that your rephrased question is concise, clear, and maintains the original intent. Use formal and respectful language.

        Chat History:
        {chat_history}

        Follow-Up Input: {question}
        Standalone Question:"""

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
            buffer = []
            for human, ai in chat_history:
                buffer.append(HumanMessage(content=human))
                buffer.append(AIMessage(content=ai))
            return buffer

        _search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                    run_name="HasChatHistoryCheck"
                ),
                RunnablePassthrough.assign(
                    chat_history=lambda x: _format_chat_history(x["chat_history"])
                )
                | CONDENSE_QUESTION_PROMPT
                | ChatGroq(temperature=0)
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x : x["question"]),
        )

        template = """You are a helpful assistant for question-answering tasks.
        Answer the question based only on the following context. If the answer is unclear or unavailable, state that you do not know.
        Structure your response using bullet points for clarity, ensuring it is concise and detailed. Include relevant references where applicable.
        Use formal and respectful language, and if the question contains derogatory remarks, kindly decline by stating, "I'm sorry, I cannot assist with such questions."

        Context:
        {context}

        Question: {question}

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnableParallel(
                {
                    "context": _search_query | retriever,
                    "question": RunnablePassthrough(),
                }
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        final_data = chain.invoke({"question": question})
        logger.info("Question processing completed.")
        return final_data

    except Exception as e:
        logger.error(f"Error occurred while processing the question: {e}")
        raise

def k_final(question):
    # final_documents = k_load()
    final_documents = []
    Unstructured = k_main_2(final_documents,question)
    final_data = process_question(Unstructured,question)
    generated_query = k_cypher(question)
    full_query = "MATCH p=()-->() RETURN p LIMIT 200"
    full_graph_path = full_graph(full_query)
    sub_graph_path = sub_graph(generated_query)
    return final_data, generated_query,full_graph_path,sub_graph_path