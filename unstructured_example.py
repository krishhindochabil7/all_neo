from langchain_community.vectorstores import Neo4jVector
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from loading import loading_graph
from logging_setup import setup_logger, getLogfile

# Set up the logger
logger = setup_logger(getLogfile())
GROQ_API_KEY="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"
def getGraphDB(final_documents):
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "kh27042001"
    GOOGLE_API_KEY="AIzaSyDQM51t_NqqH5Vc00IqPtxWzkmdtDCYgjI"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    # The Neo4jVector Module will connect to Neo4j and create a vector index if needed.
    db = Neo4jVector.from_documents(
        final_documents, embeddings, url = uri, username = user, password = password
    )

    logger.info("Created Vector Index for Unstructured Data")
    return db

def getAnswerForQuery(db, query):
    if not db:
        logger.error("No database index available")
        return []  # or handle it appropriately

    docs_with_score = db.similarity_search_with_score(query, k=2)
    logger.info("Unstructured Data is ready to be sent")
    return docs_with_score

def k_main_2(final_documents, question):
    logger.info("Creating Unstructured Data")
    neo4jdb = getGraphDB(final_documents)
    answer = getAnswerForQuery(neo4jdb, question)
    logger.info("Sending Unstructured Data")
    return answer

# if __name__ == "__main__":
#     file_path = "/home/krishhindocha/Downloads/HR Policy Manual.pdf"
#     final_documents = loading_graph(file_path)
#     question = "Can you tell who conducts exit interview"
#     answer = k_main_2(final_documents, question)
#     print(answer)
