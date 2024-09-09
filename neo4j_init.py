from langchain_community.graphs import Neo4jGraph
from logging_setup import logger  # Import the logger from logging_setup.py
import os

def init_neo4j(NEO4J_URI,NEO4J_USERNAME,NEO4J_PASSWORD):

    # NEO4J_URI = "bolt://localhost:7687"
    # NEO4J_USERNAME = "neo4j"
    # NEO4J_PASSWORD = "kh27042001"

    os.environ["NEO4J_URI"] = NEO4J_URI
    os.environ["NEO4J_USERNAME"] = NEO4J_USERNAME
    os.environ["NEO4J_PASSWORD"] = NEO4J_PASSWORD

    logger.info("Received Neo4j credentials.")
    graph = Neo4jGraph()
    logger.info("Neo4j Successfully Initialized")
    return graph