from neo4j_init import init_neo4j
from loading import loading_graph
from logging_setup import logger
from final import k_final
from dummy_final_docs import final_docs

def initialize_neo():
    # NEO4J_URI = "neo4j+s://66a5874d.databases.neo4j.io"
    # NEO4J_USERNAME = "neo4j"
    # NEO4J_PASSWORD = "bGu3bhlAksSx-KzXcbH5MgqxGLlxAIsz9px14ISGGEg"

    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "kh27042001"

    logger.info("Neo4j Credentials sent")
    graph = init_neo4j(NEO4J_URI,NEO4J_USERNAME,NEO4J_PASSWORD)
    return graph

def load_the_graph(graph,file_path):
    logger.info("Process to Load File begins.")
    final_documents = final_docs(file_path)
    logger.info("Sending Graph and Chunked documents")
    loading_graph(graph,final_documents)

def all_final(question):
    final_data, generated_query,full_graph_path,sub_graph_path = k_final(question)
    logger.info("Backend Process done.")
    return final_data,generated_query,full_graph_path,sub_graph_path

