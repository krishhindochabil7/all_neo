from logging_setup import setup_logger, getLogfile

# Set up the logger
logger = setup_logger(getLogfile())

from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
import os


def full_graph(query):
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "kh27042001"

    driver = GraphDatabase.driver(uri, auth=(user, password))

    logger.info("Entered full_graph function")
    # Run a Cypher query
    def run_query(query):
        logger.info("Running query: %s", query)
        with driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    def get_node_label(node):
        # Try common properties
        for key in ['name', 'title', 'label']:
            if key in node:
                return node[key]
        
        # Fallback to a unique identifier
        return str(node['id'])

    def extract_graph_elements(results):
        G = nx.DiGraph()
        
        for record in results:
            path = record['p']
            # Iterate over the nodes and relationships in the path
            for i in range(len(path.nodes) - 1):
                start_node = path.nodes[i]
                end_node = path.nodes[i + 1]
                relationship = path.relationships[i]
                
                # Get labels for the start and end nodes
                node_label_start = get_node_label(start_node)
                node_label_end = get_node_label(end_node)
                
                # Add nodes with labels
                G.add_node(node_label_start)
                G.add_node(node_label_end)
                
                # Add edge with relationship type as label
                G.add_edge(node_label_start, node_label_end, label=relationship.type)
        
        return G

    def draw_graph(G, file_path):
        logger.info("Drawing graph and saving to %s", file_path)
        pos = nx.spring_layout(G, seed=42)  # for consistent layout
        labels = nx.get_edge_attributes(G, 'label')
    
        plt.figure(figsize=(10, 8))  # Optional: Adjust the size of the figure
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Graph Visualization")
    
        # Save the figure to a file
        plt.savefig(file_path, format='png')  # You can specify the format as needed
        plt.close()  # Close the figure to free up memory

    try:
        results = run_query(query)
        G = extract_graph_elements(results)
        logger.info("Full graph image being sent to fastapi.")
        file_path = "/home/krishhindocha/Desktop/Main_neo4j/GROQ_GOOGLE/IMAGES_OUTPUT/Full_graph_image.png"  # Update this path to your desired location
        draw_graph(G, file_path)
        logger.info("Full graph image saved successfully")
        return file_path
    except Exception as e:
        logger.error("Error occurred: %s", e)
        return None
    
def sub_graph(query):
    logger.info("Entered sub_graph function")
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "kh27042001"
    driver = GraphDatabase.driver(uri, auth=(user, password))
    # Run a Cypher query
    def run_query(query):
        logger.info("Running query: %s", query)
        with driver.session() as session:
            result = session.run(query)
            return [record for record in result]

    def ensure_directory(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


    def extract_graph_elements(results):
        G = nx.DiGraph()
    
        for record in results:
            path = record['p']
            # Iterate over the nodes and relationships in the path
            for i in range(len(path.nodes) - 1):
                start_node = path.nodes[i]
                end_node = path.nodes[i + 1]
                relationship = path.relationships[i]
            
                node_id_start = start_node['id']
                node_id_end = end_node['id']
            
                # Extract labels and create nodes
                labels_start = start_node.labels
                labels_end = end_node.labels
            
                G.add_node(node_id_start, labels=labels_start)
                G.add_node(node_id_end, labels=labels_end)
            
                # Add edge with relationship type as label
                G.add_edge(node_id_start, node_id_end, label=relationship.type)
    
        return G

    def draw_graph(G, file_path):
        logger.info("Drawing graph and saving to %s", file_path)

        ensure_directory(file_path)
        
        pos = nx.spring_layout(G, seed=42)  # for consistent layout
        labels = nx.get_edge_attributes(G, 'label')

        plt.figure(figsize=(10, 8))  # Optional: Adjust the size of the figure
        nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', edge_color='gray')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.title("Graph Visualization")

        # Save the figure to a file
        plt.savefig(file_path, format='png')
        plt.close()  # Close the figure to free up memory


    try:
        results = run_query(query)
        G = extract_graph_elements(results)
        logger.info("Sub graph image being sent to fastapi.")
        file_path = "/home/krishhindocha/Desktop/Main_neo4j/GROQ_GOOGLE/IMAGES_OUTPUT/sub_graph_image.png"  # Update this path to your desired location
        draw_graph(G, file_path)
        logger.info("Sub graph image saved successfully")
        return file_path
    except Exception as e:
        logger.error("Error occurred: %s", e)
        return None

# full_query = "MATCH p=()-->() RETURN p LIMIT 200"
# full_graph(full_query)

# sub_query = "MATCH p=()-[r:HAS_RESOURCE]->() RETURN p"
# sub_graph(sub_query)

# def k_graph(sub_query):
#     logger.info("Entered k_graph function")
#     full_query = "MATCH p=()-->() RETURN p LIMIT 200"
#     full_graph(full_query)
#     sub_graph(sub_query)
