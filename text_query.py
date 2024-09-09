from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from neo4j.exceptions import CypherSyntaxError
import os
from logging_setup import setup_logger, getLogfile

# Set up the logger
logger = setup_logger(getLogfile())

class Neo4jGPTQuery:
    def __init__(self, url, user, password):
        GROQ_API_KEY = "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"
        logger.info("Initializing Neo4jGPTQuery with URL: %s", url)
        self.driver = GraphDatabase.driver(url, auth=(user, password))
        self.chat_model = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            groq_api_key=GROQ_API_KEY,
            )
        # Generate schema dynamically
        self.schema = self.generate_schema()

    def get_node_schema(self):
        logger.info("Fetching node schema")
        with self.driver.session() as session:
            result = session.run("""
                CALL db.labels() YIELD label
                CALL db.propertyKeys() YIELD propertyKey
                RETURN label, propertyKey
            """)
            nodes = {}
            for record in result:
                label = record['label']
                property_key = record['propertyKey']
                if label not in nodes:
                    nodes[label] = []
                nodes[label].append(property_key)
            return nodes

    def get_relationship_schema(self):
        logger.info("Fetching relationship schema")
        with self.driver.session() as session:
            result = session.run("""
                CALL db.relationshipTypes() YIELD relationshipType
                CALL db.propertyKeys() YIELD propertyKey
                RETURN relationshipType, propertyKey
            """)
            relationships = {}
            for record in result:
                rel_type = record['relationshipType']
                property_key = record['propertyKey']
                if rel_type not in relationships:
                    relationships[rel_type] = []
                relationships[rel_type].append(property_key)
            return relationships

    def generate_schema(self):
        logger.info("Generating schema")
        node_schema = self.get_node_schema()
        rel_schema = self.get_relationship_schema()
        
        node_props = "Node properties: " + ", ".join(f"{label}: {', '.join(props)}" for label, props in node_schema.items())
        rel_props = "Relationship properties: " + ", ".join(f"{rel_type}: {', '.join(props)}" for rel_type, props in rel_schema.items())
        rels = "Relationships: " + ", ".join(f"{rel_type}" for rel_type in rel_schema.keys())
        
        schema = f"{node_props}, {rel_props}, {rels}"
        return schema

    def query_database(self, cypher_query):
        logger.info("Running Cypher query: %s", cypher_query)
        with self.driver.session() as session:
            try:
                result = session.run(cypher_query)
                return [record for record in result]
            except CypherSyntaxError as e:
                logger.error("Cypher syntax error: %s", e)
                return "Invalid Cypher syntax"
            except Exception as e:
                logger.error("Error executing Cypher query: %s", e)
                raise

    def get_system_message(self):
        return f"""
        Task: Generate Cypher queries to query a Neo4j graph database based on the provided schema definition.
        Instructions:
        - Handle both direct and indirect questions.
        - Construct a Cypher query that focuses on the relationship and pattern rather than specific node properties.
        - Avoid including specific properties or filtering criteria unless explicitly asked for.
        - Use the provided relationship types and properties but keep the query as general as possible.
        - Use the `p` variable for the path pattern in the format: `MATCH p = ()-[r:REL_TYPE]->() RETURN p`.
        - If the question does not specify a particular node or property, use generalized patterns.

        Schema:
        {self.schema}

        Examples:
        1. Question: "What resources does the HR department provide?"
        Query: MATCH p=()-[r:PROVIDES_RESOURCES]->() RETURN p

        2. Question: "Tell me about the resources."
        Query: MATCH p=()-[r:PROVIDES_RESOURCES]->() RETURN p

        Note: Provide only the Cypher query, with no additional text or explanations. Format the query as follows:
        Example Query: MATCH p=()-[r:REL_TYPE]->() RETURN p. Do not deviate from the format at any cost
        """

    def construct_cypher(self, question, history=None):
        logger.info("Constructing Cypher query for question: %s", question)
        messages = [
            {"role": "system", "content": self.get_system_message()},
            {"role": "user", "content": question},
        ]
        if history:
            messages.extend(history)

        try:
            completions = self.chat_model.invoke(messages)
            response = completions.content.strip()  # Clean up the response
            cypher_query = response.split('\n')[0].strip()  # Ensure only the first line of the response is returned
            logger.info("Constructed Cypher query: %s", cypher_query)
            return cypher_query
        except Exception as e:
            logger.error("Error in constructing Cypher query: %s", e)
            raise

    def run(self, question, history=None):
        logger.info("Running the query with question: %s", question)
        try:
            cypher = self.construct_cypher(question, history)
            return cypher
        except Exception as e:
            logger.error("Error occurred: %s", e)
            raise

def k_cypher(question, history=None):
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "kh27042001"

    gds_db = Neo4jGPTQuery(
    url= NEO4J_URI,
    user=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

    # Run a query
    cypher = gds_db.run(question, history)
    return cypher


# if __name__ == "__main__":
#     question = "Can you tell me what the First year students lack experience with?"
#     answer = k_cypher(question)
#     print(answer)
