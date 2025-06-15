from mcp.server.fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

mcp = FastMCP("KnowledgeGraphMCP")
logging.info("FastMCP instance created with name 'KnowledgeGraphMCP'")

# Set your SPARQL endpoint
ENDPOINT_URL = "http://localhost:3030/dbpedia-21-05-2025/sparql"

# Add MCP tool
@mcp.tool()
def run_sparql_query(query: str) -> list[dict]:
    try:
        sparql = SPARQLWrapper(ENDPOINT_URL)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()

        # Extracting results
        bindings = results.get("results", {}).get("bindings", [])
        formatted_results = [
            {k: v["value"] for k, v in result.items()} for result in bindings
        ]
        logging.info(f"Query returned {len(formatted_results)} results.")
        return formatted_results

    except Exception as e:
        logging.error(f"SPARQL query failed: {e}")
        return [{"error": str(e)}]


logging.info("Starting KnowledgeGraphMCP server...")
try:
    mcp.run()
except KeyboardInterrupt:
    logging.info("Server stopped by user (KeyboardInterrupt).")
except Exception as e:
    logging.error(f"Server encountered an unexpected error and stopped: {e}", exc_info=True)

# if __name__ == "__main__":
#     # Example query to test against your Fuseki/SPARQL server
#     sample_query = """
    # SELECT * 
    # WHERE {
    #  ?s ?p ?o
    # } LIMIT 5
#     """

#     results = run_sparql_query(sample_query)
#     for r in results:
#         print(r)

# Let me know if you'd like to:

# Add schema browsing

# Auto-generate query suggestions

# Convert RDF results to natural language

# You're now in a great position to expand this into a smart agent interface to your knowledge graph.








