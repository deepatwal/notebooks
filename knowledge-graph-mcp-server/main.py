import logging
import os
import re

from mcp.server.fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from rdflib import Graph

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Load environment variables
# ------------------------------------------------------------------------------
try:
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
except ImportError:
    logger.error(
        "Failed to load environment variables. Make sure you have a .env file with GOOGLE_API_KEY set.")
    exit(1)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
ENDPOINT_URL = "http://localhost:3030/dbpedia-21-05-2025/sparql"
ONTOLOGY_FILE = r"C:\Users\deepa\data\workspace\notebooks\datasets\dbpedia-21-05-2025-ontology\ontology_type=parsed.owl"

# ------------------------------------------------------------------------------
# Load ontology and summarize for the LLM
# ------------------------------------------------------------------------------
def load_ontology_summary(path: str) -> str:
    g = Graph()
    g.parse(path, format="xml")
    return g.serialize(format="nt")


ontology_summary = load_ontology_summary(ONTOLOGY_FILE)

# ------------------------------------------------------------------------------
# Gemini model initialization
# ------------------------------------------------------------------------------
# chat_ollama = ChatOllama(model="gemma3:12b", temperature=0)
chat_ollama = ChatOllama(model="deepseek-r1:14b", temperature=0)
chat_ollama

# ------------------------------------------------------------------------------
# Prompt template with ontology support
# ------------------------------------------------------------------------------
sparql_prompt = ChatPromptTemplate.from_messages([(
    "system", """You are an expert SPARQL assistant.

    Convert the following natural language question into a SPARQL query.

    Use the provided ontology to understand the core elements such as:
    - Classes
    - Object Properties
    - Data Properties
    - Annotation Properties
    - Relationships between classes
    - Domain and range of properties
    - Hierarchy of classes
    - Inverse relationships
    - Cardinality constraints
    - Any other relevant information that can help in constructing the SPARQL query.
    - The ontology is provided in the form of a summary.
    Ontology: {ontology}

    Guidelines:
    - Only use IRIs that are explicitly present in the provided ontology.
    - Do not guess or hallucinate IRIs, property names, or class names.
    - If multiple IRIs could match a term (e.g., `dbo:capital` vs `dbp:capital`), choose the one **actually defined** in the ontology and aligned with class/property relationships.
    - Always include `PREFIX` declarations for every prefix you use in the query.
    - If a prefix is not declared in the ontology, use full IRIs.
    - Prefer ontology-defined object properties over instance-level data properties.
    - Use standard SPARQL syntax. Include filters (e.g., `FILTER`, `regex`) if appropriate.

    Convert the natural language question into a SPARQL query that can be executed against the provided SPARQL endpoint.
    Question: {question}

    SPARQL:
    """
)])

# ------------------------------------------------------------------------------
# Extract SPARQL from LLM output
# ------------------------------------------------------------------------------
def extract_sparql_from_markdown(markdown_text: str) -> str:
    pattern = r"```(?:\w*\n)?(.*)```"
    match = re.search(pattern, markdown_text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown_text.strip()

# ------------------------------------------------------------------------------
# Query the SPARQL endpoint
# ------------------------------------------------------------------------------
def run_sparql_query(query: str) -> list[dict]:
    try:
        sparql = SPARQLWrapper(ENDPOINT_URL)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        return [{k: v["value"] for k, v in row.items()} for row in bindings]
    except Exception as e:
        logger.error(f"SPARQL query failed: {e}", exc_info=True)
        return [{"error": str(e)}]


# ------------------------------------------------------------------------------
# FastMCP tool: ask_kg
# ------------------------------------------------------------------------------
mcp = FastMCP("KnowledgeGraphMCP")

# @mcp.tool()
def ask_kg(question: str) -> list[dict]:
    try:
        logger.info(f"Received question: {question}")

        formatted_messages = sparql_prompt.format_messages(
            question=question,
            ontology=ontology_summary
        )

        # logger.info(f"formatted_messages:\n{formatted_messages}")
        response = chat_ollama.invoke(formatted_messages)
        sparql_query = extract_sparql_from_markdown(response.content)
        logger.info(f"Extracted SPARQL:\n\n{sparql_query}")

        return run_sparql_query(sparql_query)
    except Exception as e:
        logger.error(f"Failed to process question: {e}", exc_info=True)
        return [{"error": str(e)}]


# ------------------------------------------------------------------------------
# Run MCP Server
# ------------------------------------------------------------------------------
# logger.info("Starting KnowledgeGraphMCP...")
# try:
#     mcp.run()
# except KeyboardInterrupt:
#     logger.info("MCP stopped by user.")
# except Exception as e:
#     logger.error(f"Unexpected error in MCP: {e}", exc_info=True)

# ------------------------------------------------------------------------------
# Run MCP Server in debug mode
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    results = ask_kg("What is the capital of France?")

    print("\n--- FINAL RESULTS ---")
    print(json.dumps(results, indent=2))
