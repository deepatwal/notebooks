import logging
import os
import re

from mcp.server.fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from rdflib import Graph, RDF, RDFS, OWL

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
    logger.error("Failed to load environment variables. Make sure you have a .env file with GOOGLE_API_KEY set.")
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
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0
)

# ------------------------------------------------------------------------------
# Prompt template with ontology support
# ------------------------------------------------------------------------------
prompt_template = """
You are an expert SPARQL assistant.

Convert the following natural language question into a SPARQL query.

Use the provided ontology to understand the structure and relationships of the data.
Ontology: {ontology}

Convert the question into a SPARQL query that can be executed against the SPARQL endpoint.
Question: {question}

SPARQL:
"""

prompt = PromptTemplate(
    input_variables=["question", "ontology"],
    template=prompt_template
)

# Chain
chain = prompt | model | StrOutputParser()

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
        raw_llm_output = chain.invoke({"question": question, "ontology": ontology_summary})
        logger.info(f"LLM Output:\n{raw_llm_output}")

        sparql_query = extract_sparql_from_markdown(raw_llm_output)
        logger.info(f"Extracted SPARQL:\n{sparql_query}")

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