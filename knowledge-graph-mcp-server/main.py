import logging
import os
import re

from mcp.server.fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
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
# Load Few-Shot Examples
# ------------------------------------------------------------------------------
few_shot_examples = None
few_shot_example_file = r"C:\Users\deepa\data\workspace\notebooks\datasets\few-shot-example\few_shot_example.txt"
with open(few_shot_example_file, "r", encoding="utf-8") as f:
    few_shot_examples = f.read()

# ------------------------------------------------------------------------------
# Gemini model initialization
# ------------------------------------------------------------------------------
# chat_ollama = ChatOllama(model="gemma3:12b", temperature=0)
chat_ollama = ChatOllama(model="gemma3:12b", temperature=0)
chat_ollama

# ------------------------------------------------------------------------------
# Prompt template with ontology support
# ------------------------------------------------------------------------------
sparql_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a SPARQL query generator.

You will use the following named graphs in the SPARQL query:
- Ontology Graph: <{ontology_graph}>
- Data Graph: <{data_graph}>

Given the Ontology:
{ontology}

Understand the core elements such as:
- Classes
- Object Properties
- Data Properties
- Annotation Properties
- Relationships between classes
- The complete structure of the Ontology

IMPORTANT:
- Always use all constructs from the ontology namespace `http://dbpedia.org/ontology/`, including Classes, Object Properties, Data Properties, and Annotation Properties when generating queries.
- For individuals or instance data, always use resources from the namespace `http://dbpedia.org/resource/`.
- Use the following standard PREFIX declarations at the start of every query and refer to these prefixes consistently:
  PREFIX owl: <http://www.w3.org/2002/07/owl#>  
  PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>  
  PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>  
  PREFIX dbo: <http://dbpedia.org/ontology/>  
  PREFIX dbr: <http://dbpedia.org/resource/>  
  PREFIX foaf: <http://xmlns.com/foaf/0.1/>  
  PREFIX skos: <http://www.w3.org/2004/02/skos/core#>  
  PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>  
  PREFIX text: <http://jena.apache.org/text#>

Strict requirements:
- You must generate SPARQL queries exclusively using classes, properties, and individuals defined in the provided ontology.
- Do NOT invent or use any terms or URIs that do not appear in the ontology content above.
- Do NOT use FILTER, REGEX, or STR functions for label matching instead use full-text search with `text:query` to locate URI or IRI.
- If a concept or property is not present in the ontology, DO NOT attempt to query or guess its URI or IRI.
- Always reference ontology terms with the dbo: prefix and individuals with the dbr: prefix.
- Every SPARQL query **must** include the `FROM <{ontology_graph}>` and `FROM <{data_graph}>` clauses to indicate the graphs being queried.
- User's question must be answered solely based on the ontology, generate a SPARQL query using only valid ontology terms.

Your tasks:
- Understand what the provided the ontology.
- Based on the user's question below, generate a valid SPARQL query.
- Ensure the query strictly conforms to the ontology’s structure and namespace usage.
- Output only the SPARQL query.
     
Here are some examples of questions and their SPARQL equivalents (follow these examples closely):
{examples}

The user’s question is:
{question}
""")
])

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
            ontology=ontology_summary,
            examples=few_shot_examples,
            ontology_graph="https://www.sw.org/dbpedia/ontology",
            data_graph="https://www.sw.org/dbpedia/data",
            question=question
        )

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
    # results = ask_kg("What is the capital of France?")
    results = ask_kg("What is the country code and capital of France?")

    print("\n--- FINAL RESULTS ---")
    print(json.dumps(results, indent=2))
