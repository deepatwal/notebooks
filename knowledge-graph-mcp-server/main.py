import logging
import os
import re
import json

from mcp.server.fastmcp import FastMCP
from SPARQLWrapper import SPARQLWrapper, JSON
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO)

try:
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
except ImportError:
    logging.error("dotenv package not found. Please install it to load environment variables.")
    exit(1)

# MCP and SPARQL setup
ENDPOINT_URL = "http://localhost:3030/dbpedia-21-05-2025/sparql"
mcp = FastMCP("KnowledgeGraphMCP")


# 1. Initialize the Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=google_api_key,
    temperature=0
)

# 2. Define the Prompt Template (your code)
prompt_template = """
You are a helpful assistant that converts natural language questions into SPARQL queries
targeting the DBpedia knowledge graph.

Use these standard prefixes:
PREFIX dbo: <http://dbpedia.org/ontology/>
PREFIX dbr: <http://dbpedia.org/resource/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

Question: {question}

SPARQL:
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=prompt_template
)

# LLMChain to connect Gemini with the prompt
chain = prompt | model | StrOutputParser()

def extract_sparql_from_markdown(markdown_text: str) -> str:
    """
    Extracts a SPARQL query from a Markdown code block.
    """
    pattern = r"```(?:\w*\n)?(.*)```"
    match = re.search(pattern, markdown_text.strip(), re.DOTALL)
    if match:
        return match.group(1).strip()
    return markdown_text.strip()

# Executes SPARQL against Fuseki
def run_sparql_query(query: str) -> list[dict]:
    try:
        sparql = SPARQLWrapper(ENDPOINT_URL)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        bindings = results.get("results", {}).get("bindings", [])
        return [{k: v["value"] for k, v in row.items()} for row in bindings]
    except Exception as e:
        logging.error(f"SPARQL query failed: {e}", exc_info=True)
        return [{"error": str(e)}]

# @mcp.tool()
def ask_kg(question: str) -> list[dict]:
    try:
        raw_llm_output = chain.invoke({"question": question})
        logging.info(f"Raw LLM Output:\n{raw_llm_output}")

        sparql_query = extract_sparql_from_markdown(raw_llm_output)
        logging.info(f"Extracted SPARQL for '{question}':\n{sparql_query}")

        return run_sparql_query(sparql_query)
    except Exception as e:
        logging.error(f"Processing failed: {e}", exc_info=True)
        return [{"error": str(e)}]


# # Run MCP server
# logging.info("Launching MCP with Gemini 2.5 Pro SPARQL interface...")
# try:
#     mcp.run()
# except KeyboardInterrupt:
#     logging.info("MCP stopped by user.")
# except Exception as e:
#     logging.error(f"Unhandled error in MCP: {e}", exc_info=True)


if __name__ == "__main__":
    results = ask_kg("What is the capital of France?")
    
    print("\n--- FINAL RESULTS ---")
    print(json.dumps(results, indent=2))