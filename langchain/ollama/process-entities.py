import os
import asyncio
import logging
import urllib.parse
import re
from asyncio import Lock, Semaphore
from collections import defaultdict
from typing import List, Optional
from dotenv import load_dotenv
import aiosqlite
import aiohttp
from rdflib import Graph, URIRef, Literal, BNode
from SPARQLWrapper import SPARQLWrapper, JSON
from more_itertools import chunked
import argparse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger: logging.Logger = logging.getLogger(__name__)


# Configuration
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_BASE_URL")
GRAPHDB_REPOSITORY = os.getenv("GRAPHDB_REPOSITORY")

# Validate environment variables with a user-friendly error message
if not GRAPHDB_BASE_URL or not GRAPHDB_REPOSITORY:
    logger.error("Environment variables GRAPHDB_BASE_URL and GRAPHDB_REPOSITORY are required but not set.")
    raise ValueError("GRAPHDB_BASE_URL and GRAPHDB_REPOSITORY must be set in the environment variables")

SPARQL_ENDPOINT = urllib.parse.urljoin(GRAPHDB_BASE_URL.rstrip('/') + '/', f"repositories/{GRAPHDB_REPOSITORY}")

OUTPUT_FILENAME_DIR = os.path.join("c:\\Users\\deepa\\data\\workspace\\notebooks", "datasets", "cache")
DB_PATH = os.path.join(OUTPUT_FILENAME_DIR, "cache.db")

MAX_CONCURRENT_REQUESTS = 1
MAX_CONCURRENT_CLASSES = 1
BATCH_SIZE = 50

instance_lock = Lock()
class_lock = Lock()
output_file_lock = Lock()
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
class_semaphore = Semaphore(MAX_CONCURRENT_CLASSES)

def get_sparql(return_format):
    """Initializes and returns a SPARQLWrapper instance."""
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(return_format)
    return sparql

def fetch_classes() -> List[str]:
    logger.info("Fetching ontology classes from model graph")
    try:
        return  ['http://dbpedia.org/ontology/Organisation']
    except Exception as e:
        logger.exception(f"[Error] Fetching classes: {e}")
        return []

def fetch_instances_for_class(ontology_class: str) -> List[str]:
    logger.info(f"Fetching instances of class {ontology_class}")
    instance_query = f"""
    SELECT ?instance
    WHERE {{ ?instance a <{ontology_class}> . }}
    ORDER BY ?instance
    LIMIT 10
    """
    try:
        # sparql = get_sparql(return_format=JSON)
        # sparql.setQuery(instance_query)
        # results = sparql.query().convert()
        # return [b['instance']['value'] for b in results['results']['bindings']]
        return ['http://dbpedia.org/resource/Volvo']
    except Exception as e:
        logger.exception(f"[Error] Fetching instances for {ontology_class}: {e}")
        return []

def get_label_from_uri(uri_str: str) -> str:
    if not (isinstance(uri_str, str) and uri_str.startswith('<') and uri_str.endswith('>')):
        return str(uri_str)
    uri = uri_str.strip('<>')
    try:
        parsed = urllib.parse.urlparse(uri)
        part = parsed.fragment or uri.split('/')[-1]
        decoded = urllib.parse.unquote(part)
        label = re.sub(r'(?<!^)(?=[A-Z])', ' ', decoded.replace('_', ' ').replace('-', ' '))
        return re.sub(r'\s+', ' ', label).strip() or part
    except:
        return uri

async def describe_instance(instance_iri: str, retries: int = 3, delay: float = 1.0) -> Optional[Graph]:
    """Fetches instance description using rdflib's N3Parser for robust parsing."""
    logger.info(f"Describing instance {instance_iri}")
    query = f"DESCRIBE <{instance_iri}>"
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(SPARQL_ENDPOINT, data={"query": query}, headers={"Accept": "text/n3"}) as response:
                    if response.status == 200:
                        n3_text = await response.text()
                        g = Graph()
                        g.parse(data=n3_text, format="n3")  # Use rdflib's parser
                        return g
                    else:
                        logger.warning(f"SPARQL query failed: {response.status} - {await response.text()}")
        except Exception as e:
            logger.warning(f"[Retry {attempt}/{retries}] Error describing {instance_iri}: {e}")
            if attempt < retries:
                await asyncio.sleep(delay * (2 ** (attempt - 1)))
    logger.error(f"Failed to describe {instance_iri} after {retries} attempts.")
    return None

def clean_value_rdflib(node):
    """Cleans RDF node values (URIRef, Literal, BNode)."""
    if isinstance(node, URIRef):
        return get_label_from_uri(str(node))
    elif isinstance(node, Literal):
        return str(node.value)  # Get the literal value directly
    elif isinstance(node, BNode):
        return str(node)  # Or handle blank nodes as needed
    return ""

def process_n3_with_rdflib(graph: Graph) -> str:
    """Processes the RDF graph using rdflib."""
    if not graph:
        return "No graph data provided."

    subj_iri = None
    props = defaultdict(list)
    inc = []

    for s, p, o in graph:
        if isinstance(s, URIRef):  # Identify the main subject (first URIRef encountered)
            if subj_iri is None:
                subj_iri = str(s)

        p_label = get_label_from_uri(str(p)).lower()
        o_value = clean_value_rdflib(o)

        if str(s) == subj_iri:
            props[p_label].append(o_value)
        elif str(o) == subj_iri and isinstance(s, URIRef): # Only consider incoming from URIRefs
            inc.append((clean_value_rdflib(s), p_label, get_label_from_uri(subj_iri)))

    if subj_iri is None:
        return "No URIRef subjects found in the graph."

    out = [f"IRI: {subj_iri}", f"label: {next(iter(props.get('label', [])), get_label_from_uri(subj_iri))}" ]
    if props:
        out.append("\nOutgoing Relations:")
        for k in sorted(props):
            out.append(f"{k}: {', '.join(sorted(props[k]))}")
    if inc:
        out.append("\nIncoming Relations:")
        # Ensure all elements in 'inc' are properly converted to RDF-compatible types before sorting
        inc = [(clean_value_rdflib(s), clean_value_rdflib(p), clean_value_rdflib(o)) for s, p, o in inc if isinstance(s, (URIRef, Literal, BNode)) and isinstance(p, (URIRef, Literal, BNode)) and isinstance(o, (URIRef, Literal, BNode))]
        for s, p, o in sorted(inc, key=lambda triple: (str(triple[0]), str(triple[1]), str(triple[2]))):
            out.append(f"({s}, {p}, {o})")
    return '\n'.join(out)

async def process_instance_worker(instance_iri: str, conn: aiosqlite.Connection):
    try:
        async with conn.execute("SELECT processed FROM iri_cache_organisation WHERE iri = ?", (instance_iri,)) as cursor:
            row = await cursor.fetchone()
            if row is not None and row[0] == 1:
                logger.debug(f"Skipping already processed IRI: {instance_iri}")
                return

        async with request_semaphore:
            logger.debug(f"Acquired semaphore for instance: {instance_iri}")
            graph = await describe_instance(instance_iri)

        if graph:
            desc = process_n3_with_rdflib(graph)
            await conn.execute(
                "INSERT OR REPLACE INTO iri_cache_organisation (iri, processed, data) VALUES (?, ?, ?)",
                (instance_iri, 1, desc)
            )
            await conn.commit()
            logger.info(f"Processed and saved instance: {instance_iri}")
        else:
            await conn.execute(
                "INSERT OR REPLACE INTO iri_cache_organisation (iri, processed, data) VALUES (?, ?, '')",
                (instance_iri, 0)
            )
            await conn.commit()
            logger.warning(f"No data returned for instance: {instance_iri}")
    except Exception as e:
        logger.exception(f"[Error] Worker {instance_iri}: {e}")
        await conn.execute(
            "INSERT OR REPLACE INTO iri_cache_organisation (iri, processed, data) VALUES (?, ?, '')",
            (instance_iri, 0)
        )
        await conn.commit()
    finally:
        logger.debug(f"Released semaphore for instance: {instance_iri}")

async def process_class_worker(owl_class: str, conn: aiosqlite.Connection):
    async with class_semaphore:
        instances = fetch_instances_for_class(owl_class)
        if not instances:
            return
        for chunk in chunked(instances, BATCH_SIZE):
            await asyncio.gather(*(process_instance_worker(i, conn) for i in chunk))
            logger.info(f"Processed {len(chunk)} instances for class {owl_class}")
        logger.info(f"Processed class {owl_class}: {len(instances)} instances")

async def main():
    os.makedirs(OUTPUT_FILENAME_DIR, exist_ok=True)

    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging for better concurrency
        try:
            loop = asyncio.get_running_loop()
            classes = fetch_classes()
            if not classes:
                logger.warning("No classes fetched. Ensure the SPARQL endpoint is configured correctly.")
                return

            tasks = [process_class_worker(owl_class, conn) for owl_class in classes]
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.exception(f"Error in main: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
