import os
import asyncio
import logging
import urllib.parse
import re
from asyncio import Lock, Semaphore
from collections import defaultdict
from typing import List, Optional, Tuple
from dotenv import load_dotenv
import aiosqlite
import aiohttp
from rdflib import Graph, URIRef, Literal, BNode
from SPARQLWrapper import SPARQLWrapper
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
        return ['http://dbpedia.org/resource/FC_Bihor_Oradea_(1958)']
    except Exception as e:
        logger.exception(f"[Error] Fetching instances for {ontology_class}: {e}")
        return []

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

def get_label_from_uri(node) -> str:
    """Extracts the last fragment of a URIRef or handles other RDF node types."""
    if isinstance(node, URIRef):
        return node.fragment or node.rsplit("/", 1)[-1]
    elif isinstance(node, Literal):
        return str(node.value)  # Return the literal value
    elif isinstance(node, BNode):
        return str(node)  # Handle blank nodes as needed
    return str(node)  # Fallback for other types


def process_n3_with_rdflib(graph: Graph, instance_iri) -> str:
    """Processes the RDF graph using rdflib."""
    if not graph or not instance_iri:
        return "No graph data or instance IRI provided."

    props = defaultdict(list)
    incoming: List[Tuple[str, str, str]] = []
 
    for s, p, o in graph:
        s_label = get_label_from_uri(s)
        p_label = get_label_from_uri(p)
        o_label = get_label_from_uri(o)

        s_label = s_label.replace('_', ' ').replace('__', ' ')
        p_label = p_label.replace('_', ' ').replace('__', ' ')
        o_value = o_label.replace('_', ' ').replace('__', ' ')

        if str(s) == instance_iri:
            if "homepage" in p_label or "website" in p_label:
                props[p_label].append(str(o))
            else:
                props[p_label].append(o_label)
        else:
            incoming.append((s_label, p_label, o_label))

    description = []
    description.append(f"IRI: {instance_iri}")
    for prop, values in props.items():
        description.append(f"{prop}: {', '.join(values)}")

    for s_label, p_label, o_value in incoming:
        description.append(f"({s_label} {p_label} {o_value})")

    description_str = "\n".join(description)
    logger.info(f"description_str: {description_str}")
    return description_str

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
            desc = process_n3_with_rdflib(graph, instance_iri)
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
