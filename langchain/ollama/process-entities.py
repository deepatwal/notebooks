import os
import asyncio
import logging
import urllib.parse
import re
from asyncio import Lock, Semaphore
from collections import Counter, defaultdict
from typing import List, Optional
from dotenv import load_dotenv
import concurrent.futures
import argparse
import aiosqlite
import aiohttp
from more_itertools import chunked
from SPARQLWrapper import SPARQLWrapper, JSON

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

MAX_CONCURRENT_REQUESTS = 5
MAX_CONCURRENT_CLASSES = 5
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
    # class_query = r"""
    # PREFIX owl: <http://www.w3.org/2002/07/owl#>
    # SELECT ?class
    # FROM <http://dbpedia.org/model>
    # WHERE { ?class a owl:Class . 
    #   FILTER(regex(STRAFTER(STR(?class), "http://dbpedia.org/ontology/"), "^[\\x00-\\x7F]+$")) }
    # ORDER BY ?class
    # """
    try:
    #     sparql = get_sparql(return_format=JSON)
    #     sparql.setQuery(class_query)
    #     results = sparql.query().convert()
    #     return [b['class']['value'] for b in results['results']['bindings']]
        return  ['http://dbpedia.org/ontology/Organisation']
    except Exception as e:
        logger.exception(f"[Error] Fetching classes: {e}")
        return []

def fetch_instances_for_class(ontology_class: str) -> List[str]:
    logger.info(f"Fetching instances of class {ontology_class}")
    instance_query = f"""
    SELECT ?instance
    WHERE {{ BIND(<{ontology_class}> AS ?entity) ?instance a ?entity . }}
    ORDER BY ?instance
    """
    try:
        sparql = get_sparql(return_format=JSON)
        sparql.setQuery(instance_query)
        results = sparql.query().convert()
        return [b['instance']['value'] for b in results['results']['bindings']]
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

def clean_value(rdf_term: Optional[str]) -> str:
    if rdf_term is None:
        return ""
    t = rdf_term.strip()
    if t.startswith('<') and t.endswith('>'):
        return get_label_from_uri(t)
    if t.startswith('"'):
        m = re.match(r'"(.*?)"', t)
        return m.group(1) if m else t
    return t

async def describe_instance(instance_iri: str, retries: int = 3, delay: float = 1.0) -> Optional[str]:
    """
    Fetches a description of the given instance IRI using the SPARQL DESCRIBE query.

    Args:
        instance_iri (str): The IRI of the instance to describe.
        retries (int): Number of retry attempts in case of failure.
        delay (float): Initial delay between retries, with exponential backoff.

    Returns:
        Optional[str]: The description of the instance in N3 format, or None if failed.
    """
    logger.info(f"Describing instance {instance_iri}")
    query = f"DESCRIBE <{instance_iri}>"
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(SPARQL_ENDPOINT, data={"query": query}, headers={"Accept": "text/n3"}) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"SPARQL query failed with status {response.status}: {await response.text()}")
        except Exception as e:
            logger.warning(f"[Retry {attempt}/{retries}] Error describing {instance_iri}: {e}")
            if attempt < retries:
                await asyncio.sleep(delay * (2 ** (attempt - 1)))
    logger.error(f"Failed to describe instance after {retries} attempts: {instance_iri}")
    return None

def process_n3_simplified(n3_data: str, delimiters: str = " ;,.", log_skipped: bool = True) -> str:
    """
    Process N3 data and extract triples in a simplified format.

    Args:
        n3_data (str): The N3 data to process.
        delimiters (str): Allowed delimiters for triples (default: " ;,." ).
        log_skipped (bool): Whether to log skipped triples (default: True).

    Returns:
        str: A summary of the processed triples.
    """
    triples, subjects = [], []
    # Create a regex pattern dynamically based on allowed delimiters
    delimiter_pattern = f"[{re.escape(delimiters)}]"
    pat = re.compile(rf'^\s*(<[^>]+>|_:\S+)\s+(<[^>]+>)\s+(.*)\s*{delimiter_pattern}\s*$')

    for ln in n3_data.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#'):
            continue
        # Normalize line endings to ensure consistent parsing
        if ln[-1] in delimiters:
            ln = ln[:-1] + '.'
        m = pat.match(ln)
        if m:
            s, p, o = m.groups()
            triples.append((s, p, o))
            if s.startswith('<'):
                subjects.append(s)
        elif log_skipped:
            logger.warning(f"Skipping malformed triple: {ln}")

    if not triples:
        return "No valid triples found."
    if not subjects:
        return "No URI subjects found."

    main = Counter(subjects).most_common(1)[0][0]
    subj_iri = main.strip('<>')
    main_lbl = get_label_from_uri(main)
    props, inc = defaultdict(set), set()

    for s, p, o in triples:
        lbl = get_label_from_uri(p).lower()
        o_clean = o.strip() if o else ""
        if s == main:
            props[lbl].add(clean_value(o_clean))
        elif o_clean == main:
            inc.add((clean_value(s), lbl, main_lbl))

    out = [f"IRI: {subj_iri}", f"label: {next(iter(props.get('label', [])), main_lbl)}"]
    if props:
        out.append("\nOutgoing Relationships:")
        for k in sorted(props):
            out.append(f"{k}: {', '.join(sorted(props[k]))}")
    if inc:
        out.append("\nIncoming Relationships:")
        for s, p, o in sorted(inc):
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
            data = await describe_instance(instance_iri)

        if data:
            desc = process_n3_simplified(data)
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
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
            loop.set_default_executor(executor)

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
