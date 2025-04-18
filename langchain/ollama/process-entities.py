import os
import asyncio
import logging
import aiofiles
import json
import urllib.parse
import re
from asyncio import Lock, Semaphore
from collections import Counter, defaultdict
from typing import List, Set, Optional
from SPARQLWrapper import SPARQLWrapper, N3, JSON
from more_itertools import chunked
import concurrent.futures
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

# Configuration
GRAPHDB_BASE_URL = os.getenv("GRAPHDB_BASE_URL")
GRAPHDB_REPOSITORY = os.getenv("GRAPHDB_REPOSITORY")

if not GRAPHDB_BASE_URL or not GRAPHDB_REPOSITORY:
    raise ValueError("GRAPHDB_BASE_URL and GRAPHDB_REPOSITORY must be set in the environment variables")

SPARQL_ENDPOINT = urllib.parse.urljoin(GRAPHDB_BASE_URL.rstrip('/') + '/', f"repositories/{GRAPHDB_REPOSITORY}")

OUTPUT_FILENAME_DIR = os.path.join("c:\\Users\\deepa\\data\\workspace\\notebooks", "datasets", "instance_description")
OUTPUT_FILENAME = os.path.join(OUTPUT_FILENAME_DIR, "instance_description.jsonl")

FAILED_LOG_DIR = os.path.join("c:\\Users\\deepa\\data\\workspace\\notebooks", "datasets", "failed")
FAILED_CLASS_LOG = os.path.join(FAILED_LOG_DIR, "failed_class_iri.txt")
FAILED_INSTANCE_LOG = os.path.join(FAILED_LOG_DIR, "failed_instance_iri.txt")

CPU_CORES = 24
MAX_CONCURRENT_REQUESTS = 20
MAX_CONCURRENT_CLASSES = 6  # 25% of CPU_CORES
BATCH_SIZE = 100

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

instance_lock = Lock()
class_lock = Lock()
output_file_lock = Lock()
request_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)
class_semaphore = Semaphore(MAX_CONCURRENT_CLASSES)

async def log_failed_instance(instance_iri: str):
    async with instance_lock:
        async with aiofiles.open(FAILED_INSTANCE_LOG, 'a', encoding='utf-8') as f:
            await f.write(instance_iri + "\n")

async def log_failed_class(ontology_class: str):
    async with class_lock:
        async with aiofiles.open(FAILED_CLASS_LOG, 'a', encoding='utf-8') as f:
            await f.write(ontology_class + "\n")

def get_sparql(return_format=JSON):
    sparql = SPARQLWrapper(SPARQL_ENDPOINT)
    sparql.setReturnFormat(return_format)
    return sparql

def fetch_classes() -> List[str]:
    logger.info("Fetching ontology classes from model graph")
    class_query = r"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    SELECT ?class
    FROM <http://dbpedia.org/model>
    WHERE { ?class a owl:Class . 
    FILTER(regex(STRAFTER(STR(?class), "http://dbpedia.org/ontology/"), "^[\\x00-\\x7F]+$")) }
    ORDER BY ?class
    """
    try:
        sparql = get_sparql(return_format=JSON)
        sparql.setQuery(class_query)
        results = sparql.query().convert()
        return [b['class']['value'] for b in results['results']['bindings']]
    except Exception as e:
        logger.exception(f"[Error] Fetching classes: {e}")
        return []

def fetch_instances_for_class(ontology_class: str) -> List[str]:
    logger.info(f"Fetching instances of class {ontology_class}")
    instance_query = f"""
    SELECT ?instance
    FROM <http://dbpedia.org/model>
    FROM <http://dbpedia.org/data>
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
        asyncio.create_task(log_failed_class(ontology_class))
        return []

def get_label_from_uri(uri_str: str) -> str:
    if not (isinstance(uri_str, str) and uri_str.startswith('<') and uri_str.endswith('>')):
        return str(uri_str)
    uri = uri_str.strip('<>')
    try:
        parsed = urllib.parse.urlparse(uri)
        part = parsed.fragment or uri.split('/')[-1]
        decoded = urllib.parse.unquote(part)
        label = re.sub(r'(?<!^)(?=[A-Z])', ' ', decoded.replace('_',' ').replace('-',' '))
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

def describe_instance(instance_iri: str, retries: int = 3, delay: float = 1.0) -> Optional[str]:
    logger.info(f"Describing instance {instance_iri}")
    query = f"DESCRIBE <{instance_iri}>"
    for attempt in range(1, retries + 1):
        try:
            sparql = get_sparql(return_format=N3)
            sparql.setQuery(query)
            res = sparql.query().convert()
            return res.decode('utf-8') if isinstance(res, bytes) else str(res)
        except Exception as e:
            logger.warning(f"[Retry {attempt}/{retries}] Error describing {instance_iri}: {e}")
            if attempt < retries:
                time.sleep(delay * (2 ** (attempt - 1)))
            else:
                logger.exception(f"[Error] Failed after {retries} retries: {instance_iri}")
    return None

def process_n3_simplified(n3_data: str) -> str:
    triples, subjects = [], []
    pat = re.compile(r'^\s*(<[^>]+>|_:\\S+)\s+(<[^>]+>)\s+(.*)\s*\.\s*$')
    for ln in n3_data.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith('#'): continue
        m = pat.match(ln)
        if m:
            s,p,o = m.groups(); triples.append((s,p,o))
            if s.startswith('<'): subjects.append(s)
        else:
            logger.warning(f"Skipping malformed triple: {ln}")
    if not triples: return "No valid triples found."
    if not subjects: return "No URI subjects found."
    main = Counter(subjects).most_common(1)[0][0]
    subj_iri = main.strip('<>')
    main_lbl = get_label_from_uri(main)
    props, inc = defaultdict(set), set()
    for s,p,o in triples:
        lbl = get_label_from_uri(p).lower()
        o_clean = o.strip() if o else ""
        if s == main:
            props[lbl].add(clean_value(o_clean))
        elif o_clean == main:
            inc.add((clean_value(s), lbl, main_lbl))
    out = [f"IRI: {subj_iri}", f"label: {next(iter(props.get('label', [])), main_lbl)}"]
    if props:
        out.append("\nOutgoing Relationships:")
        for k in sorted(props): out.append(f"{k}: {', '.join(sorted(props[k]))}")
    if inc:
        out.append("\nIncoming Relationships:")
        for s,p,o in sorted(inc): out.append(f"({s}, {p}, {o})")
    return '\n'.join(out)

async def process_instance_worker(instance_iri: str):
    try:
        async with request_semaphore:
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, describe_instance, instance_iri)
        if data:
            desc = process_n3_simplified(data)
            rec = {"iri": instance_iri, "description": desc}
            async with output_file_lock:
                async with aiofiles.open(OUTPUT_FILENAME, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(rec) + '\n')
            logger.debug(f"Saved: {instance_iri}")
    except Exception as e:
        logger.exception(f"[Error] Worker {instance_iri}: {e}")
        asyncio.create_task(log_failed_instance(instance_iri))

async def process_class_worker(owl_class: str, processed_iris: Set[str], resume: bool):
    async with class_semaphore:
        instances = fetch_instances_for_class(owl_class)
        if not instances:
            return

        if resume:
            instances = [iri for iri in instances if iri not in processed_iris]
            if not instances:
                logger.info(f"Skipping class {owl_class} (all instances already processed)")
                return

        for chunk in chunked(instances, BATCH_SIZE):
            await asyncio.gather(*(process_instance_worker(i) for i in chunk))
        logger.info(f"Processed class {owl_class}: {len(instances)} instances")

async def load_processed_iris() -> Set[str]:
    if not os.path.exists(OUTPUT_FILENAME):
        return set()
    processed = set()
    async with aiofiles.open(OUTPUT_FILENAME, 'r', encoding='utf-8') as f:
        async for line in f:
            try:
                record = json.loads(line)
                iri = record.get("iri")
                if iri:
                    processed.add(iri)
            except Exception as e:
                logger.warning(f"Failed to parse line in output: {e}")
    logger.info(f"Loaded {len(processed)} processed IRIs from existing output.")
    return processed

async def main(resume: bool = False):
    os.makedirs(OUTPUT_FILENAME_DIR, exist_ok=True)
    os.makedirs(FAILED_LOG_DIR, exist_ok=True)

    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
    loop.set_default_executor(executor)

    processed_iris = await load_processed_iris() if resume else set()

    classes = fetch_classes()
    tasks = [process_class_worker(owl_class, processed_iris, resume) for owl_class in classes]
    await asyncio.gather(*tasks)

    try:
        fail_i = 0
        async with aiofiles.open(FAILED_INSTANCE_LOG, 'r') as f:
            async for _ in f:
                fail_i += 1
        logger.info(f"Failed instances: {fail_i}")
    except: pass
    try:
        fail_c = 0
        async with aiofiles.open(FAILED_CLASS_LOG, 'r') as f:
            async for _ in f:
                fail_c += 1
        logger.info(f"Failed classes: {fail_c}")
    except: pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help="Skip already processed IRIs")
    args = parser.parse_args()
    asyncio.run(main(resume=args.resume))
