import os
import json
import logging
import psycopg
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Setup embedding model and LLM
embedding_model = OllamaEmbeddings(model="bge-m3:567m")
llm = ChatOllama(model="llama3.2:3b")

# Path to your JSONL file
ENTITY_DESCRIPTION_FILE = os.path.join(
    "c:/Users/deepa/data/workspace/notebooks",
    "datasets", "instance_description", "instance_description-05.jsonl"
)

# Database connection
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "dbpedia_docs"

# Connect to PGVector
logger.info(f"Connecting to PGVector collection '{COLLECTION_NAME}'...")
try:
    vectorstore = PGVector(
        connection=CONNECTION_STRING,
        embeddings=embedding_model,
        collection_name=COLLECTION_NAME,
        use_jsonb=True,
        pre_delete_collection=True
    )
    logger.info("Connection to vectorstore successful.")
except psycopg.OperationalError as e:
    logger.exception(f"Database connection error: {e}")
    exit(1)
except Exception as e:
    logger.exception(f"Unexpected error during PGVector connection: {e}")
    exit(1)

# Prompt template for summarization
summary_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an intelligent assistant tasked with summarizing entity descriptions. "
        "Your summary should be concise and cover all provided information about the entity. "
        "Include relevant details such as the entity's name, description, attributes, and any other important data. "
        "Do not add introductory phrases like 'Here is a summary'. "
        "Avoid any markdown formatting, such as bold text, bullets, or colons after labels. "
        "Your summary should be a clean, coherent text that encapsulates the entity's key characteristics without unnecessary elaboration."
    ),
    ("human", "Please summarize the following entity details:\n\n{content}")
])

# Summary generation function
def summarize_entity(description):
    messages = summary_prompt.invoke({"content": description})
    response = llm.invoke(messages)
    return response.content if hasattr(response, 'content') else str(response)

# Retrieve document by IRI
def get_embedding_by_iri(iri):
    try:
        with psycopg.connect(CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, embedding, metadata
                    FROM langchain_pg_embedding
                    WHERE metadata->>'_id' = %s
                    LIMIT 1
                """, (iri,))
                row = cur.fetchone()
                if row:
                    return row
                logger.warning(f"No embedding found for IRI: {iri}")
                return None
    except Exception as e:
        logger.error(f"Error retrieving embedding for IRI {iri}: {e}")
        return None

# Function to process and store data in parallel with retry logic and error handling
def process_and_store(filename, limit=None, batch_size=50, max_workers=4, max_retries=3, retry_delay=5):
    count = 0
    batch_docs = []

    try:
        with open(filename, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file if line.strip()]

        if limit:
            lines = lines[:limit]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for line_number, line in enumerate(lines, 1):
                try:
                    record = json.loads(line)
                    description = record.get("description", "")
                    iri = record.get("iri")
                    if not description or not iri:
                        logger.warning(f"Skipping line {line_number}: missing 'iri' or 'description'")
                        continue

                    future = executor.submit(summarize_entity, description)
                    futures[future] = (line_number, iri, description)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_number}, skipping.")

            for future in as_completed(futures):
                line_number, iri, description = futures[future]
                try:
                    summary_text = future.result()
                    full_text = f"Summary: {summary_text}\n\nOriginal: {description}"
                    doc = Document(page_content=full_text, metadata={"_id": iri})
                    batch_docs.append(doc)
                    count += 1

                    if len(batch_docs) >= batch_size:
                        success = False
                        for attempt in range(max_retries):
                            try:
                                vectorstore.add_documents(batch_docs)
                                logger.info(f"Stored batch of {len(batch_docs)} embeddings.")
                                success = True
                                break
                            except Exception as e:
                                logger.warning(f"Attempt {attempt + 1} failed to store batch: {e}")
                                time.sleep(retry_delay)
                        if not success:
                            logger.error("Failed to store batch after maximum retries.")
                        batch_docs = []

                except Exception as e:
                    logger.exception(f"Failed at line {line_number}: {e}")

        # Store any remaining docs
        if batch_docs:
            success = False
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch_docs)
                    logger.info(f"Stored final batch of {len(batch_docs)} embeddings.")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed to store final batch: {e}")
                    time.sleep(retry_delay)
            if not success:
                logger.error("Failed to store final batch after maximum retries.")

    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user. Attempting to shut down gracefully.")
        if batch_docs:
            try:
                vectorstore.add_documents(batch_docs)
                logger.info(f"Stored interrupted batch of {len(batch_docs)} embeddings.")
            except Exception as e:
                logger.error(f"Failed to store interrupted batch: {e}")

# Run with parallel processing
process_and_store(ENTITY_DESCRIPTION_FILE, limit=100, batch_size=50, max_workers=4)
