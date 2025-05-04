import os
import json
import logging
import psycopg
import time
import asyncio
import aiosqlite

from dotenv import load_dotenv
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

# Database connection for PGVector
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
COLLECTION_NAME = "dbpedia_docs_organisation_04-05-2025"

# Local file and SQLite DB path setup
OUTPUT_FILENAME_DIR = os.path.join("c:\\Users\\deepa\\data\\workspace\\notebooks", "datasets", "cache")
DB_PATH = os.path.join(OUTPUT_FILENAME_DIR, "cache.db")
SQLITE_DB_PATH = DB_PATH

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
        "Your summary should cover all provided information about the entity. "
        "Do not add introductory phrases like 'Here is a summary'. "
        "Avoid any markdown formatting, such as bold text, bullets, or colons after labels. "
        "Your summary should be a clean, coherent text that encapsulates the entity's characteristics provided without unnecessary elaboration."
    ),
    ("human", "Please summarize the following entity details:\n\n{content}")
])

# Async summary generation function
async def summarize_entity_async(description):
    messages = summary_prompt.invoke({"content": description})
    response = await llm.ainvoke(messages) 
    return response.content if hasattr(response, 'content') else str(response)

# Fetch rows from SQLite in batches with context manager
async def fetch_rows_from_sqlite_in_batches(batch_size=5, last_seen_id=0):
    try:
        async with aiosqlite.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute("SELECT id, iri, data FROM iri_cache_organisation WHERE id > ? ORDER BY id LIMIT ?", (last_seen_id, batch_size)) as cur:
                rows = [dict(row) async for row in cur]
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch rows from SQLite: {e}")
        return []

# Helper function for retry logic
def retry_operation(operation, max_retries=3, retry_delay=5, *args, **kwargs):
    for attempt in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
    logger.error("Operation failed after maximum retries.")
    return None

# Function to log failed IRIs to the database
async def log_failed_iri(iri, error_message):
    try:
        async with aiosqlite.connect(SQLITE_DB_PATH) as conn:
            await conn.execute(
                "INSERT INTO failed_iris_organisation (iri, error_message) VALUES (?, ?)",
                (iri, error_message)
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"Failed to log failed IRI {iri}: {e}")

# Function to log processed IRIs to the database
async def log_processed_iri(iri):
    try:
        async with aiosqlite.connect(SQLITE_DB_PATH) as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO processed_iris_organisation (iri) VALUES (?)",
                (iri,)
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"Failed to log processed IRI {iri}: {e}")

# Function to save the last processed ID to SQLite
async def save_last_processed_id_to_db(last_seen_id):
    try:
        async with aiosqlite.connect(SQLITE_DB_PATH) as conn:
            await conn.execute(
                "INSERT INTO checkpoint_organisation (last_seen_id, timestamp) VALUES (?, CURRENT_TIMESTAMP)",
                (last_seen_id,)
            )
            await conn.commit()
    except Exception as e:
        logger.error(f"Failed to save last processed ID to SQLite: {e}")

# Function to load the last processed ID from SQLite
async def load_last_processed_id_from_db():
    try:
        async with aiosqlite.connect(SQLITE_DB_PATH) as conn:
            async with conn.execute("SELECT last_seen_id FROM checkpoint_organisation ORDER BY timestamp DESC LIMIT 1") as cur:
                result = await cur.fetchone()
                return result[0] if result else 0
    except Exception as e:
        logger.error(f"Failed to load last processed ID from SQLite: {e}")
        return 0

# Updated process_from_sqlite to use asyncio.run with summarize_entity_async
async def process_from_sqlite(batch_size=5, limit=None):
    count = 0
    last_seen_id = await load_last_processed_id_from_db()

    while True:
        batch_start_time = time.time()  # Start timing for the batch
        batch_rows = await fetch_rows_from_sqlite_in_batches(batch_size=batch_size, last_seen_id=last_seen_id)
        if not batch_rows:
            break
        if limit and count >= limit:
            break

        tasks = []
        iri_descriptions = []

        for row in batch_rows:
            iri = row["iri"]
            try:
                description = row["data"]
                if not description or not iri:
                    logger.warning(f"Skipping row with IRI {iri}: missing 'iri' or 'description'")
                    continue
                tasks.append(summarize_entity_async(description))
                iri_descriptions.append((iri, description))
            except Exception as e:
                logger.warning(f"Failed to prepare row with IRI {iri}: {e}")

        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        batch_docs = []
        for (iri, description), summary in zip(iri_descriptions, summaries):
            if isinstance(summary, Exception) or summary is None:
                logger.exception(f"Summarization failed for IRI {iri}: {summary}")
                continue

            full_text = f"{summary}\n\n{description}"
            doc = Document(page_content=full_text, metadata={"_id": iri})
            batch_docs.append(doc)
            count += 1

        if batch_docs:
            try:
                await vectorstore.add_documents_async(batch_docs)  # Assuming vectorstore supports async
                logger.info(f"Stored batch of {len(batch_docs)} embeddings.")
            except Exception as e:
                logger.error(f"Failed to store batch: {e}")

        # Update the last_seen_id to the last record in the current batch
        last_seen_id = batch_rows[-1]["id"]

        # Save the last processed ID to the database after processing each batch
        await save_last_processed_id_to_db(last_seen_id)

        # Log batch processing time
        batch_end_time = time.time()
        logger.info(f"Batch processed in {batch_end_time - batch_start_time:.2f} seconds")

        # Log progress
        logger.info(f"Processed {count} records so far...")
        if limit:
            logger.info(f"Progress: {count}/{limit} records completed.")

# Async process rows from SQLite DB
async def process_from_sqlite_async(batch_size=1000, max_retries=3, retry_delay=5, limit=None):
    count = 0
    last_seen_id = 0
    semaphore = asyncio.Semaphore(100)  # Limit concurrency to 100 tasks

    async def summarize_entity_with_retries(description, retries=3, delay=2):
        for attempt in range(retries):
            try:
                async with semaphore:
                    return await summarize_entity_async(description)
            except Exception as e:
                logger.warning(f"Retry {attempt + 1} failed for summarization: {e}")
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
        logger.error(f"Failed to summarize after {retries} retries.")
        return None

    while True:
        batch_rows = await fetch_rows_from_sqlite_in_batches(batch_size=batch_size, last_seen_id=last_seen_id)
        if not batch_rows:
            break
        if limit and count >= limit:
            break

        batch_docs = []
        tasks = []
        iri_descriptions = []

        for row in batch_rows:
            iri = row["iri"]
            try:
                data_json = json.loads(row["data"])
                description = data_json.get("description", "")
                if not description or not iri:
                    logger.warning(f"Skipping row with IRI {iri}: missing 'iri' or 'description'")
                    continue

                tasks.append(summarize_entity_with_retries(description))
                iri_descriptions.append((iri, description))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON in row with IRI {iri}, skipping.")
                await log_failed_iri(iri, "Invalid JSON")
            except Exception as e:
                logger.warning(f"Failed to prepare row with IRI {iri}: {e}")
                await log_failed_iri(iri, str(e))

        summaries = await asyncio.gather(*tasks, return_exceptions=True)

        for (iri, description), summary in zip(iri_descriptions, summaries):
            if isinstance(summary, Exception) or summary is None:
                logger.exception(f"Summarization failed for IRI {iri}: {summary}")
                continue

            full_text = f"Summary: {summary}\n\nOriginal: {description}"
            doc = Document(page_content=full_text, metadata={"_id": iri})
            batch_docs.append(doc)
            count += 1

        if batch_docs:
            success = False
            for attempt in range(max_retries):
                try:
                    vectorstore.add_documents(batch_docs)
                    logger.info(f"Stored batch of {len(batch_docs)} embeddings.")
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1} failed to store batch: {e}")
                    await asyncio.sleep(retry_delay)
            if not success:
                logger.error("Failed to store batch after maximum retries.")

        # Update the last_seen_id to the last record in the current batch
        last_seen_id = batch_rows[-1]["id"]

        # Log progress
        logger.info(f"Processed {count} records so far...")
        if limit:
            logger.info(f"Progress: {count}/{limit} records completed.")

# Retain only the asynchronous function call
asyncio.run(process_from_sqlite(batch_size=50, limit=None))
