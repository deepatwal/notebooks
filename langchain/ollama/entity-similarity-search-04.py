import os
import logging
import psycopg
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres.vectorstores import PGVector, DistanceStrategy

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Setup LLM and Embedding Model
chat_ollama = ChatOllama(model="llama3.2:3b")
embedding_model = OllamaEmbeddings(model="bge-m3:567m")

# DB connection setup
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

if COLLECTION_NAME is None:
    raise ValueError("COLLECTION_NAME environment variable is not set.")

CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Connect to vector store
try:
    vectorstore = PGVector(
        connection=CONNECTION_STRING,
        embeddings=embedding_model,
        collection_name=COLLECTION_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        use_jsonb=True
    )
    logger.info("Connected to PGVector successfully.")
except psycopg.OperationalError as e:
    logger.exception(f"Database connection error: {e}")
except Exception as e:
    logger.exception(f"Unexpected error during PGVector connection: {e}")

# -------------------------------
# 🔍 STEP 1: Natural Language Query
# -------------------------------
user_question = "Tell me about Volvo?"

# Embed the question and get top match
query_embedding = embedding_model.embed_query(user_question)
top_doc_result = vectorstore.similarity_search_with_score_by_vector(
    embedding=query_embedding,
    k=1  # Get best match first
)

if not top_doc_result:
    print("No matching document found.")
    exit()

top_doc, top_score = top_doc_result[0]
print("✅ Most Relevant Document:\n")
print("--" * 50)
print(f"Document: {top_doc.page_content}\nScore: {top_score}")
print("--" * 50)

# -------------------------------
# 🤝 STEP 2: Recommend Similar Entities
# -------------------------------
print("\n🔁 Recommending Similar Companies...\n")

reference_embedding = embedding_model.embed_query(top_doc.page_content)

similar_docs = vectorstore.similarity_search_with_score_by_vector(
    embedding=reference_embedding,
    k=10
)

filtered_docs = [(doc, score) for doc, score in similar_docs if doc.page_content != top_doc.page_content]

print("🏭 Similar Companies:\n")
for doc, score in filtered_docs:
    print("--" * 50)
    print(f"Document: {doc.page_content}\nScore: {score}")
    print()
