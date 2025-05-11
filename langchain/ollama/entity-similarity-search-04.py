import os
import logging
import psycopg
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Setup LLM and Embedding Model
chat_ollama = ChatOllama(model="llama3.2:3b")
embedding_model = OllamaEmbeddings(model="bge-m3:567m")

# Setup DB connection
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
if COLLECTION_NAME is None:
    raise ValueError("COLLECTION_NAME environment variable is not set.")
COLLECTION_ID = os.getenv("COLLECTION_ID")
TABLE_NAME = os.getenv("TABLE_NAME")

CONNECTION_STRING = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

logger.info(f"Connecting to PGVector collection '{COLLECTION_NAME}'...")
try:
    vectorstore = PGVector(
        connection=CONNECTION_STRING,
        embeddings=embedding_model,
        collection_name=COLLECTION_NAME,
        use_jsonb=True
    )
    logger.info("Connection to vectorstore successful.")
except psycopg.OperationalError as e:
    logger.exception(f"Database connection error: {e}")
except Exception as e:
    logger.exception(f"Unexpected error during PGVector connection: {e}")

# --- Similarity Search ---
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Find similar entities to the following prompt: {prompt}"),
])

# SELECT document
# FROM public.langchain_pg_embedding
# WHERE collection_id = '72e5e3bb-7211-4197-a16e-44e4b0efb7d8'
#   AND cmetadata @> '{"_id": "http://dbpedia.org/resource/Scania_AB"}'
# LIMIT 10;
query = "Tell me about Company Scania AB based out of Sweden."

similar_docs = vectorstore.similarity_search_with_score(
    query=query,
    k=5
)


# # 2. Use a Query Embedding Directly (More Control)
# query_embedding = embedding_model.embed_query(query)

# similar_docs = vectorstore.similarity_search_with_score_by_vector(
#     embedding=query_embedding,
#     k=5
# )

# 3. Switch to max_marginal_relevance_search
# similar_docs = vectorstore.max_marginal_relevance_search(
#     query=query,
#     k=10,
#     fetch_k=20,  # retrieves more candidates before reranking
#     lambda_mult=0.5  # balance between relevance and diversity
# )
# # print the results
# print("Query:", query)
# print("Similar documents found:\n")
# for doc in similar_docs:
#     print("--" * 50)
#     print(f"Document: {doc.page_content}")
#     print("\n")

# print the results
print("Query:", query)
print("Similar documents found:\n")
for doc, score in similar_docs:
    print("--" * 50)
    print(f"Document: {doc.page_content}, \nScore: {score}")
    print("\n")
