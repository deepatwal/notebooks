import os
import ast
import logging
from typing import Optional, Tuple, cast
import numpy as np
import pandas as pd
import psycopg
import plotly.express as px
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Validate environment variables
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

if not OUTPUT_DIR:
    logging.error("OUTPUT_DIR is not set. Please configure it in the .env file.")
    exit(1)

if None in (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME):
    logging.error("Database credentials are missing. Please set them properly in the .env file.")
    exit(1)

OUTPUT_DIR = cast(str, OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure the output directory exists

# PostgreSQL connection string
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Helper functions

def extract_entity_label(entity: dict) -> str:
    name = entity.get('name')
    if name:
        return name
    iri = entity.get('IRI')
    if iri:
        return iri.split('/')[-1].replace('_', ' ')
    return "Unknown Entity"

def parse_document(document: str) -> dict:
    parsed_data = {}
    try:
        lines = document.split("\n")
        for line in lines:
            if "IRI:" in line:
                parsed_data["IRI"] = line.split("IRI:")[1].strip()
            elif "name:" in line:
                parsed_data["name"] = line.split("name:")[1].strip()
            if "IRI" in parsed_data and "name" in parsed_data:
                break
        return parsed_data
    except Exception as e:
        logging.error(f"Error parsing document: {e}")
        logging.error(f"Problematic document: {document}")
        return {}

def fetch_and_process_entities() -> Tuple[list[str], Optional[np.ndarray]]:
    """
    Fetches a sample of embeddings and documents from the database.
    """
    try:
        with psycopg.connect(CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                # Using TABLESAMPLE SYSTEM (10) to get approximately 10% of the data
                # This is a major optimization for large datasets.
                cur.execute(""" 
                    SELECT document, embedding
                    FROM public.langchain_pg_embedding
                    WHERE document IS NOT NULL
                    AND collection_id = '72e5e3bb-7211-4197-a16e-44e4b0efb7d8'
                """)
                results = cur.fetchall()
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return [], None

    labels = []
    embeddings = []

    for document, emb in results:
        if emb is None:
            continue
        try:
            emb_list = ast.literal_eval(emb)
            if not isinstance(emb_list, list):
                raise ValueError("Embedding is not a valid list.")
            embeddings.append(np.array(emb_list, dtype=np.float32))
            entity = parse_document(document)
            label = extract_entity_label(entity)
            labels.append(label)
        except Exception as e:
            logging.error(f"Error processing document or embedding: {e}")
            logging.error(f"Problematic document: {document}")

    if not embeddings:
        logging.error("No valid embeddings were fetched.")
        return [], None

    return labels, np.stack(embeddings)

def plot_tsne_3d(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    n_clusters: int = 5,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000
) -> None:
    """
    Applies t-SNE for 3D reduction and plots the result.
    """
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings provided.")
        return

    logging.info(f"Processing {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Normalize embeddings to unit vectors
    logging.info("Normalizing embeddings to unit vectors...")
    embeddings = normalize(embeddings)

    # Apply t-SNE
    logging.info(f"Applying t-SNE (perplexity={perplexity}, learning_rate={learning_rate}, n_iter={n_iter})...")
    try:
        tsne = TSNE(n_components=3, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        embedding_3d = tsne.fit_transform(embeddings)
        logging.info("t-SNE reduction complete.")
    except Exception as e:
        logging.error(f"t-SNE failed: {e}")
        return

    # Apply KMeans clustering
    logging.info("Applying KMeans clustering to embeddings...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_3d)  # Use the 3D t-SNE embeddings
    logging.info(f"KMeans clustering complete. Number of clusters: {n_clusters}")

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save reduced embeddings
    reduced_embeddings_path = os.path.join(OUTPUT_DIR, f"reduced_embeddings_tsne_{timestamp}.npy")
    np.save(reduced_embeddings_path, embedding_3d)
    logging.info(f"Reduced embeddings saved to {reduced_embeddings_path}")

    # Create DataFrame
    df = pd.DataFrame(embedding_3d, columns=["x", "y", "z"])
    df['index'] = range(len(df))

    # Truncate labels to reduce plot size and improve readability
    if labels:
        df['label'] = [label[:50] for label in labels]  # Increased truncation length slightly

    # 3D scatter plot with coloring based on cluster labels
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color=cluster_labels,  # Color by cluster labels
        title=f"3D t-SNE Projection (perplexity={perplexity}, learning_rate={learning_rate})"
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="t-SNE Component 1",
            yaxis_title="t-SNE Component 2",
            zaxis_title="t-SNE Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Save HTML plot
    plot_path = os.path.join(OUTPUT_DIR, f"3d_tsne_projection_{timestamp}.html")
    fig.write_html(plot_path, include_plotlyjs="cdn", full_html=True)
    logging.info(f"3D t-SNE plot saved at {plot_path}")

    # fig.show() # Uncomment this line if you want the plot to open automatically

# Main runner
if __name__ == "__main__":
    labels, embeddings = fetch_and_process_entities()
    if embeddings is not None:
        logging.info(f"Fetched {len(embeddings)} embeddings.")
        plot_tsne_3d(
            embeddings,
            labels=labels,
            n_clusters=5,
            perplexity=30.0,
            learning_rate=200.0,
            n_iter=1000
        )
    else:
        logging.warning("No embeddings to visualize.")
