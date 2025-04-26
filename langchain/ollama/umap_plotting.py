import os
import ast
import logging
from typing import Optional, Tuple, cast

import numpy as np
import pandas as pd
import psycopg
import umap
import plotly.express as px
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
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
    if 'label' in entity and entity['label']:
        return entity['label']
    elif 'name' in entity and entity['name']:
        return entity['name']
    elif 'IRI' in entity:
        return entity['IRI'].split('/')[-1].replace('_', ' ')
    else:
        return "Unknown Entity"

def parse_document(document: str) -> dict:
    parsed_data = {}
    try:
        lines = document.split("\n")
        for line in lines:
            if line.startswith("IRI:"):
                parsed_data["IRI"] = line.split("IRI:")[1].strip()
            elif line.startswith("label:"):
                parsed_data["label"] = line.split("label:")[1].strip()
            elif line.startswith("name:"):
                parsed_data["name"] = line.split("name:")[1].strip()
        return parsed_data
    except Exception as e:
        logging.error(f"Error parsing document: {e}")
        logging.error(f"Problematic document: {document}")
        return {}

def fetch_and_process_entities() -> Tuple[list[str], Optional[np.ndarray]]:
    try:
        with psycopg.connect(CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT document, embedding
                    FROM public.langchain_pg_embedding
                    WHERE document IS NOT NULL
                    AND collection_id = '76b3cdc3-08e6-465e-a807-31a87dc245fa'
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

    if len(set(len(e) for e in embeddings)) > 1:
        logging.error(f"Embeddings dimensions mismatch: {[len(e) for e in embeddings]}")
        return [], None

    return labels, np.stack(embeddings)

def plot_umap_3d(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    densmap: bool = True
) -> None:
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings provided.")
        return

    # Normalize
    logging.info("Normalizing embeddings to unit vectors...")
    embeddings = normalize(embeddings)

    # Apply PCA to reduce dimensionality from 1024 to 50 for faster processing
    logging.info(f"Applying PCA to reduce from {embeddings.shape[1]} dimensions to 50...")
    pca = PCA(n_components=50, random_state=42)
    embeddings = pca.fit_transform(embeddings)

    # Apply UMAP
    logging.info(f"Applying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}, densmap={densmap})...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
        n_jobs=-1,
        densmap=densmap,
        low_memory=True  # This option can help improve speed
    )
    embedding_3d = reducer.fit_transform(embeddings)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save reduced embeddings
    reduced_embeddings_path = os.path.join(OUTPUT_DIR, f"reduced_embeddings_n{n_neighbors}_{timestamp}.npy")
    np.save(reduced_embeddings_path, embedding_3d)
    logging.info(f"Reduced embeddings saved to {reduced_embeddings_path}")

    # Create DataFrame
    df = pd.DataFrame(embedding_3d, columns=["x", "y", "z"])
    df['index'] = range(len(df))
    if labels:
        df['label'] = labels

    # 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label" if labels else None,
        hover_data=["index", "label"] if labels else ["index"],
        title=f"3D UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})"
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell")
    )

    # Save HTML plot
    plot_path = os.path.join(OUTPUT_DIR, f"3d_umap_projection_{n_neighbors}_{timestamp}.html")
    fig.write_html(plot_path)
    logging.info(f"3D UMAP plot saved at {plot_path}")

    fig.show()

# Main runner
if __name__ == "__main__":
    labels, embeddings = fetch_and_process_entities()
    if embeddings is not None:
        plot_umap_3d(embeddings, labels=labels, n_neighbors=15, min_dist=0.1, metric="cosine")
    else:
        logging.warning("No embeddings to visualize.")
