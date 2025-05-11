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
            # ast.literal_eval can be slow for many calls.
            # Storing embeddings in a binary format in the DB would be faster.
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

    # Check for dimension consistency
    first_dim = len(embeddings[0])
    if any(len(e) != first_dim for e in embeddings):
         logging.error(f"Embeddings dimensions mismatch. Expected {first_dim}, found {[len(e) for e in embeddings if len(e) != first_dim][:5]}...")
         return [], None


    return labels, np.stack(embeddings)

def plot_umap_3d(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    densmap: bool = True, # Set to False for significant speedup
    pca_components: int = 20 # Number of components for initial PCA reduction
) -> None:
    """
    Applies PCA and UMAP for 3D reduction and plots the result.
    Includes parameters for UMAP optimization.
    """
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings provided.")
        return

    logging.info(f"Processing {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Normalize embeddings to unit vectors
    logging.info("Normalizing embeddings to unit vectors...")
    embeddings = normalize(embeddings)

    # Apply PCA to reduce dimensionality
    # Reducing to a lower dimension (e.g., 10-50) before UMAP speeds up UMAP.
    logging.info(f"Applying PCA to reduce from {embeddings.shape[1]} dimensions to {pca_components}...")
    try:
        pca = PCA(n_components=pca_components, random_state=42)
        embeddings = pca.fit_transform(embeddings)
        logging.info(f"PCA explained variance ratio sum: {pca.explained_variance_ratio_.sum():.4f}")
    except Exception as e:
        logging.error(f"PCA failed: {e}")
        return


    # Apply UMAP
    # n_neighbors: Smaller values (e.g., 5-10) focus on local structure and can be faster.
    # min_dist: Controls how tightly points are clustered. Doesn't impact speed as much as n_neighbors.
    # metric: 'euclidean' is often fastest, especially after normalization. 'cosine' is also common for embeddings.
    # densmap: Set to False for a significant speedup if preserving local density isn't required.
    logging.info(f"Applying UMAP (n_components=3, n_neighbors={n_neighbors}, min_dist={min_dist}, metric='{metric}', densmap={densmap})...")
    try:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
            n_jobs=-1,
            densmap=densmap,
            low_memory=True
        )
        embedding_3d = reducer.fit_transform(embeddings)
        logging.info("UMAP reduction complete.")
    except Exception as e:
        logging.error(f"UMAP failed: {e}")
        return


    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Save reduced embeddings
    reduced_embeddings_path = os.path.join(OUTPUT_DIR, f"reduced_embeddings_n{n_neighbors}_{timestamp}.npy")
    np.save(reduced_embeddings_path, embedding_3d)
    logging.info(f"Reduced embeddings saved to {reduced_embeddings_path}")

    # Create DataFrame
    df = pd.DataFrame(embedding_3d, columns=["x", "y", "z"])
    df['index'] = range(len(df))

    # Truncate labels to reduce plot size and improve readability
    if labels:
        df['label'] = [label[:50] for label in labels] # Increased truncation length slightly

    # 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        title=f"3D UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist}, metric='{metric}', densmap={densmap})"
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        # hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell") # Hover label style is not needed without hover data
    )

    # Save HTML plot
    plot_path = os.path.join(OUTPUT_DIR, f"3d_umap_projection_n{n_neighbors}_m{min_dist}_{metric}{'_densmap' if densmap else ''}_{timestamp}.html")
    fig.write_html(plot_path, include_plotlyjs="cdn", full_html=True)
    logging.info(f"3D UMAP plot saved at {plot_path}")

    # fig.show() # Uncomment this line if you want the plot to open automatically

# Main runner
if __name__ == "__main__":
    labels, embeddings = fetch_and_process_entities()
    if embeddings is not None:
        logging.info(f"Fetched {len(embeddings)} embeddings.")
        # --- Optimized UMAP Parameters ---
        # n_neighbors: Reduced from 15 to 10 for potentially faster computation.
        # min_dist: Kept at 0.1.
        # metric: Changed from 'cosine' to 'euclidean' (valid after normalization, potentially faster).
        # densmap: Set to False for a significant speedup.
        # pca_components: Kept at 20. Can experiment with this (e.g., 15, 30)
        plot_umap_3d(
            embeddings,
            labels=labels,
            n_neighbors=10,
            min_dist=0.1,
            metric="euclidean",
            densmap=False,
            pca_components=50
        )
    else:
        logging.warning("No embeddings to visualize.")