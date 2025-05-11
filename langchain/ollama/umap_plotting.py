import os
import ast
import logging
import re
import numpy as np
import pandas as pd
import psycopg
import umap
from typing import Optional, Tuple, cast
import plotly.express as px
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

# Define a set of known generic types to exclude
GENERIC_TYPES = {"Agent", "SocialPerson", "Thing", "Organization", "Group", "Organisation", "Band"}

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
            elif "type:" in line:  # Extract 'type' information
                parsed_data["type"] = line.split("type:")[1].strip()
            if "IRI" in parsed_data and "name" in parsed_data and "type" in parsed_data:
                break
        return parsed_data
    except Exception as e:
        logging.error(f"Error parsing document: {e}")
        logging.error(f"Problematic document: {document}")
        return {}

def fetch_and_process_entities() -> Tuple[list[str], Optional[np.ndarray], list[list[str]]]:
    """
    Fetches a sample of embeddings and documents from the database.
    """
    try:
        with psycopg.connect(CONNECTION_STRING) as conn:
            with conn.cursor() as cur:
                cur.execute(""" 
                    SELECT document, embedding
                    FROM public.langchain_pg_embedding
                    WHERE document IS NOT NULL
                    AND collection_id = '72e5e3bb-7211-4197-a16e-44e4b0efb7d8'
                """)
                results = cur.fetchall()
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return [], None, []

    labels = []
    embeddings = []
    types = [] 

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

            # Get the types from the entity, which is a comma-separated string
            entity_types = entity.get("type", "").split(",")
            entity_types = {et.strip() for et in entity_types}  # Unique types using a set

            # Use regex to remove types that start with "Q" followed by numeric digits (e.g., Q215380)
            entity_types = {et for et in entity_types if not re.match(r"^Q\d+$", et)}

            # Filter out generic types
            specific_types = [et for et in entity_types if et not in GENERIC_TYPES]

            if specific_types:
                types.append(specific_types)

        except Exception as e:
            logging.error(f"Error processing document or embedding: {e}")
            logging.error(f"Problematic document: {document}")

    if not embeddings:
        logging.error("No valid embeddings were fetched.")
        return [], None, []

    return labels, np.stack(embeddings), types

def assign_cluster_types(cluster_labels, types):
    """
    Assign multiple types to each cluster based on the types in that cluster.
    """
    cluster_types = {}
    for cluster_id in set(cluster_labels):
        # Get all types assigned to the current cluster
        cluster_types_in_group = [types[i] for i in range(len(types)) if cluster_labels[i] == cluster_id]
        
        # Flatten the list of types
        flat_types = [item for sublist in cluster_types_in_group for item in sublist]
        
        # Get all unique types for the current cluster
        unique_types = set(flat_types)
        
        # Assign the unique types to the cluster
        cluster_types[cluster_id] = unique_types

    return cluster_types

def plot_umap_3d(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    types: Optional[list[list[str]]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    densmap: bool = True,
    pca_components: int = 20
) -> None:
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings provided.")
        return

    logging.info(f"Processing {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    embeddings = normalize(embeddings)

    logging.info(f"Applying PCA to reduce from {embeddings.shape[1]} dimensions to {pca_components}...")
    pca = PCA(n_components=pca_components, random_state=42)
    embeddings = pca.fit_transform(embeddings)

    logging.info("Applying UMAP...")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)

    logging.info("Applying KMeans clustering...")
    n_clusters = n_neighbors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_3d)

    # Assign cluster types (optional, but now we'll not use them for the color coding)
    cluster_types = assign_cluster_types(cluster_labels, types)

    # Create DataFrame
    df = pd.DataFrame(embedding_3d, columns=["x", "y", "z"])
    df['index'] = range(len(df))

    # Add cluster labels for color coding
    df['cluster_label'] = [f"Cluster {cluster}" for cluster in cluster_labels]

    # 3D scatter plot
    fig = px.scatter_3d(
        df,
        x="x", y="y", z="z",
        color="cluster_label",
        title="3D UMAP with Clusters"
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # Timestamp for saving the plot
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plot_path = os.path.join(OUTPUT_DIR, f"3d_umap_clusters_{timestamp}.html")
    fig.write_html(plot_path, include_plotlyjs="cdn", full_html=True)
    logging.info(f"3D UMAP plot saved at {plot_path}")

# Main runner
if __name__ == "__main__":
    labels, embeddings, types = fetch_and_process_entities()
    if embeddings is not None:
        logging.info(f"Fetched {len(embeddings)} embeddings.")
        plot_umap_3d(
            embeddings,
            labels=labels,
            types=types,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            densmap=False,
            pca_components=50
        )
    else:
        logging.warning("No embeddings to visualize.")
