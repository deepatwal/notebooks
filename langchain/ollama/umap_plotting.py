import os
import ast
import logging
import re
import numpy as np
import psycopg
import umap

from typing import Optional, Tuple
from typing_extensions import cast
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
from datetime import datetime
import plotly.graph_objects as go
import plotly.io as pio
from collections import Counter

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
GENERIC_TYPES = {"Agent", "SocialPerson", "Thing", "Organization", "Group", "Organisation", "Band", "Company", "Corporation", "Person", "Place", "Location", "Event", "Concept", "Object"}

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
            elif "type:" in line: 
                parsed_data["type"] = line.split("type:")[1].strip()
            if "IRI" in parsed_data and "name" in parsed_data and "type" in parsed_data:
                break
        return parsed_data
    except Exception as e:
        logging.error(f"Error parsing document: {e}")
        logging.error(f"Problematic document: {document}")
        return {}
    

def parse_document_for_property(document: str, prop: str) -> dict:
    parsed_data = {}
    try:
        lines = document.split("\n")
        for line in lines:
            if f"{prop}:" in line:
                parsed_data[prop] = line.split(f"{prop}:")[1].strip()
            if prop in parsed_data:
                break
        return parsed_data
    except Exception as e:
        logging.error(f"Error parsing document: {e}")
        logging.error(f"Problematic document: {document}")
        return {}

def fetch_and_process_entities() -> Tuple[list[str], Optional[np.ndarray], list[list[str]]]:
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

            entity_types = entity.get("type", "").split(",")
            entity_types = {et.strip() for et in entity_types}

            entity_types = {et for et in entity_types if not re.match(r"^Q\d+$", et)}

            specific_types = [et for et in entity_types if et not in GENERIC_TYPES]

            if specific_types:
                types.append(specific_types)
            
            else:
                # Fallback to generic types if no specific types are found use industry
                specific_props = parse_document_for_property(document, "industry")
                specific_props = specific_props.get("industry", "Unknown").split(",")
                types.append([sp.strip() for sp in specific_props])

        except Exception as e:
            logging.error(f"Error processing document or embedding: {e}")
            logging.error(f"Problematic document: {document}")

    if not embeddings:
        logging.error("No valid embeddings were fetched.")
        return [], None, []

    return labels, np.stack(embeddings), types

def assign_cluster_names(cluster_labels, types) -> dict:
    cluster_names = {}
    
    for cluster_id in set(cluster_labels):
        # Collect types for the given cluster
        cluster_types_in_group = [types[i] for i in range(len(types)) if cluster_labels[i] == cluster_id]
        
        # Flatten types and count the most common ones
        flat_types = [item for sublist in cluster_types_in_group for item in sublist]
        type_counts = Counter(flat_types)

        # Get the most common type
        most_common_type = type_counts.most_common(1)[0][0] if type_counts else cluster_id
        
        # Combine the most common type as a label for a meaningful name
        cluster_names[cluster_id] = f"{most_common_type}"
    
    return cluster_names

def plot_umap_3d(
    embeddings: np.ndarray,
    labels: Optional[list[str]] = None,
    types: Optional[list[list[str]]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    pca_components: int = 50
) -> None:
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings provided.")
        return

    logging.info(f"Processing {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")

    # Standardize embeddings
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    logging.info(f"Applying PCA to reduce from {embeddings.shape[1]} dimensions to {pca_components}...")
    pca = PCA(n_components=pca_components, random_state=42)
    embeddings = pca.fit_transform(embeddings)

    logging.info("Applying UMAP...")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)

    logging.info("Applying KMeans clustering...")
    n_clusters = n_neighbors  # Number of clusters based on n_neighbors
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_3d)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(embedding_3d, cluster_labels)
    logging.info(f"Silhouette Score: {silhouette_avg:.3f}")

    # Assign names to clusters based on types and labels
    cluster_names = assign_cluster_names(cluster_labels, types)

    # Create a 3D scatter plot with Plotly Graph Objects
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=embedding_3d[:, 0],
        y=embedding_3d[:, 1],
        z=embedding_3d[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=cluster_labels,  # Color by cluster label
            colorscale='Viridis',  # Colorscale for better visualization
            opacity=0.7
        ),
        text=labels,  # Hover text with labels
        hoverinfo='text'
    ))

    # Enhance hover text with cluster names
    hover_text = [
        f"{label} - Cluster: {cluster_names[cluster_labels[i]]}"
        for i, label in enumerate(labels or [])
    ]

    fig.add_trace(go.Scatter3d(
        x=embedding_3d[:, 0],
        y=embedding_3d[:, 1],
        z=embedding_3d[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color=cluster_labels,  # Color by cluster label
            colorscale='Viridis',  # Colorscale for better visualization
            opacity=0.8
        ),
        text=hover_text,  # Enhanced hover text with cluster names
        hoverinfo='text'
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3"
        ),
        title="3D UMAP with KMeans Clusters",
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Timestamp for saving the plot
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    plot_path = os.path.join(OUTPUT_DIR, f"3d_umap_clusters_{timestamp}.html")
    pio.write_html(fig, file=plot_path, auto_open=True)

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
            n_neighbors=35,
            min_dist=0.1,
            metric="euclidean",
            pca_components=50
        )
    else:
        logging.warning("No embeddings to visualize.")
