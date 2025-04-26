import psycopg
import numpy as np
import pandas as pd
import umap
import ast
import plotly.express as px
import os
from dotenv import load_dotenv
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

# PostgreSQL connection string
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Validate environment variables
if not OUTPUT_DIR:
    logging.error("OUTPUT_DIR is not set. Please configure it in the .env file.")
    exit(1)

# Function to extract the label or name of an entity
def extract_entity_label(entity):
    """
    Extract the label or name of an entity. If neither is available, use the value after the last '/' in the IRI,
    replacing underscores with spaces.

    Args:
        entity (dict): A dictionary containing entity information with keys like 'label', 'name', and 'IRI'.

    Returns:
        str: The extracted label or name.
    """
    if 'label' in entity and entity['label']:
        return entity['label']
    elif 'name' in entity and entity['name']:
        return entity['name']
    elif 'IRI' in entity:
        # Extract the value after the last '/' in the IRI and replace underscores with spaces
        return entity['IRI'].split('/')[-1].replace('_', ' ')

# Function to fetch embeddings and their IRI from the PGVector store
def fetch_and_process_entities():
    """
    Fetch entities from the database and process their labels or names.

    Returns:
        list: A list of processed entity labels.
        np.ndarray: A NumPy array of embeddings.
    """
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
        if emb is not None:
            try:
                # Create output directory if it does not exist
                if not os.path.exists(OUTPUT_DIR):
                    os.makedirs(OUTPUT_DIR)
                    logging.info(f"Created output directory: {OUTPUT_DIR}")

                # Validate embedding deserialization
                try:
                    emb_list = ast.literal_eval(emb)
                    if not isinstance(emb_list, list):
                        raise ValueError("Embedding is not a valid list.")
                except Exception as e:
                    logging.error(f"Error deserializing embedding: {e}")
                    continue

                embeddings.append(np.array(emb_list, dtype=np.float32))

                # Process the document to extract the label or name
                entity = parse_document(document)
                label = extract_entity_label(entity)
                labels.append(label)
            except (ValueError, SyntaxError) as e:
                logging.error(f"Error processing document or embedding: {e}")
                logging.error(f"Problematic document: {document}")

    # Ensure embeddings have consistent dimensions
    if len(set(len(emb) for emb in embeddings)) > 1:
        logging.error("Embeddings have inconsistent dimensions.")
        return [], None

    return labels, np.stack(embeddings) if embeddings else None

def parse_document(document):
    """
    Custom parser to extract key-value pairs for IRI, label, name, and type from the document string.

    Args:
        document (str): The document string to parse.

    Returns:
        dict: A dictionary containing the extracted keys and values for IRI, label, name, and type.
    """
    parsed_data = {}
    try:
        # Split the document into lines
        lines = document.split("\n")
        
        # Iterate through each line to find keys and values
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
        return None

# Function to perform UMAP dimensionality reduction and plot 3D visualization using Plotly
def plot_umap_3d(embeddings, labels=None, n_neighbors=15, min_dist=0.1, metric="euclidean"):
    if embeddings is None or len(embeddings) == 0:
        logging.error("No valid embeddings found.")
        return

    # Standardize embeddings
    logging.info("Standardizing embeddings...")
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    # Use UMAP for dimensionality reduction
    logging.info(f"Applying UMAP with n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}...")
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)

    # Save reduced embeddings for reuse
    reduced_embeddings_path = os.path.join(OUTPUT_DIR, f"reduced_embeddings_n{n_neighbors}_{datetime.now().strftime('%Y-%m-%d')}.npy")
    np.save(reduced_embeddings_path, embedding_3d)
    logging.info(f"Reduced embeddings saved to {reduced_embeddings_path}")

    # Create a DataFrame for plotting with Plotly
    df = pd.DataFrame(embedding_3d, columns=["x", "y", "z"])
    if labels:
        df['label'] = labels  # Group entities by their labels

    # Create an interactive 3D scatter plot with Plotly
    logging.info("Creating 3D scatter plot...")
    fig = px.scatter_3d(
        df,
        x="x",
        y="y",
        z="z",
        color="label" if labels else None,  # Use labels for color grouping
        hover_data=["label"] if labels else None,
        title=f"3D UMAP Projection (n_neighbors={n_neighbors}, min_dist={min_dist})",
        labels={"label": "Entity Group"}
    )
    
    # Update layout for better visuals
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            zaxis_title="UMAP Component 3"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell")
    )

    # Save the plot as an HTML file
    plot_path = f"{OUTPUT_DIR}/3d_umap_projection_{datetime.now().strftime('%Y-%m-%d')}.html"
    fig.write_html(plot_path)
    logging.info(f"3D UMAP plot saved to {plot_path}")

    # Show the plot
    fig.show()

# Fetch and process entities, then plot the UMAP visualization
labels, embeddings = fetch_and_process_entities()
if embeddings is not None:
    plot_umap_3d(embeddings, labels=labels, n_neighbors=15, min_dist=0.1, metric="cosine")
else:
    logging.warning("No embeddings to visualize.")
