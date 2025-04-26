import psycopg
import numpy as np
import pandas as pd
import umap
import ast
import plotly.express as px
import os
from dotenv import load_dotenv
from datetime import datetime

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
    with psycopg.connect(CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document, embedding
                FROM public.langchain_pg_embedding
                WHERE document IS NOT NULL
                AND collection_id = '76b3cdc3-08e6-465e-a807-31a87dc245fa'
            """)
            results = cur.fetchall()

    labels = []
    embeddings = []
    for document, emb in results:
        if emb is not None:
            try:
                # Convert the string representation of the list into an actual list/array
                emb_list = ast.literal_eval(emb)
                embeddings.append(np.array(emb_list, dtype=np.float32))

                # Process the document to extract the label or name
                entity = parse_document(document)
                label = extract_entity_label(entity)
                labels.append(label)
            except (ValueError, SyntaxError) as e:
                print(f"Error processing document or embedding: {e}")
                print(f"Problematic document: {document}")

    return labels, np.stack(embeddings) if embeddings else None

def parse_document(document):
    """
    Custom parser to extract key-value pairs for IRI, label, and name from the document string.

    Args:
        document (str): The document string to parse.

    Returns:
        dict: A dictionary containing the extracted keys and values for IRI, label, and name.
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
        print(f"Error parsing document: {e}")
        print(f"Problematic document: {document}")
        return None

# Function to perform UMAP dimensionality reduction and plot 3D visualization using Plotly
def plot_umap_3d(embeddings, labels=None, n_neighbors=15):
    if embeddings is None or len(embeddings) == 0:
        print("No valid embeddings found.")
        return

    # Use UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    embedding_3d = reducer.fit_transform(embeddings)

    # Create a DataFrame for plotting with Plotly
    df = pd.DataFrame(embedding_3d, columns=["x", "y", "z"])
    if labels:
        df['label'] = labels

    # Create an interactive 3D scatter plot with Plotly
    fig = px.scatter_3d(df, x="x", y="y", z="z", hover_data=["label"],
                        title="3D UMAP Projection of Entity Embeddings",
                        labels={"label": "Entity Detail"})
    
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

    # Save the plot as an HTML file in the specified directory with n_neighbors and the current date in the name
    if not OUTPUT_DIR or not os.path.exists(OUTPUT_DIR):
        print(f"Error: The output directory '{OUTPUT_DIR}' does not exist. Please create it or update the .env file.")
        return

    current_date = datetime.now().strftime("%Y-%m-%d")
    fig.write_html(f"{OUTPUT_DIR}/3d_umap_projection_n{n_neighbors}_{current_date}.html")

    # Show the plot
    fig.show()

# Ensure the output directory exists
if OUTPUT_DIR and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Fetch and process entities, then plot the UMAP visualization
labels, embeddings = fetch_and_process_entities()
if embeddings is not None:
    plot_umap_3d(embeddings, labels=labels, n_neighbors=15)
else:
    print("No embeddings to visualize.")
