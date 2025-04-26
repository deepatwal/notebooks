import psycopg
import numpy as np
import pandas as pd
import umap
import ast
import plotly.express as px
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PostgreSQL connection string
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
CONNECTION_STRING = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Function to fetch embeddings and their IRI from the PGVector store
def fetch_embeddings():
    with psycopg.connect(CONNECTION_STRING) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cmetadata->>'_id' as iri, embedding
                FROM langchain_pg_embedding
                WHERE embedding IS NOT NULL
                AND collection_id = '76b3cdc3-08e6-465e-a807-31a87dc245fa'
            """)
            results = cur.fetchall()

    iris = []
    embeddings = []
    for iri, emb in results:
        if emb is not None:
            try:
                # Convert the string representation of the list into an actual list/array
                emb_list = ast.literal_eval(emb)
                embeddings.append(np.array(emb_list, dtype=np.float32))
                iris.append(iri)
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing embedding for IRI {iri}: {e}")

    return iris, np.stack(embeddings) if embeddings else None


# Function to perform UMAP dimensionality reduction and plot 3D visualization using Plotly
def plot_umap_3d(embeddings, labels=None):
    if embeddings is None or len(embeddings) == 0:
        print("No valid embeddings found.")
        return

    # Use UMAP for dimensionality reduction
    reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
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

    # Show the plot
    fig.show()


iris, embeddings = fetch_embeddings()
if embeddings is not None:
    plot_umap_3d(embeddings, labels=iris)
else:
    print("No embeddings to visualize.")
