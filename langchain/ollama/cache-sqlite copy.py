import json
import sqlite3
import os

# File paths
OUTPUT_FILENAME_DIR = os.path.join("c:\\Users\\deepa\\data\\workspace\\notebooks", "datasets", "instance_description")
OUTPUT_FILENAME = os.path.join(OUTPUT_FILENAME_DIR, "instance_description.jsonl")
sqlite_db = "cache.db"
BATCH_SIZE = 1000  # Adjust this based on available memory

# Connect to SQLite
conn = sqlite3.connect(sqlite_db)
cursor = conn.cursor()

# Create cache table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS iri_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE,
    processed INTEGER DEFAULT 0, -- 0 = not processed, 1 = processed
    data TEXT
)
""")

# Function to insert in batches
def insert_batch(rows_to_insert):
    if rows_to_insert:
        cursor.executemany("""
            INSERT OR REPLACE INTO iri_cache (iri, processed, data)
            VALUES (?, 1, ?)
        """, rows_to_insert)

# Open the file and process in batches
rows_to_insert = []
batch_count = 0

with open(OUTPUT_FILENAME, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line)
            iri = data.get("iri")
            if iri:
                rows_to_insert.append((iri, json.dumps(data)))  # Add data to batch
            if len(rows_to_insert) >= BATCH_SIZE:  # If batch size reached, insert
                insert_batch(rows_to_insert)
                rows_to_insert.clear()  # Clear the batch list for the next batch
                batch_count += 1
                print(f"Processed batch {batch_count}...")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON line skipped: {e}")

    # Insert any remaining rows that didn't make a full batch
    if rows_to_insert:
        insert_batch(rows_to_insert)
        batch_count += 1
        print(f"Processed final batch {batch_count}...")

# Commit changes and close connection
conn.commit()
conn.close()

print("All IRIs processed and cached in the SQLite database.")
