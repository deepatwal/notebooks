import json
import sqlite3
import os

# File paths
OUTPUT_FILENAME_DIR = os.path.join("c:\\Users\\deepa\\data\\workspace\\notebooks", "datasets", "instance_description")
OUTPUT_FILENAME = os.path.join(OUTPUT_FILENAME_DIR, "instance_description.jsonl")
sqlite_db = "cache.db"

# Connect to SQLite
conn = sqlite3.connect(sqlite_db)
cursor = conn.cursor()

# Create cache table if it doesn't exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS iri_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iri TEXT UNIQUE,
    processed INTEGER DEFAULT 0 -- 0 = not processed, 1 = processed
)
""")

# Read IRIs from JSONL, insert into cache and mark as processed
with open(OUTPUT_FILENAME, "r", encoding="utf-8") as file:
    for line in file:
        try:
            data = json.loads(line)
            iri = data.get("iri")
            if iri:
                # Insert the IRI if it doesn't exist and mark it as processed
                cursor.execute("""
                INSERT OR IGNORE INTO iri_cache (iri, processed) VALUES (?, 1)
                """, (iri,))
        except json.JSONDecodeError as e:
            print(f"Invalid JSON line skipped: {e}")

# Commit changes and close connection
conn.commit()
conn.close()

print("All IRIs processed and loaded into cache table from:", OUTPUT_FILENAME)