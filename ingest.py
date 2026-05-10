"""
Step 1: Parse the DayOne journal export into individual entries.
Step 2: Embed each entry using a local sentence-transformer model.
Step 3: Store embeddings + text in ChromaDB (local vector database).

Run this once to build the database. Takes ~1-2 minutes.
"""
import re
import os
import chromadb
from sentence_transformers import SentenceTransformer

JOURNAL_FILE = "/Users/adamseeman/claude-workspace/Journal entries/2026-4-16-DayOne.txt"
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "journal"

# The embedding model — runs locally, no API needed
# all-MiniLM-L6-v2 is small (80MB), fast, and good enough for this
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_entries(filepath: str) -> list[dict]:
    """
    Split the raw DayOne export into individual entries.
    Each entry becomes a dict with date, location, and text.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # DayOne exports use this pattern to start each entry
    entry_pattern = re.compile(
        r"Date:\s+(.+?)\n(?:.*?Weather:.*?\n)?(?:.*?Location:\s*(.+?))?\n",
        re.DOTALL
    )

    # Split the file on date headers
    parts = re.split(r"(?=\s*Date:\s+\d)", raw)
    parts = [p.strip() for p in parts if p.strip()]

    entries = []
    for part in parts:
        # Extract date
        date_match = re.search(r"Date:\s+(.+)", part)
        date = date_match.group(1).strip() if date_match else "Unknown"

        # Extract location
        location_match = re.search(r"Location:\s+(.+)", part)
        location = location_match.group(1).strip() if location_match else "Unknown"

        # Clean the text — remove metadata lines, keep the content
        text = re.sub(r"Date:.*\n", "", part)
        text = re.sub(r"Weather:.*\n", "", text)
        text = re.sub(r"Location:.*\n", "", text)
        text = text.strip()

        if len(text) < 50:  # skip near-empty entries
            continue

        entries.append({
            "date": date,
            "location": location,
            "text": text,
            "id": f"entry_{len(entries):04d}",
        })

    return entries


def build_database(entries: list[dict]):
    """
    Embed each entry and store in ChromaDB.
    ChromaDB saves to disk so this only needs to run once.
    """
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(entries)} entries...")
    texts = [e["text"] for e in entries]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print("Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Delete existing collection if rebuilding
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    collection.add(
        ids=[e["id"] for e in entries],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"date": e["date"], "location": e["location"]} for e in entries],
    )
    print(f"Done. {len(entries)} entries stored in {DB_DIR}")


if __name__ == "__main__":
    print("Parsing journal...")
    entries = parse_entries(JOURNAL_FILE)
    print(f"Found {len(entries)} entries")
    print(f"Date range: {entries[0]['date']} → {entries[-1]['date']}")
    print(f"Sample locations: {[e['location'] for e in entries[:3]]}")
    print()
    build_database(entries)
