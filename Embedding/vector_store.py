import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from Embedding.embedding import generate_embedding

# -----------------------------
# LOAD ENV VARIABLES
# -----------------------------

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# -----------------------------
# QDRANT CONNECTION
# -----------------------------

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=120
)

print("Connected to Qdrant")

# -----------------------------
# COLLECTION CONFIG
# -----------------------------

collections = {
    "text_collection": 3072,
    "table_collection": 3072,
    "image_collection": 3072
}

# -----------------------------
# CREATE COLLECTIONS
# -----------------------------

def create_collections():

    existing = [c.name for c in qdrant_client.get_collections().collections]

    for name, size in collections.items():

        if name not in existing:

            qdrant_client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=size,
                    distance=Distance.COSINE
                )
            )

            print(f"{name} created")

        else:

            print(f"{name} already exists")


# -----------------------------
# REMOVE DUPLICATES
# -----------------------------

def remove_duplicates(chunks):

    seen = set()
    unique_chunks = []

    for chunk in chunks:

        content = chunk["content"]

        if content not in seen:

            seen.add(content)
            unique_chunks.append(chunk)

    print("Duplicates removed:", len(chunks) - len(unique_chunks))

    return unique_chunks


# -----------------------------
# STORE VECTORS (SAFE BATCH MODE)
# -----------------------------

def store_vectors(chunks):

    BATCH_SIZE = 20

    collection_points = {
        "text_collection": [],
        "table_collection": [],
        "image_collection": []
    }

    for chunk in chunks:

        print("Processing chunk:", chunk["chunk_id"])

        embedding = generate_embedding(chunk["content"])

        if chunk["type"] == "text":
            collection = "text_collection"

        elif chunk["type"] == "table":
            collection = "table_collection"

        elif chunk["type"] == "image":
            collection = "image_collection"

        else:
            continue

        point = PointStruct(
            id=abs(hash(chunk["chunk_id"])) % (10**8),
            vector=embedding,
            payload=chunk
        )

        collection_points[collection].append(point)

    # Upload vectors in batches
    for collection, points in collection_points.items():

        if not points:
            continue

        for i in range(0, len(points), BATCH_SIZE):

            batch = points[i:i+BATCH_SIZE]

            qdrant_client.upsert(
                collection_name=collection,
                points=batch
            )

            print(f"Uploaded {len(batch)} vectors to {collection}")

    print("Vectors stored successfully")


# -----------------------------
# EMBEDDING PIPELINE
# -----------------------------

def run_embedding_pipeline(chunks):

    print("\nStarting Embedding Pipeline\n")

    create_collections()

    clean_chunks = remove_duplicates(chunks)

    store_vectors(clean_chunks)

    print("\nEmbedding Pipeline Completed")