import os
from unittest import result
import faiss
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer



##using os to built the whole file path for chunk_faiss_index and chunk_metadata
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(
    BASE_DIR, "data", "unstructured", "processed", "chunk_faiss_index.bin"
)
CSV_PATH = os.path.join(
    BASE_DIR, "data", "unstructured", "processed", "chunk_metadata.csv"
)


print(f"Loading FAISS index from {INDEX_PATH}...")
faiss_index = faiss.read_index(INDEX_PATH)

# Reconstruct all vectors into a NumPy array
total_vectors = faiss_index.ntotal
print(f"Extracting {total_vectors} vectors from FAISS index...")
vectors = faiss_index.reconstruct_n(0, total_vectors)

print(f"Loading metadata from {CSV_PATH}...")
df_metadata = pd.read_csv(CSV_PATH)

if len(df_metadata) != total_vectors:
    raise ValueError(
        f"Mismatch: FAISS contains {total_vectors} vectors, "
        f"but CSV contains {len(df_metadata)} rows."
    )


client = QdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "clinical_notes"
VECTOR_DIM = vectors.shape[1]  # 384 for MiniLM

# Recreate collection if already existing, or create if not present
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.EUCLID),
    )
    print(f"Created collection '{COLLECTION_NAME}' with dimension {VECTOR_DIM}.")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists.")


BATCH_SIZE = 500
records = df_metadata.to_dict(orient="records")

print(f"Beginning upsert of {total_vectors} records in batches of {BATCH_SIZE}...")

for start_idx in range(0, total_vectors, BATCH_SIZE):
    end_idx = min(start_idx + BATCH_SIZE, total_vectors)
    batch_points = []

    for i in range(start_idx, end_idx):
        item = records[i]
        
        point = PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "record_index": item.get("record_index"),
                "bucket_name": item.get("bucket_name"),
                "chunk_position": item.get("chunk_position"),
                "text": item.get("text"),
            },
        )
        batch_points.append(point)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=batch_points,
        wait=True,
    )
    print(f"Upserted {end_idx}/{total_vectors} records...")

print("Migration completed successfully.")

client.create_payload_index(
    collection_name="clinical_notes",
    field_name="bucket_name",
    field_schema="keyword"
)

model = SentenceTransformer("all-MiniLM-L6-v2")

query_text = "asthma treatment"
query_vector = model.encode(query_text).tolist()
results = client.query_points(
    collection_name="clinical_notes",
    query=query_vector,  # you generate this from your query text
    query_filter=Filter(
        must=[FieldCondition(key="bucket_name", match=MatchValue(value="transcription"))]
    ),
    limit=5
)

print(results)