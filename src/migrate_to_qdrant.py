import pandas as pd
from qdrant_client import QdrantClient, models
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "unstructured" / "processed" / "chunk_metadata.csv"

def migrate_to_hybrid_qdrant():
    print("Initializing Qdrant Client...")
    client = QdrantClient(url="http://localhost:6333")
    COLLECTION_NAME = "clinical_notes"

    # Define our models here instead of using client.set_model
    dense_model = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model = "Qdrant/bm25"

    # 1. Wipe the old collection
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Deleting old collection '{COLLECTION_NAME}' to start fresh...")
        client.delete_collection(collection_name=COLLECTION_NAME)

    # 2. Create the new collection with hybrid configuration
    print(f"Creating new hybrid collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  # The size for all-MiniLM-L6-v2
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        }
    )

    # 3. Load the chunk data
    print(f"Loading chunks from {CSV_PATH}...")
    df_metadata = pd.read_csv(CSV_PATH)
    
    documents = []
    metadata = []
    
    for idx, row in df_metadata.iterrows():
        text = str(row['text'])
        
        # We explicitly tell FastEmbed which model to use for each chunk right here
        documents.append({
            "dense": models.Document(text=text, model=dense_model),
            "sparse": models.Document(text=text, model=sparse_model),
        })
        
        metadata.append({
            "record_index": str(row.get("record_index")), # This ensures P0001 is a string
            "bucket_name": str(row.get("bucket_name")),
            "chunk_position": row.get("chunk_position"),
            "text": text,
        })

    # 4. Embed and Upload using batch processing
    print(f"Embedding and upserting chunks using FastEmbed. This might take a minute...")
    client.upload_collection(
        collection_name=COLLECTION_NAME,
        vectors=tqdm(documents, desc="Uploading Hybrid Vectors"),
        payload=metadata,
        parallel=4 # Uses CPU cores for faster encoding
    )
    
    # 5. Create Payload Indexes for our Stage 2 strict pre-filtering
    print("Building metadata payload indexes...")
    client.create_payload_index(COLLECTION_NAME, field_name="record_index", field_schema="keyword")
    client.create_payload_index(COLLECTION_NAME, field_name="bucket_name", field_schema="keyword")
    
    print("Migration complete! Database is now hybrid-ready.")

if __name__ == "__main__":
    migrate_to_hybrid_qdrant()