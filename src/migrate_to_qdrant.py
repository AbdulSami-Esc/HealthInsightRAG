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

    # Define models explicitly to bypass the missing set_model attribute
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
    
    # 4. Memory-Efficient Batch Generator
    def stream_batches(df, batch_size=500):
        for start_idx in range(0, len(df), batch_size):
            batch = df.iloc[start_idx:start_idx + batch_size]
            
            documents = []
            payloads = []
            
            for _, row in batch.iterrows():
                text = str(row['text'])
                
                # Explicitly defining the models per document like your original script
                documents.append({
                    "dense": models.Document(text=text, model=dense_model),
                    "sparse": models.Document(text=text, model=sparse_model),
                })
                
                # Keep ALL columns in the payload, including 'text', so the orchestrator can read it
                payload_dict = row.to_dict()
                
                # Ensure the text field is explicitly cast as a string to prevent JSON serialization errors
                payload_dict['text'] = str(payload_dict['text'])
                
                payloads.append(payload_dict)
                
            yield documents, payloads

    # 5. Embed and Upload using batch processing
    print(f"Embedding and upserting chunks in batches. This might take a minute...")
    total_batches = (len(df_metadata) // 500) + 1
    
    for docs, payloads in tqdm(stream_batches(df_metadata), total=total_batches, desc="Uploading Batches"):
        client.upload_collection(
            collection_name=COLLECTION_NAME,
            vectors=docs,
            payload=payloads,
            parallel=4 # Uses CPU cores for faster encoding
        )
    
    # 6. Create Payload Indexes for our Stage 2 strict pre-filtering (Updated to new schema)
    print("Building metadata payload indexes...")
    client.create_payload_index(COLLECTION_NAME, field_name="encounter_id", field_schema="keyword")
    client.create_payload_index(COLLECTION_NAME, field_name="patient_id", field_schema="keyword")
    client.create_payload_index(COLLECTION_NAME, field_name="bucket_name", field_schema="keyword")
    
    print("Migration complete! Database is now hybrid-ready.")

if __name__ == "__main__":
    migrate_to_hybrid_qdrant()