from pathlib import Path

import pandas as pd
import os
import faiss
from collections import defaultdict

from rank_bm25 import BM25Okapi
import numpy as np

from sentence_transformers import SentenceTransformer


import document_prep  # your chunking module


##Definig the function load_snippets()
##Loads the procesed dataset and return the dataframe
def load_snippets(filename):
    file_path = Path(__file__).parent.parent / 'data' / 'processed' / filename

    data = pd.read_csv(file_path)

    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())
    return data



def generate_embeddings(texts_list):
    #3This function would generate embeddings for the given list of texts using the SentenceTransformer model.
    ##Moroever it would return the embeddings as a numpy array with N row and 384 dimensions 
    vec = model.encode(texts_list, batch_size=128, show_progress_bar=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print(type(vec), vec.shape, vec.dtype)

    return vec


def build_faiss_index(embeddings):
    ##builds IndexFlatL2 space where we can add our embeddings and then search for nearest neighbors
    dimensions = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimensions)

    index.add(embeddings)

    return index


def search_index(index, embeddings, df, query_position=0, k=5):
    #3It would check the index for the nearest neighbors of the embedding at query_position and return the distances and indices of the nearest neighbors.

    query = embeddings[query_position].reshape(1, -1)
    distances, indices = index.search(query, k)

    print("Indices:", indices)
    print("Distances:", distances)

    for i in indices[0]:
        print(df['Snippet'].iloc[i])

    return distances, indices


def save_index(index, filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, '..', 'data', 'processed')
    
    # Good practice to add it here too, just in case
    os.makedirs(target_dir, exist_ok=True)
    
    output_file = os.path.join(target_dir, filename)
    faiss.write_index(index, output_file)
    print(f"Index saved to {output_file}")



def load_index(path):
    return faiss.read_index(path)


## CHANGED: now takes a plain list of strings instead of a dataframe.
def build_bm25_index(texts_list):
    lowered = [str(t).lower() for t in texts_list]
    tokenized_corpus = [doc.split() for doc in lowered]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def embed_query(query_text):
    ##Takes user query and returns the embedding vector for that query using the same model as used for the dataset embeddings.
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vector = model.encode([query_text], convert_to_numpy=True)

    return vector.astype('float32')


def reciprocal_rank_fusion(faiss_ranked_indices, bm25_ranked_indices, k=60):
    fused_scores = defaultdict(float)

    for rank, idx in enumerate(faiss_ranked_indices, start=1):
        fused_scores[idx] += 1.0 / (k + rank)

    for rank, idx in enumerate(bm25_ranked_indices, start=1):
        fused_scores[idx] += 1.0 / (k + rank)

    sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    return sorted_indices


## CHANGED: takes texts_list instead of assuming df["Snippet"] — used for
## both patient-snippet hybrid search and chunk hybrid search now.
def hybrid_search(query_text, faiss_index, bm25_index, texts_list, k=5, candidates=20):
    query_vector = embed_query(query_text)
    distances, faiss_indices = faiss_index.search(query_vector, candidates)
    faiss_ranked_indices = faiss_indices[0].tolist()

    tokenized_query = query_text.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:candidates].tolist()

    fused_indices = reciprocal_rank_fusion(faiss_ranked_indices, bm25_ranked_indices)
    top_k_indices = fused_indices[:k]

    return [texts_list[i] for i in top_k_indices], top_k_indices


# ---------------------------------------------------------------------
# NEW: chunk-specific pipeline (separate from the patient-snippet index)
# ---------------------------------------------------------------------

def load_unstructured_raw(filename="healthcare_dataset.csv"):
    """Loads the raw unstructured dataset (medical transcriptions) that
    document_prep.py will chunk."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'unstructured', 'raw', filename)
    return pd.read_csv(file_path)


def build_chunk_dataset(raw_df):
    """Runs document_prep on the raw dataframe and returns the flattened
    list of chunk dicts: [{record_index, bucket_name, chunk_position, text}, ...]"""
    return document_prep.prepare_all_chunks(raw_df)


def save_chunk_metadata(chunks, filename):
    """Saves chunk metadata + text to disk, in the SAME order the chunks
    were embedded in."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the specific directory path
    target_dir = os.path.join(current_dir, '..', 'data', 'unstructured', 'processed')
    
    # CREATE THE DIRECTORY IF IT DOESN'T EXIST (This is the fix!)
    os.makedirs(target_dir, exist_ok=True)
    
    output_file = os.path.join(target_dir, filename)
    pd.DataFrame(chunks).to_csv(output_file, index=False)
    print(f"Chunk metadata saved to {output_file}")



def load_chunk_metadata(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'data', 'unstructured', 'processed', filename)
    return pd.read_csv(file_path)


if __name__ == "__main__":
    # ---- Existing patient snippet index (unchanged behavior) ----
    dataframe = load_snippets('snippets.csv')
    print(dataframe.head())

    snippet_texts = dataframe["Snippet"].fillna("").astype(str).tolist()
    embeddings = generate_embeddings(snippet_texts)
    faiss_idx = build_faiss_index(embeddings)
    save_index(faiss_idx, 'faiss_index.bin')

    bm25_idx = build_bm25_index(snippet_texts)

    test_query = "diabetic patient emergency admission"
    results_text, _ = hybrid_search(test_query, faiss_idx, bm25_idx, snippet_texts, k=5)
    print(results_text)

    # ---- NEW: chunk index for unstructured clinical transcriptions ----
    raw_df = load_unstructured_raw('mtsamples.csv')
    chunks = build_chunk_dataset(raw_df)          # list of dicts
    chunk_texts = [c["text"] for c in chunks]

    chunk_embeddings = generate_embeddings(chunk_texts)
    chunk_faiss_idx = build_faiss_index(chunk_embeddings)
    save_index(chunk_faiss_idx, 'chunk_faiss_index.bin')

    chunk_bm25_idx = build_bm25_index(chunk_texts)
    save_chunk_metadata(chunks, 'chunk_metadata.csv')

    # Proof-of-value test: a query whose answer is buried deep inside a
    # long transcription section, not in a short field.
    proof_query = "plan for bariatric surgery patient with family history of heart disease"
    proof_results, proof_positions = hybrid_search(
        proof_query, chunk_faiss_idx, chunk_bm25_idx, chunk_texts, k=5
    )
    for pos, text in zip(proof_positions, proof_results):
        meta = chunks[pos]
        print(f"[record {meta['record_index']} | {meta['bucket_name']} | chunk {meta['chunk_position']}]")
        print(text[:200], "\n")