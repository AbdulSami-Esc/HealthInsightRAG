import os
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import document_prep

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = SentenceTransformer(MODEL_NAME)


def generate_embeddings(texts_list: list[str]) -> np.ndarray:
    """Generates 384-dimensional dense embeddings for a list of text chunks."""
    vec = _model.encode(
        texts_list,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return vec.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Initializes IndexFlatL2 for exhaustive L2 distance search and loads vectors."""
    dimensions = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatL2, filename: str) -> None:
    """Serializes the FAISS index to the processed unstructured directory."""
    target_dir = Path(__file__).resolve().parent.parent / "data" / "unstructured" / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / filename
    faiss.write_index(index, str(output_file))
    print(f"Index saved to {output_file}")


def load_index(path: str | Path) -> faiss.IndexFlatL2:
    """Loads a serialized FAISS index from disk."""
    return faiss.read_index(str(path))


def build_bm25_index(texts_list: list[str]) -> BM25Okapi:
    """Builds an in-memory Okapi BM25 index over lowercased whitespace-tokenized chunks."""
    lowered = [str(t).lower() for t in texts_list]
    tokenized_corpus = [doc.split() for doc in lowered]
    return BM25Okapi(tokenized_corpus)


def embed_query(query_text: str) -> np.ndarray:
    """Encodes a single search query into a 384-dimensional float32 vector."""
    vector = _model.encode([query_text], convert_to_numpy=True)
    return vector.astype("float32")


def reciprocal_rank_fusion(
    faiss_ranked_indices: list[int],
    bm25_ranked_indices: list[int],
    k: int = 60
) -> list[int]:
    """Combines dense and sparse ranked lists using rank reciprocal dampening."""
    fused_scores = defaultdict(float)

    for rank, idx in enumerate(faiss_ranked_indices, start=1):
        fused_scores[idx] += 1.0 / (k + rank)

    for rank, idx in enumerate(bm25_ranked_indices, start=1):
        fused_scores[idx] += 1.0 / (k + rank)

    return sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)


def hybrid_search(
    query_text: str,
    faiss_index: faiss.IndexFlatL2,
    bm25_index: BM25Okapi,
    texts_list: list[str],
    k: int = 5,
    candidates: int = 20
) -> tuple[list[str], list[int]]:
    """Executes dense and sparse searches in parallel and merges using RRF."""
    query_vector = embed_query(query_text)
    _, faiss_indices = faiss_index.search(query_vector, candidates)
    faiss_ranked_indices = faiss_indices[0].tolist()

    tokenized_query = query_text.lower().split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_indices = np.argsort(bm25_scores)[::-1][:candidates].tolist()

    fused_indices = reciprocal_rank_fusion(faiss_ranked_indices, bm25_ranked_indices)
    top_k_indices = fused_indices[:k]

    return [texts_list[i] for i in top_k_indices], top_k_indices


def load_unstructured_raw(filename: str = "mtsamples.csv") -> pd.DataFrame:
    """Loads raw clinical transcription notes."""
    file_path = Path(__file__).resolve().parent.parent / "data" / "unstructured" / "raw" / filename
    return pd.read_csv(file_path)


def build_chunk_dataset(raw_df: pd.DataFrame) -> list[dict]:
    """Applies two-pass regex and token-bounded chunking on unstructured notes."""
    return document_prep.prepare_all_chunks(raw_df)


def save_chunk_metadata(chunks: list[dict], filename: str = "chunk_metadata.csv") -> None:
    """Saves chunk text and boundary metadata in indexing sequence order."""
    target_dir = Path(__file__).resolve().parent.parent / "data" / "unstructured" / "processed"
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / filename
    pd.DataFrame(chunks).to_csv(output_file, index=False)
    print(f"Chunk metadata saved to {output_file}")


def load_chunk_metadata(filename: str = "chunk_metadata.csv") -> pd.DataFrame:
    """Loads serialized chunk metadata from disk."""
    file_path = Path(__file__).resolve().parent.parent / "data" / "unstructured" / "processed" / filename
    return pd.read_csv(file_path)


if __name__ == "__main__":
    print("Loading raw clinical dataset (mtsamples.csv)...")
    raw_df = load_unstructured_raw("mtsamples.csv")

    print("Executing two-pass chunking...")
    chunks = build_chunk_dataset(raw_df)
    chunk_texts = [str(c["text"]) for c in chunks]

    print(f"Total chunks created: {len(chunk_texts)}")
    print("Generating dense vector embeddings...")
    chunk_embeddings = generate_embeddings(chunk_texts)

    print("Building and saving FAISS index...")
    chunk_faiss_idx = build_faiss_index(chunk_embeddings)
    save_index(chunk_faiss_idx, "chunk_faiss_index.bin")

    print("Building BM25 index...")
    chunk_bm25_idx = build_bm25_index(chunk_texts)

    print("Persisting metadata...")
    save_chunk_metadata(chunks, "chunk_metadata.csv")

    proof_query = "plan for bariatric surgery patient with family history of heart disease"
    print(f"\nTesting hybrid search for query: '{proof_query}'")
    proof_results, proof_positions = hybrid_search(
        proof_query, chunk_faiss_idx, chunk_bm25_idx, chunk_texts, k=5
    )

    for pos, text in zip(proof_positions, proof_results):
        meta = chunks[pos]
        print(f"\n[Record {meta['record_index']} | Bucket: {meta['bucket_name']} | Chunk: {meta['chunk_position']}]")
        print(f"{text[:250]}...")