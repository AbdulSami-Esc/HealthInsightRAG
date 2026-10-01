from typing import Any, Dict, List, Optional
from sentence_transformers import CrossEncoder

# Model identifier
DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Process-level singleton instance
_reranker_model: Optional[CrossEncoder] = None


def load_reranker_model(model_name: str = DEFAULT_MODEL_NAME) -> CrossEncoder:
    """Load the CrossEncoder model into memory. 
    
    Call this once during FastAPI application startup.
    """
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(model_name)
    return _reranker_model


def get_reranker_model() -> CrossEncoder:
    """Retrieve the cached model instance."""
    if _reranker_model is None:
        raise RuntimeError("Reranker model has not been initialized. Call load_reranker_model() on startup.")
    return _reranker_model


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    text_key: str = "text",
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rerank candidate chunks using the loaded CrossEncoder.

    Args:
        query: The user search query.
        candidates: List of chunk dictionaries (containing text and payload metadata).
        text_key: The dictionary key holding the chunk text.
        top_k: Optional count to slice down after sorting.

    Returns:
        The candidates list re-sorted descending by cross-encoder score.
    """
    if not candidates:
        return []

    model = get_reranker_model()

    # Build batched (query, chunk_text) pairs and fail loudly on missing text
    sentence_pairs = []
    for i, item in enumerate(candidates):
        text = item.get(text_key)
        if not text or not str(text).strip():
            raise ValueError(
                f"Reranker Error: Candidate at index {i} is missing valid text under the key '{text_key}'. "
                f"Check upstream payload extraction. Candidate data: {item}"
            )
        sentence_pairs.append((query, str(text)))

    # Predict scores in a single batched pass
    scores = model.predict(sentence_pairs)

    # Attach score to candidate object (preserving existing payload fields)
    scored_candidates = []
    for candidate, score in zip(candidates, scores):
        updated = dict(candidate)
        updated["rerank_score"] = float(score)
        scored_candidates.append(updated)

    # Sort descending by score
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    if top_k is not None:
        return scored_candidates[:top_k]

    return scored_candidates