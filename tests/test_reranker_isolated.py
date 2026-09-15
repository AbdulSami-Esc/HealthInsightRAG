import sys
from pathlib import Path

# Add the parent folder of the current file to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from src.reranker import load_reranker_model, rerank

# Explicitly trigger startup load
print("Initializing model...")
load_reranker_model()

# 4 synthetic chunks: 1 highly relevant, 1 keyword-stuffed, 2 noise/tangential
mock_candidates = [
    {
        "record_index": 101,
        "bucket_name": "sample_name_keywords",
        "text": "asthma asthma asthma treatment clinic asthma protocol inhaler asthma",
    },
    {
        "record_index": 204,
        "bucket_name": "transcription",
        "text": (
            "The patient presents with persistent wheezing and dyspnea. "
            "We initiated inhaled corticosteroids and a rescue albuterol inhaler. "
            "Peak flow monitoring will be tracked daily."
        ),
    },
    {
        "record_index": 305,
        "bucket_name": "medical_specialty",
        "text": "Pulmonology and Respiratory Medicine - general clinic overview and office hours.",
    },
    {
        "record_index": 409,
        "bucket_name": "transcription",
        "text": "Patient was examined for knee pain following a sports injury. Physical therapy recommended.",
    },
]

query = "asthma treatment"
results = rerank(query=query, candidates=mock_candidates, text_key="text", top_k=4)

print(f"\nQuery: '{query}'\n" + "=" * 60)
for rank, item in enumerate(results, start=1):
    print(
        f"Rank {rank} | Score: {item['rerank_score']:+.4f} | "
        f"Bucket: {item['bucket_name']} (Record: {item['record_index']})\n"
        f"Text snippet: {item['text'][:85]}...\n"
    )

# Sanity assertion
assert results[0]["record_index"] == 204, "Failure: Genuine transcription chunk did not rank #1!"
print("Step 1 validation passed: Semantic transcription ranked over keyword spam.")