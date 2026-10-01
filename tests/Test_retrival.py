import sys
from pathlib import Path
import pandas as pd

# Setup paths so we can import directly from src
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from orchestrator import search_clinical_documents, load_reranker_model

# Test Cases for Lane 2: Query and the strict ID we expect to isolate
LANE_2_TESTS = [
    ("What medications was this patient prescribed?", "P01305"),
    ("Summarize the operative report.", "P00745"),
    ("What were the findings of this patient's imaging?", "P01485"),
    ("Review the physical examination notes.", "P02085"),
    ("Any recorded drug allergies for this patient?", "P01604"),
]

def run_retrieval_eval():
    load_reranker_model()

    results = []
    leakage_failures = 0
    empty_failures = 0

    print("\n" + "=" * 95)
    print(f"{'#':<3} | {'Target ID':<10} | {'Chunks Found':<14} | {'Data Leakage':<15} | {'Status'}")
    print("=" * 95)

    for idx, (query, target_id) in enumerate(LANE_2_TESTS, start=1):
        try:
            # Hit the Qdrant database + Reranker directly
            response = search_clinical_documents(query=query, patient_id=target_id, top_k=3)
            
            # The tool returns a dict with a 'results' list
            retrieved_chunks = response.get("results", [])
            chunk_count = len(retrieved_chunks)
            
            # 1. Check for Empty Retrieval
            if chunk_count == 0:
                status = "FAIL [EMPTY]"
                leak_status = "N/A"
                empty_failures += 1
                match_status = "FAIL_EMPTY"
            else:
                # 2. Check for Data Leakage (Did any chunk belong to a different patient?)
                leak_detected = False
                for chunk in retrieved_chunks:
                    if chunk.get("patient_id") != target_id:
                        leak_detected = True
                        break
                
                if leak_detected:
                    status = "FAIL [LEAK]"
                    leak_status = "DETECTED 🚨"
                    leakage_failures += 1
                    match_status = "FAIL_LEAK"
                else:
                    status = "PASS [✓]"
                    leak_status = "SECURE"
                    match_status = "PASS"

        except Exception as e:
            chunk_count = "ERR"
            leak_status = "ERR"
            status = f"ERROR: {str(e)[:10]}"
            match_status = "ERROR"

        print(f"{idx:<3} | {target_id:<10} | {chunk_count:<14} | {leak_status:<15} | {status}")

        results.append({
            "query": query,
            "target_id": target_id,
            "chunks_retrieved": chunk_count,
            "leak_detected": leak_status,
            "match_status": match_status
        })

    print("=" * 95)
    print(f"\nEvaluation Summary:")
    print(f"- Total Tests: {len(LANE_2_TESTS)}")
    print(f"- Data Leakage Failures: {leakage_failures} (Should be 0)")
    print(f"- Empty Retrieval Failures: {empty_failures} (Indicates missing data or bad routing)")

    # Save to CSV
    df = pd.DataFrame(results)
    csv_filename = ROOT_DIR / "tests" / "retrieval_integrity_evaluation.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nResults successfully saved to: {csv_filename}\n")

if __name__ == "__main__":
    run_retrieval_eval()