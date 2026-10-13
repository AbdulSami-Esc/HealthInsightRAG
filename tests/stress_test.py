import sys
from pathlib import Path
from collections import Counter

# Constructed the file path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

#Here I am calling the exact function that FastAPI server calls when a user hits ask
from src.orchestrator import search_clinical_documents, ask, load_reranker_model

def run_stress_test():
    print("=" * 70)
    print("PHASE 1: DATA STRESS & TEMPORAL INTEGRITY TEST")
    print("=" * 70)

    patient_id = "P03982"
    #I have already written about choodsing the above patient id in the file of Data_stress_log
    query = (
        "Trace the progression of erythema nodosum for patient P03982 across all visits. "
        "Did the condition improve or worsen over time, and were there any new acute injuries "
        "noted at the most recent visit?"
    )


    load_reranker_model()

    #Retrieval recall
    print("\n[STEP 1] Running hybrid retrieval + reranker for Patient",patient_id," (top_k=5)...")
    #pass paramters to the function called from orchestrator
    search_response = search_clinical_documents(query=query, patient_id=patient_id, top_k=5)
    retrieved_chunks = search_response.get("results", [])

    # Extract which visits actually made it into the top 5
    retrieved_visits = [chunk.get("visit_number") for chunk in retrieved_chunks]
    visit_counts = Counter(retrieved_visits)

    print("\n--- Retrieval Funnel Diagnosis ---")
    print("Total chunks retrieved in final top-5:" ,len(retrieved_chunks))
    print(f"Visit distribution in top-5 context:",dict(visit_counts))
    
    # Print the specific visits captured
    v1_found = 1 in retrieved_visits
    v2_found = 2 in retrieved_visits
    v3_found = 3 in retrieved_visits

    print(f"  • Visit 1 (Baseline) captured:   {'✅ YES' if v1_found else '❌ DROPPED'}")
    print(f"  • Visit 2 (Improvement) captured: {'✅ YES' if v2_found else '❌ DROPPED'}")
    print(f"  • Visit 3 (Relapse/Injury) captured: {'✅ YES' if v3_found else '❌ DROPPED'}")

    # Inspect the exact chunk snippets provided to the LLM
    print("\n--- Chunks Selected by Cross-Encoder ---")
    for idx, chunk in enumerate(retrieved_chunks):
        print("[",idx+1,"] Visit ",chunk.get('visit_number'),"| Date: ",chunk.get('visit_date'))
        snippet = chunk.get("text", "").replace("\n", " ")[:140]
        print(f"    Excerpt: \"{snippet}...\"\n")

    #end to end synthesis test
    print("[STEP 2] Running full ask() ReAct/Synthesis pipeline...")
    final_answer, used_chunks = ask(query)

    print("\n" + "=" * 70)
    print("FINAL LLM SYNTHESIS OUTPUT")
    print("=" * 70)
    print(final_answer)
    print("=" * 70)

    #Diagnostic scoring for phase 1 dev log

    answer_lower = final_answer.lower()
    
    
    # Ground truth clinical fact checks
    captured_improvement = any(k in answer_lower for k in ["half as bad", "improved", "improvement", "improving", "decreased itching"])
    captured_worsening = any(k in answer_lower for k in ["worsened", "progressed", "larger", "relapse", "increased erythema"])
    captured_injury = any(k in answer_lower for k in ["ankle", "twisted", "twisted the ankle"])

    ## 
    print("\n--- Diagnostic Scorecard ---")
    print(f"1. Captured Visit 2 Improvement: {'PASS' if captured_improvement else 'FAIL (Missed factual state)'}")
    print(f"2. Captured Visit 3 Worsening:   {'PASS' if captured_worsening else 'FAIL (Missed factual state)'}")
    print(f"3. Captured Acute Ankle Injury:  {'PASS' if captured_injury else 'FAIL (Lost critical acute event)'}")

    print(f"Total chunks actually passed to LLM in Step 2: {len(used_chunks)}")
    visits_in_synthesis = [c.get("visit_number") for c in used_chunks]
    print(f"Visits present in synthesis context: {Counter(visits_in_synthesis)}")

if __name__ == "__main__":
    run_stress_test()