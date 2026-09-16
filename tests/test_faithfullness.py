import sys
import json
import time
from pathlib import Path
import pandas as pd

# Setup paths
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

for p in [str(ROOT_DIR), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import your pipeline elements
from orchestrator import (
    ask, 
    search_clinical_documents, 
    load_reranker_model,
    client,       # Reusing your Groq client for the Judge
    MODEL_NAME    # Reusing gpt-oss-120b to act as the Judge
)

# Expanded Test Matrix (20 Queries)
FAITHFULNESS_TESTS = [
    ("What medications was patient P0001 prescribed upon discharge?", "P0001"),
    ("Summarize the operative report for P0015.", "P0015"),
    ("What were the findings of the chest CT scan for P0042?", "P0042"),
    ("Review the physical examination notes for patient P0104.", "P0104"),
    ("Did patient P0023 have any recorded drug allergies?", "P0023"),
    ("What was the initial diagnosis for patient P0088?", "P0088"),
    ("Retrieve the post-operative recovery plan for P0201.", "P0201"),
    ("What was the blood pressure reading during admission for P0005?", "P0005"),
    ("Show me the family medical history mentioned in P0310.", "P0310"),
    ("What dosage of lisinopril was given to patient P0012?", "P0012"),
    ("What was the primary complaint for patient P0111?", "P0111"),
    ("Did patient P0222 require intubation during the procedure?", "P0222"),
    ("What were the lab results for hemoglobin for patient P0333?", "P0333"),
    ("Was patient P0444 referred to physical therapy?", "P0444"),
    ("What surgical instruments were used for patient P0555?", "P0555"),
    ("Summarize the discharge instructions for P0666.", "P0666"),
    ("What was the estimated blood loss during P0777's surgery?", "P0777"),
    ("Did patient P0888 have a history of smoking?", "P0888"),
    ("What anesthesia was administered to P0999?", "P0999"),
    ("List the post-op vital signs for patient P1000.", "P1000")
]

def judge_faithfulness(question, context_chunks, generated_answer):
    """Uses the LLM to grade if the generated answer hallucinated."""
    
    # Combine chunks into a single readable string
    context_text = "\n\n".join([f"Chunk {i+1}: {c['text']}" for i, c in enumerate(context_chunks)])
    
    judge_prompt = f"""You are an objective Clinical AI Evaluator. 
    Your job is to determine if the 'Generated Answer' contains ANY information, medications, diagnoses, or facts that are NOT explicitly stated in the 'Retrieved Context'.

    User Question: {question}

    Retrieved Context:
    {context_text}

    Generated Answer:
    {generated_answer}

    Output a strict JSON object with two keys:
    1. 'score': Either "PASS" (100% faithful) or "FAIL" (contains hallucinations/outside info).
    2. 'reasoning': A brief 1-sentence explanation of why it passed or failed.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"score": "ERROR", "reasoning": str(e)}


def run_faithfulness_eval():
    load_reranker_model()
    
    results = []
    print("\n" + "=" * 105)
    print(f"{'#':<3} | {'Target ID':<10} | {'Pipeline Status':<18} | {'Faithfulness':<15} | {'Judge Reasoning'}")
    print("=" * 105)

    for idx, (query, target_id) in enumerate(FAITHFULNESS_TESTS, start=1):
        try:
            # 1. Run the full ReAct pipeline to get the final answer
            final_answer = ask(query, max_rounds=3)
            
            # 2. Manually grab the exact context the LLM used to judge it
            context_response = search_clinical_documents(query, target_id, top_k=5)
            context_chunks = context_response.get("results", [])

            if not context_chunks:
                pipeline_status = "NO CONTEXT"
                score = "N/A"
                reasoning = "Database returned empty."
            else:
                pipeline_status = "ANSWER GENERATED"
                # 3. Ask the Judge LLM to grade the answer
                judgement = judge_faithfulness(query, context_chunks, final_answer)
                score = judgement.get("score", "ERROR").upper()
                reasoning = judgement.get("reasoning", "Failed to parse reasoning.")

        except Exception as e:
            pipeline_status = "PIPELINE CRASH"
            score = "ERROR"
            reasoning = str(e)[:40]
            final_answer = ""

        # Print truncated reasoning to fit the terminal
        short_reasoning = reasoning[:45] + "..." if len(reasoning) > 45 else reasoning
        print(f"{idx:<3} | {target_id:<10} | {pipeline_status:<18} | {score:<15} | {short_reasoning}")

        results.append({
            "query": query,
            "target_id": target_id,
            "final_answer": final_answer,
            "faithfulness_score": score,
            "judge_reasoning": reasoning
        })

        time.sleep(1) # Prevents rate-limiting from Groq

    print("=" * 105)
    
    # Save the detailed report to CSV
    df = pd.DataFrame(results)
    csv_filename = ROOT_DIR / "tests" / "faithfulness_evaluation.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nDetailed faithfulness report saved to: {csv_filename}\n")


if __name__ == "__main__":
    run_faithfulness_eval()