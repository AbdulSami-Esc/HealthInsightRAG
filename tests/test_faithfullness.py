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
    load_reranker_model,
    client,       
    MODEL_NAME    
)

# Expanded Test Matrix (20 Queries)
FAITHFULNESS_TESTS = [
    ("What medications was patient P01854 prescribed?", "P01854"),
    ("Summarize the operative report for P03910.", "P03910"),
    ("What were the findings for patient P04897's cardiac catheterization?", "P04897"),
    ("Review the physical examination notes for patient P01305.", "P01305"),
    ("Did patient P01604 have any recorded drug allergies?", "P01604"),
    ("What was the initial diagnosis for patient P00997?", "P00997"),
    ("Retrieve the post-operative recovery plan for P04543.", "P04543"),
    ("What was recorded during admission for P01787?", "P01787"),
]

def judge_faithfulness(question, context_chunks, generated_answer):
    """Uses the LLM to grade if the generated answer hallucinated."""
    
    # Combine chunks into a single readable string
    
    context_text = "\n\n".join([
    f"Chunk {i+1} (visit_date: {c.get('visit_date')}, visit_number: {c.get('visit_number')}): {c['text']}"
    for i, c in enumerate(context_chunks)
    ])
    
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
            # 1. Run the full ReAct pipeline and capture the exact context it used
            final_answer, context_chunks = ask(query, max_rounds=3)

            if not context_chunks:
                pipeline_status = "NO CONTEXT"
                score = "N/A"
                reasoning = "Database returned empty or tool not called."
            else:
                pipeline_status = "ANSWER GENERATED"
                # 2. Ask the Judge LLM to grade the answer against that exact context
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

        # Increased delay to 5 seconds to actively prevent HTTP 429 Too Many Requests errors
        time.sleep(5)

    print("=" * 105)
    
    # Save the detailed report to CSV
    df = pd.DataFrame(results)
    csv_filename = ROOT_DIR / "tests" / "faithfulness_evaluation.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\nDetailed faithfulness report saved to: {csv_filename}\n")


if __name__ == "__main__":
    run_faithfulness_eval()