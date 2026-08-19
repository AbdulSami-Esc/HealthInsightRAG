"""
The LLM (via Groq) decides whether to:
  1. Call get_condition_stats (math/counting questions)
  2. Call find_high_risk_patients (high-risk filter questions)
  3. Call search_patient_records (hybrid FAISS + BM25 retrieval)
  4. Answer directly with no tool (generic knowledge questions)
"""

import os
import json
import pandas as pd
from groq import Groq
from dotenv import load_dotenv 

load_dotenv() 

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "llama-3.3-70b-versatile"

# Extract functions from other files (including the new hybrid search and BM25 builder)
from tools import get_condition_stats, find_high_risk_patients
from embeddings import load_index, load_snippets, build_bm25_index, hybrid_search

SYSTEM_PROMPT = (
    "You are HealthInsight, a healthcare operations assistant. "
    "You have access to tools for exact statistics, high-risk patient lookups, "
    "and hybrid search (semantic + exact keyword matching) over patient records. "
    "Use a tool when the question requires computation or specific patient data. "
    "Answer directly, without a tool, for general medical knowledge questions "
    "(e.g. 'what is diabetes', 'symptoms of asthma') that don't require this dataset. "
    "Always be clear and concise, and never invent numbers — only report "
    "values that came from a tool result."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_condition_stats",
            "description": (
                "Get the count or average billing amount for patients with a specific "
                "medical condition (e.g. Diabetes, Asthma, Obesity, Cancer, Arthritis, "
                "Hypertension). Use this when the user asks how many patients have a "
                "condition, or asks for average cost/billing for a condition. "
                "Do NOT use this for general questions about what a condition is, "
                "its symptoms, or its causes -- those should be answered directly "
                "without calling a tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "condition": {
                        "type": "string",
                        "description": "The medical condition name, e.g. 'Diabetes'"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["count", "average"],
                        "description": "'count' for number of patients, 'average' for mean billing amount"
                    },
                    "column": {
                        "type": "string",
                        "description": "Column to average, defaults to 'Billing Amount'. Only used when operation='average'."
                    }
                },
                "required": ["condition", "operation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_high_risk_patients",
            "description": (
                "Find patients classified as high-risk, defined as: Test Results is "
                "Abnormal AND Admission Type is Emergency. Use this when the user asks "
                "about high-risk patients, emergency cases with abnormal results, or "
                "wants a list of patients matching that risk criteria. "
                "Do NOT use this for general condition statistics -- use "
                "get_condition_stats for that instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max number of patient records to return, defaults to 20"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_patient_records",
            "description": (
                "Hybrid search patient record snippets using natural language and keywords. "
                "Use this when the user asks an open-ended or descriptive question about "
                "patient records, or searches for specific patient IDs, rare codes, or conditions -- "
                "e.g. 'tell me about patients with unusual complications', 'find records "
                "similar to a 45 year old male with heart issues'. "
                "Do NOT use this for questions with an exact computable answer -- "
                "prefer get_condition_stats or find_high_risk_patients for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or keyword search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top hybrid results to retrieve, defaults to 5"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_CSV_PATH = os.path.join(DATA_DIR, "raw", "healthcare_dataset.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "processed", "faiss_index.bin")
SNIPPETS_CSV_PATH = os.path.join(DATA_DIR, "processed", "snippets.csv")

_df = None
_faiss_index = None
_snippets_df = None
_bm25_index = None


def _load_resources():
    """Lazy load raw data, FAISS index, snippets dataframe, and build BM25 index into memory."""
    global _df, _faiss_index, _snippets_df, _bm25_index
    if _df is None:
        _df = pd.read_csv(RAW_CSV_PATH)
    if _faiss_index is None:
        _faiss_index = load_index(FAISS_INDEX_PATH)
    if _snippets_df is None:
        _snippets_df = load_snippets(SNIPPETS_CSV_PATH)
    if _bm25_index is None:
        # Build the sparse BM25 index on-the-fly from the processed snippets
        _bm25_index = build_bm25_index(_snippets_df)


def search_patient_records(query: str, top_k: int = 5):
    """Executes hybrid search (FAISS + BM25 via RRF) over the patient snippets."""
    _load_resources()
    
    # Call the new hybrid_search function from embeddings.py
    results_df = hybrid_search(
        query_text=query,
        faiss_index=_faiss_index,
        bm25_index=_bm25_index,
        df=_snippets_df,
        k=top_k,
        candidates=20
    )
    
    # Extract the snippet strings to feed back into the LLM
    snippets_list = results_df["Snippet"].tolist()
    
    return {
        "status": "ok",
        "query": query,
        "results": snippets_list
    }


TOOL_REGISTRY = {
    "get_condition_stats": lambda **kwargs: get_condition_stats(_df, **kwargs),
    "find_high_risk_patients": lambda **kwargs: find_high_risk_patients(_df, **kwargs),
    "search_patient_records": lambda **kwargs: search_patient_records(**kwargs),
}


def ask(user_question: str, max_rounds: int = 5) -> str:
    _load_resources()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message

        # Case 1: model is done, gave a final text answer
        if not message.tool_calls:
            return message.content

        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        })

        # Execute each requested tool call and append its result
        for tc in message.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            if fn_name not in TOOL_REGISTRY:
                result = {"status": "error", "message": f"Unknown tool '{fn_name}'"}
            else:
                try:
                    result = TOOL_REGISTRY[fn_name](**fn_args)
                except Exception as e:
                    result = {"status": "error", "message": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "I wasn't able to fully answer that within the allowed number of tool-call rounds. Please try rephrasing your question."


if __name__ == "__main__":
    # Quick manual tests verifying hybrid search integration via the LLM router
    print("Test 1 (Stats):")
    print(ask("What's the average billing amount for Obesity patients?"))
    print("\nTest 2 (Direct Knowledge):")
    print(ask("What is diabetes?"))
    print("\nTest 3 (Hybrid Search / Patient Records):")
    print(ask("Find patient records mentioning emergency admission for diabetes"))