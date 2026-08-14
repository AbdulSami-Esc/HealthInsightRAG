"""
The LLM (via Groq) decides whether to:
  1. Call get_condition_stats (math/counting questions)
  2. Call find_high_risk_patients (high-risk filter questions)
  3. Call search_patient_records (semantic FAISS retrieval)
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

##Extract functions from other files
##Extract functions from other files
from tools import get_condition_stats, find_high_risk_patients
from embeddings import load_index, search_index, load_snippets

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "llama-3.3-70b-versatile"


##System prompt is sue to establish boundaries,tone,role and safety rules to guide every response model amkes

SYSTEM_PROMPT = (
    "You are HealthInsight, a healthcare operations assistant. "
    "You have access to tools for exact statistics, high-risk patient lookups, "
    "and semantic search over patient records. Use a tool when the question "
    "requires computation or specific patient data. Answer directly, without "
    "a tool, for general medical knowledge questions (e.g. 'what is diabetes', "
    "'symptoms of asthma') that don't require this dataset. "
    "Always be clear and concise, and never invent numbers — only report "
    "values that came from a tool result."
)

##Geenrated using AI
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
                "Semantically search patient record snippets using natural language. "
                "Use this when the user asks an open-ended or descriptive question about "
                "patient records that isn't a specific count/average or the high-risk rule -- "
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
                        "description": "Natural language search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar snippets to retrieve, defaults to 5"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_CSV_PATH = os.path.join(DATA_DIR, "raw", "healthcare_dataset.csv")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "processed", "faiss_index.bin")
SNIPPETS_CSV_PATH = os.path.join(DATA_DIR, "processed", "snippets.csv")

_df = None
_index = None
_snippets_df = None


def _load_resources():
    global _df, _index, _snippets_df
    if _df is None:
        _df = pd.read_csv(RAW_CSV_PATH)
    if _index is None:
        _index = load_index(FAISS_INDEX_PATH)
    if _snippets_df is None:
        _snippets_df = load_snippets(SNIPPETS_CSV_PATH)

def search_patient_records(query: str, top_k: int = 5):
    _load_resources()
    results = search_index(_index, _snippets_df, query, top_k=top_k)
    return {
        "status": "ok",
        "query": query,
        "results": results
    }


TOOL_REGISTRY = {
    "get_condition_stats": lambda **kwargs: get_condition_stats(_df, **kwargs),
    "find_high_risk_patients": lambda **kwargs: find_high_risk_patients(_df, **kwargs),
    "search_patient_records": lambda **kwargs: search_patient_records(**kwargs),
}


##orchestration loop
def ask(user_question: str, max_rounds: int = 5) -> str:
    _load_resources()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]
    ##Loop up the max round time
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
                    # never let a tool crash the whole request -- feed the
                    # error back to the model so it can explain gracefully
                    result = {"status": "error", "message": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

    return "I wasn't able to fully answer that within the allowed number of tool-call rounds. Please try rephrasing your question."


if __name__ == "__main__":
    # quick manual test
    print(ask("What's the average billing amount for Obesity patients?"))
    print(ask("What is diabetes?"))
    print(ask("Show me high-risk patients"))