import sys
import os
from pathlib import Path


# 1. Resolve project root (HealthInsightRAG) and src directories dynamically
CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = SRC_DIR.parent

# 2. Inject both into sys.path
for path_dir in [str(ROOT_DIR), str(SRC_DIR)]:
    if path_dir not in sys.path:
        sys.path.insert(0, path_dir)

import json
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, FusionQuery, Fusion, SparseVector
from fastembed import TextEmbedding, SparseTextEmbedding

# 3. Direct import because reranker.py lives in the same folder
try:
    from src.reranker import rerank, load_reranker_model
except ModuleNotFoundError:
    from reranker import rerank, load_reranker_model

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "openai/gpt-oss-120b"

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION_NAME = "clinical_notes"

CANDIDATE_POOL_SIZE = 20
DEFAULT_FINAL_TOP_K = 5

SYSTEM_PROMPT = (
    "You are HealthInsight, an intelligent clinical research and operations assistant. "
    "You have access to a hybrid search tool that indexes detailed clinical transcription notes. "
    "Use the search_clinical_documents tool whenever the user asks for specific clinical "
    "details about a specific patient. "
    "You MUST extract the record index (e.g., 'P0001', 'P1200') from the user's prompt and pass it to the search tool. "
    "If a search does not return an exact match, synthesize the most relevant findings "
    "from the closest retrieved documents. "
    "Answer directly, without calling any tool, for general medical questions "
    "that do not require specific patient files. "
    "Always remain factual, objective, and cite the clinical section where applicable."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_clinical_documents",
            "description": (
                "Hybrid search (BM25 + Semantic) over chunked clinical transcription notes. "
                "Use this tool to find patient history, physical exams, and treatment plans."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language medical query targeting clinical notes",
                    },
                    "record_index": {
                        "type": "string",
                        "description": "The exact alphanumeric record index (e.g., 'P0001', 'P0104') required to isolate the search.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to return after reranking. Defaults to 5.",
                    },
                },
                "required": ["query", "record_index"],
            },
        },
    }
]

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Lazy singletons ---
_qdrant_client = None
_dense_model = None
_sparse_model = None

def _load_resources():
    global _qdrant_client, _dense_model, _sparse_model
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    if _dense_model is None:
        _dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding("Qdrant/bm25")

def _embed_query(query: str):
    """Helper function to explicitly generate vectors using FastEmbed."""
    _load_resources()
    
    dense_gen = _dense_model.query_embed(query)
    dense_vector = next(dense_gen).tolist()
    
    sparse_gen = _sparse_model.query_embed(query)
    sparse_res = next(sparse_gen)
    sparse_vector = SparseVector(
        indices=sparse_res.indices.tolist(),
        values=sparse_res.values.tolist()
    )
    
    return dense_vector, sparse_vector

def search_clinical_documents(query: str, record_index: str, top_k: int = DEFAULT_FINAL_TOP_K) -> dict:
    """Lane 2: Native Qdrant Hybrid Search (Dense + Sparse RRF) -> Reranker."""
    _load_resources()
    dense_vector, sparse_vector = _embed_query(query)

    # 1. Strict Patient & Document Type Pre-filtering
    strict_filter = Filter(
        must=[
            FieldCondition(key="bucket_name", match=MatchValue(value="transcription")),
            FieldCondition(key="record_index", match=MatchValue(value=record_index))
        ]
    )

    # 2. Native Hybrid Search
    search_result = _qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                filter=strict_filter,
                limit=CANDIDATE_POOL_SIZE,
            ),
            Prefetch(
                query=sparse_vector,
                using="sparse",
                filter=strict_filter,
                limit=CANDIDATE_POOL_SIZE,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=CANDIDATE_POOL_SIZE,
    )

    candidates = [
        {
            "text": point.payload.get("text", ""),
            "record_index": point.payload.get("record_index"),
            "section": point.payload.get("bucket_name"),
        }
        for point in search_result.points
    ]

    reranked = rerank(query=query, candidates=candidates, text_key="text", top_k=top_k)

    results = [
        {
            "text": item["text"],
            "record_index": item["record_index"],
            "section": item["section"],
        }
        for item in reranked
    ]

    # --- DEBUG PRINT ---
    print(f"\n🔍 [QDRANT SEARCH] Looked for Patient: {record_index} | Found {len(results)} chunks.\n")

    return {
        "status": "ok",
        "query": query,
        "record_index": record_index,
        "results": results,
    }

def search_clinical_cohorts(query: str, top_k: int = DEFAULT_FINAL_TOP_K) -> dict:
    """Lane 3: Qdrant Hybrid Search WITHOUT the patient ID filter for cohort analysis."""
    _load_resources()
    dense_vector, sparse_vector = _embed_query(query)

    cohort_filter = Filter(
        must=[FieldCondition(key="bucket_name", match=MatchValue(value="transcription"))]
    )

    search_result = _qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vector, using="dense", filter=cohort_filter, limit=CANDIDATE_POOL_SIZE),
            Prefetch(query=sparse_vector, using="sparse", filter=cohort_filter, limit=CANDIDATE_POOL_SIZE),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=CANDIDATE_POOL_SIZE,
    )

    candidates = [
        {
            "text": point.payload.get("text", ""),
            "record_index": point.payload.get("record_index"),
            "section": point.payload.get("bucket_name"),
        }
        for point in search_result.points
    ]

    reranked = rerank(query=query, candidates=candidates, text_key="text", top_k=top_k)

    formatted_results = []
    for item in reranked:
        formatted_results.append(f"Patient ID: {item['record_index']} | Finding: {item['text']}")

    return {
        "status": "ok",
        "query": query,
        "results": formatted_results,
    }

def answer_analytical_query(question: str) -> str:
    """Lane 4: Bypasses Qdrant entirely and dynamically executes native Pandas operations."""
    csv_path = BASE_DIR / "data" / "unstructured" / "processed" / "chunk_metadata.csv"
    df = pd.read_csv(csv_path)

    prompt = (
        f"You are an expert Data Analyst. A pandas DataFrame 'df' is loaded in memory. "
        f"The columns are: {list(df.columns)}\n"
        f"Write EXACTLY the Python pandas code to answer this question: '{question}'\n"
        f"Do NOT include markdown, backticks, explanations, or print statements. ONLY raw python code. "
        f"You MUST assign the final textual or numerical answer to a variable named 'result'."
    )

    try:
        # Use the global working client and gpt-oss-120b
        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        # Strip any accidental markdown formatting from the LLM
        code = response.choices[0].message.content.strip()
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Execute the pandas code dynamically
        local_vars = {"df": df, "pd": pd}
        exec(code, {}, local_vars)
        
        return str(local_vars.get("result", "Execution successful, but no 'result' variable was created."))
    except Exception as e:
        return f"Error executing analytical math: {str(e)}\nAttempted Code:\n{code}"


# ReAct loop elements
TOOL_REGISTRY = {
    "search_clinical_documents": lambda **kwargs: search_clinical_documents(**kwargs),
}
def ask(user_question: str, max_rounds: int = 5) -> str:
    """ReAct execution loop."""
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
        if not message.tool_calls:
            return message.content

        messages.append({
            "role": "assistant",
            "content": message.content,
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
                    print(f"\n🚨 [CRITICAL TOOL ERROR]: {str(e)} 🚨\n")
                    result = {"status": "error", "message": str(e)}
            
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": json.dumps(result),
            })

    collected_results = [m["content"] for m in messages if m.get("role") == "tool"]

    synthesis_prompt = (
        f"Original question: {user_question}\n\n"
        f"Retrieved clinical context:\n"
        f"{chr(10).join(collected_results)}\n\n"
        f"Using solely the retrieved information above, formulate a concise, factual answer. "
        f"Do not call any additional tools."
    )

    final_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": synthesis_prompt},
        ],
    )
    return final_response.choices[0].message.content


if __name__ == "__main__":
    load_reranker_model()

    print("--- Test 1: Direct Medical Knowledge ---")
    print(ask("What are common complications of untreated hypertension?"))

    print("\n--- Test 2: Secure Patient Retrieval ---")
    # Using P0001 to ensure we hit a real record in your dataset
    print(ask("What was the treatment plan for patient P0001?"))