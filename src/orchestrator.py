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
try:
    from src.fsm_observer import FSMObserver, Lane, State
except ModuleNotFoundError:
    from fsm_observer import FSMObserver, Lane, State

try:
    from src.reranker import rerank, load_reranker_model
except ModuleNotFoundError:
    from reranker import rerank, load_reranker_model

try:
    from src.prompt_manager import build_managed_context,stratified_temporal_window
except ModuleNotFoundError:
    from prompt_manager import build_managed_context,stratified_temporal_window

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
    "You MUST extract the patient ID (e.g., 'P01854') from the user's prompt and pass it to the search tool. "
    "Once you have called the tool for a given patient and query and received results, do NOT call it "
    "again with the same patient and query — synthesize your final answer directly from those results. "
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
                    "patient_id": {
                        "type": "string",
                        "description": "The exact alphanumeric patient ID (e.g., 'P01854') required to isolate the search.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant chunks to return after reranking. Defaults to 5.",
                    },
                },
                "required": ["query", "patient_id"],
            },
        },
    }
]

BASE_DIR = Path(__file__).resolve().parent.parent


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
    
    try:
        load_reranker_model()
    except Exception:
        pass  


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

def search_clinical_documents(query: str, patient_id: str, top_k: int = DEFAULT_FINAL_TOP_K) -> dict:
    observer = FSMObserver(max_retries=2)
    observer.transition_to(State.ROUTING)
    observer.set_lane(Lane.SINGLE_PATIENT)
    
    observer.transition_to(State.EXTRACT_RECORD_INDEX)
    _load_resources()
    dense_vector, sparse_vector = _embed_query(query)

    strict_filter = Filter(
        must=[
            FieldCondition(key="bucket_name", match=MatchValue(value="transcription")),
            FieldCondition(key="patient_id", match=MatchValue(value=patient_id))
        ]
    )

    current_pool_size = CANDIDATE_POOL_SIZE
    search_result = None

    for attempt in range(observer.max_retries + 1):
        observer.transition_to(State.QDRANT_HYBRID_RRF)
        try:
            search_result = _qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_vector, using="dense", filter=strict_filter, limit=current_pool_size),
                    Prefetch(query=sparse_vector, using="sparse", filter=strict_filter, limit=current_pool_size),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=current_pool_size,
            )
            
            if search_result and len(search_result.points) > 0:
                break
            else:
                raise ValueError("Zero points returned from Qdrant hybrid search.")

        except Exception as e:
            error_msg = str(e)
            print(f"\n🚨 [RETRIEVAL WARNING on attempt {attempt+1}]: {error_msg} 🚨\n")
            if observer.retry_count < observer.max_retries:
                observer.trigger_healing(f"Retrieval zero-hit/error: {error_msg}")
                current_pool_size *= 2  # Expand candidate pool size for fallback
                observer.transition_to(State.QDRANT_HYBRID_RRF)
            else:
                observer.transition_to(State.FAILED)
                search_result = None
                break

    candidates = []
    if search_result and hasattr(search_result, 'points'):
        for point in search_result.points:
            text_content = point.document if hasattr(point, 'document') and point.document else point.payload.get("text", "")
            candidates.append({
                "text": text_content,
                "patient_id": point.payload.get("patient_id"),
                "section": point.payload.get("bucket_name"),
                "visit_date": point.payload.get("visit_date"),
                "visit_number": point.payload.get("visit_number_for_patient"),
            })

    observer.transition_to(State.CROSS_ENCODER_RERANK)
    reranked = rerank(query=query, candidates=candidates, text_key="text", top_k=top_k)

    observer.transition_to(State.FORMAT_PATIENT_CONTEXT)
    results = [
        {
            "text": item["text"],
            "patient_id": item["patient_id"],
            "section": item["section"],
            "visit_date": item["visit_date"],
            "visit_number": item["visit_number"],
        }
        for item in reranked
    ]
    print(f"\n🔍 [QDRANT SEARCH] Looked for Patient: {patient_id} | Found {len(results)} chunks.\n")

    observer.transition_to(State.COMPLETED)
    return {
        "status": "ok",
        "query": query,
        "patient_id": patient_id,
        "results": results,
    }


def search_clinical_cohorts(query: str, top_k: int = DEFAULT_FINAL_TOP_K) -> dict:
    observer = FSMObserver(max_retries=2)
    observer.transition_to(State.ROUTING)
    observer.set_lane(Lane.COHORT)
    
    observer.transition_to(State.GLOBAL_HYBRID_SEARCH)
    _load_resources()
    dense_vector, sparse_vector = _embed_query(query)

    cohort_filter = Filter(
        must=[FieldCondition(key="bucket_name", match=MatchValue(value="transcription"))]
    )

    current_pool_size = CANDIDATE_POOL_SIZE
    search_result = None

    for attempt in range(observer.max_retries + 1):
        observer.transition_to(State.QDRANT_SEARCH)
        try:
            search_result = _qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_vector, using="dense", filter=cohort_filter, limit=current_pool_size),
                    Prefetch(query=sparse_vector, using="sparse", filter=cohort_filter, limit=current_pool_size),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=current_pool_size,
            )
            
            if search_result and len(search_result.points) > 0:
                break
            else:
                raise ValueError("Zero points returned from Qdrant cohort search.")

        except Exception as e:
            error_msg = str(e)
            print(f"\n🚨 [COHORT RETRIEVAL WARNING on attempt {attempt+1}]: {error_msg} 🚨\n")
            if observer.retry_count < observer.max_retries:
                observer.trigger_healing(f"Cohort retrieval zero-hit/error: {error_msg}")
                current_pool_size *= 2  # Expand candidate pool size for fallback
                observer.transition_to(State.QDRANT_SEARCH)
            else:
                observer.transition_to(State.FAILED)
                search_result = None
                break

    candidates = []
    if search_result and hasattr(search_result, 'points'):
        for point in search_result.points:
            text_content = point.document if hasattr(point, 'document') and point.document else point.payload.get("text", "")
            candidates.append({
                "text": text_content,
                "patient_id": point.payload.get("patient_id"),
                "section": point.payload.get("bucket_name"),
            })

    reranked = rerank(query=query, candidates=candidates, text_key="text", top_k=top_k)

    observer.transition_to(State.MARKDOWN_TABLE_OUTPUT)
    formatted_results = []
    for item in reranked:
        formatted_results.append(f"Patient ID: {item['patient_id']} | Finding: {item['text']}")

    observer.transition_to(State.COMPLETED)
    return {
        "status": "ok",
        "query": query,
        "results": formatted_results,
    }











def answer_analytical_query(question: str) -> str:
    """Lane 4: Dynamic Pandas execution via `exec` equipped with FSM Self-Healing retry loop."""
    observer = FSMObserver(max_retries=2)
    observer.transition_to(State.ROUTING)
    observer.set_lane(Lane.ANALYTICAL)
    observer.transition_to(State.PANDAS_EXECUTION)

    csv_path = BASE_DIR / "data" / "unstructured" / "processed" / "chunk_metadata.csv"
    df = pd.read_csv(csv_path)

    prompt = (
        f"You are an expert Data Analyst. A pandas DataFrame 'df' is loaded in memory. "
        f"The columns are: {list(df.columns)}\n"
        f"Write EXACTLY the Python pandas code to answer this question: '{question}'\n"
        f"Do NOT include markdown, backticks, explanations, or print statements. ONLY raw python code. "
        f"You MUST assign the final textual or numerical answer to a variable named 'result'."
    )

    current_prompt = prompt
    code = ""
    for attempt in range(observer.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=[{"role": "user", "content": current_prompt}],
                temperature=0
            )
            
            code = response.choices[0].message.content.strip()
            code = code.replace("```python", "").replace("```", "").strip()
            
            local_vars = {"df": df, "pd": pd}
            exec(code, {}, local_vars)
            
            result_str = str(local_vars.get("result", "Execution successful, but no 'result' variable was created."))
            observer.transition_to(State.QUANTITATIVE_RESULT)
            observer.transition_to(State.COMPLETED)
            return result_str

        except Exception as e:
            error_msg = str(e)
            print(f"\n🚨 [PANDAS EXECUTION ERROR on attempt {attempt+1}]: {error_msg} 🚨\n")
            if observer.retry_count < observer.max_retries:
                observer.trigger_healing(f"Pandas execution error: {error_msg}")
                observer.transition_to(State.PANDAS_EXECUTION)
                current_prompt = (
                    f"{prompt}\n\n"
                    f"PREVIOUS ATTEMPT FAILED with error: {error_msg}.\n"
                    f"Failed code was:\n{code}\n"
                    f"Fix the error, ensure valid pandas syntax/column names, and output ONLY raw python code assigning to 'result'."
                )
            else:
                observer.transition_to(State.FAILED)
                return f"Error executing analytical math after retries: {error_msg}\nAttempted Code:\n{code}"

            
# ReAct loop elements
TOOL_REGISTRY = {
    "search_clinical_documents": lambda **kwargs: search_clinical_documents(**kwargs),
}

def ask(user_question: str, max_rounds: int = 5) -> tuple:
    print("\n🚀 [DEBUG] RUNNING NEW ASK() WITH FSM OBSERVER 🚀\n")
    _load_resources()

    observer = FSMObserver(max_retries=2)
    observer.transition_to(State.ROUTING)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    # Quick peek to see if tools are likely needed (or let the first LLM turn decide)
    # For clean lane setting: if it's general, set LANE.GENERAL; if patient-specific, set SINGLE_PATIENT
    # We can inspect the prompt text for a patient ID pattern (e.g., 'P0')
    is_general = not any(p in user_question for p in ["P0", "patient"])
    
    if is_general:
        observer.set_lane(Lane.GENERAL)
    else:
        observer.set_lane(Lane.SINGLE_PATIENT)

    used_context_chunks = []

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        message = response.choices[0].message
        
        # Lane 1: General Medical Question (Direct Response)
        if not message.tool_calls:
            if observer.active_lane == Lane.GENERAL:
                observer.transition_to(State.DIRECT_LLM_RESPONSE)
            observer.transition_to(State.COMPLETED)
            return message.content, used_context_chunks

        # If tools are called, ensure lane is SINGLE_PATIENT
        if observer.active_lane != Lane.SINGLE_PATIENT:
            observer.active_lane = Lane.SINGLE_PATIENT
        observer.transition_to(State.REACT_LOOP)

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
                    if isinstance(result, dict) and "results" in result:
                        used_context_chunks.extend(result["results"])
                except Exception as e:
                    print(f"\n🚨 [CRITICAL TOOL ERROR]: {str(e)} 🚨\n")
                    observer.trigger_healing(str(e))
                    result = {"status": "error", "message": str(e)}

        break

    managed_context = build_managed_context(
        retrieved_chunks=used_context_chunks,
        client=client,
        model_name=MODEL_NAME
    )

    synthesis_prompt = (
        f"Original question: {user_question}\n\n"
        f"Retrieved clinical context:\n"
        f"{managed_context}\n\n"
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

    observer.transition_to(State.COMPLETED)
    final_context_chunks = stratified_temporal_window(used_context_chunks)

    return final_response.choices[0].message.content, final_context_chunks

if __name__ == "__main__":
    load_reranker_model()

    print("--- Test 1: Direct Medical Knowledge ---")
    ans, _ = ask("What are common complications of untreated hypertension?")
    print(ans)

    print("\n--- Test 2: Secure Patient Retrieval ---")
    ans, _ = ask("What was the treatment plan for patient P01854?")
    print(ans)

    print("Test 3---multi-visit chronology, no skips (P01305)")
    ans, _ = ask("What new health issue came up at P01305's most recent visit, and how is that different from what was new at the visit before it?")
    print(ans)

    print("Test 4 — the batch-skip case (P03910)")
    ans, _ = ask("How did P03910's fracture recovery progress between their two recorded visits, and roughly how much time passed between them?")
    print(ans)

    print("Test 5 — single-visit edge case (P00997)")
    ans, _ = ask("What follow-up did the doctor plan for P00997 after their most recent visit?")
    print(ans)

    print("Test 6 — another skip-batch case, different specialty (P04897)")
    ans, _ = ask("Was there a large gap between P04897's two visits, and what was the status of their cardiac catheterization by the later one?")
    print(ans)