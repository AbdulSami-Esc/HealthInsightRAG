import json
import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from embeddings import build_bm25_index, hybrid_search, load_chunk_metadata, load_index

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are HealthInsight, an intelligent clinical research and operations assistant. "
    "You have access to a hybrid search tool (combining dense semantic retrieval and sparse "
    "BM25 keyword matching) that indexes detailed clinical transcription notes, including "
    "patient histories, physical examination findings, SOAP notes, and treatment plans. "
    "Use the search_clinical_documents tool whenever the user asks for specific clinical "
    "details, historical records, diagnoses, or management strategies documented by doctors. "
    "If a search does not return an exact match, synthesize the most relevant findings "
    "from the closest retrieved documents instead of making redundant repeated calls. "
    "Answer directly, without calling any tool, for general medical questions "
    "(e.g., 'what is hypertension', 'symptoms of asthma') that do not require specific clinical files. "
    "Always remain factual, objective, and cite the clinical section or record details where applicable."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_clinical_documents",
            "description": (
                "Hybrid search over chunked clinical transcription notes (history of present illness, "
                "physical exam findings, treatment plans, medications, and family/social history). "
                "Use this tool when the query asks about real patient cases, doctor evaluations, "
                "clinical recommendations, or medical transcriptions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language or keyword search query targeting clinical notes",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of relevant hybrid chunks to retrieve. Defaults to 5.",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "unstructured" / "processed"

CHUNK_FAISS_INDEX_PATH = DATA_DIR / "chunk_faiss_index.bin"
CHUNK_METADATA_PATH = DATA_DIR / "chunk_metadata.csv"

_chunk_faiss_index = None
_chunk_metadata_df = None
_chunk_bm25_index = None
_chunk_texts = None


def _load_resources():
    """Lazy loads the FAISS index, metadata, and BM25 index for clinical text search."""
    global _chunk_faiss_index, _chunk_metadata_df, _chunk_bm25_index, _chunk_texts

    if _chunk_faiss_index is None:
        _chunk_faiss_index = load_index(CHUNK_FAISS_INDEX_PATH)
    if _chunk_metadata_df is None:
        _chunk_metadata_df = load_chunk_metadata(CHUNK_METADATA_PATH)
    if _chunk_texts is None:
        _chunk_texts = _chunk_metadata_df["text"].fillna("").astype(str).tolist()
    if _chunk_bm25_index is None:
        _chunk_bm25_index = build_bm25_index(_chunk_texts)


def search_clinical_documents(query: str, top_k: int = 5) -> dict:
    """Executes hybrid search (FAISS + BM25 with RRF) over unstructured medical chunks."""
    _load_resources()

    chunk_texts_result, positions = hybrid_search(
        query_text=query,
        faiss_index=_chunk_faiss_index,
        bm25_index=_chunk_bm25_index,
        texts_list=_chunk_texts,
        k=top_k,
        candidates=20,
    )

    results = []
    for pos, text in zip(positions, chunk_texts_result):
        meta_row = _chunk_metadata_df.iloc[pos]
        results.append({
            "text": text,
            "record_index": int(meta_row["record_index"]),
            "section": meta_row["bucket_name"],
        })

    return {
        "status": "ok",
        "query": query,
        "results": results,
    }


TOOL_REGISTRY = {
    "search_clinical_documents": lambda **kwargs: search_clinical_documents(**kwargs),
}


def ask(user_question: str, max_rounds: int = 5) -> str:
    """ReAct execution loop routing queries between clinical search and direct answers."""
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
                    result = {"status": "error", "message": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fn_name,
                "content": json.dumps(result),
            })

    # If max rounds are reached without a final synthesis, assemble gathered context
    collected_results = [
        m["content"] for m in messages if m.get("role") == "tool"
    ]

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
    print("--- Test 1: Direct Medical Knowledge ---")
    print(ask("What are common complications of untreated hypertension?"))

    print("\n--- Test 2: Unstructured Clinical Document Retrieval ---")
    print(ask("What was the treatment plan for a bariatric surgery patient with family history of heart disease?"))