from itertools import count
import os
from pyexpat import model
from transformers import TOKENIZER_MAPPING, AutoTokenizer
from groq import Groq

TOKENIZER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Default thresholds
TOKEN_THRESHOLD_SMALL = 1200   # Below this: fast pass-through (no summarization needed)
MAX_CHUNKS_PER_VISIT = 2       # Quota per visit during stratified windowing


def count_context_tokens(text_or_chunks: str | list[dict]) -> int:
    ##Checking whther the dictionmary is passed or dict in
    Total_token=0
    if(isinstance(text_or_chunks,str)):
        Total_token=len(tokenizer.encode(text_or_chunks,add_special_tokens=False))
        return Total_token
    elif(isinstance(text_or_chunks,list)):
        for chunk in text_or_chunks:
            Text=chunk.get("text","")
            Single_text_token=len(tokenizer.encode(Text,add_special_tokens=False))
            Total_token+=Single_text_token

    return Total_token


def stratified_temporal_window(chunks: list[dict], max_per_visit: int = MAX_CHUNKS_PER_VISIT) -> list[dict]:
    # Prevents a single dense visit (e.g., Visit 1 with 16 chunks) from starving 
    # out other visits by enforcing a maximum chunk quota per visit.

    ##a DICT IS used to make the list of chunks with same number of visits
    visit_groups={}
    for chunk in chunks:
        visit_num=chunk.get("visit_number")
        if visit_num not in visit_groups:
            visit_groups[visit_num]=[]
        visit_groups[visit_num].append(chunk)

    ##We are only going to take in the data of chunks till max like if for vist 1 there are 5 chunks so we only take upto 2 only   
    balanced_chunks=[]
    for chunk in visit_groups.values():
        balanced_chunks.extend(chunk[:max_per_visit])

    balanced_chunks.sort(key=lambda item: item.get("visit_number", 0))

    return balanced_chunks

    


def summarize_single_visit(
    visit_number: int, 
    visit_chunks: list[dict], 
    client: Groq, 
    model_name: str = "openai/gpt-oss-120b") -> str:
    # Compresses multiple raw chunks of a single visit into a concise clinical digest.
    Text=""
    Text="\n".join([c.get("text","") for c in visit_chunks])

    prompt=(
        f"Distill these clinical notes for Visit {visit_number} into 3 concise bullet points:\n"
        f"1. Chief complaint & diagnosis\n"
        f"2. Key exam findings/observations\n"
        f"3. Medications and plan\n\n"
        f"Clinical Notes:\n{Text}"
    )

    answer=client.chat.completions.create(temperature=0,model=model_name,messages=[
        {"role": "system", "content": "You are a clinical summarizer. Distill notes into concise bullet points."},
        {"role": "user", "content": f"{prompt}\n\nClinical Notes:\n{Text}"}
    ])

    return f"--- VISIT {visit_number} CLINICAL DIGEST ---\n" + answer.choices[0].message.content.strip()





def build_managed_context(
    retrieved_chunks: list[dict], 
    client: Groq, 
    model_name: str = "openai/gpt-oss-120b"
) -> str:
    if not retrieved_chunks:
        return "No clinical records found"

    windowed_chunks = stratified_temporal_window(retrieved_chunks)
    # Fast path: below token threshold, format windowed chunks directly
    if count_context_tokens(windowed_chunks) <= TOKEN_THRESHOLD_SMALL:
        formatted = []
        for chunk in windowed_chunks:
            header = f"[Visit {chunk.get('visit_number')} | Date: {chunk.get('visit_date', 'Unknown')}]"
            formatted.append(f"{header}\n{chunk.get('text', '')}")
        return "\n\n".join(formatted)
    visit_groups = {}
    for chunk in retrieved_chunks:
        v_num = chunk.get("visit_number")
        if v_num not in visit_groups:
            visit_groups[v_num] = []
        visit_groups[v_num].append(chunk)

    summaries = []
    # Sort visit numbers so digests remain chronological
    for v_num in sorted(visit_groups.keys(), key=lambda x: x or 0):
        digest = summarize_single_visit(v_num, visit_groups[v_num], client, model_name)
        summaries.append(digest)

    return "\n\n".join(summaries)

# ---------------------------------------------------------------------------
# LOCAL TEST RUNNER (To verify your code before connecting to orchestrator.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Dummy mock data to test your data structures locally without calling APIs
    sample_chunks = [
        {"visit_number": 1, "text": "Visit 1 note piece 1: Rash evaluated."},
        {"visit_number": 1, "text": "Visit 1 note piece 2: Biopsy planned."},
        {"visit_number": 1, "text": "Visit 1 note piece 3: Followup arranged."},
        {"visit_number": 2, "text": "Visit 2 note: Rash is 50% better on Prednisone."},
        {"visit_number": 3, "text": "Visit 3 note piece 1: Relapse of rash noted."},
        {"visit_number": 3, "text": "Visit 3 note piece 2: Twisted ankle while walking."},
    ]

    print("--- Testing Token Counter ---")
    print(count_context_tokens(sample_chunks))
    

    print("\n--- Testing Stratified Windowing (max 1 chunk per visit) ---")
    windowed = stratified_temporal_window(sample_chunks, max_per_visit=1)
    for c in windowed:
        print(f"Visit {c['visit_number']}: {c['text']}")