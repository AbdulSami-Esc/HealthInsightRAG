# HealthInsight — Architectural Log

## What this actually is

HealthInsight is a clinical operations assistant — the kind of thing a doctor, nurse, or clinic admin could type a plain-English question into and get back the right patient records, without manually reading through thousands of clinical notes. It handles two kinds of questions in two different ways: semantic questions get searched intelligently against the clinical record corpus, and general medical questions get answered directly. Underneath, it's an agent that decides which of those two paths a question needs, then goes and does it.

This log tracks not just what got built, but *why* — including the decisions that turned out to matter more than expected, and the ones I got wrong on the first pass.

---

## In plain terms

Imagine a clinic with thousands of patient visit notes on file. A doctor wants to know something like *"which patients have been treated for asthma"* — normally that means someone manually reading through file after file. HealthInsight lets them just ask, in a normal sentence, and get back the actual relevant notes instead of a pile of paperwork to sift through.

It doesn't just match exact words either — ask about "breathing problems" and it can still find notes that say "wheezing" or "shortness of breath," because it understands meaning, not just keywords. Right now it's built and tested against a synthetic dataset of clinical transcriptions, not real patient data, while the underlying retrieval system gets hardened.

---

## The original build

The first version worked, but it carried some quiet fragility that only became obvious once I started trying to scale it.

Text got normalized into clean, standalone clinical records (`src/data_prep.py`) — this decoupling turned out to be a good call, since it meant the embedding model and index type could change later without re-touching the source text. That decision paid off directly during the Qdrant migration.

From there, `src/embeddings.py` did the heavy lifting: 384-dimensional dense vectors from `all-MiniLM-L6-v2`, stored in a FAISS `IndexFlatL2` index, sitting alongside a BM25 keyword index. Every query hit both, and a hand-written Reciprocal Rank Fusion function (`k=60`) merged the two ranked lists into one. It's a genuinely solid retrieval design — the problem was never the *logic*, it was where all of it *lived*.

Everything — the FAISS index, the BM25 index, the metadata used for filtering — existed only inside the running Python process's memory. Restart the server, and it all had to be rebuilt from a CSV and a `.bin` file on disk. Filtering by category (like restricting a search to actual clinical notes rather than one-line summaries) meant pulling back a batch of results and then throwing most of them away in Python — post-filtering, which quietly caps how much useful signal ever makes it back.

`src/orchestrator.py` and `src/main.py` haven't changed yet — the agentic loop and the FastAPI gateway are untouched. This log is specifically about the retrieval layer underneath them.

---

## Phase 1 — Moving retrieval out of memory, into Qdrant

The first real architectural shift: the 32,989 clinical chunks that used to live inside a FAISS index in RAM now live in **Qdrant**, running in a Docker container with a persistent volume mount. That last part matters more than it sounds — without it, deleting or restarting the container would silently wipe every embedded record. With it, the data survives independently of whether my Python process, or even the container itself, is currently running.

The migration script (`src/migrate_to_qdrant.py`) reconstructs the vectors straight out of the existing FAISS index, pairs each one with its metadata row, and upserts everything into a single `clinical_notes` collection in batches.

One decision worth recording here: I kept the collection's distance metric as **Euclidean**, matching the original `IndexFlatL2` behavior, rather than switching to Cosine. The embeddings were never normalized during the original build, so Euclidean and Cosine would rank results differently — switching metrics silently, mid-migration, would have changed retrieval quality without any visible signal that something had shifted. Migrating storage shouldn't quietly change what the system considers a "good match." That felt like the wrong kind of surprise to introduce.

## Phase 2 — Filtering inside the database, not after it

Every point in `clinical_notes` carries a payload: `record_index`, `bucket_name`, `chunk_position`, and the actual `text`. A payload index was added on `bucket_name`, which turns filtering from "check all 32,989 records one at a time" into a near-instant lookup.

The difference isn't theoretical — I tested it directly. Querying *"asthma treatment"* without any filter returned a mix of real treatment notes and short one-line diagnostic summaries — technically related, practically useless to a doctor looking for what was actually prescribed. The same query, filtered to `bucket_name == "transcription"`, returned only substantive clinical narrative. That's the whole argument for pre-filtering in one side-by-side comparison: broad semantic similarity isn't the same thing as *useful* similarity, and filtering at the database level is what closes that gap — rather than fetching a small batch and hoping enough of it survives a filter applied afterward.

`chunk_metadata.csv` is now effectively retired — everything it held lives natively in Qdrant's payload.

---

## What's still ahead

The hand-written RRF fusion in `embeddings.py` hasn't moved yet — that's the next piece, once dense and sparse vectors both live natively inside Qdrant instead of two separate indexes glued together in Python. After that: cross-encoder reranking to tighten precision on the final candidates before they ever reach the LLM, and a self-correcting retrieval loop that can detect a bad retrieval and retry with a reformulated query rather than silently returning a weak answer.

---

## Execution Interface

```bash
# 1. Prepare clinical text records
python src/data_prep.py

# 2. Generate embeddings and construct FAISS index
python src/embeddings.py

# 3. Run Qdrant (Docker, persistent storage)
docker run -d --name qdrant -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant

# 4. Migrate embeddings into Qdrant
python src/migrate_to_qdrant.py

# 5. Launch production API server
uvicorn src.main:app --reload --port 8000
```