# HealthInsight — Phase Log

Each phase below gets filled in as it's completed: what was built, what
concepts were learned, and what "done" looked like. Keep entries short —
this is a log, not an essay.

---

## Phase 0 — Setup
**Status:** _pending_
**Goal:** Environment ready — all core libraries importable, FastAPI serves a hello-world.
**Concepts learned:**

**What I built:**
**Done looked like:**

---

## Phase 1 — Data → Text
**Status:** _in progress_
**Goal:** Convert structured patient rows into plain-English text snippets so they can be embedded.
**Concepts learned:**
- AI models are trained on human language, meaning raw spreadsheet columns
- When we change data in our rows to text, we add verbs and prepositions that explicitly tell the AI how the data connects.



**Why this matters specifically once FAISS enters:**
Phase 2 will  `read data/processed/snippets.csv` — it never touches raw data. That means we can completely rebuild our FAISS index (re-embed, different model, different index type) without re-running Phase 1 at all, as long as the snippets file doesn't need to change.If the snippets require any change so we can re run both phases one after another

**Why it matters for scaling later (multiple internships from now):** once you have more data sources like lab results or new hospital data each new source likely needs its own row_to_text equivalent, but they'd all still feed into the same embedding + FAISS pipeline. If Phase 1 and Phase 2 logic were written together, adding a new data source would mean rewriting embedding logic too. Kept separate, adding a new source just means writing one new small function that outputs snippets in the same shape.
**What I built:**
- `load_data()` — loads raw CSV into a dataframe
- `row_to_text()` — converts one row into a natural-language sentence
- Applied across the dataset, saved to `data/processed/snippets.csv`

---

## Phase 2 — Embeddings + FAISS Index
**Status:** done

**Goal:** Convert patient text snippets into numerical vectors and build a searchable FAISS index, so semantically similar patient records can be retrieved by meaning, not just exact keyword matches.

**Concepts learned:**

Embeddings are how AI models represent meaning as numbers.A sentence gets converted into a fixed length vector (384 numbers for this  model), and sentences with similar meaning end up numerically close to each other in that vector space.

This fixed length (384) comes from the model itself, not the input, 3-word sentence and a 60-word sentence both produce a 384 number vector. That's what makes comparison between any two vectors possible in the first place.

FAISS doesn't understand text — it only understands vectors and distance. IndexFlatL2 compares vectors using Euclidean distance: smaller distance means more similar meaning. It checks every vector against every other one (brute force).

**Why this step matters:** once patient records are represented as vectors, "find me records like this one" becomes a pure math problem measuring distance instead of relying on exact keyword overlap the way traditional search does.

**Why this matters specifically once the LLM enters the picture:** the LLM never touches raw patient data directly. Instead, a user's question gets embedded the same way, FAISS finds the closest matching patient snippets by distance, and only those relevant snippets get handed to the LLM as context. This is what keeps the LLM's answers completely related to real data instead of guessing.

**Why it matters for scaling later:** the embedding model and FAISS index don't care where snippets came from — they just need text in the same shape. If more data sources get added later (lab results, a second hospital), they just need to produce snippets like Phase 1 does; this embedding step stays untouched.

What I built:

load_snippets() — loads the processed text snippets from Phase 1
generate_embeddings() — encodes all snippets into 384 dimension vectors using all-MiniLM-L6-v2, batched for speed
build_faiss_index() — creates an IndexFlatL2 index and adds all vectors to it
search_index() — tested retrieval by querying with a known vector and confirming it matched itself first, then returned genuinely similar patients (same condition/urgency) as the next-closest results
save_index() / load_index() — persists the built index to disk so it never needs to be rebuilt from scratch

## Phase 3 — Tool Functions
**Status:** done

**Goal:** Give the system a way to answer questions that FAISS genuinely can't like exact counts, averages, and rule-based filtering. Vector search is great at "find me things similar to this," but it's the wrong tool entirely for "how many patients have Diabetes" or "give me exact billing average." Those need real computation on the real dataframe.

**Concepts learned:**

FAISS retrieval and structured functions solve two completely different problems. Semantic search answers "what's related to this," math answers "what's the exact number." A healthcare assistant needs both, so Phase 3 exists to give the LLM (in Phase 4) a way to reach for real computation instead of trying to interpreting for example an average from retrieved text snippets, which would just be wrong.

I also learned that the shape of what a function *returns* matters as much as what it does. Every tool here returns a dict with a `status` key (`ok` / `not_found` / `error`) instead of raising exceptions.An LLM consuming these results needs something consistent it can branch on, not five different versions of "something went wrong" it has to guess about.

Case-insensitive matching on the condition name was a small decision- the dataset stores exact strings like `"Diabetes"`, but a user (or an LLM guessing the argument) won't always match casing exactly. Handling that at the tool level meant I didn't have to push that responsibility up to the LLM or the user.



**Why it matters for scaling later:** because every tool follows the same `status`-dict pattern, adding a third, fourth, or tenth tool later means writing one more function in the same shape — not redesigning how tools communicate their results.

**What I built:**
- `get_condition_stats()` — count or average (billing amount by default, configurable column) for a given medical condition, case-insensitive match, graceful "not found" for unknown conditions
- `find_high_risk_patients()` — filters on a rule I decided on logically (`Test Results == 'Abnormal' AND Admission Type == 'Emergency'`) since EDA showed no actual statistical correlation between these fields in this dataset — documented that honestly instead of pretending there was a data-driven threshold
- Both capped/validated their inputs and outputs so nothing ever throws an unhandled exception

---

## Phase 4 — LLM Tool-Calling Orchestration
**Status:** done

**Goal:** Get an LLM (via Groq) to actually decide, per question, whether it needs a structured tool, a FAISS search, or just its own general knowledge and connecting them to make tool calling work

**Concepts learned:**

The biggest mental shift in this phase was realizing the API call itself doesn't *execute* anything. The model just tells you, in a structured way, "I want you to call this function with these arguments" — and then it's on my code to actually run it. It's not one request response, it's a loop: ask the model, check if it wants a tool, run the tool myself if so, feed the result back in, ask again, repeat until it gives a real answer.

I sketched the message list by hand for one example question before writing any code, which made the actual loop way less abstract — it's really just building that same list programmatically, round by round.

Tool descriptions turned out to be a bigger deal than I expected. A vague description like "does math operations" isn't enough — the model needs to know both when *to* use a tool and when *not to*, or it'll misroute questions like "what is diabetes" into a stats call that makes no sense.

I decided to expose FAISS search as a third *tool*, callable on demand, rather than always running retrieval on every question. That was a deliberate token/latency tradeoff — running a vector search for a pure calculation question wastes time and context for no benefit, so letting the model choose keeps the system efficient.


**Why this matters specifically once FastAPI enters the picture:** the orchestrator's `ask()` function is a single, self-contained entry point — call it with a question, get an answer back. That's exactly the shape FastAPI needs to wrap around an endpoint; none of the tool-calling complexity needs to leak into the API layer at all.

**Why it matters for scaling later:** the `TOOL_REGISTRY` dispatch pattern means adding a new tool later is just: write the function, add one line mapping its name to a lambda that injects the dataframe, add its schema. Nothing else about the loop changes.

**What I built:**
- Tool schemas (name, description, JSON-schema parameters) for all three tools, written to be specific about when each should and shouldn't fire
- `TOOL_REGISTRY` — a name-to-function dispatch map using lambdas to inject the shared dataframe invisibly, so the LLM never needs to know it exists
- `search_patient_records()` — FAISS search wrapped as a third tool, same `status`-dict return shape as the other two
- `ask()` — the full orchestration loop: build messages, call Groq, check for tool calls vs. final answer, execute requested tools, feed results back, repeat up to a max-round safety cap so a confused model can't loop forever

---

## Phase 5 — FastAPI Endpoint
**Status:** done

**Goal:** Turn the orchestrator into an actual service — something that starts up once, stays ready, and answers questions over HTTP instead of only running from a Python script.

**Concepts learned:**

Pydantic models aren't just type hints, they're validation happening automatically before my code even runs. Setting `min_length=1` on the question field means an empty string gets rejected at the door, with a clean 422, before it ever reaches the orchestrator, the Groq API, or any tool.That ensres that wrong errorful messages or values dont flow through the systems and are discarded immediately.

The concept I had to actually go learn for this phase was `lifespan`. Before this, my instinct was that lazy loading (loading resources on first use) was good enough — and it is, for correctness. But it means the very first user to hit the API eats the full cost of loading a 55K-row dataset and a FAISS index while they wait. `lifespan` runs setup code once, at server startup, before any request is accepted, so by the time anyone can actually call `/ask`, everything's already warm. The `yield` inside it is the same idea as a context manager — code above the yield is setup, code below is shutdown/cleanup, and the server just runs in between.(This learning was purely done using Claude AI.I still face issue in this and would be going deep)


**Why this matters for scaling later:** the `/ask` endpoint doesn't know or care that there's a multi-round tool-calling loop, a FAISS index, or three separate tools underneath it — it just calls `ask()` and returns the result. If the orchestration logic changes completely later, the API layer doesn't have to.

**What I built:**
- `AskRequest` / `AskResponse` — Pydantic models with field validation on the question
- `lifespan` — eager loading of the dataframe, FAISS index, and snippets at server startup instead of on first request
- `POST /ask` — calls the orchestrator, wraps the answer, and returns a clean generic error (not a raw exception) if anything fails
- `GET /health` — a lightweight endpoint separate from the expensive path, so the service's liveness can be checked without triggering a full LLM round trip






## Scaling HealthInsight: Exploring New Search Techniques

As a next step beyond the initial 5 phases, this project is being scaled to explore
different retrieval strategies within the RAG pipeline. The first technique implemented is **hybrid search**.

### Why hybrid search

The original retrieval path used only dense vector search (FAISS `IndexFlatL2`
over MiniLM embeddings). Vector search is strong at matching *meaning* — a query
like "patients with heart problems" will correctly surface records mentioning
cardiac conditions even without an exact word match. But it's weaker at *exact*
terms: patient IDs, precise medical condition names, or hospital names can rank
poorly if their embedding isn't "semantically close" to the query, even when the
literal word is right there in the text.

Hybrid search fixes this by running two retrieval methods in parallel and merging
the results:

- **Dense retrieval (FAISS)** — unchanged from before, finds semantically similar snippets.
- **Sparse retrieval (BM25)** — a keyword-based ranking algorithm (via `rank_bm25`)
  that scores documents on exact term overlap with the query.

The two ranked lists are combined using **Reciprocal Rank Fusion (RRF)**, which
merges them by rank position rather than raw score 

### How it works now

`src/embeddings.py` now exposes:
- `build_bm25_index(df)` — builds a BM25 index over the same snippets used for FAISS
- `embed_query(query_text)` — embeds a live user query
- `reciprocal_rank_fusion(...)` — merges FAISS and BM25 ranked results
- `hybrid_search(query_text, faiss_index, bm25_index, df, k, candidates)` — the full
  pipeline: embed + search both indexes, fuse, return top-k matching rows

The BM25 index is rebuilt from `snippets.csv` at startup rather than saved to disk,
since rebuilding it is cheap even at ~55,500 rows.

### What this means for the bot

Once wired into the orchestrator's `search_patient_records` tool, the assistant will
be able to correctly retrieve records for exact-term queries (patient IDs, specific
condition names) that pure vector search could previously miss, while keeping the
existing semantic search strength for descriptive, natural-language questions.

### Wiring into Orchestrator (`src/orchestrator.py`)

The orchestrator has been updated to integrate the hybrid retrieval pipeline into the Groq tool-calling loop:

- **Startup & Resource Management:** `_load_resources()` now lazy-loads the FAISS index, raw dataset, and snippets DataFrame, while building the `BM25Okapi` index in memory on startup via `build_bm25_index(_snippets_df)`.
- **Hybrid Tool Execution:** `search_patient_records` was upgraded from pure FAISS search to call `hybrid_search(...)`, querying both FAISS and BM25 simultaneously and returning fused RRF candidate records.


**Status:** Core hybrid search is fully wired and integrated into `orchestrator.py`.