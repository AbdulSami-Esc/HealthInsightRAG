### Baseline Observation: Candidate Pool Boundary
- Patient ID: P03982
- Total Chunks: 21 (Visits: 1, 2, 3)
- Architectural Parameter: CANDIDATE_POOL_SIZE = 20 in orchestrator.py
- Finding: Patient chunk count (21) exceeds prefetch limit (20). 
  At least 1 chunk is permanently truncated prior to Cross-Encoder reranking.


### Stress Test Subject Profile
- Patient ID: P03982
- Chunks: Visit 1 (16 chunks), Visit 2 (1 chunk), Visit 3 (1 chunk)
- Ground Truth Progression:
  - Visit 1: Baseline erythema nodosum history
  - Visit 2: Improvement (itching 50% better)
  - Visit 3: Relapse/progression (larger lesions) + Acute injury (twisted ankle)
- Hypothesis: Because Visit 1 dominates the chunk distribution (16/18 transcription chunks), top_k=5 reranking will drop Visit 2 and Visit 3, causing the LLM to miss the relapse and ankle injury.



### Key Discovery: Agent-Driven Context Expansion
- At fixed top_k=5, Visit 2 was completely dropped by Cross-Encoder reranking due to Visit 1 chunk dominance (16 chunks).
- In the ReAct loop, the LLM autonomously expanded the retrieval request to top_k=10 to cover the progression query.
- Outcome: Visit 2 was recovered, but at the cost of doubling the prompt token size (5 chunks -> 10 chunks).
- Architectural Implication: Confirms the necessity of Phase 2 (Prompt Management & Context Windowing). Without explicit windowing, the agent inflates top_k to compensate for chunk imbalance, hitting token ceilings and Qdrant candidate pool limits (CANDIDATE_POOL_SIZE=20).