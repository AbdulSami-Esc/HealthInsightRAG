# HealthInsight

A multi-intent Retrieval-Augmented Generation (RAG) system for querying clinical 
transcription records — built to deeply understand production RAG architecture

## Problem Statement

Clinical transcription records aren't uniform prose. Each patient record is 
divided into distinct sections — Medical Specialty, Transcription, Sample Name, 
Keywords — each marked by an all-caps header ending in a colon or slash 
(e.g. `HISTORY OF PRESENT ILLNESS:`).

Naive fixed-size chunking ignores this structure. If a chunk boundary falls 
mid-sentence or mid-thought, the retrieved context is incomplete — and an LLM 
given an incomplete fragment is far more likely to hallucinate the missing 
piece rather than admit it's missing.

HealthInsight instead chunks **by section first**. Only if an individual 
section exceeds a maximum token size (measured in tokens, not words or 
characters — to make full use of the embedding model's context window) is it 
further split, using a fallback strategy that overlaps a small window of 
tokens between the resulting sub-chunks so no clinical detail is lost at 
the split point.