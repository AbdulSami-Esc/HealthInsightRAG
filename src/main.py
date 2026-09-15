import sys
import os
from pathlib import Path

# Dynamically find the root directory (HealthInsightRAG) and force Python to see it
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from groq import Groq

from src.orchestrator import (  # Note: you can now safely use src.orchestrator if it's in the src folder
    _load_resources, 
    ask, 
    search_clinical_cohorts, 
    answer_analytical_query
)
from src.reranker import load_reranker_model

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL_NAME = "openai/gpt-oss-120b"
ROUTER_MODEL = "llama3-8b-8192"

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

class AskResponse(BaseModel):
    question: str
    answer: str
    route_taken: str 
    extracted_id: str | None = None

def route_query(question: str) -> dict:
    """Analyzes the query and returns a JSON routing command."""
    system_prompt = """You are a medical API traffic router. Analyze the user's query and output a strict JSON object with exactly two keys: 'intent' and 'patient_id'.
    
    The 'intent' must be exactly one of these strings:
    - "GENERAL": Asking basic medical definitions, symptoms, or textbook questions.
    - "SINGLE_PATIENT": Asking about a specific patient's history, treatment, or records.
    - "COHORT": Asking to find clinical records or patients matching a certain disease or procedure.
    - "ANALYTICAL": Asking for counts, averages, math, or statistics (e.g., "how many patients").
    
    The 'patient_id' must be the exact alphanumeric ID (e.g., "P0012") if mentioned, otherwise null.
    """
    
    response = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0,
        response_format={"type": "json_object"} # Forces the LLM to return valid JSON
    )
    
    return json.loads(response.choices[0].message.content)


def ask_general_medical(question: str) -> str:
    """Lane 1: Direct, fast LLM generation."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant. Answer general questions accurately and concisely."},
            {"role": "user", "content": question}
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def ask_cohort_synthesis(question: str) -> str:
    """Lane 3: Synthesizes the cohort retrieval into a safe, structured table."""
    raw_results = search_clinical_cohorts(question)
    
    synthesis_prompt = (
        f"Original question: {question}\n\n"
        f"Retrieved Records:\n{chr(10).join(raw_results['results'])}\n\n"
        f"Synthesize this data into a structured Markdown table. You MUST cite the 'Patient ID' for every finding."
    )
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a clinical data synthesizer. Never mix up patient records."},
            {"role": "user", "content": synthesis_prompt}
        ],
        temperature=0,
    )
    return response.choices[0].message.content


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading API resources...")
    _load_resources()
    load_reranker_model()
    yield
    print("Shutting down...")

app = FastAPI(title="HealthInsight 4-Lane API", lifespan=lifespan)

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    try:
        # 1. The JSON Router makes the decision
        routing_data = route_query(request.question)
        intent = routing_data.get("intent", "GENERAL")
        patient_id = routing_data.get("patient_id")
        
        # 2. Execute the correct lane
        if intent == "GENERAL":
            answer_text = ask_general_medical(request.question)
            
        elif intent == "SINGLE_PATIENT":
            # If the user forgot to provide an ID but asked a single patient question, fall back to general
            if not patient_id:
                answer_text = "Error: Please provide a specific Patient ID (e.g., P0012) to access secure records."
            else:
                answer_text = ask(request.question) # Your original ReAct loop
                
        elif intent == "COHORT":
            answer_text = ask_cohort_synthesis(request.question)
            
        elif intent == "ANALYTICAL":
            answer_text = answer_analytical_query(request.question)
            
        else:
            answer_text = ask_general_medical(request.question)

        return AskResponse(
            question=request.question, 
            answer=answer_text,
            route_taken=intent,
            extracted_id=patient_id
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error.")