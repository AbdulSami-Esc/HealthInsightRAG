from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from orchestrator import _load_resources, ask


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="The clinical inquiry or medical knowledge question.",
        examples=["What was the treatment plan for the bariatric surgery patient?"]
    )


class AskResponse(BaseModel):
    question: str
    answer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Pre-loading unstructured clinical FAISS and BM25 indexes...")
    _load_resources()
    print("Clinical search resources successfully cached in memory.")
    
    yield
    
    print("Shutting down clinical intelligence service...")


app = FastAPI(
    title="HealthInsight Clinical Assistant API",
    description="Production endpoint serving hybrid search over unstructured clinical transcriptions.",
    version="2.0.0",
    lifespan=lifespan
)


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    try:
        answer_text = ask(request.question)
        return AskResponse(
            question=request.question,
            answer=answer_text
        )
    except Exception as e:
        print(f"Server execution error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the clinical query."
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline": "unstructured-clinical-hybrid"
    }