from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import the loading and the ask function from orchestrator
from orchestrator import _load_resources, ask

class AskRequest(BaseModel):
    question: str = Field(
        ..., 
        min_length=1,# Declaring field to be of minimum 1 length 
        description="The medical data question you want to ask."
    )

class AskResponse(BaseModel):
    question: str
    answer: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading resources...")
    _load_resources()
    print("Resources loaded. Application is ready.")
    
    yield  # Application runs while yielded
    
    # Shutdown logic (if any cleanup were needed, it would go here)
    print("Shutting down application...")

app = FastAPI(lifespan=lifespan, title="Healthcare Data AI API")

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    try:
        # Call your orchestrator
        answer_text = ask(request.question)
        
        # Echo back the question along with the generated answer
        return AskResponse(
            question=request.question,
            answer=answer_text
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail="An internal service error occurred while processing your request. Please try again later."
        )

@app.get("/health")
async def health_check():
    return {"status": "ok"}