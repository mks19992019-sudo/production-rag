from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from app.graph import build_graph, compile
from app.state import State
from langchain_redis import RedisChatMessageHistory
import os

app = FastAPI(title="Agentic RAG API", version="0.1.0")

# Pydantic models for request/response
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    context: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict

# Global variables for the compiled graph
compiled_app = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global compiled_app
    try:
        # Build the graph
        graph = build_graph(State)
        # Compile with checkpointer (using in-memory for simplicity)
        compiled_app = compile(graph, checkpointer=None)
        print("RAG system initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        # Don't raise here to allow health checks to still work

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {
        "api": "running",
        "rag_system": "initialized" if compiled_app else "not initialized"
    }

    # Check if we can connect to Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
        client.get_collections()
        services["qdrant"] = "connected"
    except Exception as e:
        services["qdrant"] = f"disconnected: {str(e)}"

    return HealthResponse(
        status="healthy" if compiled_app else "degraded",
        version="0.1.0",
        services=services
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint that processes user queries through the RAG system"""
    if not compiled_app:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        # Prepare the initial state
        initial_state = State(
            msg=request.query,
            context="",
            ans=""
        )

        # Run the graph
        result = compiled_app.invoke(initial_state)

        # Extract the answer from the result
        answer = result.get("ans", "I couldn't generate an answer.")
        context = result.get("context", "")

        return ChatResponse(
            answer=answer,
            context=context if context else None
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)