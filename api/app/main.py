"""
Minimal Skeleton only
"""
import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
WEAVIATE_HTTP_PORT = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
WEAVIATE_URL = f"http://weaviate:8080"  # service name on the compose network
LLM_MODEL = os.environ["LLM_MODEL"]

app = FastAPI(title="Blaai API")


class Query(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps():
    """Confirm the API can actually reach its two dependencies."""
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Weaviate readiness
        try:
            r = await client.get(f"{WEAVIATE_URL}/v1/.well-known/ready")
            results["weaviate"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as e:
            results["weaviate"] = f"unreachable: {e.__class__.__name__}"

        # Ollama tags endpoint
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            results["ollama"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as e:
            results["ollama"] = f"unreachable: {e.__class__.__name__}"

    return results


@app.post("/ask")
async def ask(query: Query):
    """
    STUB. To build - RAG Flow
      1. Embed query.question via Ollama (EMBED_MODEL)
      2. Hybrid search Weaviate (vector + BM25)
      3. Build a grounded prompt from retrieved chunks
      4. Stream the LLM response back
    """
    return {"received": query.question, "answer": "not implemented yet"}