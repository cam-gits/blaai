import os
import httpx
import weaviate
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from app import retrieval
from config import DISTANCE_CUTOFF

OLLAMA_BASE_URL = os.environ["OLLAMA_BASE_URL"]
WEAVIATE_HTTP_PORT = os.environ.get("WEAVIATE_HTTP_PORT", "8080")
WEAVIATE_HOST = os.environ["WEAVIATE_HOST"]
WEAVIATE_URL = f"http://{WEAVIATE_HOST}:8080"
LLM_MODEL = os.environ["LLM_MODEL"]

COLLECTION_NAME = "blaai_collection"

@asynccontextmanager
async def lifespan(app: FastAPI):
    #Open one Weaviate connection for the life of the process
    app.state.client = weaviate.connect_to_local(host=WEAVIATE_HOST)
    app.state.collection = app.state.client.collections.get(COLLECTION_NAME)
    try:
        yield
    finally:
        app.state.client.close()

app = FastAPI(title="Blaai API", lifespan=lifespan)

class Query(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps():
    #Confirm the API can reach its dependencies
    results = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        #Weaviate readiness
        try:
            r = await client.get(f"{WEAVIATE_URL}/v1/.well-known/ready")
            results["weaviate"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as e:
            results["weaviate"] = f"unreachable: {e.__class__.__name__}"

        #Ollama tags endpoint
        try:
            r = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            results["ollama"] = "ok" if r.status_code == 200 else f"status {r.status_code}"
        except Exception as e:
            results["ollama"] = f"unreachable: {e.__class__.__name__}"

    return results


@app.post("/ask")
def ask(query: Query):
    """
    To do:
      3. Build a grounded prompt from retrieved chunks
      4. Stream the LLM response back
    """
    chunks, top_distance = retrieval.search_db(app.state.collection, query.question)
    top_url = chunks[0]["url"] if (chunks and top_distance is not None and top_distance < DISTANCE_CUTOFF) else "None"

    return {"received": query.question, "answer": top_url}